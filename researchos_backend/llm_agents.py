from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pydantic import ValidationError

from .agents import SubmissionReviewAgent, TopicToRoadmapAgent, UnderstandingGateAgent, clamp
from .modal_llm import ModalOpenAIChat
from .schemas import (
    EvidenceRequirement,
    LearningResearchPackage,
    ProjectState,
    ResearchRoadmap,
    ReviewDecision,
    ReviewFinding,
    ReviewReport,
    StudentProfile,
    StudentSubmission,
    UnderstandingAnswer,
    UnderstandingReport,
)


JSON_ONLY_SYSTEM = """You are a backend agent in a research-training supervision system.
Return valid JSON only. Do not use Markdown fences. Do not add prose before or after JSON.
All decisions are recommendations for mentor/PI review; do not claim final scientific authority."""


def _as_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_as_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _as_jsonable(v) for k, v in obj.items()}
    return obj


def _safe_list(value: Any, fallback: List[str], max_items: int = 8) -> List[str]:
    if not isinstance(value, list):
        return fallback
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text[:500])
    return out[:max_items] or fallback


def _safe_text(value: Any, fallback: str, max_chars: int = 600) -> str:
    if not isinstance(value, str):
        return fallback
    text = value.strip()
    return text[:max_chars] if text else fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return clamp(float(value))
    except Exception:
        return fallback


def _safe_decision(value: Any, fallback: ReviewDecision = ReviewDecision.revise) -> ReviewDecision:
    try:
        return ReviewDecision(str(value).strip().lower())
    except Exception:
        return fallback


class LLMTopicToRoadmapAgent:
    """Hybrid planner: deterministic schema + LLM topic-specific customization.

    Full roadmap JSON is large and easy for smaller models to break. For a stable demo,
    this agent first builds a valid roadmap with TopicToRoadmapAgent, then asks the Modal
    endpoint for topic-specific risks, decision points, and package customization.
    """

    def __init__(self, llm: ModalOpenAIChat, *, fallback_on_error: bool = True) -> None:
        self.llm = llm
        self.fallback_on_error = fallback_on_error
        self.fallback_agent = TopicToRoadmapAgent()

    def run(
        self,
        project_id: str,
        topic: str,
        target_standard: str,
        students: List[StudentProfile],
        constraints: str = "",
    ) -> ResearchRoadmap:
        base = self.fallback_agent.run(project_id, topic, target_standard, students, constraints)
        try:
            payload = {
                "topic": topic,
                "target_standard": target_standard,
                "constraints": constraints,
                "students": _as_jsonable(students),
                "existing_package_ids": [p.package_id for p in base.packages],
                "roadmap_schema_note": "Do not change IDs. Only customize content fields.",
                "expected_json_schema": {
                    "topic_summary": "string",
                    "candidate_contribution_angles": ["string"],
                    "novelty_risks": ["string"],
                    "pi_decision_points": ["string"],
                    "package_customizations": {
                        "P1": {
                            "title": "string",
                            "learning_objectives": ["string"],
                            "research_questions": ["string"],
                            "background_reading": ["string"],
                            "student_instructions": ["string"],
                            "expected_outputs": ["string"],
                            "acceptance_criteria": ["string"],
                            "understanding_questions": ["string"],
                        }
                    },
                },
            }
            user = """Create topic-specific customization for a research training roadmap.
The system goal is twofold:
1) teach students systematic research practice;
2) reduce PI workload while maintaining publication-oriented novelty/evidence standards.

Return JSON matching expected_json_schema. Keep each list concise. Use package IDs P1-P6 exactly.

Input:
""" + json.dumps(payload, ensure_ascii=False, indent=2)
            data = self.llm.complete_json(JSON_ONLY_SYSTEM, user, temperature=0.2, max_tokens=4096)
            if not isinstance(data, dict):
                raise ValueError("Planner JSON must be an object")
            return self._merge_customization(base, data)
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            base.novelty_risks.append(f"LLM planner fallback used because endpoint output failed: {type(exc).__name__}: {str(exc)[:200]}")
            base.pi_decision_points.append("Check LLM planner logs before trusting topic-specific customization.")
            return base

    def _merge_customization(self, base: ResearchRoadmap, data: Mapping[str, Any]) -> ResearchRoadmap:
        topic_summary = _safe_text(data.get("topic_summary"), "")
        angles = _safe_list(data.get("candidate_contribution_angles"), [], max_items=5)
        if topic_summary:
            base.charter += f" LLM topic summary: {topic_summary}"
        if angles:
            base.charter += " Candidate contribution angles: " + "; ".join(angles) + "."
        base.novelty_risks = _safe_list(data.get("novelty_risks"), base.novelty_risks, max_items=8)
        base.pi_decision_points = _safe_list(data.get("pi_decision_points"), base.pi_decision_points, max_items=8)

        custom = data.get("package_customizations")
        if isinstance(custom, dict):
            for package in base.packages:
                patch = custom.get(package.package_id)
                if not isinstance(patch, dict):
                    continue
                package.title = _safe_text(patch.get("title"), package.title, max_chars=180)
                package.learning_objectives = _safe_list(patch.get("learning_objectives"), package.learning_objectives, max_items=6)
                package.research_questions = _safe_list(patch.get("research_questions"), package.research_questions, max_items=6)
                package.background_reading = _safe_list(patch.get("background_reading"), package.background_reading, max_items=8)
                package.student_instructions = _safe_list(patch.get("student_instructions"), package.student_instructions, max_items=8)
                package.expected_outputs = _safe_list(patch.get("expected_outputs"), package.expected_outputs, max_items=8)
                package.acceptance_criteria = _safe_list(patch.get("acceptance_criteria"), package.acceptance_criteria, max_items=8)
                package.understanding_questions = _safe_list(patch.get("understanding_questions"), package.understanding_questions, max_items=8)
        return base


class LLMSubmissionReviewAgent:
    """Review a student submission using the Modal endpoint with deterministic fallback."""

    def __init__(self, llm: ModalOpenAIChat, *, fallback_on_error: bool = True) -> None:
        self.llm = llm
        self.fallback_on_error = fallback_on_error
        self.fallback_agent = SubmissionReviewAgent()

    def run(self, state: ProjectState, submission: StudentSubmission) -> ReviewReport:
        package = state.get_package(submission.package_id)
        deterministic = self.fallback_agent.run(state, submission)
        try:
            context = {
                "package": _as_jsonable(package),
                "submission": {
                    "submission_id": submission.submission_id,
                    "student_id": submission.student_id,
                    "artifact_paths": submission.artifact_paths,
                    "report_text": submission.report_text[:6000],
                },
                "project_claims": _as_jsonable(state.roadmap.claims if state.roadmap else []),
                "evidence_requirements": _as_jsonable(state.roadmap.evidence_requirements if state.roadmap else []),
                "expected_json_schema": {
                    "decision": "accept|revise|reject",
                    "scores": {
                        "format": "float 0..1",
                        "reproducibility": "float 0..1",
                        "scientific": "float 0..1",
                        "evidence_alignment": "float 0..1",
                    },
                    "findings": [
                        {
                            "area": "format|reproducibility|scientific_validity|evidence_alignment|understanding|other",
                            "severity": "low|medium|high",
                            "message": "string",
                            "recommendation": "string",
                        }
                    ],
                    "generated_understanding_questions": ["string"],
                    "recommended_next_action": "string",
                },
            }
            user = """Review this student research submission.
Prioritize false-accept prevention: weak artifacts should be revised, not accepted.
Assess whether the submission follows the package, links outputs to parent claims/evidence, records reproducibility details, and avoids overclaiming.
Return JSON only.

Input:
""" + json.dumps(context, ensure_ascii=False, indent=2)
            data = self.llm.complete_json(JSON_ONLY_SYSTEM, user, temperature=0.1, max_tokens=4096)
            if not isinstance(data, dict):
                raise ValueError("Review JSON must be an object")
            return self._to_review_report(data, deterministic, submission.submission_id, package.package_id)
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            deterministic.findings.append(
                ReviewFinding(
                    area="llm_fallback",
                    severity="medium",
                    message=f"LLM review fallback used because endpoint output failed: {type(exc).__name__}: {str(exc)[:200]}",
                    recommendation="Inspect endpoint response/logs; deterministic review was used for continuity.",
                )
            )
            return deterministic

    def _to_review_report(
        self,
        data: Mapping[str, Any],
        fallback: ReviewReport,
        submission_id: str,
        package_id: str,
    ) -> ReviewReport:
        scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        decision = _safe_decision(data.get("decision"), fallback.decision)
        findings: List[ReviewFinding] = []
        raw_findings = data.get("findings")
        if isinstance(raw_findings, list):
            for item in raw_findings[:8]:
                if not isinstance(item, dict):
                    continue
                try:
                    findings.append(
                        ReviewFinding(
                            area=_safe_text(item.get("area"), "other", max_chars=80),
                            severity=_safe_text(item.get("severity"), "medium", max_chars=20),
                            message=_safe_text(item.get("message"), "LLM finding", max_chars=600),
                            recommendation=_safe_text(item.get("recommendation"), "Revise according to reviewer feedback.", max_chars=600),
                        )
                    )
                except ValidationError:
                    continue
        if not findings:
            findings = fallback.findings

        questions = _safe_list(
            data.get("generated_understanding_questions"),
            fallback.generated_understanding_questions,
            max_items=6,
        )

        return ReviewReport(
            submission_id=submission_id,
            package_id=package_id,
            decision=decision,
            score_format=round(_safe_float(scores.get("format"), fallback.score_format), 3),
            score_reproducibility=round(_safe_float(scores.get("reproducibility"), fallback.score_reproducibility), 3),
            score_scientific=round(_safe_float(scores.get("scientific"), fallback.score_scientific), 3),
            score_evidence_alignment=round(_safe_float(scores.get("evidence_alignment"), fallback.score_evidence_alignment), 3),
            findings=findings,
            generated_understanding_questions=questions,
            recommended_next_action=_safe_text(data.get("recommended_next_action"), fallback.recommended_next_action, max_chars=700),
        )


class LLMUnderstandingGateAgent:
    """Grade student understanding with the Modal endpoint and fallback heuristics."""

    def __init__(self, llm: ModalOpenAIChat, *, fallback_on_error: bool = True) -> None:
        self.llm = llm
        self.fallback_on_error = fallback_on_error
        self.fallback_agent = UnderstandingGateAgent()

    def run(
        self,
        state: ProjectState,
        review: ReviewReport,
        student_id: str,
        answers: List[UnderstandingAnswer],
    ) -> UnderstandingReport:
        fallback = self.fallback_agent.run(state, review, student_id, answers)
        package = state.get_package(review.package_id)
        try:
            context = {
                "package": _as_jsonable(package),
                "review": _as_jsonable(review),
                "student_answers": _as_jsonable(answers),
                "expected_json_schema": {
                    "decision": "accept|revise|reject",
                    "score": "float 0..1",
                    "feedback": "string",
                    "weak_points": ["string"],
                    "passed": "boolean",
                },
            }
            user = """Evaluate whether the student understands the research task.
Do not grade only correctness of prose. Check whether the student can explain:
- parent claim and evidence requirement;
- what artifact they produced;
- why the artifact matters for research/paper-readiness;
- limitations, failure modes, or reproducibility risks.
Return JSON only.

Input:
""" + json.dumps(context, ensure_ascii=False, indent=2)
            data = self.llm.complete_json(JSON_ONLY_SYSTEM, user, temperature=0.1, max_tokens=2048)
            if not isinstance(data, dict):
                raise ValueError("Understanding JSON must be an object")
            decision = _safe_decision(data.get("decision"), fallback.decision)
            score = round(_safe_float(data.get("score"), fallback.score), 3)
            passed_raw = data.get("passed")
            if isinstance(passed_raw, bool):
                passed = passed_raw
            else:
                passed = decision == ReviewDecision.accept and score >= 0.55
            # Never pass if the review itself rejected the artifact.
            if review.decision == ReviewDecision.reject:
                passed = False
                if decision == ReviewDecision.accept:
                    decision = ReviewDecision.revise
            return UnderstandingReport(
                package_id=package.package_id,
                student_id=student_id,
                review_id=review.review_id,
                decision=decision,
                score=score,
                feedback=_safe_text(data.get("feedback"), fallback.feedback, max_chars=900),
                weak_points=_safe_list(data.get("weak_points"), fallback.weak_points, max_items=8),
                passed=passed,
            )
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            fallback.weak_points.append(f"LLM understanding fallback used: {type(exc).__name__}: {str(exc)[:200]}")
            return fallback
