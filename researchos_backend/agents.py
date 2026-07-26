from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .schemas import (
    ArtifactRecord,
    ArtifactStatus,
    ClaimEvidenceRow,
    Criticality,
    Difficulty,
    EvidenceRequirement,
    LearningResearchPackage,
    PIDashboard,
    PackageStatus,
    ProjectState,
    ResearchClaim,
    ResearchRoadmap,
    ReviewDecision,
    ReviewFinding,
    ReviewReport,
    RoadmapStage,
    StudentLevel,
    StudentProfile,
    StudentSubmission,
    UnderstandingAnswer,
    UnderstandingReport,
)


REPORT_TEMPLATE = [
    "Task summary",
    "Parent research question and claim",
    "Method or procedure",
    "Output artifacts",
    "Result interpretation",
    "Limitations and confounders",
    "Reproducibility commands",
    "Next step recommendation",
]


LEVEL_ORDER = {
    StudentLevel.junior: 0,
    StudentLevel.intermediate: 1,
    StudentLevel.strong: 2,
    StudentLevel.senior: 3,
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", normalize_text(text)))


def score_keywords(text: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    normalized = normalize_text(text)
    hits = 0
    for keyword in keywords:
        words = normalize_text(keyword).split()
        if any(word in normalized for word in words if len(word) >= 4):
            hits += 1
    return hits / max(1, len(keywords))


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def pick_student(students: List[StudentProfile], required: StudentLevel) -> str | None:
    if not students:
        return None
    required_rank = LEVEL_ORDER[required]
    eligible = [s for s in students if LEVEL_ORDER[s.level] >= required_rank]
    if not eligible:
        eligible = students
    # Prefer higher availability, then lower level that still satisfies the task.
    eligible = sorted(eligible, key=lambda s: (-s.availability_hours_per_week, LEVEL_ORDER[s.level]))
    return eligible[0].student_id


class TopicToRoadmapAgent:
    """Deterministic first-pass planner.

    The goal is not to be clever. The goal is to create a stable, inspectable
    graph that shows the backend workflow before adding live retrieval or LLMs.
    """

    def run(
        self,
        project_id: str,
        topic: str,
        target_standard: str,
        students: List[StudentProfile],
        constraints: str = "",
    ) -> ResearchRoadmap:
        topic_short = topic.strip().rstrip(".")
        charter = (
            f"PI topic: {topic_short}. Target: {target_standard}. "
            "The system will train students through a structured research pipeline while reducing PI workload. "
            "It must maintain publication-oriented standards: prior-art defense, novelty risk checks, claim-evidence alignment, reproducibility, and PI gates."
        )
        if constraints.strip():
            charter += f" Constraints: {constraints.strip()}"

        claims = [
            ResearchClaim(
                claim_id="C1",
                text=f"The topic '{topic_short}' can be decomposed into a defensible research roadmap after prior-art and gap analysis.",
                claim_type="roadmap",
                novelty_risk="medium",
            ),
            ResearchClaim(
                claim_id="C2",
                text="A structured claim-evidence graph reduces orphan tasks and makes student work traceable to research goals.",
                claim_type="supervision",
                novelty_risk="low",
            ),
            ResearchClaim(
                claim_id="C3",
                text="Learning-and-research packages improve student understanding by linking tasks to readings, research questions, templates, rubrics, and understanding checks.",
                claim_type="education",
                novelty_risk="medium",
            ),
            ResearchClaim(
                claim_id="C4",
                text="Publication-oriented evidence requirements can turn a broad topic into testable artifacts such as baselines, ablations, failure analysis, and paper-readiness reports.",
                claim_type="paper_readiness",
                novelty_risk="medium-high",
            ),
        ]

        evidence = [
            EvidenceRequirement(
                evidence_id="E1",
                claim_id="C1",
                description="Literature map, closest prior-art list, gap matrix, and reviewer-objection memo.",
                required_artifact_types=["paper_cards", "prior_art_matrix", "gap_cards", "reviewer_attack_memo"],
                success_criteria=[
                    "Dangerous prior work is explicit",
                    "Gap statements are specific and not marketing claims",
                    "PI can approve, mutate, or reject candidate directions",
                ],
            ),
            EvidenceRequirement(
                evidence_id="E2",
                claim_id="C2",
                description="Claim-evidence-task graph with no orphan packages and clear stage dependencies.",
                required_artifact_types=["claim_graph", "stage_plan", "package_plan"],
                success_criteria=[
                    "Every package links to a parent claim or evidence requirement",
                    "Dependency order is clear",
                    "PI dashboard can show missing evidence",
                ],
            ),
            EvidenceRequirement(
                evidence_id="E3",
                claim_id="C3",
                description="Learning-and-research packages with objectives, research questions, templates, rubrics, and understanding questions.",
                required_artifact_types=["learning_package", "rubric", "understanding_check"],
                success_criteria=[
                    "Student can explain why the task matters",
                    "Submission follows a standard report structure",
                    "Understanding gate detects weak explanations",
                ],
            ),
            EvidenceRequirement(
                evidence_id="E4",
                claim_id="C4",
                description="Experiment, benchmark, method, or analysis artifacts that are reproducible and linked to claims.",
                required_artifact_types=["experiment_plan", "results_table", "ablation", "failure_analysis", "readiness_report"],
                success_criteria=[
                    "Commands, configs, and seeds are recorded",
                    "Baselines are fair enough for mentor review",
                    "Limitations and missing evidence are explicit",
                ],
            ),
        ]

        packages = [
            LearningResearchPackage(
                package_id="P1",
                title="Build literature map and dangerous prior-art matrix",
                stage_id="S1",
                stage_name="Topic and prior-art gate",
                assigned_to=pick_student(students, StudentLevel.junior),
                owner_level_required=StudentLevel.junior,
                difficulty=Difficulty.D1_reproduction,
                criticality=Criticality.C2_validity,
                parent_claims=["C1"],
                evidence_requirements=["E1"],
                learning_objectives=[
                    "Learn how to read papers systematically",
                    "Distinguish problem, method, data, metric, claim, and limitation",
                    "Identify papers that could make the project look incremental",
                ],
                research_questions=[
                    "What has already been solved in this topic?",
                    "Which papers would a reviewer cite against this project?",
                    "Which gap statements are concrete enough to investigate?",
                ],
                background_reading=["Seed papers from PI", "Recent surveys", "Top venue papers in the last 3 years"],
                student_instructions=[
                    "Create 10-20 paper cards.",
                    "For each card, extract problem, method, data, metric, claims, and limitations.",
                    "Rank the top dangerous prior-art papers and explain why they are dangerous.",
                    "Propose 3-5 gap cards with evidence from the literature map.",
                ],
                report_template=REPORT_TEMPLATE,
                expected_outputs=["paper_cards.md", "dangerous_prior_art_matrix.csv", "gap_cards.md"],
                acceptance_criteria=[
                    "No fake citations or unsupported claims",
                    "Each paper card includes limitation and relation to our topic",
                    "Top dangerous prior-art list is specific",
                    "Gap cards cite supporting papers",
                ],
                understanding_questions=[
                    "What makes a paper dangerous prior art for this project?",
                    "Which proposed gap is weakest and why?",
                    "How does this package help the PI decide whether the topic is worth pursuing?",
                ],
                status=PackageStatus.assigned,
            ),
            LearningResearchPackage(
                package_id="P2",
                title="Create research question, claim, and evidence matrix",
                stage_id="S2",
                stage_name="Claim-evidence graph",
                assigned_to=pick_student(students, StudentLevel.intermediate),
                owner_level_required=StudentLevel.intermediate,
                difficulty=Difficulty.D3_scientific_analysis,
                criticality=Criticality.C3_main_claim,
                parent_claims=["C1", "C2"],
                evidence_requirements=["E2"],
                learning_objectives=[
                    "Learn how to convert a broad topic into research questions and testable claims",
                    "Understand the difference between a task, a claim, and evidence",
                    "Learn to detect orphan tasks and missing evidence",
                ],
                research_questions=[
                    "What are the main scientific claims the project could make?",
                    "What evidence is required before each claim can be trusted?",
                    "Which claim should not be assigned to students until PI approval?",
                ],
                background_reading=["Gap cards from P1", "PI notes", "Artifact review checklist"],
                student_instructions=[
                    "Convert approved gaps into research questions.",
                    "Write candidate claims and map each claim to evidence requirements.",
                    "Mark each claim as low, medium, or high novelty risk.",
                ],
                report_template=REPORT_TEMPLATE,
                expected_outputs=["research_questions.md", "claim_evidence_matrix.csv", "risk_notes.md"],
                acceptance_criteria=[
                    "Every claim has at least one evidence requirement",
                    "Risk notes mention prior-art objections",
                    "No task is proposed without a parent claim",
                ],
                understanding_questions=[
                    "Why is a claim not the same as a task?",
                    "Which evidence requirement is most critical for paper-readiness?",
                    "What would make a claim unsupported?",
                ],
                status=PackageStatus.assigned,
            ),
            LearningResearchPackage(
                package_id="P3",
                title="Design method, benchmark, or evaluation protocol",
                stage_id="S3",
                stage_name="Contribution design",
                assigned_to=pick_student(students, StudentLevel.strong),
                owner_level_required=StudentLevel.strong,
                difficulty=Difficulty.D4_core_method_or_benchmark,
                criticality=Criticality.C4_core_contribution,
                parent_claims=["C4"],
                evidence_requirements=["E4"],
                learning_objectives=[
                    "Learn how to design a contribution that can be evaluated",
                    "Translate novelty into required evidence and baselines",
                    "Anticipate reviewer objections before implementation",
                ],
                research_questions=[
                    "What exact contribution is being evaluated?",
                    "Which baselines and ablations are needed to defend the claim?",
                    "How can shortcut solving, leakage, or overclaiming be detected?",
                ],
                background_reading=["Claim-evidence matrix from P2", "Closest method/benchmark papers", "Evaluation guidelines"],
                student_instructions=[
                    "Write a design document for the candidate contribution.",
                    "Specify inputs, outputs, metrics, baselines, ablations, and failure cases.",
                    "List PI decision points before implementation begins.",
                ],
                report_template=REPORT_TEMPLATE,
                expected_outputs=["design_doc.md", "baseline_plan.md", "ablation_plan.md", "failure_risk.md"],
                acceptance_criteria=[
                    "Design maps directly to C4 and E4",
                    "Baselines are fair and executable",
                    "Failure modes and limitations are explicit",
                    "PI approval required before implementation",
                ],
                understanding_questions=[
                    "Why is this design potentially publishable rather than a simple engineering task?",
                    "Which reviewer objection is most dangerous?",
                    "What evidence would falsify the main claim?",
                ],
                status=PackageStatus.assigned,
            ),
            LearningResearchPackage(
                package_id="P4",
                title="Implement reproducible prototype or baseline run",
                stage_id="S4",
                stage_name="Implementation and reproducibility",
                assigned_to=pick_student(students, StudentLevel.intermediate),
                owner_level_required=StudentLevel.intermediate,
                difficulty=Difficulty.D2_engineering,
                criticality=Criticality.C2_validity,
                parent_claims=["C4"],
                evidence_requirements=["E4"],
                learning_objectives=[
                    "Learn how to produce reproducible research artifacts",
                    "Record configs, seeds, versions, commands, and metrics",
                    "Separate implementation success from scientific evidence strength",
                ],
                research_questions=[
                    "Can the protocol be executed by another student?",
                    "Do the logs and result files support the reported numbers?",
                    "Which result should not be interpreted as final evidence yet?",
                ],
                background_reading=["Design doc from P3", "Code completeness checklist", "Experiment protocol"],
                student_instructions=[
                    "Run the baseline or prototype according to the protocol.",
                    "Record commands, environment, seed, config, and result files.",
                    "Do not change metrics after seeing results without documenting the change.",
                ],
                report_template=REPORT_TEMPLATE,
                expected_outputs=["run_commands.md", "config.yaml", "results.csv", "logs/"],
                acceptance_criteria=[
                    "One-command reproduction path is documented",
                    "Result table is machine-readable",
                    "Logs and configs are linked in artifact manifest",
                ],
                understanding_questions=[
                    "Why is reproducibility necessary before PI can use the artifact?",
                    "What is the difference between a successful run and strong evidence?",
                    "Which hidden variable could make the result unfair?",
                ],
                status=PackageStatus.assigned,
            ),
            LearningResearchPackage(
                package_id="P5",
                title="Analyze results, limitations, and missing evidence",
                stage_id="S5",
                stage_name="Analysis and review",
                assigned_to=pick_student(students, StudentLevel.strong),
                owner_level_required=StudentLevel.strong,
                difficulty=Difficulty.D3_scientific_analysis,
                criticality=Criticality.C3_main_claim,
                parent_claims=["C4", "C2"],
                evidence_requirements=["E2", "E4"],
                learning_objectives=[
                    "Learn how to interpret results without overclaiming",
                    "Identify confounders, failure cases, and missing evidence",
                    "Prepare artifacts for mentor and PI review",
                ],
                research_questions=[
                    "Which claims are supported, weakened, or still missing evidence?",
                    "What failure modes appear in the artifacts?",
                    "What should the PI read first?",
                ],
                background_reading=["Accepted or revised artifacts from P1-P4", "Reviewer attack memo"],
                student_instructions=[
                    "Create a claim-evidence matrix from current artifacts.",
                    "Write a failure and limitation analysis.",
                    "Separate accepted learning artifacts from PI-ready research evidence.",
                ],
                report_template=REPORT_TEMPLATE,
                expected_outputs=["claim_evidence_matrix.csv", "failure_analysis.md", "missing_evidence.md"],
                acceptance_criteria=[
                    "No unsupported claim is marked ready",
                    "Failure examples are linked to metrics or artifacts",
                    "Missing evidence is explicit",
                ],
                understanding_questions=[
                    "Which result is easiest to overclaim?",
                    "What is the strongest limitation of the current project?",
                    "What should PI decide next?",
                ],
                status=PackageStatus.assigned,
            ),
            LearningResearchPackage(
                package_id="P6",
                title="Prepare PI-ready evidence package and paper-readiness memo",
                stage_id="S6",
                stage_name="PI assembly support",
                assigned_to=pick_student(students, StudentLevel.senior),
                owner_level_required=StudentLevel.senior,
                difficulty=Difficulty.D5_paper_strategy,
                criticality=Criticality.C4_core_contribution,
                parent_claims=["C1", "C2", "C3", "C4"],
                evidence_requirements=["E1", "E2", "E3", "E4"],
                learning_objectives=[
                    "Learn how to synthesize a research project for PI review",
                    "Separate paper framing from raw task completion",
                    "Understand which decisions belong to PI only",
                ],
                research_questions=[
                    "Which accepted artifacts are strong enough for a paper or report?",
                    "Which claims still need PI judgement?",
                    "What is the safest paper angle based on evidence rather than optimism?",
                ],
                background_reading=["All accepted artifacts", "Claim-evidence matrix", "Readiness report template"],
                student_instructions=[
                    "Assemble the evidence map, unresolved risks, and PI decision list.",
                    "Draft a paper/report outline, but do not mark claims as final.",
                    "Flag any unsupported or high-risk statements.",
                ],
                report_template=REPORT_TEMPLATE,
                expected_outputs=["pi_ready_evidence_package.md", "paper_readiness_report.md", "draft_outline.md"],
                acceptance_criteria=[
                    "Every suggested claim links to accepted evidence",
                    "Every missing claim is marked missing",
                    "PI-only decisions are explicitly listed",
                ],
                understanding_questions=[
                    "Why should PI receive evidence rather than raw files?",
                    "Which decision should not be made by the agent or student?",
                    "What evidence is still missing before a paper claim is safe?",
                ],
                status=PackageStatus.assigned,
            ),
        ]

        stages = [
            RoadmapStage(
                stage_id="S1",
                name="Topic and prior-art gate",
                objective="Determine what is already known, what is risky, and what gaps are worth PI attention.",
                exit_condition="PI sees literature map, dangerous prior art, and gap cards.",
                package_ids=["P1"],
            ),
            RoadmapStage(
                stage_id="S2",
                name="Claim-evidence graph",
                objective="Convert broad topic and gaps into research questions, claims, and evidence requirements.",
                exit_condition="Every candidate claim has evidence requirements and risk notes.",
                package_ids=["P2"],
                depends_on=["S1"],
            ),
            RoadmapStage(
                stage_id="S3",
                name="Contribution design",
                objective="Design the benchmark, method, analysis, or evaluation protocol needed for paper-readiness.",
                exit_condition="PI approves core design and required baselines/ablations.",
                package_ids=["P3"],
                depends_on=["S2"],
            ),
            RoadmapStage(
                stage_id="S4",
                name="Implementation and reproducibility",
                objective="Produce runnable artifacts, logs, configs, and results that can be checked.",
                exit_condition="Artifacts have reproducibility metadata and result files.",
                package_ids=["P4"],
                depends_on=["S3"],
            ),
            RoadmapStage(
                stage_id="S5",
                name="Analysis and review",
                objective="Evaluate what results mean, where they fail, and what is still missing.",
                exit_condition="Claim-evidence matrix and failure analysis are ready for mentor review.",
                package_ids=["P5"],
                depends_on=["S4"],
            ),
            RoadmapStage(
                stage_id="S6",
                name="PI assembly support",
                objective="Prepare the PI-ready evidence package and paper-readiness memo.",
                exit_condition="PI receives accepted artifacts, missing evidence, and decision points.",
                package_ids=["P6"],
                depends_on=["S5"],
            ),
        ]

        return ResearchRoadmap(
            project_id=project_id,
            topic=topic_short,
            target_standard=target_standard,
            charter=charter,
            stages=stages,
            claims=claims,
            evidence_requirements=evidence,
            packages=packages,
            novelty_risks=[
                "The broad topic may already be crowded; dangerous prior-art mapping is required before heavy implementation.",
                "Benchmark or method novelty must be defended against simple baselines and near-miss papers.",
                "Student outputs may pass learning objectives but still fail research-evidence readiness.",
            ],
            pi_decision_points=[
                "Approve or mutate candidate gaps after P1.",
                "Approve claim-evidence graph after P2.",
                "Approve core method/benchmark/evaluation design after P3.",
                "Decide whether accepted evidence is sufficient for a paper, report, thesis chapter, or pivot after P6.",
            ],
        )


class SubmissionReviewAgent:
    def run(self, state: ProjectState, submission: StudentSubmission) -> ReviewReport:
        package = state.get_package(submission.package_id)
        report = submission.report_text or ""
        normalized = normalize_text(report)

        section_keywords = [
            "summary",
            "parent claim",
            "research question",
            "method",
            "procedure",
            "output",
            "result",
            "interpretation",
            "limitation",
            "confounder",
            "reproducibility",
            "command",
            "next",
        ]
        format_score = clamp(score_keywords(report, section_keywords))

        repro_keywords = ["command", "seed", "config", "environment", "version", "log", "reproduce", "commit"]
        repro_score = clamp(score_keywords(report, repro_keywords))

        evidence_terms = package.parent_claims + package.evidence_requirements + [
            "claim",
            "evidence",
            "artifact",
            "metric",
            "baseline",
            "limitation",
        ]
        evidence_score = clamp(score_keywords(report, evidence_terms))

        scientific_terms = [
            "baseline",
            "metric",
            "ablation",
            "fair",
            "failure",
            "risk",
            "overclaim",
            "novelty",
            "prior art",
            "confounder",
        ]
        scientific_score = clamp(0.4 * score_keywords(report, scientific_terms) + 0.6 * evidence_score)

        expected_hits = 0
        for output in package.expected_outputs:
            output_key = output.lower().replace("/", "").replace(".", "")
            text_key = normalized.replace("/", "").replace(".", "")
            if output_key in text_key or output in submission.artifact_paths:
                expected_hits += 1
        expected_score = expected_hits / max(1, len(package.expected_outputs))

        score_evidence_alignment = clamp(0.6 * evidence_score + 0.4 * expected_score)
        total = (format_score + repro_score + scientific_score + score_evidence_alignment) / 4.0

        findings: List[ReviewFinding] = []
        if format_score < 0.7:
            findings.append(
                ReviewFinding(
                    area="format",
                    severity="medium",
                    message="Report does not cover enough required sections from the package template.",
                    recommendation="Rewrite using the package report template exactly.",
                )
            )
        if repro_score < 0.5 and package.criticality in {Criticality.C2_validity, Criticality.C3_main_claim, Criticality.C4_core_contribution}:
            findings.append(
                ReviewFinding(
                    area="reproducibility",
                    severity="high",
                    message="Reproducibility metadata is weak for a validity/main-claim package.",
                    recommendation="Add commands, config, seed, environment, model/data versions, and log paths.",
                )
            )
        if score_evidence_alignment < 0.65:
            findings.append(
                ReviewFinding(
                    area="evidence_alignment",
                    severity="high",
                    message="Submission does not clearly link outputs to parent claims and evidence requirements.",
                    recommendation="Add a claim-evidence paragraph and mention which output supports which evidence requirement.",
                )
            )
        if scientific_score < 0.55:
            findings.append(
                ReviewFinding(
                    area="scientific_validity",
                    severity="medium",
                    message="Scientific reasoning is too shallow or missing baseline/metric/limitation discussion.",
                    recommendation="Add limitation, baseline fairness, failure risk, and interpretation details.",
                )
            )

        if len(normalized) < 200:
            decision = ReviewDecision.reject
            next_action = "Reject for now: submission is too short to review. Student must resubmit a full report."
        elif total >= 0.72 and not any(f.severity == "high" for f in findings):
            decision = ReviewDecision.accept
            next_action = "Proceed to Understanding Gate. Agent acceptance is not PI approval."
        else:
            decision = ReviewDecision.revise
            next_action = "Revise according to findings, then resubmit or run Understanding Gate only as a diagnostic."

        questions = list(package.understanding_questions[:3])
        if score_evidence_alignment < 0.8:
            questions.append("Which exact artifact supports which parent claim and evidence requirement?")
        if scientific_score < 0.8:
            questions.append("What is the most likely failure mode, confounder, or overclaim risk in your submission?")
        if repro_score < 0.8:
            questions.append("Which command, config, seed, or environment detail would another student need to reproduce your result?")

        return ReviewReport(
            submission_id=submission.submission_id,
            package_id=package.package_id,
            decision=decision,
            score_format=round(format_score, 3),
            score_reproducibility=round(repro_score, 3),
            score_scientific=round(scientific_score, 3),
            score_evidence_alignment=round(score_evidence_alignment, 3),
            findings=findings,
            generated_understanding_questions=questions,
            recommended_next_action=next_action,
        )


class UnderstandingGateAgent:
    def run(
        self,
        state: ProjectState,
        review: ReviewReport,
        student_id: str,
        answers: List[UnderstandingAnswer],
    ) -> UnderstandingReport:
        package = state.get_package(review.package_id)
        answer_text = "\n".join(a.answer for a in answers)
        answer_tokens = token_set(answer_text)
        claim_lookup = {claim.claim_id: claim for claim in (state.roadmap.claims if state.roadmap else [])}
        evidence_lookup = {evidence.evidence_id: evidence for evidence in (state.roadmap.evidence_requirements if state.roadmap else [])}
        claim_texts = [claim_lookup[cid].text for cid in package.parent_claims if cid in claim_lookup]
        evidence_texts = [evidence_lookup[eid].description for eid in package.evidence_requirements if eid in evidence_lookup]

        package_context = " ".join(
            package.parent_claims
            + package.evidence_requirements
            + claim_texts
            + evidence_texts
            + package.learning_objectives
            + package.research_questions
            + package.acceptance_criteria
        )
        context_tokens = token_set(package_context)
        overlap = len(answer_tokens & context_tokens) / max(1, min(len(context_tokens), 60))

        concept_keywords = [
            "claim",
            "evidence",
            "research question",
            "limitation",
            "baseline",
            "metric",
            "artifact",
            "reproduce",
            "failure",
            "prior art",
            "pi",
        ]
        keyword_score = score_keywords(answer_text, concept_keywords)
        length_score = clamp(len(answer_text) / 1200.0)
        specificity_terms = package.parent_claims + package.evidence_requirements + claim_texts + evidence_texts
        specificity_score = score_keywords(answer_text, specificity_terms)
        score = clamp(0.30 * keyword_score + 0.35 * specificity_score + 0.20 * overlap + 0.15 * length_score)

        weak_points: List[str] = []
        if keyword_score < 0.45:
            weak_points.append("Answer does not use enough research concepts such as claim, evidence, baseline, limitation, or reproducibility.")
        if specificity_score < 0.35:
            weak_points.append("Answer does not clearly reference the parent claim or evidence requirement.")
        if length_score < 0.25:
            weak_points.append("Answer is too short to verify understanding.")
        if "limitation" not in normalize_text(answer_text) and "failure" not in normalize_text(answer_text):
            weak_points.append("Student did not discuss limitations or failure risks.")

        passed = score >= 0.55 and len(weak_points) <= 1 and review.decision != ReviewDecision.reject
        decision = ReviewDecision.accept if passed else ReviewDecision.revise
        if passed:
            feedback = "Student explanation is sufficient for the demo Understanding Gate. Mentor/PI review is still required for high-criticality artifacts."
        else:
            feedback = "Student should revise the explanation before the artifact can become PI-ready evidence."

        return UnderstandingReport(
            package_id=package.package_id,
            student_id=student_id,
            review_id=review.review_id,
            decision=decision,
            score=round(score, 3),
            feedback=feedback,
            weak_points=weak_points,
            passed=passed,
        )


class EvidenceRegistryAgent:
    def update_after_understanding(
        self,
        state: ProjectState,
        submission: StudentSubmission,
        review: ReviewReport,
        understanding: UnderstandingReport,
    ) -> ArtifactRecord:
        if review.decision == ReviewDecision.accept and understanding.passed:
            status = ArtifactStatus.agent_accepted_pending_pi
            accepted_for_pi = True
            notes = "Passed agent review and Understanding Gate. Pending PI or mentor final approval."
            package_status = PackageStatus.ready_for_pi
        elif review.decision == ReviewDecision.reject:
            status = ArtifactStatus.rejected
            accepted_for_pi = False
            notes = "Rejected by review agent."
            package_status = PackageStatus.rejected
        else:
            status = ArtifactStatus.revise_required
            accepted_for_pi = False
            notes = "Revision required before artifact can be PI-ready."
            package_status = PackageStatus.revise_required

        artifact = ArtifactRecord(
            package_id=submission.package_id,
            student_id=submission.student_id,
            submission_id=submission.submission_id,
            review_id=review.review_id,
            understanding_id=understanding.understanding_id,
            status=status,
            accepted_for_pi=accepted_for_pi,
            artifact_paths=submission.artifact_paths,
            notes=notes,
        )
        state.artifacts.append(artifact)
        package = state.get_package(submission.package_id)
        package.status = package_status
        state.touch()
        return artifact


class PIDashboardBuilder:
    def run(self, state: ProjectState) -> PIDashboard:
        if not state.roadmap:
            raise ValueError("Project has no roadmap")

        package_by_id = {p.package_id: p for p in state.roadmap.packages}
        packages_by_evidence: Dict[str, List[str]] = defaultdict(list)
        for package in state.roadmap.packages:
            for evidence_id in package.evidence_requirements:
                packages_by_evidence[evidence_id].append(package.package_id)

        accepted_by_package: Dict[str, List[str]] = defaultdict(list)
        revise_by_package: Dict[str, List[str]] = defaultdict(list)
        for artifact in state.artifacts:
            if artifact.accepted_for_pi:
                accepted_by_package[artifact.package_id].append(artifact.artifact_id)
            elif artifact.status == ArtifactStatus.revise_required:
                revise_by_package[artifact.package_id].append(artifact.artifact_id)

        claim_by_id = {claim.claim_id: claim for claim in state.roadmap.claims}
        rows: List[ClaimEvidenceRow] = []
        missing: List[str] = []
        for evidence in state.roadmap.evidence_requirements:
            package_ids = packages_by_evidence.get(evidence.evidence_id, [])
            accepted_ids: List[str] = []
            for package_id in package_ids:
                accepted_ids.extend(accepted_by_package.get(package_id, []))
            if accepted_ids:
                status = "has_agent_accepted_artifact_pending_pi"
            else:
                status = "missing_or_not_ready"
                missing.append(f"{evidence.evidence_id}: {evidence.description}")
            claim = claim_by_id[evidence.claim_id]
            rows.append(
                ClaimEvidenceRow(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    evidence_id=evidence.evidence_id,
                    evidence_description=evidence.description,
                    package_ids=package_ids,
                    accepted_artifact_ids=accepted_ids,
                    status=status,
                )
            )

        stage_summary: Dict[str, str] = {}
        for stage in state.roadmap.stages:
            statuses = [package_by_id[pid].status.value for pid in stage.package_ids if pid in package_by_id]
            if statuses and all(s == PackageStatus.ready_for_pi.value for s in statuses):
                stage_summary[stage.stage_id] = "ready_for_pi"
            elif any(s in {PackageStatus.revise_required.value, PackageStatus.rejected.value} for s in statuses):
                stage_summary[stage.stage_id] = "needs_attention"
            elif any(s in {PackageStatus.submitted.value, PackageStatus.agent_reviewed.value, PackageStatus.understanding_passed.value} for s in statuses):
                stage_summary[stage.stage_id] = "in_review"
            else:
                stage_summary[stage.stage_id] = "planned_or_assigned"

        package_summary = {p.package_id: p.status.value for p in state.roadmap.packages}
        pi_attention_items: List[str] = []
        for package in state.roadmap.packages:
            if package.criticality in {Criticality.C3_main_claim, Criticality.C4_core_contribution}:
                if package.status == PackageStatus.ready_for_pi:
                    pi_attention_items.append(f"Review high-criticality package {package.package_id}: {package.title}")
                elif package.status == PackageStatus.revise_required:
                    pi_attention_items.append(f"Package {package.package_id} is high-criticality but needs revision: {package.title}")

        accepted_artifacts = [a.artifact_id for a in state.artifacts if a.accepted_for_pi]
        revise_items = [f"{pid}: {ids}" for pid, ids in revise_by_package.items()]

        return PIDashboard(
            project_id=state.project_id,
            topic=state.roadmap.topic,
            stage_summary=stage_summary,
            package_summary=package_summary,
            claim_evidence_matrix=rows,
            missing_evidence=missing,
            pi_attention_items=pi_attention_items,
            accepted_artifacts=accepted_artifacts,
            revise_items=revise_items,
        )
