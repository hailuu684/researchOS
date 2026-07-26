from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .agents import (
    EvidenceRegistryAgent,
    PIDashboardBuilder,
    SubmissionReviewAgent,
    TopicToRoadmapAgent,
    UnderstandingGateAgent,
)
from .llm_agents import LLMSubmissionReviewAgent, LLMTopicToRoadmapAgent, LLMUnderstandingGateAgent
from .modal_llm import ModalOpenAIChat
from .schemas import (
    ArtifactRecord,
    PIDashboard,
    PackageStatus,
    ProjectState,
    ReviewReport,
    StudentProfile,
    StudentSubmission,
    UnderstandingAnswer,
    UnderstandingReport,
)


def _maybe_llm(use_llm: bool, llm: Optional[ModalOpenAIChat]) -> Optional[ModalOpenAIChat]:
    if not use_llm:
        return None
    return llm or ModalOpenAIChat.from_env()


def create_project(
    topic: str,
    students: List[StudentProfile],
    target_standard: str = "paper-ready / Q1-oriented",
    constraints: str = "",
    *,
    use_llm: bool = False,
    llm: Optional[ModalOpenAIChat] = None,
    fallback_on_llm_error: bool = True,
) -> ProjectState:
    """Create a project and generate the first-pass roadmap.

    use_llm=False: deterministic backend, no endpoint required.
    use_llm=True: calls the configured Modal/OpenAI-compatible endpoint to customize the roadmap.
    """
    state = ProjectState(students=students)
    modal_llm = _maybe_llm(use_llm, llm)
    if modal_llm is not None:
        planner = LLMTopicToRoadmapAgent(modal_llm, fallback_on_error=fallback_on_llm_error)
    else:
        planner = TopicToRoadmapAgent()
    state.roadmap = planner.run(
        project_id=state.project_id,
        topic=topic,
        target_standard=target_standard,
        students=students,
        constraints=constraints,
    )
    state.touch()
    return state


def submit_package(
    state: ProjectState,
    package_id: str,
    student_id: str,
    report_text: str,
    artifact_paths: Sequence[str] | None = None,
) -> StudentSubmission:
    """Student submits a package report and artifact references."""
    package = state.get_package(package_id)
    state.get_student(student_id)
    submission = StudentSubmission(
        package_id=package_id,
        student_id=student_id,
        report_text=report_text,
        artifact_paths=list(artifact_paths or []),
    )
    state.submissions.append(submission)
    package.status = PackageStatus.submitted
    state.touch()
    return submission


def review_submission(
    state: ProjectState,
    submission_id: str,
    *,
    use_llm: bool = False,
    llm: Optional[ModalOpenAIChat] = None,
    fallback_on_llm_error: bool = True,
) -> ReviewReport:
    """Run the review gate for a submitted package."""
    submission = state.get_submission(submission_id)
    modal_llm = _maybe_llm(use_llm, llm)
    if modal_llm is not None:
        reviewer = LLMSubmissionReviewAgent(modal_llm, fallback_on_error=fallback_on_llm_error)
    else:
        reviewer = SubmissionReviewAgent()
    report = reviewer.run(state, submission)
    state.review_reports.append(report)
    package = state.get_package(submission.package_id)
    package.status = PackageStatus.agent_reviewed
    state.touch()
    return report


def answer_understanding(
    state: ProjectState,
    review_id: str,
    answers: Iterable[str] | Iterable[UnderstandingAnswer],
    *,
    use_llm: bool = False,
    llm: Optional[ModalOpenAIChat] = None,
    fallback_on_llm_error: bool = True,
) -> tuple[UnderstandingReport, ArtifactRecord]:
    """Run Understanding Gate and update the evidence registry."""
    review = state.get_review(review_id)
    submission = state.get_submission(review.submission_id)

    normalized_answers: List[UnderstandingAnswer] = []
    for idx, answer in enumerate(answers):
        if isinstance(answer, UnderstandingAnswer):
            normalized_answers.append(answer)
        else:
            question = review.generated_understanding_questions[idx] if idx < len(review.generated_understanding_questions) else f"Question {idx+1}"
            normalized_answers.append(UnderstandingAnswer(question=question, answer=str(answer)))

    modal_llm = _maybe_llm(use_llm, llm)
    if modal_llm is not None:
        gate = LLMUnderstandingGateAgent(modal_llm, fallback_on_error=fallback_on_llm_error)
    else:
        gate = UnderstandingGateAgent()

    understanding = gate.run(
        state=state,
        review=review,
        student_id=submission.student_id,
        answers=normalized_answers,
    )
    state.understanding_reports.append(understanding)

    registry = EvidenceRegistryAgent()
    artifact = registry.update_after_understanding(state, submission, review, understanding)
    state.touch()
    return understanding, artifact


def build_dashboard(state: ProjectState) -> PIDashboard:
    """Build a PI-facing summary from the current project state."""
    builder = PIDashboardBuilder()
    return builder.run(state)


def save_state(state: ProjectState, path: str | Path) -> Path:
    """Persist the state to JSON.

    For Modal Notebook demos, a local path is enough. Later, place this under a
    Modal Volume if you need persistence across sessions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_state(path: str | Path) -> ProjectState:
    path = Path(path)
    return ProjectState.model_validate_json(path.read_text(encoding="utf-8"))
