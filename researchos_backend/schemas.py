from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


class StudentLevel(str, Enum):
    junior = "junior"
    intermediate = "intermediate"
    strong = "strong"
    senior = "senior"


class Difficulty(str, Enum):
    D0_operational = "D0_operational"
    D1_reproduction = "D1_reproduction"
    D2_engineering = "D2_engineering"
    D3_scientific_analysis = "D3_scientific_analysis"
    D4_core_method_or_benchmark = "D4_core_method_or_benchmark"
    D5_paper_strategy = "D5_paper_strategy"


class Criticality(str, Enum):
    C0_support = "C0_support"
    C1_useful = "C1_useful"
    C2_validity = "C2_validity"
    C3_main_claim = "C3_main_claim"
    C4_core_contribution = "C4_core_contribution"


class PackageStatus(str, Enum):
    planned = "planned"
    assigned = "assigned"
    submitted = "submitted"
    agent_reviewed = "agent_reviewed"
    understanding_passed = "understanding_passed"
    ready_for_pi = "ready_for_pi"
    revise_required = "revise_required"
    rejected = "rejected"


class ReviewDecision(str, Enum):
    accept = "accept"
    revise = "revise"
    reject = "reject"


class ArtifactStatus(str, Enum):
    submitted = "submitted"
    agent_accepted_pending_pi = "agent_accepted_pending_pi"
    revise_required = "revise_required"
    rejected = "rejected"
    pi_accepted = "pi_accepted"


class StudentProfile(BaseModel):
    student_id: str = Field(default_factory=lambda: new_id("stu"))
    name: str
    level: StudentLevel = StudentLevel.junior
    skills: List[str] = Field(default_factory=list)
    availability_hours_per_week: int = 5
    notes: str = ""

    @field_validator("availability_hours_per_week")
    @classmethod
    def positive_hours(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("availability_hours_per_week must be positive")
        return value


class ResearchClaim(BaseModel):
    claim_id: str
    text: str
    claim_type: str = "scientific"
    novelty_risk: str = "medium"
    approved_by_pi: bool = False


class EvidenceRequirement(BaseModel):
    evidence_id: str
    claim_id: str
    description: str
    required_artifact_types: List[str]
    success_criteria: List[str]


class LearningResearchPackage(BaseModel):
    package_id: str
    title: str
    stage_id: str
    stage_name: str
    assigned_to: Optional[str] = None
    owner_level_required: StudentLevel = StudentLevel.junior
    difficulty: Difficulty = Difficulty.D1_reproduction
    criticality: Criticality = Criticality.C1_useful
    parent_claims: List[str]
    evidence_requirements: List[str]
    learning_objectives: List[str]
    research_questions: List[str]
    background_reading: List[str] = Field(default_factory=list)
    student_instructions: List[str]
    report_template: List[str]
    expected_outputs: List[str]
    acceptance_criteria: List[str]
    understanding_questions: List[str]
    status: PackageStatus = PackageStatus.planned


class RoadmapStage(BaseModel):
    stage_id: str
    name: str
    objective: str
    exit_condition: str
    package_ids: List[str]
    depends_on: List[str] = Field(default_factory=list)


class ResearchRoadmap(BaseModel):
    project_id: str
    topic: str
    target_standard: str = "paper-ready / Q1-oriented"
    charter: str
    stages: List[RoadmapStage]
    claims: List[ResearchClaim]
    evidence_requirements: List[EvidenceRequirement]
    packages: List[LearningResearchPackage]
    novelty_risks: List[str]
    pi_decision_points: List[str]


class StudentSubmission(BaseModel):
    submission_id: str = Field(default_factory=lambda: new_id("sub"))
    package_id: str
    student_id: str
    report_text: str
    artifact_paths: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class ReviewFinding(BaseModel):
    area: str
    severity: str = "medium"  # low, medium, high
    message: str
    recommendation: str


class ReviewReport(BaseModel):
    review_id: str = Field(default_factory=lambda: new_id("rev"))
    submission_id: str
    package_id: str
    decision: ReviewDecision
    score_format: float
    score_reproducibility: float
    score_scientific: float
    score_evidence_alignment: float
    findings: List[ReviewFinding]
    generated_understanding_questions: List[str]
    recommended_next_action: str
    created_at: str = Field(default_factory=utc_now_iso)


class UnderstandingAnswer(BaseModel):
    question: str
    answer: str


class UnderstandingReport(BaseModel):
    understanding_id: str = Field(default_factory=lambda: new_id("und"))
    package_id: str
    student_id: str
    review_id: str
    decision: ReviewDecision
    score: float
    feedback: str
    weak_points: List[str]
    passed: bool
    created_at: str = Field(default_factory=utc_now_iso)


class ArtifactRecord(BaseModel):
    artifact_id: str = Field(default_factory=lambda: new_id("art"))
    package_id: str
    student_id: str
    submission_id: str
    review_id: Optional[str] = None
    understanding_id: Optional[str] = None
    status: ArtifactStatus = ArtifactStatus.submitted
    accepted_for_pi: bool = False
    artifact_paths: List[str] = Field(default_factory=list)
    notes: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class ClaimEvidenceRow(BaseModel):
    claim_id: str
    claim_text: str
    evidence_id: str
    evidence_description: str
    package_ids: List[str]
    accepted_artifact_ids: List[str]
    status: str


class PIDashboard(BaseModel):
    project_id: str
    topic: str
    stage_summary: Dict[str, str]
    package_summary: Dict[str, str]
    claim_evidence_matrix: List[ClaimEvidenceRow]
    missing_evidence: List[str]
    pi_attention_items: List[str]
    accepted_artifacts: List[str]
    revise_items: List[str]
    generated_at: str = Field(default_factory=utc_now_iso)


class ProjectState(BaseModel):
    project_id: str = Field(default_factory=lambda: new_id("proj"))
    roadmap: Optional[ResearchRoadmap] = None
    students: List[StudentProfile] = Field(default_factory=list)
    submissions: List[StudentSubmission] = Field(default_factory=list)
    review_reports: List[ReviewReport] = Field(default_factory=list)
    understanding_reports: List[UnderstandingReport] = Field(default_factory=list)
    artifacts: List[ArtifactRecord] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def get_package(self, package_id: str) -> LearningResearchPackage:
        if not self.roadmap:
            raise ValueError("Project has no roadmap")
        for package in self.roadmap.packages:
            if package.package_id == package_id:
                return package
        raise KeyError(f"Package not found: {package_id}")

    def get_student(self, student_id: str) -> StudentProfile:
        for student in self.students:
            if student.student_id == student_id:
                return student
        raise KeyError(f"Student not found: {student_id}")

    def get_submission(self, submission_id: str) -> StudentSubmission:
        for submission in self.submissions:
            if submission.submission_id == submission_id:
                return submission
        raise KeyError(f"Submission not found: {submission_id}")

    def get_review(self, review_id: str) -> ReviewReport:
        for review in self.review_reports:
            if review.review_id == review_id:
                return review
        raise KeyError(f"Review not found: {review_id}")
