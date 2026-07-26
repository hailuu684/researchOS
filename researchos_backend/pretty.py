from __future__ import annotations

from typing import Any, Iterable


def print_roadmap(state: Any) -> None:
    if not state.roadmap:
        print("No roadmap")
        return
    print(f"Project: {state.project_id}")
    print(f"Topic: {state.roadmap.topic}")
    print(f"Target: {state.roadmap.target_standard}\n")
    print("Stages")
    print("------")
    for stage in state.roadmap.stages:
        print(f"{stage.stage_id}. {stage.name}")
        print(f"  objective: {stage.objective}")
        print(f"  packages: {', '.join(stage.package_ids)}")
        print(f"  exit: {stage.exit_condition}")
    print("\nPackages")
    print("--------")
    for package in state.roadmap.packages:
        owner = package.assigned_to or "unassigned"
        print(f"{package.package_id}: {package.title} -> {owner} | {package.difficulty.value} | {package.criticality.value}")


def print_package(package: Any) -> None:
    print(f"{package.package_id}: {package.title}")
    print(f"Stage: {package.stage_id} - {package.stage_name}")
    print(f"Assigned to: {package.assigned_to}")
    print(f"Parent claims: {', '.join(package.parent_claims)}")
    print(f"Evidence requirements: {', '.join(package.evidence_requirements)}")
    print("\nLearning objectives:")
    for item in package.learning_objectives:
        print(f"- {item}")
    print("\nResearch questions:")
    for item in package.research_questions:
        print(f"- {item}")
    print("\nExpected outputs:")
    for item in package.expected_outputs:
        print(f"- {item}")
    print("\nUnderstanding questions:")
    for item in package.understanding_questions:
        print(f"- {item}")


def print_review(review: Any) -> None:
    print(f"Review {review.review_id}: {review.decision.value}")
    print(
        "Scores: "
        f"format={review.score_format}, "
        f"repro={review.score_reproducibility}, "
        f"scientific={review.score_scientific}, "
        f"evidence={review.score_evidence_alignment}"
    )
    print(f"Next action: {review.recommended_next_action}")
    if review.findings:
        print("\nFindings:")
        for finding in review.findings:
            print(f"- [{finding.severity}] {finding.area}: {finding.message}")
            print(f"  recommendation: {finding.recommendation}")
    print("\nUnderstanding questions:")
    for question in review.generated_understanding_questions:
        print(f"- {question}")


def print_dashboard(dashboard: Any) -> None:
    print(f"PI Dashboard for {dashboard.project_id}")
    print(f"Topic: {dashboard.topic}")
    print("\nStage summary:")
    for stage_id, status in dashboard.stage_summary.items():
        print(f"- {stage_id}: {status}")
    print("\nPackage summary:")
    for package_id, status in dashboard.package_summary.items():
        print(f"- {package_id}: {status}")
    print("\nClaim-evidence matrix:")
    for row in dashboard.claim_evidence_matrix:
        print(f"- {row.claim_id}/{row.evidence_id}: {row.status} | artifacts={row.accepted_artifact_ids}")
    print("\nMissing evidence:")
    for item in dashboard.missing_evidence:
        print(f"- {item}")
    print("\nPI attention items:")
    for item in dashboard.pi_attention_items:
        print(f"- {item}")
