from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

from .config import default_fallback_on_llm_error, default_use_llm, load_local_env
from .modal_llm import ModalOpenAIChat, extract_json
from .orchestrator import (
    answer_understanding,
    build_dashboard,
    create_project,
    load_state,
    review_submission,
    save_state,
    submit_package,
)
from .pretty import print_dashboard, print_package, print_review, print_roadmap
from .sample_data import GOOD_P1_SUBMISSION, GOOD_UNDERSTANDING_ANSWERS, SAMPLE_STUDENTS
from .schemas import ProjectState, StudentProfile

DEFAULT_TOPIC = "LLM-driven point cloud reasoning with verifiable geometric evidence"


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _llm_or_none(use_llm: bool) -> ModalOpenAIChat | None:
    return ModalOpenAIChat.from_env() if use_llm else None


def _load_students(path: str | None) -> List[StudentProfile]:
    if not path:
        return SAMPLE_STUDENTS
    raw = json.loads(_read_text(path))
    if not isinstance(raw, list):
        raise ValueError("students JSON must be a list")
    return [StudentProfile.model_validate(item) for item in raw]


def cmd_smoke(args: argparse.Namespace) -> int:
    load_local_env(args.env_file)
    llm = ModalOpenAIChat.from_env()
    print("Configured endpoint")
    print("  base_url:", llm.config.base_url)
    print("  model:", llm.config.model)
    print("  modal headers present:", bool(llm.config.modal_key and llm.config.modal_secret))
    print("  reasoning_enabled:", llm.config.reasoning_enabled)
    print("\nRaw response:")
    raw = llm.smoke_test()
    print(raw)
    print("\nParsed JSON:")
    print(json.dumps(extract_json(raw), ensure_ascii=False, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    load_local_env(args.env_file)
    use_llm = args.use_llm if args.use_llm is not None else default_use_llm()
    fallback = args.fallback if args.fallback is not None else default_fallback_on_llm_error()
    llm = _llm_or_none(use_llm)

    if llm:
        print("== Endpoint ==")
        print("base_url:", llm.config.base_url)
        print("model:", llm.config.model)
        print("headers present:", bool(llm.config.modal_key and llm.config.modal_secret))
        print("\n== Smoke test ==")
        print(llm.smoke_test())
    else:
        print("== Running deterministic mode: no LLM endpoint ==")

    state = create_project(
        topic=args.topic,
        students=_load_students(args.students_file),
        target_standard=args.target_standard,
        constraints=args.constraints,
        use_llm=use_llm,
        llm=llm,
        fallback_on_llm_error=fallback,
    )
    print("\n== Roadmap ==")
    print_roadmap(state)

    package = state.get_package(args.package_id)
    print(f"\n== Package {args.package_id} ==")
    print_package(package)

    report_text = _read_text(args.report_file) if args.report_file else GOOD_P1_SUBMISSION
    student_id = args.student_id or (package.assigned_to or SAMPLE_STUDENTS[0].student_id)
    submission = submit_package(
        state,
        package_id=args.package_id,
        student_id=student_id,
        report_text=report_text,
        artifact_paths=args.artifacts,
    )
    print("\n== Submission ==")
    print("submission_id:", submission.submission_id)

    review = review_submission(
        state,
        submission.submission_id,
        use_llm=use_llm,
        llm=llm,
        fallback_on_llm_error=fallback,
    )
    print("\n== Review ==")
    print_review(review)

    answers = GOOD_UNDERSTANDING_ANSWERS
    if args.answers_file:
        raw_answers = json.loads(_read_text(args.answers_file))
        if isinstance(raw_answers, list):
            answers = [str(x) for x in raw_answers]
        else:
            raise ValueError("answers JSON must be a list of strings")

    understanding, artifact = answer_understanding(
        state,
        review.review_id,
        answers,
        use_llm=use_llm,
        llm=llm,
        fallback_on_llm_error=fallback,
    )
    print("\n== Understanding ==")
    print("decision:", understanding.decision.value)
    print("score:", understanding.score)
    print("passed:", understanding.passed)
    print("artifact_status:", artifact.status.value)
    print("accepted_for_pi:", artifact.accepted_for_pi)

    dashboard = build_dashboard(state)
    print("\n== PI Dashboard ==")
    print_dashboard(dashboard)

    output = save_state(state, args.state_path)
    print("\nSaved state:", output)
    if args.dashboard_json:
        _write_json(args.dashboard_json, dashboard.model_dump())
        print("Saved dashboard:", args.dashboard_json)
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    load_local_env(args.env_file)
    use_llm = args.use_llm if args.use_llm is not None else default_use_llm()
    fallback = args.fallback if args.fallback is not None else default_fallback_on_llm_error()
    llm = _llm_or_none(use_llm)
    state = create_project(
        topic=args.topic,
        students=_load_students(args.students_file),
        target_standard=args.target_standard,
        constraints=args.constraints,
        use_llm=use_llm,
        llm=llm,
        fallback_on_llm_error=fallback,
    )
    print_roadmap(state)
    print("\nSaved state:", save_state(state, args.state_path))
    return 0


def cmd_show_package(args: argparse.Namespace) -> int:
    load_local_env(args.env_file)
    state = load_state(args.state_path)
    print_package(state.get_package(args.package_id))
    return 0


def cmd_submit_review(args: argparse.Namespace) -> int:
    load_local_env(args.env_file)
    use_llm = args.use_llm if args.use_llm is not None else default_use_llm()
    fallback = args.fallback if args.fallback is not None else default_fallback_on_llm_error()
    llm = _llm_or_none(use_llm)
    state = load_state(args.state_path)
    report = _read_text(args.report_file)
    submission = submit_package(
        state,
        package_id=args.package_id,
        student_id=args.student_id,
        report_text=report,
        artifact_paths=args.artifacts,
    )
    review = review_submission(state, submission.submission_id, use_llm=use_llm, llm=llm, fallback_on_llm_error=fallback)
    print("submission_id:", submission.submission_id)
    print_review(review)
    print("\nSaved state:", save_state(state, args.state_path))
    return 0


def cmd_understanding(args: argparse.Namespace) -> int:
    load_local_env(args.env_file)
    use_llm = args.use_llm if args.use_llm is not None else default_use_llm()
    fallback = args.fallback if args.fallback is not None else default_fallback_on_llm_error()
    llm = _llm_or_none(use_llm)
    state = load_state(args.state_path)
    answers = json.loads(_read_text(args.answers_file))
    if not isinstance(answers, list):
        raise ValueError("answers JSON must be a list of strings")
    understanding, artifact = answer_understanding(
        state,
        args.review_id,
        [str(x) for x in answers],
        use_llm=use_llm,
        llm=llm,
        fallback_on_llm_error=fallback,
    )
    print("decision:", understanding.decision.value)
    print("score:", understanding.score)
    print("passed:", understanding.passed)
    print("feedback:", understanding.feedback)
    print("artifact_status:", artifact.status.value)
    print("accepted_for_pi:", artifact.accepted_for_pi)
    print("\nSaved state:", save_state(state, args.state_path))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    load_local_env(args.env_file)
    state = load_state(args.state_path)
    dashboard = build_dashboard(state)
    print_dashboard(dashboard)
    if args.output_json:
        _write_json(args.output_json, dashboard.model_dump())
        print("\nSaved dashboard:", args.output_json)
    return 0


def _add_common_llm_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-llm", dest="use_llm", action="store_true", help="Call the Modal endpoint.")
    group.add_argument("--no-llm", dest="use_llm", action="store_false", help="Run deterministic mode without endpoint.")
    parser.set_defaults(use_llm=None)

    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument("--fallback", dest="fallback", action="store_true", help="Fallback if LLM returns invalid JSON.")
    fallback_group.add_argument("--no-fallback", dest="fallback", action="store_false", help="Raise LLM errors instead of fallback.")
    parser.set_defaults(fallback=None)
    parser.add_argument("--env-file", default=None, help="Path to .env file. Defaults to .env if present.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchos", description="Backend-only ResearchOS demo for VSCode/SSH testing.")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Smoke-test the Modal OpenAI-compatible endpoint.")
    smoke.add_argument("--env-file", default=None)
    smoke.set_defaults(func=cmd_smoke)

    demo = sub.add_parser("demo", help="Run the full backend pipeline and save state.")
    _add_common_llm_flags(demo)
    demo.add_argument("--topic", default=DEFAULT_TOPIC)
    demo.add_argument("--target-standard", default="paper-ready / Q1-oriented research training demo")
    demo.add_argument("--constraints", default="VSCode backend test. LLM assists; PI remains final decision maker.")
    demo.add_argument("--students-file", default=None, help="Optional JSON list of StudentProfile objects.")
    demo.add_argument("--package-id", default="P1")
    demo.add_argument("--student-id", default=None)
    demo.add_argument("--report-file", default=None)
    demo.add_argument("--answers-file", default=None)
    demo.add_argument("--artifacts", nargs="*", default=["paper_cards.md", "dangerous_prior_art_matrix.csv", "gap_cards.md"])
    demo.add_argument("--state-path", default="outputs/demo_state.json")
    demo.add_argument("--dashboard-json", default="outputs/demo_dashboard.json")
    demo.set_defaults(func=cmd_demo)

    create = sub.add_parser("create", help="Create project roadmap and save state.")
    _add_common_llm_flags(create)
    create.add_argument("--topic", default=DEFAULT_TOPIC)
    create.add_argument("--target-standard", default="paper-ready / Q1-oriented")
    create.add_argument("--constraints", default="")
    create.add_argument("--students-file", default=None)
    create.add_argument("--state-path", default="outputs/project_state.json")
    create.set_defaults(func=cmd_create)

    show_package = sub.add_parser("show-package", help="Print a package from a saved state.")
    show_package.add_argument("--state-path", default="outputs/project_state.json")
    show_package.add_argument("--package-id", default="P1")
    show_package.add_argument("--env-file", default=None)
    show_package.set_defaults(func=cmd_show_package)

    submit_review = sub.add_parser("submit-review", help="Submit a report file and run review gate.")
    _add_common_llm_flags(submit_review)
    submit_review.add_argument("--state-path", default="outputs/project_state.json")
    submit_review.add_argument("--package-id", required=True)
    submit_review.add_argument("--student-id", required=True)
    submit_review.add_argument("--report-file", required=True)
    submit_review.add_argument("--artifacts", nargs="*", default=[])
    submit_review.set_defaults(func=cmd_submit_review)

    understanding = sub.add_parser("understanding", help="Run Understanding Gate for a review.")
    _add_common_llm_flags(understanding)
    understanding.add_argument("--state-path", default="outputs/project_state.json")
    understanding.add_argument("--review-id", required=True)
    understanding.add_argument("--answers-file", required=True, help="JSON list of answer strings.")
    understanding.set_defaults(func=cmd_understanding)

    dashboard = sub.add_parser("dashboard", help="Build PI dashboard from saved state.")
    dashboard.add_argument("--state-path", default="outputs/project_state.json")
    dashboard.add_argument("--output-json", default="outputs/dashboard.json")
    dashboard.add_argument("--env-file", default=None)
    dashboard.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
