from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import default_fallback_on_llm_error, default_use_llm, load_local_env, state_dir
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
from .sample_data import SAMPLE_STUDENTS
from .schemas import PIDashboard, ProjectState, ReviewReport, StudentProfile, StudentSubmission, UnderstandingReport

load_local_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state_dir().mkdir(parents=True, exist_ok=True)
    _load_saved_projects()
    yield


app = FastAPI(
    title="ResearchOS Backend Demo",
    version="0.2.0",
    description="Backend-only API for testing the ResearchOS vertical slice with a Modal OpenAI-compatible endpoint.",
    lifespan=lifespan,
)

_PROJECTS: Dict[str, ProjectState] = {}


def _project_path(project_id: str) -> Path:
    return state_dir() / f"{project_id}.json"


def _save(project: ProjectState) -> None:
    save_state(project, _project_path(project.project_id))


def _load_saved_projects() -> None:
    directory = state_dir()
    if not directory.exists():
        return
    for path in directory.glob("*.json"):
        try:
            project = load_state(path)
            _PROJECTS[project.project_id] = project
        except Exception:
            continue


def _get_project(project_id: str) -> ProjectState:
    project = _PROJECTS.get(project_id)
    if project:
        return project
    path = _project_path(project_id)
    if path.exists():
        project = load_state(path)
        _PROJECTS[project_id] = project
        return project
    raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")


def _llm_or_none(use_llm: Optional[bool]) -> ModalOpenAIChat | None:
    enabled = default_use_llm() if use_llm is None else use_llm
    return ModalOpenAIChat.from_env() if enabled else None


class CreateProjectRequest(BaseModel):
    topic: str = "LLM-driven point cloud reasoning with verifiable geometric evidence"
    target_standard: str = "paper-ready / Q1-oriented"
    constraints: str = "Backend API test. LLM assists; PI remains final decision maker."
    students: Optional[List[StudentProfile]] = None
    use_llm: Optional[bool] = None
    fallback_on_llm_error: Optional[bool] = None


class SubmitPackageRequest(BaseModel):
    package_id: str = "P1"
    student_id: str = "S_JUNIOR_01"
    report_text: str
    artifact_paths: List[str] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    use_llm: Optional[bool] = None
    fallback_on_llm_error: Optional[bool] = None


class UnderstandingRequest(BaseModel):
    answers: List[str]
    use_llm: Optional[bool] = None
    fallback_on_llm_error: Optional[bool] = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "projects_in_memory": len(_PROJECTS),
        "state_dir": str(state_dir()),
        "default_use_llm": default_use_llm(),
    }


@app.get("/llm/smoke")
def llm_smoke() -> dict:
    try:
        llm = ModalOpenAIChat.from_env()
        raw = llm.smoke_test()
        parsed = extract_json(raw)
        return {
            "ok": True,
            "base_url": llm.config.base_url,
            "model": llm.config.model,
            "modal_headers_present": bool(llm.config.modal_key and llm.config.modal_secret),
            "raw": raw,
            "parsed": parsed,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM smoke test failed: {type(exc).__name__}: {exc}") from exc


@app.post("/projects")
def create_project_endpoint(req: CreateProjectRequest) -> dict:
    fallback = default_fallback_on_llm_error() if req.fallback_on_llm_error is None else req.fallback_on_llm_error
    llm = _llm_or_none(req.use_llm)
    state = create_project(
        topic=req.topic,
        students=req.students or SAMPLE_STUDENTS,
        target_standard=req.target_standard,
        constraints=req.constraints,
        use_llm=llm is not None,
        llm=llm,
        fallback_on_llm_error=fallback,
    )
    _PROJECTS[state.project_id] = state
    _save(state)
    return {
        "project_id": state.project_id,
        "topic": state.roadmap.topic if state.roadmap else req.topic,
        "stage_count": len(state.roadmap.stages) if state.roadmap else 0,
        "package_count": len(state.roadmap.packages) if state.roadmap else 0,
        "state_path": str(_project_path(state.project_id)),
        "roadmap": state.roadmap.model_dump() if state.roadmap else None,
    }


@app.get("/projects")
def list_projects() -> dict:
    _load_saved_projects()
    return {
        "project_ids": sorted(_PROJECTS.keys()),
        "count": len(_PROJECTS),
    }


@app.get("/projects/{project_id}")
def get_project(project_id: str) -> dict:
    state = _get_project(project_id)
    return state.model_dump()


@app.get("/projects/{project_id}/packages/{package_id}")
def get_package(project_id: str, package_id: str) -> dict:
    state = _get_project(project_id)
    try:
        return state.get_package(package_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/projects/{project_id}/submissions")
def submit_package_endpoint(project_id: str, req: SubmitPackageRequest) -> dict:
    state = _get_project(project_id)
    try:
        submission = submit_package(
            state,
            package_id=req.package_id,
            student_id=req.student_id,
            report_text=req.report_text,
            artifact_paths=req.artifact_paths,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Submission failed: {type(exc).__name__}: {exc}") from exc
    _save(state)
    return submission.model_dump()


@app.post("/projects/{project_id}/submissions/{submission_id}/review")
def review_submission_endpoint(project_id: str, submission_id: str, req: ReviewRequest = ReviewRequest()) -> dict:
    state = _get_project(project_id)
    fallback = default_fallback_on_llm_error() if req.fallback_on_llm_error is None else req.fallback_on_llm_error
    llm = _llm_or_none(req.use_llm)
    try:
        review = review_submission(
            state,
            submission_id,
            use_llm=llm is not None,
            llm=llm,
            fallback_on_llm_error=fallback,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Review failed: {type(exc).__name__}: {exc}") from exc
    _save(state)
    return review.model_dump()


@app.post("/projects/{project_id}/reviews/{review_id}/understanding")
def understanding_endpoint(project_id: str, review_id: str, req: UnderstandingRequest) -> dict:
    state = _get_project(project_id)
    fallback = default_fallback_on_llm_error() if req.fallback_on_llm_error is None else req.fallback_on_llm_error
    llm = _llm_or_none(req.use_llm)
    try:
        understanding, artifact = answer_understanding(
            state,
            review_id,
            req.answers,
            use_llm=llm is not None,
            llm=llm,
            fallback_on_llm_error=fallback,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Understanding gate failed: {type(exc).__name__}: {exc}") from exc
    _save(state)
    return {
        "understanding": understanding.model_dump(),
        "artifact": artifact.model_dump(),
    }


@app.get("/projects/{project_id}/dashboard")
def dashboard_endpoint(project_id: str) -> dict:
    state = _get_project(project_id)
    dashboard = build_dashboard(state)
    return dashboard.model_dump()


@app.get("/projects/{project_id}/submissions")
def list_submissions(project_id: str) -> dict:
    state = _get_project(project_id)
    return {"submissions": [s.model_dump() for s in state.submissions]}


@app.get("/projects/{project_id}/reviews")
def list_reviews(project_id: str) -> dict:
    state = _get_project(project_id)
    return {"reviews": [r.model_dump() for r in state.review_reports]}


def main() -> None:
    import uvicorn

    uvicorn.run("researchos_backend.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
