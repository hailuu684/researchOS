"""Backend-only ResearchOS demo.

Core flow:
PI topic -> roadmap -> learning-and-research packages -> student submission
-> agent review -> understanding gate -> evidence registry -> PI dashboard.

This package can run deterministically or with an OpenAI-compatible Modal endpoint.
"""

from .modal_llm import ModalLLMConfig, ModalOpenAIChat
from .orchestrator import (
    answer_understanding,
    build_dashboard,
    create_project,
    load_state,
    review_submission,
    save_state,
    submit_package,
)

__all__ = [
    "ModalLLMConfig",
    "ModalOpenAIChat",
    "create_project",
    "submit_package",
    "review_submission",
    "answer_understanding",
    "build_dashboard",
    "save_state",
    "load_state",
]
