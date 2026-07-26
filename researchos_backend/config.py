from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def load_local_env(env_file: str | Path | None = None) -> None:
    """Load .env for local VSCode/SSH development.

    Existing environment variables win over .env values. This lets you run the same
    code locally, over SSH, or inside a container without changing source code.
    """
    if env_file is None:
        load_dotenv(override=False)
    else:
        load_dotenv(dotenv_path=env_file, override=False)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def state_dir() -> Path:
    return Path(os.getenv("RESEARCHOS_STATE_DIR", "outputs/state"))


def default_use_llm() -> bool:
    return env_bool("RESEARCHOS_DEFAULT_USE_LLM", True)


def default_fallback_on_llm_error() -> bool:
    return env_bool("RESEARCHOS_FALLBACK_ON_LLM_ERROR", True)
