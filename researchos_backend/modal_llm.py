from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class LLMConfigError(RuntimeError):
    """Raised when the Modal/OpenAI-compatible endpoint is not configured."""


class LLMResponseError(RuntimeError):
    """Raised when the endpoint returns a response that cannot be used."""


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


@dataclass
class ModalLLMConfig:
    """Configuration for an OpenAI-compatible Modal endpoint.

    The user's Modal endpoint follows this pattern:

        OpenAI(
            base_url="https://...modal.direct/v1",
            api_key="unused",
            default_headers={"Modal-Key": "...", "Modal-Secret": "..."},
        )

    Keep secrets outside source code. Set them in the notebook/session using env vars.
    """

    base_url: str
    api_key: str = "unused"
    model: str = "google/gemma-4-E4B-it"
    modal_key: Optional[str] = None
    modal_secret: Optional[str] = None
    timeout: float = 180.0
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9
    reasoning_enabled: bool = True

    @classmethod
    def from_env(cls) -> "ModalLLMConfig":
        base_url = os.getenv("MODAL_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or ""
        return cls(
            base_url=base_url,
            api_key=os.getenv("MODAL_LLM_API_KEY") or os.getenv("LLM_API_KEY") or "unused",
            model=os.getenv("MODAL_LLM_MODEL") or os.getenv("LLM_MODEL") or "google/gemma-4-E4B-it",
            modal_key=os.getenv("MODAL_KEY") or os.getenv("MODAL_LLM_KEY"),
            modal_secret=os.getenv("MODAL_SECRET") or os.getenv("MODAL_LLM_SECRET"),
            timeout=float(os.getenv("MODAL_LLM_TIMEOUT", "180")),
            max_tokens=int(os.getenv("MODAL_LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("MODAL_LLM_TEMPERATURE", "0.2")),
            top_p=float(os.getenv("MODAL_LLM_TOP_P", "0.9")),
            reasoning_enabled=_env_bool("MODAL_LLM_REASONING", True),
        )

    def validate(self) -> None:
        if not self.base_url:
            raise LLMConfigError(
                "MODAL_LLM_BASE_URL or LLM_BASE_URL is required. Example: "
                "https://your-app--endpoint.modal.direct/v1"
            )
        if not self.model:
            raise LLMConfigError("MODAL_LLM_MODEL or LLM_MODEL is required.")

    def default_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.modal_key:
            headers["Modal-Key"] = self.modal_key
        if self.modal_secret:
            headers["Modal-Secret"] = self.modal_secret
        return headers


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json(text: str) -> Any:
    """Extract a JSON object or array from a chat response.

    This accepts:
    - raw JSON
    - ```json fenced JSON```
    - prose before/after one JSON object/array
    """
    if not text or not text.strip():
        raise LLMResponseError("Empty LLM response.")

    stripped = _strip_markdown_fence(text)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    candidates: List[str] = []
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(open_char)
        end = stripped.rfind(close_char)
        if start >= 0 and end > start:
            candidates.append(stripped[start : end + 1])

    errors: List[str] = []
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

    preview = stripped[:800].replace("\n", " ")
    raise LLMResponseError(f"Could not parse JSON from LLM response. Preview: {preview}. Errors: {errors}")


class ModalOpenAIChat:
    """Small wrapper around the OpenAI Python client for Modal endpoints."""

    def __init__(self, config: Optional[ModalLLMConfig] = None) -> None:
        self.config = config or ModalLLMConfig.from_env()
        self.config.validate()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError("Install openai: pip install openai") from exc

        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            default_headers=self.config.default_headers(),
            timeout=self.config.timeout,
        )

    @classmethod
    def from_env(cls) -> "ModalOpenAIChat":
        return cls(ModalLLMConfig.from_env())

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        reasoning_enabled: Optional[bool] = None,
    ) -> str:
        enabled = self.config.reasoning_enabled if reasoning_enabled is None else reasoning_enabled
        extra_body = {"reasoning": {"enabled": True}} if enabled else None
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "top_p": self.config.top_p if top_p is None else top_p,
            "stream": False,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        completion = self._client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if content is None:
            raise LLMResponseError("LLM completion returned no message content.")
        return content

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )

    def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        text = self.complete(system, user, **kwargs)
        return extract_json(text)

    def smoke_test(self) -> str:
        return self.complete(
            system="You are a concise technical assistant.",
            user="Return exactly this JSON object with no prose: {\"status\": \"ok\", \"message\": \"modal endpoint works\"}",
            temperature=0.0,
            max_tokens=256,
        )
