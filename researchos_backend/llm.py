from __future__ import annotations

from typing import Any, Dict, Optional

from .modal_llm import ModalLLMConfig, ModalOpenAIChat, LLMConfigError, LLMResponseError, extract_json


class LLMUnavailable(RuntimeError):
    pass


class OpenAICompatibleLLM:
    """Backward-compatible adapter.

    Prefer ModalOpenAIChat for new code. This wrapper keeps the old API working.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 180,
    ) -> None:
        config = ModalLLMConfig.from_env()
        if base_url is not None:
            config.base_url = base_url
        if api_key is not None:
            config.api_key = api_key
        if model is not None:
            config.model = model
        config.timeout = timeout
        self.config = config

    @property
    def configured(self) -> bool:
        return bool(self.config.base_url and self.config.model)

    def _client(self) -> ModalOpenAIChat:
        if not self.configured:
            raise LLMUnavailable("LLM_BASE_URL/MODAL_LLM_BASE_URL and LLM_MODEL/MODAL_LLM_MODEL must be set")
        try:
            return ModalOpenAIChat(self.config)
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc

    def generate_text(self, system: str, user: str, temperature: float = 0.2) -> str:
        return self._client().complete(system, user, temperature=temperature)

    def generate_json(self, system: str, user: str, temperature: float = 0.1) -> Dict[str, Any]:
        data = self._client().complete_json(system, user, temperature=temperature)
        if not isinstance(data, dict):
            raise ValueError("LLM response JSON must be an object")
        return data
