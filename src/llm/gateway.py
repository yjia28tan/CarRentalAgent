"""
Provider-agnostic LLM interface.

Messages use a neutral shape that adapters translate to their provider's format:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str | None, "tool_calls": [{"id", "name", "arguments": dict}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str (JSON)}
"""
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str = "stop"
    usage: dict[str, int] | None = None


class LLMError(Exception):
    """The LLM could not produce a response (after retries). Message is safe to log, not to show."""


class LLMGateway(Protocol):
    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse: ...


class UnconfiguredLLM:
    """Used when LLM settings are missing, so the app still starts and replies with the fallback message."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        raise LLMError(self.reason)


def create_llm_gateway() -> LLMGateway:
    from src.utils.config import get_settings

    settings = get_settings()
    provider = settings.llm_provider.strip().lower()
    if provider == "azure_openai":
        missing = [
            name for name, value in (
                ("LLM_MODEL", settings.llm_model),
                ("LLM_ENDPOINT", settings.llm_endpoint),
                ("LLM_API_VERSION", settings.llm_api_version),
                ("LLM_API_KEY", settings.llm_api_key.get_secret_value()),
            ) if not value
        ]
        if missing:
            return UnconfiguredLLM(f"Azure OpenAI not configured; missing {', '.join(missing)}")
        from src.llm.azure_openai_adapter import AzureOpenAIAdapter

        return AzureOpenAIAdapter(settings)
    return UnconfiguredLLM(f"Unsupported LLM_PROVIDER {settings.llm_provider!r}")
