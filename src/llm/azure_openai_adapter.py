"""Azure OpenAI chat-completions adapter with timeouts and bounded retries on transient errors."""
import asyncio
import json
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import openai
from openai import AsyncAzureOpenAI

from src.llm.gateway import LLMError, LLMResponse, ToolCall, ToolSpec
from src.utils.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRANSIENT = (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
_BACKOFF_BASE_SECONDS = 0.5


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content"),
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in m["tool_calls"]
                ],
            })
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in tools
    ]


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        # Surfaces to the model as a VALIDATION_ERROR so it can retry with valid JSON.
        return {"_invalid_json": True}
    return parsed if isinstance(parsed, dict) else {"_invalid_json": True}


def proxied_http_client(settings: Settings) -> httpx.AsyncClient | None:
    http_proxy = settings.http_proxy or settings.https_proxy
    https_proxy = settings.https_proxy or settings.http_proxy
    if not http_proxy:
        return None
    return httpx.AsyncClient(
        mounts={
            "http://": httpx.AsyncHTTPTransport(proxy=http_proxy),
            "https://": httpx.AsyncHTTPTransport(proxy=https_proxy),
        },
        timeout=settings.llm_timeout_seconds,
    )


class AzureOpenAIAdapter:
    provider = "azure_openai"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self.max_retries = settings.llm_max_retries
        http_client = proxied_http_client(settings)
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.llm_endpoint,
            api_key=settings.llm_api_key.get_secret_value(),
            api_version=settings.llm_api_version,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retries are handled here so each attempt is logged
            http_client=http_client,
        )
        logger.info("llm_client_ready", provider=self.provider, model=self.model,
                    endpoint_host=urlparse(settings.llm_endpoint).hostname, proxied=http_client is not None)

    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": to_openai_messages(system, messages)}
        if tools:
            kwargs["tools"] = to_openai_tools(tools)

        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                completion = await self._client.chat.completions.create(**kwargs)
            except _TRANSIENT as exc:
                logger.warning("llm_call_failed", provider=self.provider, model=self.model, attempt=attempt + 1,
                               error=type(exc).__name__, latency_ms=_ms(started))
                if attempt < self.max_retries:
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * 2**attempt)
                    continue
                raise LLMError(f"LLM unavailable after {attempt + 1} attempts: {type(exc).__name__}") from exc
            except openai.OpenAIError as exc:
                logger.error("llm_call_failed", provider=self.provider, model=self.model, attempt=attempt + 1,
                             error=type(exc).__name__, status=getattr(exc, "status_code", None),
                             provider_error_code=getattr(exc, "code", None), latency_ms=_ms(started))
                raise LLMError(f"LLM request rejected: {type(exc).__name__}") from exc

            choice = completion.choices[0]
            usage = completion.usage
            logger.info(
                "llm_call",
                provider=self.provider,
                model=self.model,
                latency_ms=_ms(started),
                stop_reason=choice.finish_reason,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                tool_calls=len(choice.message.tool_calls or []),
            )
            return LLMResponse(
                text=choice.message.content,
                tool_calls=[
                    ToolCall(id=c.id, name=c.function.name, arguments=_parse_arguments(c.function.arguments))
                    for c in choice.message.tool_calls or []
                ],
                stop_reason=choice.finish_reason or "stop",
                usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
                if usage else None,
            )
        raise LLMError("unreachable")


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
