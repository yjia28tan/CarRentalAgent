"""Scripted LLM for deterministic tests."""
import copy
from collections.abc import Callable
from itertools import count
from typing import Any

from src.llm.gateway import LLMError, LLMResponse, ToolCall, ToolSpec

ScriptStep = LLMResponse | Exception | Callable[[str, list[dict[str, Any]], list[ToolSpec]], LLMResponse]

_ids = count(1)


def say(text: str) -> LLMResponse:
    return LLMResponse(text=text, stop_reason="stop")


def call(name: str, text: str | None = None, **arguments: Any) -> LLMResponse:
    return LLMResponse(
        text=text, tool_calls=[ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)], stop_reason="tool_calls"
    )


def calls(*items: tuple[str, dict[str, Any]]) -> LLMResponse:
    """Several tool calls in one step: calls(("get_my_reservation", {...}), ("search_policy", {...}))."""
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{next(_ids)}", name=n, arguments=a) for n, a in items], stop_reason="tool_calls"
    )


class FakeLLM:
    """
    Returns scripted steps in order. A step may be an LLMResponse, an Exception to raise, or a
    callable(system, messages, tools) -> LLMResponse for assertions on what the model was sent.
    """

    def __init__(self, script: list[ScriptStep] | None = None) -> None:
        self.script = list(script or [])
        self.requests: list[dict[str, Any]] = []

    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        self.requests.append({"system": system, "messages": copy.deepcopy(messages), "tools": [t.name for t in tools]})
        if not self.script:
            raise LLMError("FakeLLM script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, LLMResponse):
            return step(system, messages, tools)
        return step

    def tool_results(self, request_index: int = -1) -> list[dict[str, Any]]:
        """Tool messages the model saw in a given request."""
        return [m for m in self.requests[request_index]["messages"] if m["role"] == "tool"]
