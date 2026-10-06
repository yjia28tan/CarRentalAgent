"""
Tool-calling loop (spec section 7.2), framework-free by design.

Each turn calls the LLM up to AGENT_MAX_STEPS times. Requested tools run concurrently through
execute_tool, which validates arguments and enforces TOOL_TIMEOUT_SECONDS per call; validation and
upstream errors go back to the LLM as structured results so it can recover or ask the user.
The whole turn is bounded by TURN_TIMEOUT_SECONDS, and no exception escapes the loop: LLM failures,
timeouts and step exhaustion all return a fixed, friendly reply.
Why no LangChain/LangGraph by default, and how this maps to LangGraph: see README, "Orchestration: why
not LangChain/LangGraph?". The LangGraph variant (AGENT_ORCHESTRATOR=langgraph) is in langgraph_orchestrator.py.
"""
import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from src.agent.context import apply_tool_result
from src.agent.prompts import build_system_prompt
from src.agent.tools import TOOLS, Tool, ToolContext, ToolExecution, execute_tool, redact_args
from src.llm.gateway import LLMError, LLMGateway
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_STEPS_REPLY = "I wasn't able to complete that request. Could you rephrase or break it into smaller steps?"
LLM_ERROR_REPLY = "Sorry, I'm having trouble reaching our assistant service right now. Please try again in a moment."
TIMEOUT_REPLY = "Sorry, that took longer than expected. Please try again."
EMPTY_REPLY = "Sorry, I couldn't come up with an answer. Could you rephrase your question?"


@dataclass
class TurnResult:
    reply: str
    cards: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    executions: list[ToolExecution] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None

    @property
    def tool_trace(self) -> list[dict[str, Any]]:
        return [e.trace() for e in self.executions]

    @property
    def tool_calls_for_history(self) -> list[dict[str, Any]]:
        """Stored in Mongo: no result payloads, arguments redacted."""
        return [
            {"id": e.call.id, "name": e.call.name, "args": redact_args(e.call.arguments), "status": e.status,
             "error_code": e.error_code, "latency_ms": e.latency_ms}
            for e in self.executions
        ]

    def absorb(self, ex: ToolExecution) -> None:
        """Collect a tool execution's trace, cards, de-duplicated sources and pending action."""
        self.executions.append(ex)
        self.cards.extend(ex.cards)
        seen = {(s["doc_id"], s["section"]) for s in self.sources}
        self.sources.extend(s for s in ex.sources if (s["doc_id"], s["section"]) not in seen)
        if ex.pending_action is not None:
            self.pending_action = ex.pending_action


class TurnOrchestrator(Protocol):
    async def run_turn(
        self, *, user_message: str, history: list[dict[str, Any]], trip: dict[str, Any], ctx: ToolContext
    ) -> TurnResult:
        """Mutates `trip` in place from successful tool results. Never raises."""
        ...


def system_prompt_for(ctx: ToolContext, trip: dict[str, Any]) -> str:
    s = ctx.settings
    return build_system_prompt(company=s.company_name, now_local=ctx.now, timezone=s.default_timezone,
                               customer_name=ctx.request.customer_name, trip=trip)


async def record_llm_error(ctx: ToolContext, exc: LLMError) -> None:
    logger.error("LLM unavailable; returning fallback reply", error=str(exc))
    await ctx.audit.record("LLM_ERROR", session_id=ctx.session_id, customer_id=ctx.request.customer_id,
                           outcome="fallback", error=str(exc)[:300])


def build_orchestrator(llm: LLMGateway, kind: str = "custom") -> TurnOrchestrator:
    """AGENT_ORCHESTRATOR=custom|langgraph. Both share tools, validation, prompts and fallbacks."""
    if kind == "langgraph":
        return _langgraph_orchestrator(llm)
    return Orchestrator(llm)


@lru_cache(maxsize=8)
def _langgraph_orchestrator(llm: LLMGateway) -> TurnOrchestrator:
    # Compiling the graph once per gateway, not per turn.
    from src.agent.langgraph_orchestrator import LangGraphOrchestrator

    return LangGraphOrchestrator(llm)


class Orchestrator:
    def __init__(self, llm: LLMGateway, tools: dict[str, Tool] | None = None) -> None:
        self.llm = llm
        self.tools = tools or TOOLS

    async def run_turn(
        self, *, user_message: str, history: list[dict[str, Any]], trip: dict[str, Any], ctx: ToolContext
    ) -> TurnResult:
        """Mutates `trip` in place from successful tool results. Never raises."""
        result = TurnResult(reply="")
        try:
            async with asyncio.timeout(ctx.settings.turn_timeout_seconds):
                result.reply = await self._loop(user_message, history, trip, ctx, result)
        except TimeoutError:
            logger.warning("Turn timed out", steps=len(result.executions))
            result.reply = TIMEOUT_REPLY
        return result

    async def _loop(
        self, user_message: str, history: list[dict[str, Any]], trip: dict[str, Any], ctx: ToolContext,
        result: TurnResult,
    ) -> str:
        settings = ctx.settings
        messages: list[dict[str, Any]] = [*history, {"role": "user", "content": user_message}]
        specs = [t.spec for t in self.tools.values()]

        for _ in range(settings.agent_max_steps):
            try:
                response = await self.llm.complete(system_prompt_for(ctx, trip), messages, specs)
            except LLMError as exc:
                await record_llm_error(ctx, exc)
                return LLM_ERROR_REPLY

            if not response.tool_calls:
                return (response.text or "").strip() or EMPTY_REPLY

            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": [c.model_dump() for c in response.tool_calls],
            })
            executions = await asyncio.gather(*(execute_tool(c, ctx, self.tools) for c in response.tool_calls))
            for ex in executions:
                messages.append(ex.message())
                result.absorb(ex)
                if ex.status == "ok" and ex.args is not None:
                    apply_tool_result(trip, ex.call.name, ex.args, ex.result)

        logger.warning("Step limit reached", steps=settings.agent_max_steps)
        return MAX_STEPS_REPLY
