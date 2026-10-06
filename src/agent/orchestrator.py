"""Tool-calling loop (spec section 7.2)."""
import asyncio
from dataclasses import dataclass, field
from typing import Any

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
            logger.warning("turn_timeout", steps=len(result.executions))
            result.reply = TIMEOUT_REPLY
        return result

    async def _loop(
        self, user_message: str, history: list[dict[str, Any]], trip: dict[str, Any], ctx: ToolContext,
        result: TurnResult,
    ) -> str:
        settings = ctx.settings
        messages: list[dict[str, Any]] = [*history, {"role": "user", "content": user_message}]
        specs = [t.spec for t in self.tools.values()]
        seen_sources: set[tuple[str, str]] = set()

        for _ in range(settings.agent_max_steps):
            system = build_system_prompt(
                company=settings.company_name, now_local=ctx.now, timezone=settings.default_timezone,
                customer_name=ctx.request.customer_name, trip=trip,
            )
            try:
                response = await self.llm.complete(system, messages, specs)
            except LLMError as exc:
                logger.error("llm_error", error=str(exc))
                await ctx.audit.record("LLM_ERROR", session_id=ctx.session_id, customer_id=ctx.request.customer_id,
                                       outcome="fallback", error=str(exc)[:300])
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
                result.executions.append(ex)
                result.cards.extend(ex.cards)
                for s in ex.sources:
                    key = (s["doc_id"], s["section"])
                    if key not in seen_sources:
                        seen_sources.add(key)
                        result.sources.append(s)
                if ex.status == "ok" and ex.args is not None:
                    apply_tool_result(trip, ex.call.name, ex.args, ex.result)
                if ex.pending_action is not None:
                    result.pending_action = ex.pending_action

        logger.warning("max_steps_reached", steps=settings.agent_max_steps)
        return MAX_STEPS_REPLY
