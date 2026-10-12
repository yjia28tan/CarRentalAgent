"""
LangGraph implementation of the agent turn (AGENT_ORCHESTRATOR=langgraph).

Graph: START -> agent -> (tool calls? -> tools -> agent) | END. It reuses the same LLMGateway, tool
registry, validation, per-tool timeouts, prompts and fallback replies as Orchestrator. ToolContext
(and so the verified RequestContext) travels in the runtime config, never in tool arguments.
No checkpointer: trip context and history stay in Redis/Mongo, and writes still happen only through
the authorized /actions confirm endpoint.
"""
import asyncio
import copy
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from src.agent.context import apply_tool_result
from src.agent.orchestrator import (
    EMPTY_REPLY,
    LLM_ERROR_REPLY,
    MAX_STEPS_REPLY,
    TIMEOUT_REPLY,
    TurnResult,
    record_llm_error,
    system_prompt_for,
)
from src.agent.tools import TOOLS, Tool, ToolContext, ToolExecution, execute_tool
from src.llm.gateway import LLMError, LLMGateway, ToolCall
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TurnState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    trip: dict[str, Any]
    executions: Annotated[list[ToolExecution], operator.add]
    steps: int
    reply: str | None


def _tool_ctx(config: RunnableConfig) -> ToolContext:
    return config["configurable"]["tool_ctx"]


def _route(state: TurnState) -> str:
    return END if state.get("reply") is not None else "tools"


class LangGraphOrchestrator:
    def __init__(self, llm: LLMGateway, tools: dict[str, Tool] | None = None) -> None:
        self.llm = llm
        self.tools = tools or TOOLS
        self._specs = [t.spec for t in self.tools.values()]
        self.graph = self._build()

    def _build(self):
        graph = StateGraph(TurnState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        return graph.compile()

    async def _agent_node(self, state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _tool_ctx(config)
        if state["steps"] >= ctx.settings.agent_max_steps:
            logger.warning("Step limit reached", steps=state["steps"])
            return {"reply": MAX_STEPS_REPLY}
        # LLMError propagates out of the graph and is turned into the fallback reply in run_turn.
        response = await self.llm.complete(system_prompt_for(ctx, state["trip"]), state["messages"], self._specs)
        if not response.tool_calls:
            return {"reply": (response.text or "").strip() or EMPTY_REPLY, "steps": state["steps"] + 1}
        assistant = {"role": "assistant", "content": response.text,
                     "tool_calls": [c.model_dump() for c in response.tool_calls]}
        return {"messages": [assistant], "steps": state["steps"] + 1}

    async def _tools_node(self, state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        ctx = _tool_ctx(config)
        calls = [ToolCall(**c) for c in state["messages"][-1]["tool_calls"]]
        executions = await asyncio.gather(*(execute_tool(c, ctx, self.tools) for c in calls))
        trip = copy.deepcopy(state["trip"])
        for ex in executions:
            if ex.status == "ok" and ex.args is not None:
                apply_tool_result(trip, ex.call.name, ex.args, ex.result)
        return {"messages": [ex.message() for ex in executions], "executions": list(executions), "trip": trip}

    async def run_turn(
        self, *, user_message: str, history: list[dict[str, Any]], trip: dict[str, Any], ctx: ToolContext
    ) -> TurnResult:
        """Same contract as Orchestrator.run_turn: mutates `trip` in place, never raises."""
        initial: TurnState = {
            "messages": [*history, {"role": "user", "content": user_message}],
            "trip": copy.deepcopy(trip),
            "executions": [],
            "steps": 0,
            "reply": None,
        }
        # The step limit is enforced in _agent_node; the recursion limit is only a backstop.
        config: RunnableConfig = {
            "configurable": {"tool_ctx": ctx},
            "recursion_limit": 2 * ctx.settings.agent_max_steps + 3,
        }
        last: dict[str, Any] = dict(initial)
        try:
            async with asyncio.timeout(ctx.settings.turn_timeout_seconds):
                async for snapshot in self.graph.astream(initial, config, stream_mode="values"):
                    last = snapshot
            reply = last.get("reply") or EMPTY_REPLY
        except LLMError as exc:
            await record_llm_error(ctx, exc)
            reply = LLM_ERROR_REPLY
        except GraphRecursionError:
            logger.warning("Step limit reached (graph recursion limit)", steps=last.get("steps"))
            reply = MAX_STEPS_REPLY
        except TimeoutError:
            logger.warning("Turn timed out", steps=len(last["executions"]))
            reply = TIMEOUT_REPLY

        trip.clear()
        trip.update(last["trip"])
        result = TurnResult(reply=reply)
        for ex in last["executions"]:
            result.absorb(ex)
        return result
