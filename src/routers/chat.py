"""Chat sessions and the per-turn flow (spec section 4)."""
from datetime import datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo.errors import PyMongoError
from redis.exceptions import RedisError

from src.agent.context import empty_context
from src.agent.orchestrator import build_orchestrator
from src.agent.pending import current_pending_for_session
from src.agent.tools import ToolContext
from src.clients.business_api import BusinessApiClient
from src.clients.kb_client import KbClient
from src.deps import RequestContext, Services, get_request_context, get_services
from src.shared.errors import AppError, ErrorCode
from src.utils.logger import bind_log_context, get_logger
from src.utils.utils import new_id, utcnow

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

SessionId = Annotated[str, Path(pattern=r"^sess_[0-9A-Z]{26}$")]


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: Annotated[str, Field(min_length=1, max_length=2000)]
    client_message_id: Annotated[str | None, Field(max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")] = None

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v.strip()


async def _deny(services: Services, ctx: RequestContext, sid: str) -> AppError:
    await services.audit.record("AUTHZ_DENIED", session_id=sid, customer_id=ctx.customer_id,
                                outcome="SESSION_NOT_FOUND", resource="chat_session")
    return AppError(ErrorCode.SESSION_NOT_FOUND)


async def _require_active_session(services: Services, ctx: RequestContext, sid: str) -> None:
    session = await services.sessions.get_session(sid)
    if session is None:
        raise AppError(ErrorCode.SESSION_NOT_FOUND)
    if session["customer_id"] != ctx.customer_id:
        raise await _deny(services, ctx, sid)


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


@router.post("/sessions")
async def create_session(
    ctx: RequestContext = Depends(get_request_context), services: Services = Depends(get_services)
):
    sid = new_id("sess")
    now = utcnow()
    await services.sessions.create_session(sid, ctx.customer_id, now)
    await services.history.create_conversation(sid, ctx.customer_id, now)
    logger.info("session_created", session_id=sid)
    return {"session_id": sid, "created_at": now.isoformat()}


@router.get("/sessions/{sid}/messages")
async def list_messages(
    sid: SessionId,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ctx: RequestContext = Depends(get_request_context),
    services: Services = Depends(get_services),
):
    conversation = await services.history.get_conversation(sid)
    if conversation is None:
        raise AppError(ErrorCode.SESSION_NOT_FOUND)
    if conversation["customer_id"] != ctx.customer_id:
        raise await _deny(services, ctx, sid)
    messages = await services.history.list_messages(sid, limit)
    keys = ("message_id", "turn_id", "role", "content", "cards", "sources", "tool_calls", "created_at")
    return {"messages": [{k: _iso(m.get(k)) for k in keys} for m in messages]}


@router.post("/sessions/{sid}/messages")
async def send_message(
    sid: SessionId,
    body: SendMessageRequest,
    ctx: RequestContext = Depends(get_request_context),
    services: Services = Depends(get_services),
):
    settings = services.settings
    await _require_active_session(services, ctx, sid)
    bind_log_context(session_id=sid)

    received_at = utcnow()
    await services.sessions.touch(sid, received_at)
    trip = await services.sessions.get_context(sid) or empty_context()
    history = await services.history.recent(sid)

    api = BusinessApiClient(services.http, ctx.raw_token, ctx.request_id)
    tool_ctx = ToolContext(
        request=ctx, session_id=sid, api=api, kb=KbClient(api, services.sessions), sessions=services.sessions,
        audit=services.audit, settings=settings, now=received_at.astimezone(ZoneInfo(settings.default_timezone)),
    )
    result = await build_orchestrator(services.llm, settings.agent_orchestrator).run_turn(
        user_message=body.message, history=history, trip=trip, ctx=tool_ctx
    )

    assistant_message_id = new_id("msg")
    try:
        await services.sessions.save_context(sid, trip)
        turn_id = await services.sessions.next_turn(sid)
        base = {"session_id": sid, "customer_id": ctx.customer_id, "turn_id": turn_id, "request_id": ctx.request_id}
        await services.history.append({
            **base, "message_id": new_id("msg"), "role": "user", "content": body.message,
            "client_message_id": body.client_message_id, "tool_calls": [], "sources": [], "cards": [],
            "created_at": received_at,
        })
        await services.history.append({
            **base, "message_id": assistant_message_id, "role": "assistant", "content": result.reply,
            "tool_calls": result.tool_calls_for_history, "sources": result.sources, "cards": result.cards,
            "created_at": utcnow(),
        })
    except (RedisError, PyMongoError) as exc:
        # The customer still gets the answer; only the history write is lost.
        logger.error("history_persist_failed", error=type(exc).__name__)

    # Keep showing an unconfirmed change (e.g. after the customer types "yes, confirm"): the LLM can't apply it.
    pending_action = result.pending_action or await current_pending_for_session(
        services.sessions, sid, ctx.customer_id
    )

    return {
        "request_id": ctx.request_id,
        "message_id": assistant_message_id,
        "reply": result.reply,
        "cards": result.cards,
        "pending_action": pending_action,
        "sources": result.sources,
        "tool_trace": result.tool_trace,
    }
