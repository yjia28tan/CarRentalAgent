"""
Second phase of the two-phase write: only these endpoints (UI buttons) can execute a pending action.
There is deliberately no LLM tool that reaches this code.
"""
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Path
from pydantic import BaseModel, ConfigDict, Field
from pymongo.errors import PyMongoError
from redis.exceptions import RedisError

from src.agent.pending import EXPIRED_GRACE_SECONDS, is_expired, pending_key, session_pending_key
from src.clients.business_api import BusinessApiClient, UpstreamError
from src.deps import RequestContext, Services, get_request_context, get_services
from src.shared.errors import AppError, ErrorCode
from src.utils.logger import get_logger
from src.utils.utils import new_id, redact_code, utcnow

logger = get_logger(__name__)

router = APIRouter(prefix="/actions", tags=["actions"])

ActionId = Annotated[str, Path(pattern=r"^act_[0-9A-Z]{26}$")]
IdempotencyKey = Annotated[
    str, Header(alias="Idempotency-Key", min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_\-:.]+$")
]
IDEMPOTENCY_TTL_SECONDS = 24 * 3600
# After these the action can never succeed, so it is dropped instead of kept for a retry.
FINAL_REJECTIONS = {
    ErrorCode.RESERVATION_NOT_FOUND, ErrorCode.INVALID_STATE, ErrorCode.VERSION_CONFLICT,
    ErrorCode.NOT_AVAILABLE, ErrorCode.PREVIEW_EXPIRED, ErrorCode.VALIDATION_ERROR,
}


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: Annotated[str | None, Field(pattern=r"^sess_[0-9A-Z]{26}$")] = None


def _result_key(action_id: str) -> str:
    return f"action_result:{action_id}"


def _short_date(iso: str) -> str:
    d = datetime.fromisoformat(iso)
    return f"{d:%b} {d.day}"


async def _load_owned(services: Services, ctx: RequestContext, action_id: str, session_id: str | None) -> dict:
    pending = await services.sessions.get_json(pending_key(action_id))
    if pending is None:
        raise AppError(ErrorCode.ACTION_NOT_FOUND)
    if pending["customer_id"] != ctx.customer_id or (session_id and pending["session_id"] != session_id):
        await services.audit.record("AUTHZ_DENIED", session_id=session_id, customer_id=ctx.customer_id,
                                    outcome="ACTION_NOT_FOUND", resource="pending_action", action_id=action_id)
        raise AppError(ErrorCode.ACTION_NOT_FOUND)
    return pending


async def _clear_session_pointer(services: Services, pending: dict[str, Any]) -> None:
    key = session_pending_key(pending["session_id"])
    if await services.sessions.get_json(key) == pending["action_id"]:
        await services.sessions.delete(key)


async def _system_note(services: Services, ctx: RequestContext, pending: dict[str, Any], text: str) -> None:
    try:
        await services.history.append({
            "message_id": new_id("msg"), "session_id": pending["session_id"], "customer_id": ctx.customer_id,
            "turn_id": None, "role": "system", "content": text, "tool_calls": [], "sources": [], "cards": [],
            "request_id": ctx.request_id, "created_at": utcnow(),
        })
    except (RedisError, PyMongoError) as exc:
        logger.error("history_persist_failed", error=type(exc).__name__)


@router.post("/{action_id}/confirm")
async def confirm_action(
    action_id: ActionId,
    idempotency_key: IdempotencyKey,
    body: ActionRequest | None = None,
    ctx: RequestContext = Depends(get_request_context),
    services: Services = Depends(get_services),
):
    sessions = services.sessions

    replay = await sessions.get_json(f"idem:{idempotency_key}")
    if replay is not None:
        if replay["customer_id"] != ctx.customer_id or replay["action_id"] != action_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "This Idempotency-Key was already used for another request.")
        return replay["response"]
    done = await sessions.get_json(_result_key(action_id))
    if done is not None and done["customer_id"] == ctx.customer_id:
        return done["response"]  # double click with a fresh key: same outcome, no second write

    pending = await _load_owned(services, ctx, action_id, body.session_id if body else None)
    if is_expired(pending):
        await sessions.delete(pending_key(action_id))
        await _clear_session_pointer(services, pending)
        raise AppError(ErrorCode.ACTION_EXPIRED)
    if await sessions.pop_json(pending_key(action_id)) is None:
        raise AppError(ErrorCode.ACTION_NOT_FOUND, "This change is already being processed.")

    code = pending["confirmation_code"]
    audit_common = {"session_id": pending["session_id"], "customer_id": ctx.customer_id,
                    "args_redacted": {"confirmation_code": redact_code(code)}, "action_id": action_id}
    api = BusinessApiClient(services.http, ctx.raw_token, ctx.request_id)
    try:
        # The action id is the downstream idempotency key, so any retry of this action applies at most once.
        applied = await api.apply_change(code, pending["preview_id"], pending["expected_version"], action_id)
    except UpstreamError as exc:
        await services.audit.record("WRITE_REJECTED", outcome=str(exc.code), **audit_common)
        final = exc.code in FINAL_REJECTIONS
        logger.warning("Reservation change rejected", action_id=action_id, confirmation_code=redact_code(code),
                       error_code=str(exc.code), retryable=not final)
        if final:
            await _clear_session_pointer(services, pending)
            await _system_note(services, ctx, pending, f"Reservation {code} was not changed: {exc.message}")
        else:
            remaining = (datetime.fromisoformat(pending["expires_at"]) - utcnow()).total_seconds()
            await sessions.set_json(pending_key(action_id), pending, int(remaining) + EXPIRED_GRACE_SECONDS)
        raise AppError(exc.code, exc.message) from None

    response = {"status": "APPLIED", "action_id": action_id, "reservation": applied["reservation"]}
    record = {"customer_id": ctx.customer_id, "action_id": action_id, "response": response}
    await sessions.set_json(f"idem:{idempotency_key}", record, IDEMPOTENCY_TTL_SECONDS)
    await sessions.set_json(_result_key(action_id), record, IDEMPOTENCY_TTL_SECONDS)
    await _clear_session_pointer(services, pending)
    await services.audit.record("WRITE_CONFIRMED", outcome="APPLIED",
                                new_version=applied["reservation"]["version"], **audit_common)
    logger.info("Reservation change applied", action_id=action_id, confirmation_code=redact_code(code),
                new_version=applied["reservation"]["version"])
    await _system_note(services, ctx, pending,
                       f"Reservation {code} updated: return date {_short_date(pending['new_return_at'])}.")
    return response


@router.post("/{action_id}/cancel")
async def cancel_action(
    action_id: ActionId,
    body: ActionRequest | None = None,
    ctx: RequestContext = Depends(get_request_context),
    services: Services = Depends(get_services),
):
    pending = await _load_owned(services, ctx, action_id, body.session_id if body else None)
    await services.sessions.delete(pending_key(action_id))
    await _clear_session_pointer(services, pending)
    await services.audit.record(
        "WRITE_REJECTED", session_id=pending["session_id"], customer_id=ctx.customer_id,
        args_redacted={"confirmation_code": redact_code(pending["confirmation_code"])},
        outcome="DISCARDED_BY_CUSTOMER", action_id=action_id,
    )
    logger.info("Pending reservation change discarded", action_id=action_id,
                confirmation_code=redact_code(pending["confirmation_code"]))
    await _system_note(services, ctx, pending,
                       f"Proposed change to reservation {pending['confirmation_code']} was discarded.")
    return {"status": "DISCARDED"}
