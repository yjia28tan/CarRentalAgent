"""
Pending actions for the two-phase write (spec section 7.4). The LLM can only create them;
only POST /actions/{id}/confirm (a UI button) can execute one.
"""
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.storage.session_store import SessionStore
from src.utils.utils import new_id, redact_code, utcnow

if TYPE_CHECKING:
    from src.agent.tools import ToolContext

# Keep the key a bit past expiry so a late confirm gets ACTION_EXPIRED rather than ACTION_NOT_FOUND.
EXPIRED_GRACE_SECONDS = 600


def pending_key(action_id: str) -> str:
    return f"pending:{action_id}"


def session_pending_key(session_id: str) -> str:
    return f"pending_for:{session_id}"


def _short_date(iso: str) -> str:
    d = datetime.fromisoformat(iso)
    return f"{d:%b} {d.day}"


def public_pending_action(pending: dict[str, Any]) -> dict[str, Any]:
    """What the UI and the LLM see (no customer/session ids, no preview internals)."""
    return {
        "action_id": pending["action_id"],
        "type": pending["type"],
        "confirmation_code": pending["confirmation_code"],
        "summary": f"{pending['confirmation_code']}: return date "
                   f"{_short_date(pending['old_return_at'])} → {_short_date(pending['new_return_at'])}",
        "old_return_at": pending["old_return_at"],
        "new_return_at": pending["new_return_at"],
        "old_total": pending["old_total"],
        "new_total": pending["new_total"],
        "price_delta": pending["price_delta"],
        "currency": pending["currency"],
        "expires_at": pending["expires_at"],
    }


def is_expired(pending: dict[str, Any]) -> bool:
    return datetime.fromisoformat(pending["expires_at"]) <= utcnow()


async def create_pending_action(ctx: "ToolContext", preview: dict[str, Any]) -> dict[str, Any]:
    ttl = ctx.settings.pending_action_ttl_seconds
    pending = {
        "action_id": new_id("act"),
        "type": "MODIFY_RETURN_DATE",
        "customer_id": ctx.request.customer_id,
        "session_id": ctx.session_id,
        "confirmation_code": preview["confirmation_code"],
        "preview_id": preview["preview_id"],
        "expected_version": preview["expected_version"],
        "old_return_at": preview["old_return_at"],
        "new_return_at": preview["new_return_at"],
        "old_total": preview["old_total"],
        "new_total": preview["new_total"],
        "price_delta": preview["price_delta"],
        "currency": preview["currency"],
        "expires_at": (utcnow() + timedelta(seconds=ttl)).isoformat(),
    }
    await ctx.sessions.set_json(pending_key(pending["action_id"]), pending, ttl + EXPIRED_GRACE_SECONDS)
    await ctx.sessions.set_json(session_pending_key(ctx.session_id), pending["action_id"], ttl)
    await ctx.audit.record(
        "WRITE_PROPOSED",
        session_id=ctx.session_id,
        customer_id=ctx.request.customer_id,
        tool="propose_reservation_change",
        args_redacted={"confirmation_code": redact_code(pending["confirmation_code"]),
                       "new_return_at": pending["new_return_at"]},
        outcome="ok",
        action_id=pending["action_id"],
        price_delta=pending["price_delta"],
    )
    return pending


async def current_pending_for_session(
    sessions: SessionStore, session_id: str, customer_id: str
) -> dict[str, Any] | None:
    """The session's still-valid pending action, so the Confirm button survives a 'yes, confirm' turn."""
    action_id = await sessions.get_json(session_pending_key(session_id))
    if not action_id:
        return None
    pending = await sessions.get_json(pending_key(action_id))
    if not pending or pending["customer_id"] != customer_id or is_expired(pending):
        return None
    return public_pending_action(pending)
