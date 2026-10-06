"""Audit trail in MongoDB `audit_logs`. Writing an audit record must never break a request."""
from typing import Any, Protocol

from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from src.utils.logger import get_logger, get_request_id
from src.utils.utils import utcnow

logger = get_logger(__name__)

AUDIT_EVENTS = {
    "TOOL_CALL", "AUTHZ_DENIED", "WRITE_PROPOSED", "WRITE_CONFIRMED", "WRITE_REJECTED",
    "LLM_ERROR", "RATE_LIMITED", "VALIDATION_ERROR",
}


def build_record(event: str, **fields: Any) -> dict[str, Any]:
    if event not in AUDIT_EVENTS:
        raise ValueError(f"unknown audit event {event}")
    return {
        "ts": utcnow(),
        "request_id": get_request_id(),
        "session_id": fields.pop("session_id", None),
        "customer_id": fields.pop("customer_id", None),
        "event": event,
        "tool": fields.pop("tool", None),
        "args_redacted": fields.pop("args_redacted", None),
        "outcome": fields.pop("outcome", None),
        "detail": fields or None,
    }


class AuditLog(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def record(self, event: str, **fields: Any) -> None: ...


class MongoAuditLog:
    def __init__(self, db: AsyncDatabase) -> None:
        self._col = db["audit_logs"]

    async def ensure_indexes(self) -> None:
        await self._col.create_index([("customer_id", ASCENDING), ("ts", DESCENDING)])
        await self._col.create_index([("event", ASCENDING), ("ts", DESCENDING)])

    async def record(self, event: str, **fields: Any) -> None:
        record = build_record(event, **fields)
        try:
            await self._col.insert_one(record)
        except Exception as exc:  # audit failures are logged, never raised
            logger.error("audit_write_failed", audit_event=event, error=type(exc).__name__)
