"""In-memory store implementations for tests and local runs without Redis/Mongo."""
import copy
import json
import time
from datetime import datetime
from typing import Any

from src.storage.audit import build_record
from src.storage.history_store import HISTORY_WINDOW, window_entry
from src.storage.session_store import SESSION_TTL_SECONDS, dumps


class InMemorySessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, str]] = {}
        self.kv: dict[str, tuple[str, float]] = {}
        self._session_expiry: dict[str, float] = {}

    async def ping(self) -> None:
        return None

    async def create_session(self, sid: str, customer_id: str, now: datetime) -> None:
        self.sessions[sid] = {"customer_id": customer_id, "created_at": now.isoformat(),
                              "last_active": now.isoformat(), "turns": "0"}
        self._session_expiry[sid] = time.monotonic() + SESSION_TTL_SECONDS

    async def get_session(self, sid: str) -> dict[str, str] | None:
        if self._session_expiry.get(sid, 0) <= time.monotonic():
            self.sessions.pop(sid, None)
            return None
        return dict(self.sessions[sid])

    async def touch(self, sid: str, now: datetime) -> None:
        if sid in self.sessions:
            self.sessions[sid]["last_active"] = now.isoformat()
            self._session_expiry[sid] = time.monotonic() + SESSION_TTL_SECONDS

    def expire_session(self, sid: str) -> None:
        self._session_expiry[sid] = 0

    async def next_turn(self, sid: str) -> int:
        turns = int(self.sessions[sid]["turns"]) + 1
        self.sessions[sid]["turns"] = str(turns)
        return turns

    async def get_context(self, sid: str) -> dict[str, Any] | None:
        return await self.get_json(f"ctx:{sid}")

    async def save_context(self, sid: str, context: dict[str, Any]) -> None:
        await self.set_json(f"ctx:{sid}", context, 24 * 3600)

    async def get_json(self, key: str) -> Any | None:
        item = self.kv.get(key)
        if item is None or item[1] <= time.monotonic():
            self.kv.pop(key, None)
            return None
        return json.loads(item[0])

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.kv[key] = (dumps(value), time.monotonic() + ttl_seconds)

    async def delete(self, key: str) -> None:
        self.kv.pop(key, None)


class InMemoryHistoryStore:
    def __init__(self) -> None:
        self.conversations: dict[str, dict[str, Any]] = {}
        self.messages: list[dict[str, Any]] = []

    async def ensure_indexes(self) -> None:
        return None

    async def create_conversation(self, sid: str, customer_id: str, now: datetime) -> None:
        self.conversations[sid] = {"_id": sid, "customer_id": customer_id, "title": "New conversation",
                                   "status": "active", "created_at": now, "updated_at": now}

    async def get_conversation(self, sid: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.conversations.get(sid))

    async def append(self, message: dict[str, Any]) -> None:
        self.messages.append(copy.deepcopy(message))

    async def recent(self, sid: str, limit: int = HISTORY_WINDOW) -> list[dict[str, Any]]:
        return [window_entry(m) for m in self.messages if m["session_id"] == sid][-limit:]

    async def list_messages(self, sid: str, limit: int) -> list[dict[str, Any]]:
        return copy.deepcopy([m for m in self.messages if m["session_id"] == sid][-limit:])


class InMemoryAuditLog:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def ensure_indexes(self) -> None:
        return None

    async def record(self, event: str, **fields: Any) -> None:
        self.records.append(build_record(event, **fields))

    def events(self, event: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.records if event is None or r["event"] == event]
