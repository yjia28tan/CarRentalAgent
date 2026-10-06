"""Chat history: MongoDB is the system of record; Redis keeps the last 12 messages hot for the LLM."""
import json
from datetime import datetime
from typing import Any, Protocol

import redis.asyncio as aioredis
from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from src.storage.session_store import dumps

HISTORY_WINDOW = 12
HISTORY_TTL_SECONDS = 24 * 3600


def window_entry(message: dict[str, Any]) -> dict[str, Any]:
    """What the LLM history window keeps: role and text only (tool payloads are not replayed)."""
    return {"role": message["role"], "content": message["content"]}


class HistoryStore(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def create_conversation(self, sid: str, customer_id: str, now: datetime) -> None: ...

    async def get_conversation(self, sid: str) -> dict[str, Any] | None: ...

    async def append(self, message: dict[str, Any]) -> None: ...

    async def recent(self, sid: str, limit: int = HISTORY_WINDOW) -> list[dict[str, Any]]:
        """Last `limit` messages as {role, content}, oldest first."""
        ...

    async def list_messages(self, sid: str, limit: int) -> list[dict[str, Any]]: ...


def _public(doc: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k != "_id"}
    out["message_id"] = doc["_id"]
    return out


class MongoHistoryStore:
    def __init__(self, db: AsyncDatabase, redis: aioredis.Redis) -> None:
        self._conversations = db["conversations"]
        self._messages = db["messages"]
        self._r = redis

    async def ensure_indexes(self) -> None:
        await self._messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
        await self._conversations.create_index([("customer_id", ASCENDING), ("updated_at", DESCENDING)])

    async def create_conversation(self, sid: str, customer_id: str, now: datetime) -> None:
        await self._conversations.insert_one({
            "_id": sid, "customer_id": customer_id, "title": "New conversation", "status": "active",
            "created_at": now, "updated_at": now,
        })

    async def get_conversation(self, sid: str) -> dict[str, Any] | None:
        return await self._conversations.find_one({"_id": sid})

    async def append(self, message: dict[str, Any]) -> None:
        doc = {**message, "_id": message["message_id"]}
        doc.pop("message_id")
        await self._messages.insert_one(doc)
        await self._conversations.update_one({"_id": message["session_id"]},
                                             {"$set": {"updated_at": message["created_at"]}})
        key = f"hist:{message['session_id']}"
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.rpush(key, dumps(window_entry(message)))
            pipe.ltrim(key, -HISTORY_WINDOW, -1)
            pipe.expire(key, HISTORY_TTL_SECONDS)
            await pipe.execute()

    async def recent(self, sid: str, limit: int = HISTORY_WINDOW) -> list[dict[str, Any]]:
        cached = await self._r.lrange(f"hist:{sid}", -limit, -1)
        if cached:
            return [json.loads(item) for item in cached]
        cursor = self._messages.find({"session_id": sid}).sort("created_at", DESCENDING).limit(limit)
        docs = [doc async for doc in cursor]
        return [window_entry(d) for d in reversed(docs)]

    async def list_messages(self, sid: str, limit: int) -> list[dict[str, Any]]:
        cursor = self._messages.find({"session_id": sid}).sort("created_at", DESCENDING).limit(limit)
        docs = [doc async for doc in cursor]
        return [_public(d) for d in reversed(docs)]
