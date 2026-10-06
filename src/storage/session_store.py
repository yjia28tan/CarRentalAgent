"""Redis: session validity, trip context, and small JSON caches (quotes, KB results, ...)."""
import json
from datetime import datetime
from typing import Any, Protocol

import redis.asyncio as aioredis

SESSION_TTL_SECONDS = 30 * 60
CONTEXT_TTL_SECONDS = 24 * 3600


def dumps(value: Any) -> str:
    return json.dumps(value, default=str)


class SessionStore(Protocol):
    async def ping(self) -> None: ...

    async def create_session(self, sid: str, customer_id: str, now: datetime) -> None: ...

    async def get_session(self, sid: str) -> dict[str, str] | None: ...

    async def touch(self, sid: str, now: datetime) -> None:
        """Sliding expiry: extend the session TTL."""
        ...

    async def next_turn(self, sid: str) -> int: ...

    async def get_context(self, sid: str) -> dict[str, Any] | None: ...

    async def save_context(self, sid: str, context: dict[str, Any]) -> None: ...

    async def get_json(self, key: str) -> Any | None: ...

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None: ...

    async def delete(self, key: str) -> None: ...


class RedisSessionStore:
    def __init__(self, redis: aioredis.Redis) -> None:
        self._r = redis

    async def ping(self) -> None:
        await self._r.ping()

    async def create_session(self, sid: str, customer_id: str, now: datetime) -> None:
        key = f"session:{sid}"
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping={"customer_id": customer_id, "created_at": now.isoformat(),
                                    "last_active": now.isoformat(), "turns": 0})
            pipe.expire(key, SESSION_TTL_SECONDS)
            await pipe.execute()

    async def get_session(self, sid: str) -> dict[str, str] | None:
        data = await self._r.hgetall(f"session:{sid}")
        return data or None

    async def touch(self, sid: str, now: datetime) -> None:
        key = f"session:{sid}"
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hset(key, "last_active", now.isoformat())
            pipe.expire(key, SESSION_TTL_SECONDS)
            await pipe.execute()

    async def next_turn(self, sid: str) -> int:
        return int(await self._r.hincrby(f"session:{sid}", "turns", 1))

    async def get_context(self, sid: str) -> dict[str, Any] | None:
        return await self.get_json(f"ctx:{sid}")

    async def save_context(self, sid: str, context: dict[str, Any]) -> None:
        await self.set_json(f"ctx:{sid}", context, CONTEXT_TTL_SECONDS)

    async def get_json(self, key: str) -> Any | None:
        raw = await self._r.get(key)
        return json.loads(raw) if raw is not None else None

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        await self._r.set(key, dumps(value), ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._r.delete(key)
