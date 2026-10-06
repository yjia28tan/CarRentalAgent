"""KB search with a Redis-backed cache (kb:{sha256(query+filters)}, 1 h)."""
import hashlib
import json
from typing import Any

from src.clients.business_api import BusinessApiClient
from src.storage.session_store import SessionStore

KB_CACHE_TTL_SECONDS = 3600


def cache_key(query: str, top_k: int, categories: list[str] | None, plan_code: str | None) -> str:
    payload = json.dumps(
        {"q": query.strip().lower(), "k": top_k, "c": sorted(categories or []), "p": plan_code}, sort_keys=True
    )
    return "kb:" + hashlib.sha256(payload.encode()).hexdigest()


class KbClient:
    def __init__(self, api: BusinessApiClient, cache: SessionStore) -> None:
        self._api = api
        self._cache = cache

    async def search(
        self, query: str, top_k: int, categories: list[str] | None, plan_code: str | None
    ) -> list[dict[str, Any]]:
        key = cache_key(query, top_k, categories, plan_code)
        cached = await self._cache.get_json(key)
        if cached is not None:
            return cached
        results = (await self._api.kb_search(query, top_k, categories, plan_code)).get("results", [])
        await self._cache.set_json(key, results, KB_CACHE_TTL_SECONDS)
        return results
