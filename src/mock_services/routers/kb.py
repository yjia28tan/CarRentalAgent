"""POST /v1/kb/search: semantic search over policy chunks (pgvector)."""
import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from src.mock_services.deps import Embedder, get_embedder, get_repo
from src.mock_services.repository import Repository
from src.shared.auth import require_principal

router = APIRouter(prefix="/v1/kb", tags=["kb"], dependencies=[Depends(require_principal)])

KbCategory = Literal["INSURANCE", "ACCIDENT", "CANCELLATION", "ROADSIDE", "TERMS"]


class KbFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categories: Annotated[list[KbCategory] | None, Field(max_length=5)] = None
    plan_code: Annotated[str | None, Field(pattern=r"^[A-Z_]{2,40}$")] = None


class KbSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, Field(min_length=3, max_length=300)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
    filters: KbFilters = KbFilters()


@router.post("/search")
async def kb_search(
    body: KbSearchRequest,
    repo: Repository = Depends(get_repo),
    embed: Embedder = Depends(get_embedder),
):
    vector = (await asyncio.to_thread(embed, [body.query]))[0]
    hits = await repo.kb_search(vector, body.top_k, body.filters.categories, body.filters.plan_code)
    return {
        "results": [
            {"doc_id": h.doc_id, "title": h.title, "section": h.section, "content": h.content,
             "score": round(h.score, 4)}
            for h in hits
        ]
    }
