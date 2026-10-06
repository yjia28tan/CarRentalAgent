"""Request-scoped identity and app-wide service wiring for the agent API."""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
from fastapi import Depends, Request

from src.llm.gateway import LLMGateway
from src.shared.auth import Principal, require_principal
from src.storage.audit import AuditLog
from src.storage.history_store import HistoryStore
from src.storage.session_store import SessionStore
from src.utils.logger import bind_log_context


@dataclass(frozen=True)
class RequestContext:
    """Identity comes only from the verified JWT; never from the request body or the LLM."""

    customer_id: str
    customer_name: str
    request_id: str
    raw_token: str


@dataclass
class Services:
    sessions: SessionStore
    history: HistoryStore
    audit: AuditLog
    llm: LLMGateway
    http: httpx.AsyncClient  # to mock_services
    ping_mongo: Callable[[], Awaitable[object]]


def get_services(request: Request) -> Services:
    return request.app.state.services


async def get_request_context(request: Request, principal: Principal = Depends(require_principal)) -> RequestContext:
    customer_id = str(principal.customer_id)
    bind_log_context(customer_id=customer_id)
    return RequestContext(
        customer_id=customer_id,
        customer_name=principal.name,
        request_id=request.state.request_id,
        raw_token=principal.token,
    )
