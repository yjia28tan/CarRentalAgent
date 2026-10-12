"""
httpx client for the (mock) enterprise business APIs. Forwards the caller's JWT and X-Request-ID,
and turns every failure into an UpstreamError with an error code from spec section 11.3.
"""
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import httpx

from src.shared.errors import DEFAULT_MESSAGES, ErrorCode
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UpstreamError(Exception):
    def __init__(self, code: ErrorCode, message: str, status: int | None = None, details: dict | None = None) -> None:
        self.code = code
        self.message = message
        self.status = status
        self.details = details
        super().__init__(f"{code}: {message}")


def _param(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


class BusinessApiClient:
    def __init__(self, http: httpx.AsyncClient, token: str | None, request_id: str | None) -> None:
        self._http = http
        self._headers = {"X-Request-ID": request_id or ""}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        clean_params = {k: _param(v) for k, v in (params or {}).items() if v is not None and v != []}
        try:
            resp = await self._http.request(
                method, path, params=clean_params, json=json, headers={**self._headers, **(headers or {})}
            )
        except httpx.TimeoutException:
            logger.warning("upstream_timeout", path=path)
            raise UpstreamError(ErrorCode.UPSTREAM_TIMEOUT, DEFAULT_MESSAGES[ErrorCode.UPSTREAM_TIMEOUT]) from None
        except httpx.HTTPError as exc:
            logger.warning("upstream_unreachable", path=path, error=type(exc).__name__)
            raise UpstreamError(ErrorCode.UPSTREAM_ERROR, DEFAULT_MESSAGES[ErrorCode.UPSTREAM_ERROR]) from None

        if resp.status_code >= 500:
            logger.warning("upstream_error", path=path, status=resp.status_code)
            code = ErrorCode.UPSTREAM_TIMEOUT if resp.status_code == 504 else ErrorCode.UPSTREAM_ERROR
            raise UpstreamError(code, DEFAULT_MESSAGES[code], resp.status_code)
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error", {})
            except ValueError:
                err = {}
            code = ErrorCode.__members__.get(err.get("code", ""), ErrorCode.UPSTREAM_ERROR)
            raise UpstreamError(code, err.get("message") or DEFAULT_MESSAGES[code], resp.status_code, err.get("details"))
        try:
            return resp.json()
        except ValueError:
            raise UpstreamError(ErrorCode.UPSTREAM_ERROR, DEFAULT_MESSAGES[ErrorCode.UPSTREAM_ERROR]) from None

    # --- auth ---
    async def verify_credentials(self, email: str, password: str) -> dict[str, Any]:
        return await self._request("POST", "/v1/auth/verify-credentials", json={"email": email, "password": password})

    # --- catalog ---
    async def search_availability(self, **params: Any) -> dict[str, Any]:
        return await self._request("GET", "/v1/availability", params=params)

    async def vehicle_models(self, ids: list[int] | None = None, names: list[str] | None = None) -> dict[str, Any]:
        params = {"ids": ",".join(map(str, ids)) if ids else None, "names": ",".join(names) if names else None}
        return await self._request("GET", "/v1/vehicle-models", params=params)

    async def create_quote(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/v1/quotes", json={k: _param(v) for k, v in body.items()})

    # --- the caller's own reservations ---
    async def list_reservations(self, from_date: date | None, to_date: date | None, status: list[str] | None) -> dict:
        return await self._request(
            "GET", "/v1/me/reservations", params={"from_date": from_date, "to_date": to_date, "status": status}
        )

    async def get_reservation(self, code: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/me/reservations/{code}")

    async def preview_change(self, code: str, new_return_at: datetime) -> dict[str, Any]:
        return await self._request(
            "POST", f"/v1/me/reservations/{code}/change-previews", json={"new_return_at": new_return_at.isoformat()}
        )

    async def apply_change(
        self, code: str, preview_id: str, expected_version: int, idempotency_key: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/me/reservations/{code}/changes",
            json={"preview_id": preview_id, "expected_version": expected_version},
            headers={"Idempotency-Key": idempotency_key},
        )

    # --- knowledge base ---
    async def kb_search(self, query: str, top_k: int, categories: list[str] | None, plan_code: str | None) -> dict:
        body = {"query": query, "top_k": top_k, "filters": {"categories": categories, "plan_code": plan_code}}
        return await self._request("POST", "/v1/kb/search", json=body)

    async def ping(self) -> dict[str, Any]:
        return await self._request("GET", "/health")
