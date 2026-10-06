"""Thin synchronous client for the agent API, used by the Streamlit UI."""
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_TIMEOUT_SECONDS = 60  # above the agent's 45 s turn timeout


@dataclass
class ApiError(Exception):
    status: int
    code: str
    message: str
    request_id: str | None = None

    def __str__(self) -> str:
        return f"{self.message} (code: {self.code}, request_id: {self.request_id or 'n/a'})"


class AgentApiClient:
    def __init__(self, base_url: str | None = None, http: httpx.Client | None = None) -> None:
        self.base_url = base_url or str(getattr(http, "base_url", ""))
        self._http = http or httpx.Client(base_url=base_url or "", timeout=DEFAULT_TIMEOUT_SECONDS)

    def _call(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        all_headers = dict(headers or {})
        if token:
            all_headers["Authorization"] = f"Bearer {token}"
        try:
            resp = self._http.request(method, path, json=json, headers=all_headers)
        except httpx.TimeoutException:
            raise ApiError(0, "UPSTREAM_TIMEOUT", "The assistant took too long to respond. Please try again.") from None
        except httpx.HTTPError:
            raise ApiError(0, "UNREACHABLE", f"Cannot reach the agent API at {self.base_url}.") from None

        request_id = resp.headers.get("x-request-id")
        try:
            body = resp.json()
        except ValueError:
            body = None
        if resp.status_code >= 400:
            err = (body or {}).get("error", {}) if isinstance(body, dict) else {}
            raise ApiError(
                resp.status_code,
                err.get("code", "HTTP_ERROR"),
                err.get("message", f"Request failed with status {resp.status_code}."),
                err.get("request_id") or request_id,
            )
        if not isinstance(body, dict):
            raise ApiError(resp.status_code, "BAD_RESPONSE", "The agent API returned an unexpected response.", request_id)
        return body

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._call("POST", "/auth/login", json={"email": email, "password": password})

    def create_session(self, token: str) -> dict[str, Any]:
        return self._call("POST", "/chat/sessions", token=token)

    def send_message(self, token: str, session_id: str, message: str) -> dict[str, Any]:
        return self._call("POST", f"/chat/sessions/{session_id}/messages", token=token, json={"message": message})

    def confirm_action(self, token: str, action_id: str, session_id: str, idempotency_key: str) -> dict[str, Any]:
        return self._call("POST", f"/actions/{action_id}/confirm", token=token, json={"session_id": session_id},
                          headers={"Idempotency-Key": idempotency_key})

    def cancel_action(self, token: str, action_id: str, session_id: str) -> dict[str, Any]:
        return self._call("POST", f"/actions/{action_id}/cancel", token=token, json={"session_id": session_id})
