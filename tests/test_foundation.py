from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.shared.errors import AppError, ErrorCode, install_exception_handlers
from src.shared.middleware import RequestContextMiddleware
from src.utils.config import Settings, get_settings


# ---------- settings ----------

def test_settings_layering_yaml_and_env():
    s = get_settings()
    assert s.company_name == "Maple Car Rental"  # from config.yaml
    assert s.mock_services_url == "http://mock-services.test"  # env overrides yaml
    assert s.agent_max_steps == 6


def test_secrets_are_not_leaked_in_repr():
    s = get_settings()
    assert "test-secret" not in repr(s)
    assert "test-secret" not in str(s.model_dump())


def test_default_jwt_secret_rejected_outside_dev():
    with pytest.raises(ValidationError):
        Settings(app_env="prod", jwt_secret="change-me")


# ---------- middleware + error format ----------

@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("db password=hunter2")

    @app.get("/not-found")
    async def not_found():
        raise AppError(ErrorCode.RESERVATION_NOT_FOUND)

    @app.get("/typed")
    async def typed(n: int):
        return {"n": n}

    return TestClient(app)


def test_request_id_generated_and_returned(client):
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert resp.headers["x-request-id"].startswith("req_")


def test_valid_incoming_request_id_is_propagated(client):
    resp = client.get("/ok", headers={"X-Request-ID": "req_upstream-123"})
    assert resp.headers["x-request-id"] == "req_upstream-123"


def test_unsafe_incoming_request_id_is_replaced(client):
    resp = client.get("/ok", headers={"X-Request-ID": "bad id\nINJECTED"})
    assert resp.headers["x-request-id"].startswith("req_")
    assert "INJECTED" not in resp.headers["x-request-id"]


def test_app_error_format(client):
    resp = client.get("/not-found")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "RESERVATION_NOT_FOUND"
    assert err["message"]
    assert err["request_id"] == resp.headers["x-request-id"]


def test_unhandled_exception_returns_generic_500_without_internals(client):
    resp = client.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "hunter2" not in resp.text
    assert body["error"]["request_id"] == resp.headers["x-request-id"]


def test_validation_error_format(client):
    resp = client.get("/typed", params={"n": "abc"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "abc" not in resp.json()["error"]["message"]


def test_unknown_route_uses_error_format(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "HTTP_ERROR"


# ---------- agent /health ----------

def _agent_client(*, redis_ok=True, mongo_ok=True, pg_status="ok", seen_headers=None) -> TestClient:
    import app_main

    def handler(request: httpx.Request) -> httpx.Response:
        if seen_headers is not None:
            seen_headers.update(request.headers)
        code = 200 if pg_status == "ok" else 503
        return httpx.Response(code, json={"checks": {"postgres": pg_status}})

    state = app_main.app.state
    state.redis = SimpleNamespace(
        ping=AsyncMock(return_value=True) if redis_ok else AsyncMock(side_effect=ConnectionError())
    )
    state.mongo = SimpleNamespace(
        admin=SimpleNamespace(
            command=AsyncMock(return_value={"ok": 1}) if mongo_ok else AsyncMock(side_effect=TimeoutError())
        )
    )
    state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock")
    return TestClient(app_main.app)  # no context manager -> lifespan (real connections) not run


def test_health_all_ok_and_propagates_request_id():
    seen: dict = {}
    resp = _agent_client(seen_headers=seen).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "checks": {"postgres": "ok", "mongo": "ok", "redis": "ok"}}
    assert seen["x-request-id"] == resp.headers["x-request-id"]


def test_health_degraded_when_dependency_down():
    resp = _agent_client(redis_ok=False, pg_status="down").get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"] == {"postgres": "down", "mongo": "ok", "redis": "down"}
