"""Mock services: JWT verification and per-customer ownership on every /v1/me route."""
import base64
import json
from datetime import timedelta
from uuid import uuid4

import jwt
import pytest

from src.shared.auth import create_access_token
from src.utils.utils import utcnow
from tests.helpers import local

PROTECTED = [
    ("GET", "/v1/locations", None),
    ("GET", "/v1/availability?location=YVR&pickup_at=2030-01-01T10:00:00&return_at=2030-01-02T10:00:00", None),
    ("GET", "/v1/vehicle-models?names=RAV4", None),
    ("POST", "/v1/quotes", {"model_id": 3, "location": "YVR", "pickup_at": "2030-01-01T10:00:00",
                            "return_at": "2030-01-02T10:00:00"}),
    ("GET", "/v1/me/reservations", None),
    ("GET", "/v1/me/reservations/DLRAV4", None),
    ("POST", "/v1/me/reservations/DLRAV4/change-previews", {"new_return_at": "2030-01-01T10:00:00"}),
    ("POST", "/v1/kb/search", {"query": "deductible"}),
]


def _codes(resp) -> list[str]:
    return [r["confirmation_code"] for r in resp.json()["reservations"]]


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


# ---------- authentication ----------

@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_every_v1_route_requires_a_token(mock_client, method, path, body):
    resp = mock_client.request(method, path, json=body)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHENTICATED"


def _bad_tokens(customers) -> dict[str, str]:
    daniel = customers["daniel"]
    good_claims = {
        "sub": str(daniel["id"]), "name": "Daniel Lee", "iss": "car-rental-agent",
        "iat": utcnow(), "exp": utcnow() + timedelta(hours=1),
    }
    expired, _ = create_access_token(daniel["id"], "Daniel Lee", now=utcnow() - timedelta(hours=2))
    return {
        "garbage": "not-a-jwt",
        "wrong_secret": jwt.encode(good_claims, "some-other-secret-0123456789abcdef", algorithm="HS256"),
        "expired": expired,
        "wrong_issuer": jwt.encode({**good_claims, "iss": "evil"}, "test-secret-not-for-production-0123456789"),
        "missing_sub": jwt.encode(
            {k: v for k, v in good_claims.items() if k != "sub"}, "test-secret-not-for-production-0123456789"
        ),
        "non_uuid_sub": jwt.encode({**good_claims, "sub": "admin"}, "test-secret-not-for-production-0123456789"),
        "alg_none": _b64({"alg": "none", "typ": "JWT"}) + "." + _b64({**good_claims, "iat": 0, "exp": 4102444800}) + ".",
    }


@pytest.mark.parametrize(
    "kind", ["garbage", "wrong_secret", "expired", "wrong_issuer", "missing_sub", "non_uuid_sub", "alg_none"]
)
def test_invalid_tokens_rejected(mock_client, customers, kind):
    token = _bad_tokens(customers)[kind]
    resp = mock_client.get("/v1/me/reservations", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHENTICATED"


def test_non_bearer_scheme_rejected(mock_client):
    resp = mock_client.get("/v1/me/reservations", headers={"Authorization": "Basic ZGFuaWVsOmRlbW8xMjM="})
    assert resp.status_code == 401


def test_unknown_customer_sees_nothing(mock_client):
    token, _ = create_access_token(uuid4(), "Nobody")
    resp = mock_client.get("/v1/me/reservations", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert _codes(resp) == []


def test_health_is_public(mock_client):
    assert mock_client.get("/health").status_code == 200


# ---------- ownership ----------

def test_list_returns_only_own_reservations(mock_client, auth):
    assert _codes(mock_client.get("/v1/me/reservations", headers=auth("daniel"))) == ["DLCOR1", "DLRAV4"]
    assert _codes(mock_client.get("/v1/me/reservations", headers=auth("emily"))) == ["ELBMW3"]
    assert _codes(mock_client.get("/v1/me/reservations", headers=auth("guest"))) == []


def test_owner_can_read_reservation(mock_client, auth):
    resp = mock_client.get("/v1/me/reservations/ELBMW3", headers=auth("emily"))
    assert resp.status_code == 200
    assert resp.json()["confirmation_code"] == "ELBMW3"


def test_lowercase_code_is_accepted(mock_client, auth):
    assert mock_client.get("/v1/me/reservations/dlrav4", headers=auth("daniel")).status_code == 200


@pytest.mark.parametrize("code", ["DLRAV", "DLRAV44", "DL-RV4", "..%2F.."])
def test_malformed_code_is_validation_error(mock_client, auth, code):
    resp = mock_client.get(f"/v1/me/reservations/{code}", headers=auth("daniel"))
    assert resp.status_code in (404, 422)
    assert resp.json()["error"]["code"] in ("VALIDATION_ERROR", "HTTP_ERROR")


def test_other_customers_code_looks_like_unknown_code(mock_client, auth):
    other = mock_client.get("/v1/me/reservations/ELBMW3", headers=auth("daniel"))
    unknown = mock_client.get("/v1/me/reservations/ZZZZZZ", headers=auth("daniel"))
    assert other.status_code == unknown.status_code == 404
    assert other.json()["error"]["code"] == unknown.json()["error"]["code"] == "RESERVATION_NOT_FOUND"
    assert other.json()["error"]["message"] == unknown.json()["error"]["message"]
    assert "ELBMW3" not in other.text and "Emily" not in other.text


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def warning(self, event, **kw):
        self.events.append(("warning", event, kw))

    def info(self, event, **kw):
        self.events.append(("info", event, kw))


def test_cross_customer_access_logs_authz_denied(mock_client, auth, customers, monkeypatch):
    import src.mock_services.routers.reservations as reservations_router

    recorder = _RecordingLogger()
    monkeypatch.setattr(reservations_router, "logger", recorder)
    mock_client.get("/v1/me/reservations/ELBMW3", headers=auth("daniel"))

    denied = [kw for level, event, kw in recorder.events if event == "AUTHZ_DENIED"]
    assert len(denied) == 1
    assert denied[0]["customer_id"] == str(customers["daniel"]["id"])
    assert denied[0]["owner_customer_id"] == str(customers["emily"]["id"])
    assert denied[0]["confirmation_code"] == "EL***"


def test_cannot_preview_or_apply_change_on_other_customers_reservation(mock_client, auth, anchor):
    body = {"new_return_at": local(anchor, 8).isoformat()}
    preview = mock_client.post("/v1/me/reservations/ELBMW3/change-previews", json=body, headers=auth("daniel"))
    assert preview.status_code == 404
    assert preview.json()["error"]["code"] == "RESERVATION_NOT_FOUND"

    # Emily creates a real preview; Daniel tries to apply it to her reservation.
    emily_preview = mock_client.post(
        "/v1/me/reservations/ELBMW3/change-previews", json=body, headers=auth("emily")
    ).json()
    applied = mock_client.post(
        "/v1/me/reservations/ELBMW3/changes",
        json={"preview_id": emily_preview["preview_id"], "expected_version": 1},
        headers={**auth("daniel"), "Idempotency-Key": "key-daniel-steals-0001"},
    )
    assert applied.status_code == 404

    after = mock_client.get("/v1/me/reservations/ELBMW3", headers=auth("emily")).json()
    assert after["version"] == 1


def test_preview_cannot_be_applied_to_a_different_reservation(mock_client, auth, anchor):
    emily_preview = mock_client.post(
        "/v1/me/reservations/ELBMW3/change-previews",
        json={"new_return_at": local(anchor, 8).isoformat()},
        headers=auth("emily"),
    ).json()
    resp = mock_client.post(
        "/v1/me/reservations/DLRAV4/changes",
        json={"preview_id": emily_preview["preview_id"], "expected_version": 1},
        headers={**auth("daniel"), "Idempotency-Key": "key-cross-preview-0001"},
    )
    assert resp.status_code == 410
    assert resp.json()["error"]["code"] == "PREVIEW_EXPIRED"
    assert mock_client.get("/v1/me/reservations/DLRAV4", headers=auth("daniel")).json()["version"] == 1


def test_request_id_is_propagated(mock_client, auth):
    resp = mock_client.get("/v1/me/reservations", headers={**auth("daniel"), "X-Request-ID": "req_from_agent_1"})
    assert resp.headers["x-request-id"] == "req_from_agent_1"
