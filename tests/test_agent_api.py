"""Agent HTTP API: login, sessions, chat turn plumbing (FakeLLM + in-memory stores + in-process mock)."""
import jwt

from src.agent.orchestrator import LLM_ERROR_REPLY
from src.llm.fake import call, say
from src.llm.gateway import UnconfiguredLLM
from tests.helpers import naive


# ---------- login ----------

def test_login_issues_jwt(agent, customers):
    resp = agent.client.post("/auth/login", json={"email": "Daniel@Example.com", "password": "demo123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 3600
    assert body["customer"] == {"id": str(customers["daniel"]["id"]), "name": "Daniel Lee"}
    claims = jwt.decode(body["access_token"], options={"verify_signature": False})
    assert claims["sub"] == str(customers["daniel"]["id"])
    assert {"sub", "name", "iat", "exp", "iss"} <= set(claims)


def test_wrong_password_and_unknown_email_look_the_same(agent):
    wrong = agent.client.post("/auth/login", json={"email": "daniel@example.com", "password": "nope"})
    unknown = agent.client.post("/auth/login", json={"email": "nobody@example.com", "password": "demo123"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"] == "Invalid email or password."


def test_login_validation(agent):
    assert agent.client.post("/auth/login", json={"email": "daniel@example.com"}).status_code == 422


# ---------- sessions ----------

def test_session_endpoints_require_auth(agent):
    assert agent.client.post("/chat/sessions").status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert agent.client.post("/chat/sessions", headers=bad).json()["error"]["code"] == "UNAUTHENTICATED"


def test_chat_turn_payload_and_history(agent):
    headers = agent.login("daniel")
    sid = agent.new_session(headers)
    agent.llm.script = [say("Hello Daniel!")]

    resp = agent.send(headers, sid, "  hi  ")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"request_id", "message_id", "reply", "cards", "pending_action", "sources", "tool_trace"}
    assert body["reply"] == "Hello Daniel!"
    assert body["request_id"] == resp.headers["x-request-id"]
    assert body["pending_action"] is None and body["cards"] == [] and body["tool_trace"] == []

    messages = agent.client.get(f"/chat/sessions/{sid}/messages", headers=headers).json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [("user", "hi"), ("assistant", "Hello Daniel!")]
    assert messages[1]["message_id"] == body["message_id"]


def test_tool_turn_persists_trace_without_payloads(agent, anchor):
    headers = agent.login("daniel")
    sid = agent.new_session(headers)
    agent.llm.script = [call("get_my_reservation", confirmation_code="DLRAV4"), say("Found it.")]
    body = agent.send(headers, sid, "check DLRAV4").json()
    assert body["tool_trace"][0] | {"latency_ms": 0} == {
        "tool": "get_my_reservation", "status": "ok", "latency_ms": 0, "error_code": None}
    assert [c["type"] for c in body["cards"]] == ["reservation", "insurance"]

    stored = agent.services.history.messages[-1]
    assert stored["tool_calls"][0]["args"] == {"confirmation_code": "DL***"}
    assert "result" not in stored["tool_calls"][0]


def test_trip_context_and_history_carry_over_turns(agent, anchor):
    headers = agent.login("daniel")
    sid = agent.new_session(headers)
    agent.llm.script = [
        call("search_vehicles", location="YVR", pickup_at=naive(anchor, 5), return_at=naive(anchor, 10),
             passengers=4, large_bags=3),
        say("RAV4, Highlander or Pacifica."),
        say("The RAV4 fits."),
    ]
    agent.send(headers, sid, "car for 4 with 3 bags")
    agent.send(headers, sid, "which is better?")

    second = agent.llm.requests[-1]
    assert '"location": "YVR"' in second["system"] and '"passengers": 4' in second["system"]
    assert [m["content"] for m in second["messages"]] == [
        "car for 4 with 3 bags", "RAV4, Highlander or Pacifica.", "which is better?"]


def test_other_customers_session_is_not_found_and_audited(agent):
    daniel, emily = agent.login("daniel"), agent.login("emily")
    sid = agent.new_session(emily)
    resp = agent.send(daniel, sid, "hi")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"
    assert agent.client.get(f"/chat/sessions/{sid}/messages", headers=daniel).status_code == 404
    assert len(agent.services.audit.events("AUTHZ_DENIED")) == 2
    assert agent.llm.requests == []


def test_unknown_malformed_and_expired_sessions(agent):
    headers = agent.login("daniel")
    assert agent.send(headers, "sess_01ARZ3NDEKTSV4RRFFQ69G5FAV", "hi").status_code == 404
    assert agent.send(headers, "not-a-session", "hi").status_code == 422
    sid = agent.new_session(headers)
    agent.services.sessions.expire_session(sid)
    assert agent.send(headers, sid, "hi").json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_message_length_limits(agent):
    headers = agent.login("daniel")
    sid = agent.new_session(headers)
    for text in ("", "   ", "x" * 2001, "x" * 5000):
        resp = agent.send(headers, sid, text)
        assert resp.status_code == 422, text[:10]
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"
    assert agent.llm.requests == []


def test_unconfigured_llm_gives_friendly_reply(agent):
    agent.services.llm = UnconfiguredLLM("missing key")
    headers = agent.login("daniel")
    sid = agent.new_session(headers)
    resp = agent.send(headers, sid, "hello")
    assert resp.status_code == 200
    assert resp.json()["reply"] == LLM_ERROR_REPLY
    assert agent.services.audit.events("LLM_ERROR")
