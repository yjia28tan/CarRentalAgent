"""Streamlit UI: pure formatting, the API client, and the app itself driven headlessly with AppTest
against the real agent API (in-process, FakeLLM, in-memory stores and mock services)."""
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src.llm.fake import call, say
from tests.helpers import month_day
from ui.api_client import AgentApiClient, ApiError
from ui.views import comparison_rows, money, quote_rows, signed_money, vehicle_option_rows, when

APP = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")


# ---------------------------------------------------------------- formatting

def test_money_and_dates():
    assert money(1305.92, "CAD") == "CAD 1,305.92"
    assert money(None, "CAD") == "-"
    assert signed_money(712.32, "CAD") == "+CAD 712.32"
    assert signed_money(-123.2, "CAD") == "CAD -123.20"
    assert when("2026-10-17T10:00:00-07:00") == "Sat Oct 17 2026, 10:00"
    assert when("not a date") == "not a date"


def test_table_rows():
    options = vehicle_option_rows({"options": [{"name": "Toyota RAV4", "category": "MIDSIZE_SUV", "seats": 5,
                                                "large_bags": 3, "small_bags": 2, "available_count": 2,
                                                "daily_rate": 74.0, "currency": "CAD"}]})
    assert options == [{"Model": "Toyota RAV4", "Category": "MIDSIZE_SUV", "Seats": 5, "Large bags": 3,
                        "Small bags": 2, "Available": 2, "Daily rate": "CAD 74.00"}]

    comparison = comparison_rows({"models": [
        {"name": "Toyota RAV4", "category": "MIDSIZE_SUV", "seats": 5, "large_bags": 3, "features": {"awd": True}},
        {"name": "Toyota Corolla", "category": "COMPACT", "seats": 5, "large_bags": 2, "features": {}},
    ]})
    bags = next(r for r in comparison if r["Spec"] == "Large bags")
    assert bags == {"Spec": "Large bags", "Toyota RAV4": "3", "Toyota Corolla": "2"}
    assert next(r for r in comparison if r["Spec"] == "awd") == {"Spec": "awd", "Toyota RAV4": "True",
                                                                 "Toyota Corolla": "-"}

    quote = quote_rows({"total": 593.6, "currency": "CAD", "breakdown": {
        "days": 5, "daily_rate": 74.0, "base": 370.0, "taxes_fees": 63.6,
        "insurance": [{"code": "PREMIUM_PROTECT", "daily_price": 32.0, "amount": 160.0}]}})
    assert [r["Amount"] for r in quote] == ["CAD 370.00", "CAD 160.00", "CAD 63.60", "CAD 593.60"]


# ---------------------------------------------------------------- API client

def test_api_error_carries_code_and_request_id():
    def handler(request):
        return httpx.Response(404, headers={"x-request-id": "req_hdr"},
                              json={"error": {"code": "SESSION_NOT_FOUND", "message": "gone", "request_id": "req_1"}})

    api = AgentApiClient(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x"))
    with pytest.raises(ApiError) as exc:
        api.send_message("t", "sess_x", "hi")
    assert (exc.value.status, exc.value.code, exc.value.request_id) == (404, "SESSION_NOT_FOUND", "req_1")
    assert "req_1" in str(exc.value)


@pytest.mark.parametrize(
    ("handler", "code"),
    [
        (lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")), "UNREACHABLE"),
        (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow")), "UPSTREAM_TIMEOUT"),
        (lambda r: httpx.Response(502, text="bad gateway"), "HTTP_ERROR"),
        (lambda r: httpx.Response(200, text="not json"), "BAD_RESPONSE"),
    ],
)
def test_api_transport_failures(handler, code):
    api = AgentApiClient(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x"))
    with pytest.raises(ApiError) as exc:
        api.create_session("t")
    assert exc.value.code == code


def test_confirm_sends_idempotency_key_and_session():
    seen = {}

    def handler(request):
        seen.update(headers=dict(request.headers), body=request.content)
        return httpx.Response(200, json={"status": "APPLIED"})

    api = AgentApiClient(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x"))
    api.confirm_action("tok", "act_1", "sess_1", "key-123")
    assert seen["headers"]["idempotency-key"] == "key-123"
    assert seen["headers"]["authorization"] == "Bearer tok"
    assert b'"session_id":"sess_1"' in seen["body"].replace(b" ", b"")


# ---------------------------------------------------------------- the app (AppTest)

@pytest.fixture
def app(agent):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api"] = AgentApiClient(http=agent.client)
    at.run()
    assert not at.exception, at.exception
    return at


def _send(at: AppTest, text: str) -> AppTest:
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def test_signs_in_default_user_and_opens_session(app):
    assert app.session_state["customer"]["name"] == "Daniel Lee"
    assert app.session_state["session_id"].startswith("sess_")
    assert not app.error


def test_chat_turn_renders_reply_cards_and_trace(app, agent, anchor):
    agent.llm.script = [
        call("search_vehicles", location="YVR", pickup_at=month_day(anchor, 4), return_at=month_day(anchor, 9),
             passengers=4, large_bags=3),
        say("RAV4, Highlander or Pacifica."),
    ]
    _send(app, "Car for 4 with 3 suitcases")

    roles = [m.name for m in app.chat_message]
    assert roles == ["user", "assistant"]
    assert app.chat_message[1].markdown[0].value == "RAV4, Highlander or Pacifica."
    assert len(app.dataframe) >= 2  # vehicle options + tool trace
    assert any("Tool trace (1 call)" in e.label for e in app.expander)
    assert any("request_id: req_" in c.value for c in app.caption)


def _propose(app, agent, anchor):
    agent.llm.script = [
        call("propose_reservation_change", confirmation_code="DLRAV4", new_return_at=month_day(anchor, 12)),
        say("Please review and press Confirm."),
    ]
    return _send(app, f"Change my return date to {month_day(anchor, 12)}.")


def _rav4(agent):
    return next(r for r in agent.repo.reservations.values() if r["confirmation_code"] == "DLRAV4")


def test_confirm_button_applies_change(app, agent, anchor):
    _propose(app, agent, anchor)
    assert any("awaiting your confirmation" in w.value for w in app.warning)
    assert any("+CAD 712.32" in m.value for m in app.markdown)
    assert _rav4(agent)["version"] == 1

    app.button(key="confirm_action").click().run()
    assert not app.exception
    assert _rav4(agent)["version"] == 2
    assert app.session_state["pending"] is None
    assert app.chat_message[-1].markdown[0].value.startswith("Confirmed. Reservation **DLRAV4**")
    assert not app.warning


def test_cancel_button_discards_change(app, agent, anchor):
    _propose(app, agent, anchor)
    app.button(key="cancel_action").click().run()
    assert app.session_state["pending"] is None
    assert "was discarded" in app.chat_message[-1].markdown[0].value
    assert _rav4(agent)["version"] == 1 and agent.repo.changes == []


def test_typing_yes_keeps_confirm_button(app, agent, anchor):
    _propose(app, agent, anchor)
    agent.llm.script = [say("Please press the Confirm button.")]
    _send(app, "yes, confirm it")
    assert app.session_state["pending"] is not None
    assert app.button(key="confirm_action").label == "Confirm"
    assert _rav4(agent)["version"] == 1


def test_expired_session_shows_error_with_request_id(app, agent):
    old_sid = app.session_state["session_id"]
    agent.services.sessions.expire_session(old_sid)
    _send(app, "hello")
    (error,) = app.error
    assert "session expired" in error.value and "request_id: req_" in error.value
    assert app.session_state["session_id"] != old_sid


def test_switching_user_signs_in_and_starts_new_session(app, agent):
    first_sid = app.session_state["session_id"]
    app.selectbox(key="selected_user").select("Emily Lee").run()
    assert app.session_state["customer"]["name"] == "Emily Lee"
    assert app.session_state["session_id"] != first_sid
    assert len(app.chat_message) == 0


def test_api_down_shows_error(agent):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api"] = AgentApiClient(
        http=httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))),
                          base_url="http://down"))
    at.run()
    assert not at.exception
    assert "Cannot reach the agent API" in at.error[0].value
