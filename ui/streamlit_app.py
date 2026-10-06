"""
Car Rental Assistant demo UI (spec section 15). Run: streamlit run ui/streamlit_app.py
The agent API URL comes from AGENT_API_URL (default http://localhost:8000).
"""
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from ui.api_client import AgentApiClient, ApiError  # noqa: E402
from ui.views import money, reservation_card_from_api, render_cards, render_sources, render_tool_trace, signed_money, when  # noqa: E402

DEMO_USERS = {
    "Daniel Lee": "daniel@example.com",
    "Emily Lee": "emily@example.com",
    "Guest User": "guest@example.com",
}
DEMO_PASSWORD = "demo123"
# The action can no longer succeed after these, so the Confirm box is removed.
FINAL_ACTION_ERRORS = {"ACTION_NOT_FOUND", "ACTION_EXPIRED", "NOT_AVAILABLE", "VERSION_CONFLICT", "PREVIEW_EXPIRED",
                       "INVALID_STATE", "RESERVATION_NOT_FOUND"}

ss = st.session_state


def _init_state() -> None:
    if "api" not in ss:
        ss.api = AgentApiClient(os.getenv("AGENT_API_URL", "http://localhost:8000"))
    defaults: dict[str, Any] = {
        "user": None, "token": None, "customer": None, "session_id": None,
        "messages": [], "last": None, "pending": None, "error": None,
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)


def _reset_conversation() -> None:
    ss.messages, ss.last, ss.pending, ss.error = [], None, None, None


def _login(user: str) -> None:
    resp = ss.api.login(DEMO_USERS[user], DEMO_PASSWORD)
    ss.user, ss.token, ss.customer = user, resp["access_token"], resp["customer"]
    _new_session()


def _new_session() -> None:
    try:
        ss.session_id = _authed(lambda token: ss.api.create_session(token))["session_id"]
        _reset_conversation()
    except ApiError as exc:
        ss.error = exc


def _authed(call: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Run an API call; on an expired token, sign in again once and retry."""
    try:
        return call(ss.token)
    except ApiError as exc:
        if exc.status != 401 or not ss.user:
            raise
        resp = ss.api.login(DEMO_USERS[ss.user], DEMO_PASSWORD)
        ss.token = resp["access_token"]
        return call(ss.token)


def _on_submit() -> None:
    text = (ss.get("chat_input") or "").strip()
    if not text or not ss.session_id:
        return
    ss.error = None
    ss.messages.append({"role": "user", "content": text})
    try:
        resp = _authed(lambda token: ss.api.send_message(token, ss.session_id, text))
    except ApiError as exc:
        ss.error = exc
        if exc.code == "SESSION_NOT_FOUND":
            _new_session()
            ss.error = ApiError(exc.status, exc.code, "Your chat session expired, so a new one was started. "
                                "Please send your message again.", exc.request_id)
        return
    ss.messages.append({"role": "assistant", "content": resp["reply"], "request_id": resp["request_id"]})
    ss.last = {"cards": resp["cards"], "sources": resp["sources"], "tool_trace": resp["tool_trace"],
               "request_id": resp["request_id"]}
    ss.pending = resp["pending_action"]


def _confirm() -> None:
    pending = ss.pending
    ss.error = None
    try:
        resp = _authed(lambda token: ss.api.confirm_action(token, pending["action_id"], ss.session_id,
                                                         str(uuid.uuid4())))
    except ApiError as exc:
        ss.error = exc
        if exc.code in FINAL_ACTION_ERRORS:
            ss.pending = None
        return
    r = resp["reservation"]
    ss.messages.append({"role": "assistant", "content": (
        f"Confirmed. Reservation **{r['confirmation_code']}** now returns on **{when(r['return_at'])}**. "
        f"New total: **{money(r['total_amount'], r['currency'])}**.")})
    ss.last = {"cards": [reservation_card_from_api(r)], "sources": [], "tool_trace": [], "request_id": None}
    ss.pending = None


def _cancel() -> None:
    pending = ss.pending
    ss.error = None
    try:
        _authed(lambda token: ss.api.cancel_action(token, pending["action_id"], ss.session_id))
    except ApiError as exc:
        ss.error = exc
        if exc.code not in FINAL_ACTION_ERRORS:
            return
    ss.messages.append({"role": "assistant", "content": f"The proposed change to {pending['confirmation_code']} "
                                                        "was discarded. Nothing was changed."})
    ss.pending = None


def _sidebar() -> None:
    with st.sidebar:
        st.header("Demo")
        user = st.selectbox("Signed-in customer", list(DEMO_USERS), key="selected_user")
        if user != ss.user:
            try:
                _login(user)
            except ApiError as exc:
                ss.user, ss.token, ss.session_id = None, None, None
                ss.error = exc
        st.button("New session", on_click=_new_session, disabled=not ss.token)
        st.caption(f"API: {ss.api.base_url}")
        if ss.customer:
            st.caption(f"Customer: {ss.customer['name']}")
        if ss.session_id:
            st.caption(f"Session: {ss.session_id}")


def _pending_box(pending: dict[str, Any]) -> None:
    with st.container(border=True):
        st.warning(f"**Change awaiting your confirmation:** {pending['summary']}")
        currency = pending["currency"]
        st.write(f"Total {money(pending['old_total'], currency)} → {money(pending['new_total'], currency)} "
                 f"(**{signed_money(pending['price_delta'], currency)}**)")
        st.caption(f"Expires {when(pending['expires_at'])}")
        confirm_col, cancel_col = st.columns(2)
        confirm_col.button("Confirm", key="confirm_action", type="primary", on_click=_confirm)
        cancel_col.button("Cancel", key="cancel_action", on_click=_cancel)


def main() -> None:
    st.set_page_config(page_title="Car Rental Assistant", layout="wide")
    _init_state()
    _sidebar()

    st.title("Car Rental Assistant")
    chat_col, info_col = st.columns([3, 2], gap="large")

    with chat_col:
        if ss.error:
            st.error(str(ss.error))
        for m in ss.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if ss.pending:
            _pending_box(ss.pending)

    with info_col:
        st.subheader("Details")
        last = ss.last
        if last:
            render_cards(last["cards"])
            render_sources(last["sources"])
            render_tool_trace(last["tool_trace"])
            if last.get("request_id"):
                st.caption(f"request_id: {last['request_id']}")
        else:
            st.caption("Vehicle options, quotes, reservations and policy sources appear here.")

    st.chat_input("Ask about cars, prices, your reservations or policies", key="chat_input",
                  on_submit=_on_submit, max_chars=2000, disabled=not ss.session_id)


main()
