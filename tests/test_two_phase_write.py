"""Spec section 7.4 / scenario 8: the LLM can only propose; only the Confirm endpoint writes."""
import asyncio
import uuid

import httpx
import pytest

from src.agent.pending import pending_key
from src.agent.tools import TOOLS
from src.llm.fake import call, say
from src.utils.utils import utcnow
from tests.helpers import local, month_day, reply_from


def _reservation(repo, code="DLRAV4"):
    return next(r for r in repo.reservations.values() if r["confirmation_code"] == code)


def _confirm(agent, headers, action_id, key=None, body=None):
    return agent.client.post(f"/actions/{action_id}/confirm", json=body,
                             headers={**headers, "Idempotency-Key": key or str(uuid.uuid4())})


@pytest.fixture
def proposed(agent, anchor):
    """Daniel asks to move DLRAV4's return from anchor+6 to anchor+12 (spec: Oct 11 -> Oct 17)."""
    headers = agent.login("daniel")
    sid = agent.new_session(headers)
    agent.llm.script = [
        call("propose_reservation_change", confirmation_code="DLRAV4", new_return_at=month_day(anchor, 12)),
        reply_from(lambda rs: f"{rs[0]['summary']}. The price changes by {rs[0]['currency']} "
                              f"{rs[0]['price_delta']:.2f}. Please review and press Confirm."),
    ]
    body = agent.send(headers, sid, f"Change my return date to {month_day(anchor, 12)}.").json()
    return agent, headers, sid, body


# ---------------------------------------------------------------- propose

def test_proposal_returns_pending_action_and_changes_nothing(proposed, anchor):
    agent, _, sid, body = proposed
    pa = body["pending_action"]
    assert pa["action_id"].startswith("act_")
    assert pa["type"] == "MODIFY_RETURN_DATE" and pa["confirmation_code"] == "DLRAV4"
    assert (pa["old_total"], pa["new_total"], pa["price_delta"], pa["currency"]) == (593.6, 1305.92, 712.32, "CAD")
    assert "→" in pa["summary"]
    assert [c["type"] for c in body["cards"]] == ["pending_action"]
    assert "press Confirm" in body["reply"]

    rav = _reservation(agent.repo)
    assert rav["version"] == 1 and rav["return_at"] == local(anchor, 6)
    assert agent.repo.changes == []

    (proposed_event,) = agent.services.audit.events("WRITE_PROPOSED")
    assert proposed_event["args_redacted"]["confirmation_code"] == "DL***"
    stored = asyncio.run(agent.services.sessions.get_json(pending_key(pa["action_id"])))
    assert stored["customer_id"] == str(agent.repo.reservations[rav["id"]]["customer_id"])
    assert stored["session_id"] == sid


def test_no_tool_can_execute_a_write():
    assert not any(word in name for name in TOOLS for word in ("confirm", "apply", "commit", "execute"))


# ---------------------------------------------------------------- confirm

def test_confirm_applies_change_once(proposed, anchor):
    agent, headers, sid, body = proposed
    action_id = body["pending_action"]["action_id"]
    key = str(uuid.uuid4())

    first = _confirm(agent, headers, action_id, key)
    assert first.status_code == 200
    res = first.json()["reservation"]
    assert first.json()["status"] == "APPLIED"
    assert res["version"] == 2
    assert res["return_at"] == local(anchor, 12).isoformat()
    assert res["total_amount"] == 1305.92

    rav = _reservation(agent.repo)
    assert rav["version"] == 2 and rav["return_at"] == local(anchor, 12)
    (change,) = agent.repo.changes
    assert change["idempotency_key"] == action_id and float(change["price_delta"]) == pytest.approx(712.32)

    replay = _confirm(agent, headers, action_id, key)
    assert replay.status_code == 200 and replay.json() == first.json()
    double_click = _confirm(agent, headers, action_id)  # fresh key, same action
    assert double_click.status_code == 200 and double_click.json() == first.json()
    assert len(agent.repo.changes) == 1

    assert len(agent.services.audit.events("WRITE_CONFIRMED")) == 1
    note = agent.services.history.messages[-1]
    assert note["role"] == "system" and note["session_id"] == sid
    assert note["content"].startswith("Reservation DLRAV4 updated: return date")


def test_typing_yes_does_not_write(proposed):
    agent, headers, sid, body = proposed
    agent.llm.script = [say("The change is ready. Please press the Confirm button to apply it.")]
    resp = agent.send(headers, sid, "yes, confirm it").json()

    assert resp["tool_trace"] == []
    assert resp["pending_action"]["action_id"] == body["pending_action"]["action_id"]  # button stays visible
    assert _reservation(agent.repo)["version"] == 1
    assert agent.repo.changes == []
    offered = agent.llm.requests[-1]["tools"]
    assert "propose_reservation_change" in offered
    assert not any("confirm" in t for t in offered)


def test_other_customer_cannot_confirm_or_cancel(proposed):
    agent, headers, _, body = proposed
    action_id = body["pending_action"]["action_id"]
    emily = agent.login("emily")

    for resp in (_confirm(agent, emily, action_id), agent.client.post(f"/actions/{action_id}/cancel", headers=emily)):
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "ACTION_NOT_FOUND"
    assert len(agent.services.audit.events("AUTHZ_DENIED")) == 2
    assert _reservation(agent.repo)["version"] == 1

    assert _confirm(agent, headers, action_id).status_code == 200  # still usable by its owner


def test_confirm_from_another_session_is_rejected(proposed):
    agent, headers, _, body = proposed
    other_sid = agent.new_session(headers)
    resp = _confirm(agent, headers, body["pending_action"]["action_id"], body={"session_id": other_sid})
    assert resp.status_code == 404


def test_expired_action(proposed):
    agent, headers, _, body = proposed
    action_id = body["pending_action"]["action_id"]
    sessions = agent.services.sessions
    pending = asyncio.run(sessions.get_json(pending_key(action_id)))
    pending["expires_at"] = utcnow().isoformat()
    asyncio.run(sessions.set_json(pending_key(action_id), pending, 600))

    resp = _confirm(agent, headers, action_id)
    assert resp.status_code == 410
    assert resp.json()["error"]["code"] == "ACTION_EXPIRED"
    assert _confirm(agent, headers, action_id).status_code == 404
    assert agent.repo.changes == []


def test_cancel_discards_action(proposed):
    agent, headers, sid, body = proposed
    action_id = body["pending_action"]["action_id"]
    resp = agent.client.post(f"/actions/{action_id}/cancel", headers=headers)
    assert resp.status_code == 200 and resp.json() == {"status": "DISCARDED"}
    assert _confirm(agent, headers, action_id).status_code == 404
    assert agent.repo.changes == []

    agent.llm.script = [say("OK")]
    assert agent.send(headers, sid, "thanks").json()["pending_action"] is None


def test_conflict_after_proposal_is_rejected_and_dropped(proposed, anchor):
    agent, headers, _, body = proposed
    repo = agent.repo
    for v in repo.vehicles.values():
        if v["model_id"] == 3 and v["home_location_id"] == 1:
            rid = uuid.uuid4()
            repo.reservations[rid] = {
                "id": rid, "confirmation_code": f"BLK{v['id']:03d}", "customer_id": next(iter(repo.customers)),
                "vehicle_id": v["id"], "pickup_location_id": 1, "return_location_id": 1,
                "pickup_at": local(anchor, 8), "return_at": local(anchor, 9), "status": "CONFIRMED",
                "payment_status": "AUTHORIZED", "total_amount": 0, "currency": "CAD", "version": 1,
            }

    action_id = body["pending_action"]["action_id"]
    resp = _confirm(agent, headers, action_id)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "NOT_AVAILABLE"
    assert _reservation(repo)["version"] == 1 and repo.changes == []
    assert agent.services.audit.events("WRITE_REJECTED")[0]["outcome"] == "NOT_AVAILABLE"
    assert _confirm(agent, headers, action_id).status_code == 404


def test_transient_failure_keeps_action_for_retry(proposed):
    agent, headers, _, body = proposed
    action_id = body["pending_action"]["action_id"]
    real_http = agent.services.http
    agent.services.http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503)), base_url="http://mock")

    failed = _confirm(agent, headers, action_id)
    assert failed.status_code == 502
    assert failed.json()["error"]["code"] == "UPSTREAM_ERROR"

    agent.services.http = real_http
    assert _confirm(agent, headers, action_id).status_code == 200
    assert len(agent.repo.changes) == 1


def test_confirm_request_validation(proposed):
    agent, headers, _, body = proposed
    action_id = body["pending_action"]["action_id"]
    no_key = agent.client.post(f"/actions/{action_id}/confirm", headers=headers)
    assert no_key.status_code == 422
    assert _confirm(agent, headers, "act_not-valid").status_code == 422
    assert agent.client.post(f"/actions/{action_id}/confirm", headers={"Idempotency-Key": "k" * 10}).status_code == 401


def test_idempotency_key_cannot_be_reused_for_another_action(proposed, anchor):
    agent, headers, sid, body = proposed
    key = str(uuid.uuid4())
    assert _confirm(agent, headers, body["pending_action"]["action_id"], key).status_code == 200

    agent.llm.script = [call("propose_reservation_change", confirmation_code="DLRAV4",
                             new_return_at=month_day(anchor, 13)), say("Press Confirm.")]
    second = agent.send(headers, sid, "Actually make it one more day.").json()["pending_action"]
    resp = _confirm(agent, headers, second["action_id"], key)
    assert resp.status_code == 422


# ---------------------------------------------------------------- proposal errors

def test_cannot_propose_on_someone_elses_reservation(agent, anchor):
    headers = agent.login("daniel")
    sid = agent.new_session(headers)
    agent.llm.script = [
        call("propose_reservation_change", confirmation_code="ELBMW3", new_return_at=month_day(anchor, 8)),
        say("I can't find that reservation."),
    ]
    body = agent.send(headers, sid, "Extend ELBMW3").json()
    assert body["tool_trace"][0]["error_code"] == "RESERVATION_NOT_FOUND"
    assert body["pending_action"] is None
    assert agent.services.audit.events("WRITE_PROPOSED") == []


def test_ambiguous_reservation_asks_instead_of_proposing(agent):
    headers = agent.login("daniel")
    sid = agent.new_session(headers)
    agent.llm.script = [
        call("list_my_reservations"),
        reply_from(lambda rs: "Which reservation should I change: " + " or ".join(
            r["confirmation_code"] for r in rs[0]["reservations"]) + "?"),
    ]
    body = agent.send(headers, sid, "Change my return date to October 17.").json()
    assert body["reply"] == "Which reservation should I change: DLCOR1 or DLRAV4?"
    assert body["pending_action"] is None
