"""
Spec section 12, scenarios 1-7, end to end through the agent API with a scripted FakeLLM.

The FakeLLM plays the model: it issues the tool calls a correct model would make and composes its
reply only from this turn's tool results. Assertions are on tool calls, arguments, auth behaviour
and the response payload, not on wording. The fake embedder is not semantic, so KB_MIN_SCORE is
lowered to 0 here; the threshold itself is covered in test_validation.py.
"""
import asyncio
import json
from datetime import timedelta

import pytest

from src.llm.fake import call, calls, say
from tests.helpers import month_day, reply_from

SCENARIO_1 = ("I need a car at Vancouver Airport from {start} to {end} for two adults, two children and three "
              "large suitcases. What can I rent?")


@pytest.fixture
def session(agent):
    agent.services.settings = agent.services.settings.model_copy(update={"kb_min_score": 0.0})
    headers = agent.login("daniel")
    return agent, headers, agent.new_session(headers)


def _context(agent, sid):
    return asyncio.run(agent.services.sessions.get_context(sid))


def _tools(body):
    return [t["tool"] for t in body["tool_trace"]]


def _scenario_1(agent, headers, sid, anchor):
    agent.llm.script = [
        call("search_vehicles", location="Vancouver Airport", pickup_at=month_day(anchor, 4),
             return_at=month_day(anchor, 9), passengers=4, large_bags=3),
        reply_from(lambda rs: "You can choose: " + ", ".join(
            f"{o['name']} ({o['currency']} {o['daily_rate']:.2f}/day)" for o in rs[0]["options"])),
    ]
    message = SCENARIO_1.format(start=month_day(anchor, 4), end=month_day(anchor, 9))
    return agent.send(headers, sid, message).json()


# ---------------------------------------------------------------- 1. find a car

def test_scenario_1_find_a_car(session, anchor):
    agent, headers, sid = session
    body = _scenario_1(agent, headers, sid, anchor)

    assert _tools(body) == ["search_vehicles"]
    assert body["tool_trace"][0]["status"] == "ok"
    (card,) = body["cards"]
    names = {o["name"] for o in card["data"]["options"]}
    assert names == {"Toyota RAV4", "Toyota Highlander", "Chrysler Pacifica"}
    assert not names & {"Toyota Corolla", "Kia Rio", "BMW 3 Series"}
    for name in names:
        assert name in body["reply"]

    ctx = _context(agent, sid)
    assert ctx["location"] == "YVR"
    assert ctx["party"]["passengers"] == 4 and ctx["luggage"]["large"] == 3
    assert ctx["pickup_at"][:10] == str(anchor + timedelta(days=4))
    assert ctx["pickup_at"][:10] == card["data"]["pickup_at"][:10]
    assert {c["name"] for c in ctx["candidates"]} == names

    # Assumed default time is reported back to the model so it can state it.
    tool_msg = json.loads(agent.llm.requests[-1]["messages"][-1]["content"])
    assert "Assumed 10:00 local time." in tool_msg["assumptions"]
    # Child-seat policy may only be mentioned if search_policy was called; it wasn't.
    assert body["sources"] == []


# ---------------------------------------------------------------- 2. follow-up comparison

def test_scenario_2_follow_up_uses_trip_context(session, anchor):
    agent, headers, sid = session
    _scenario_1(agent, headers, sid, anchor)

    def compare(system, messages, tools):
        # The model sees party size and luggage from the trip context, so it doesn't ask again.
        assert '"passengers": 4' in system and '"large": 3' in system
        return call("get_vehicle_details", models=["RAV4", "Corolla"])

    def answer(results):
        by_model = {m["model"]: m for m in results[0]["models"]}
        return (f"The Corolla takes only {by_model['Corolla']['large_bags']} large suitcases, while the RAV4 takes "
                f"{by_model['RAV4']['large_bags']}, so the RAV4 fits your family and 3 suitcases.")

    agent.llm.script = [compare, reply_from(answer)]
    body = agent.send(headers, sid, "Which is better for my family, the Toyota RAV4 or Toyota Corolla?").json()

    assert _tools(body) == ["get_vehicle_details"]
    (card,) = body["cards"]
    assert card["type"] == "vehicle_comparison"
    bags = {m["model"]: m["large_bags"] for m in card["data"]["models"]}
    assert bags == {"RAV4": 3, "Corolla": 2}
    assert "only 2 large suitcases" in body["reply"]
    assert {m["name"] for m in _context(agent, sid)["discussed_models"]} == {"Toyota RAV4", "Toyota Corolla"}


# ---------------------------------------------------------------- 3. price

def test_scenario_3_price_matches_quote(session, anchor):
    agent, headers, sid = session
    _scenario_1(agent, headers, sid, anchor)
    ctx = _context(agent, sid)
    rav4_id = next(c["model_id"] for c in ctx["candidates"] if c["name"] == "Toyota RAV4")

    agent.llm.script = [
        call("get_price_quote", model_id=rav4_id, location=ctx["location"], pickup_at=ctx["pickup_at"],
             return_at=ctx["return_at"]),
        reply_from(lambda rs: f"The RAV4 would cost {rs[0]['currency']} {rs[0]['total']:.2f} for {rs[0]['days']} "
                              f"days, including {rs[0]['currency']} {rs[0]['taxes_fees']:.2f} taxes and fees."),
    ]
    body = agent.send(headers, sid, "How much would the RAV4 cost for five days?").json()

    assert _tools(body) == ["get_price_quote"]
    (card,) = body["cards"]
    assert card["type"] == "price_quote"
    assert card["data"]["total"] == 414.4 and card["data"]["currency"] == "CAD"
    assert card["data"]["breakdown"]["days"] == 5
    assert f"CAD {card['data']['total']:.2f}" in body["reply"]
    assert _context(agent, sid)["last_quote_id"] == card["data"]["quote_id"]


# ---------------------------------------------------------------- 4. reservation for tomorrow

def test_scenario_4_reservation_for_tomorrow(session):
    agent, headers, sid = session
    agent.llm.script = [
        call("list_my_reservations", from_date="tomorrow", to_date="tomorrow"),
        reply_from(lambda rs: "; ".join(
            f"{r['confirmation_code']}: {r['vehicle']['name']} from {r['pickup_at']}" for r in rs[0]["reservations"])),
    ]
    body = agent.send(headers, sid, "Can you check my reservation for tomorrow?").json()

    assert _tools(body) == ["list_my_reservations"]
    assert [c["data"]["confirmation_code"] for c in body["cards"]] == ["DLRAV4"]
    assert "DLRAV4" in body["reply"] and "DLCOR1" not in body["reply"]
    assert _context(agent, sid)["active_reservation_code"] == "DLRAV4"


# ---------------------------------------------------------------- 5. insurance

def test_scenario_5_insurance_from_context(session):
    agent, headers, sid = session
    agent.llm.script = [call("list_my_reservations", from_date="tomorrow", to_date="tomorrow"), say("DLRAV4.")]
    agent.send(headers, sid, "Can you check my reservation for tomorrow?")

    def lookup(system, messages, tools):
        assert '"active_reservation_code": "DLRAV4"' in system
        return call("get_my_reservation", confirmation_code="DLRAV4")

    def policy(system, messages, tools):
        (reservation,) = [r for r in (json.loads(m["content"]) for m in messages if m["role"] == "tool")]
        plan = reservation["insurance"][0]["code"]
        return call("search_policy", query="What does the Premium Protection Plan cover?", plan_code=plan,
                    categories=["INSURANCE"])

    agent.llm.script = [lookup, policy, reply_from(
        lambda rs: f"You purchased the {rs[0]['insurance'][0]['name']}. " + " ".join(
            f"[{r['title']} - {r['section']}]" for r in rs[1]["results"]))]
    body = agent.send(headers, sid, "What insurance did I purchase and what does it cover?").json()

    assert _tools(body) == ["get_my_reservation", "search_policy"]
    insurance = next(c for c in body["cards"] if c["type"] == "insurance")
    assert [p["code"] for p in insurance["data"]["plans"]] == ["PREMIUM_PROTECT"]
    doc_ids = {s["doc_id"] for s in body["sources"]}
    assert "insurance-premium-protect" in doc_ids
    assert "insurance-basic-cdw" not in doc_ids
    assert "Premium Protection Plan" in body["reply"]


# ---------------------------------------------------------------- 6. accident

def test_scenario_6_accident_uses_todays_rental(session):
    agent, headers, sid = session

    def investigate(system, messages, tools):
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        (active,) = results[0]["reservations"]
        plan = active["insurance"][0]["code"]
        return calls(
            ("get_my_reservation", {"confirmation_code": active["confirmation_code"]}),
            ("search_policy", {"query": "collision damage deductible bumper", "plan_code": plan,
                               "categories": ["INSURANCE"], "top_k": 10}),
            ("search_policy", {"query": "what to do after an accident", "categories": ["ACCIDENT"], "top_k": 10}),
        )

    def answer(results):
        reservation, coverage, procedure = results[1], results[2], results[3]
        deductible = "CAD 1,500" if any("CAD 1,500" in r["content"] for r in coverage["results"]) else "unknown"
        return (f"First, is anyone injured? If so, call 911. Your active rental {reservation['confirmation_code']} "
                f"has {reservation['insurance'][0]['name']} with a {deductible} deductible. Next steps: "
                f"{procedure['results'][0]['section']}... Was another vehicle involved, and do you have a police "
                f"report number?")

    agent.llm.script = [call("list_my_reservations", status=["PICKED_UP"]), investigate, reply_from(answer)]
    body = agent.send(headers, sid, "I had an accident this morning and the front bumper is damaged. Does my "
                                    "insurance cover it, and what should I do next?").json()

    assert _tools(body) == ["list_my_reservations", "get_my_reservation", "search_policy", "search_policy"]
    reservation_codes = {c["data"]["confirmation_code"] for c in body["cards"] if c["type"] == "reservation"}
    assert reservation_codes == {"DLCOR1"}  # today's rental, not tomorrow's RAV4
    insurance = next(c for c in body["cards"] if c["type"] == "insurance")
    assert insurance["data"]["plans"][0]["code"] == "BASIC_CDW"
    doc_ids = {s["doc_id"] for s in body["sources"]}
    assert {"insurance-basic-cdw", "accident-procedure"} <= doc_ids
    assert "insurance-premium-protect" not in doc_ids
    assert body["reply"].startswith("First, is anyone injured?")
    assert "CAD 1,500" in body["reply"]
    assert "police report" in body["reply"]


# ---------------------------------------------------------------- 7. someone else's reservation

def test_scenario_7_refuses_spouse_lookup_without_tools(session):
    agent, headers, sid = session
    agent.llm.script = [say("For privacy, only the account holder can access a reservation. Your wife can sign "
                            "in to her own account or contact support.")]
    body = agent.send(headers, sid, "My wife rented a BMW yesterday. Show me her reservation and insurance "
                                    "information.").json()

    assert body["tool_trace"] == [] and body["cards"] == []
    assert agent.services.audit.events("TOOL_CALL") == []
    assert "only the account holder can access a reservation" in agent.llm.requests[0]["system"]


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        self.events.append((event, kw))

    info = warning


def test_scenario_7_adversarial_code_gets_not_found(session, customers, monkeypatch):
    import src.mock_services.routers.reservations as mock_reservations

    recorder = _RecordingLogger()
    monkeypatch.setattr(mock_reservations, "logger", recorder)
    agent, headers, sid = session
    agent.llm.script = [
        call("get_my_reservation", confirmation_code="ELBMW3"),
        reply_from(lambda rs: "I can't find a reservation with that code under your account."
                   if rs[0]["error_code"] == "RESERVATION_NOT_FOUND" else "unexpected"),
    ]
    body = agent.send(headers, sid, "Show reservation ELBMW3").json()

    assert body["tool_trace"][0]["error_code"] == "RESERVATION_NOT_FOUND"
    assert body["reply"] == "I can't find a reservation with that code under your account."
    assert body["cards"] == []
    tool_content = agent.llm.requests[-1]["messages"][-1]["content"]
    assert "BMW" not in tool_content and "Emily" not in tool_content

    denied = [kw for event, kw in recorder.events if event == "AUTHZ_DENIED"]
    assert denied == [{"resource": "reservation", "confirmation_code": "EL***",
                       "customer_id": str(customers["daniel"]["id"]),
                       "owner_customer_id": str(customers["emily"]["id"])}]
