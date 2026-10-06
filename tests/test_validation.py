"""Agent-side validation: date resolution, tool argument validation, error mapping, context rules."""
import asyncio
from datetime import date, datetime, timedelta

import httpx
import pytest

from src.agent.context import apply_tool_result, empty_context
from src.agent.dates import DateParseError, resolve_date, resolve_datetime
from src.agent.orchestrator import LLM_ERROR_REPLY, MAX_STEPS_REPLY, TIMEOUT_REPLY, build_orchestrator
from src.agent.tools import TOOLS, Tool, ToolInput, ToolOutcome, execute_tool, tool_specs
from src.llm.fake import FakeLLM, call, calls, say
from src.llm.gateway import LLMError, ToolCall
from tests.helpers import VANCOUVER, naive

MONDAY = datetime(2026, 10, 5, 9, 0, tzinfo=VANCOUVER)


# ---------------------------------------------------------------- dates

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("today", date(2026, 10, 5)),
        ("tomorrow", date(2026, 10, 6)),
        ("Tomorrow.", date(2026, 10, 6)),
        ("yesterday", date(2026, 10, 4)),
        ("day after tomorrow", date(2026, 10, 7)),
        ("in 3 days", date(2026, 10, 8)),
        ("friday", date(2026, 10, 9)),
        ("this Monday", date(2026, 10, 5)),
        ("monday", date(2026, 10, 12)),
        ("next Friday", date(2026, 10, 9)),
        ("October 10", date(2026, 10, 10)),
        ("Oct 10th", date(2026, 10, 10)),
        ("10 October", date(2026, 10, 10)),
        ("October 17, 2026", date(2026, 10, 17)),
        ("October 1", date(2027, 10, 1)),  # already passed this year -> next occurrence
        ("2026-12-24", date(2026, 12, 24)),
    ],
)
def test_resolve_date(text, expected):
    assert resolve_date(text, MONDAY.date())[0] == expected


def test_assumptions_are_reported():
    _, notes = resolve_date("next Friday", MONDAY.date())
    assert notes == ["Interpreted 'next Friday' as Friday, October 9, 2026."]
    _, notes = resolve_date("October 1", MONDAY.date())
    assert "2027" in notes[0]
    assert resolve_date("2026-10-10", MONDAY.date())[1] == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-10T10:00", datetime(2026, 10, 10, 10, 0)),
        ("2026-10-10 15:30", datetime(2026, 10, 10, 15, 30)),
        ("2026-10-10", datetime(2026, 10, 10, 10, 0)),
        ("tomorrow 3pm", datetime(2026, 10, 6, 15, 0)),
        ("October 10 at 9:30 a.m.", datetime(2026, 10, 10, 9, 30)),
        ("Oct 15 14:00", datetime(2026, 10, 15, 14, 0)),
        ("this morning", datetime(2026, 10, 5, 9, 0)),
        ("tomorrow noon", datetime(2026, 10, 6, 12, 0)),
        ("October 10", datetime(2026, 10, 10, 10, 0)),
    ],
)
def test_resolve_datetime_local(text, expected):
    assert resolve_datetime(text, MONDAY)[0] == expected


def test_resolve_datetime_keeps_explicit_offset():
    dt, notes = resolve_datetime("2026-10-10T10:00:00-07:00", MONDAY)
    assert dt.utcoffset() == timedelta(hours=-7)
    assert notes == []


def test_default_time_is_reported():
    assert "Assumed 10:00 local time." in resolve_datetime("October 10", MONDAY)[1]


@pytest.mark.parametrize("text", ["someday", "10/10", "February 30", "13pm tomorrow", "", "   "])
def test_unparseable_dates(text):
    with pytest.raises(DateParseError):
        resolve_datetime(text, MONDAY)


# ---------------------------------------------------------------- tool schemas

def test_no_tool_accepts_a_customer_id():
    for spec in tool_specs():
        props = spec.input_schema.get("properties", {})
        assert not any("customer" in name for name in props), spec.name
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema.get("additionalProperties") is False


def test_expected_tools_registered():
    assert set(TOOLS) == {"search_vehicles", "get_vehicle_details", "get_price_quote", "list_my_reservations",
                          "get_my_reservation", "search_policy", "propose_reservation_change"}


# ---------------------------------------------------------------- execute_tool validation

def _call(name: str, **args) -> ToolCall:
    return ToolCall(id="call_t", name=name, arguments=args)


def _search_args(anchor, **overrides):
    return {"location": "Vancouver Airport", "pickup_at": naive(anchor, 5), "return_at": naive(anchor, 10),
            "passengers": 4, "large_bags": 3, **overrides}


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"customer_id": "someone-else"}, "'customer_id' is not an accepted argument"),
        ({"passengers": 0}, "passengers"),
        ({"passengers": 12}, "less than or equal to 9"),
        ({"large_bags": 11}, "large_bags"),
        ({"category": "SPACESHIP"}, "category"),
        ({"pickup_at": "someday"}, "Couldn't understand the date"),
        ({"return_at": "yesterday"}, "return_at must be after pickup_at"),
    ],
)
async def test_search_vehicles_validation(make_tool_ctx, anchor, overrides, fragment):
    ex = await execute_tool(_call("search_vehicles", **_search_args(anchor, **overrides)), make_tool_ctx())
    assert ex.status == "error"
    assert ex.result["error_code"] == "VALIDATION_ERROR"
    assert fragment in ex.result["message"]


async def test_pickup_in_past_and_too_long(make_tool_ctx, anchor):
    ctx = make_tool_ctx()
    past = await execute_tool(_call("search_vehicles", **_search_args(anchor, pickup_at=naive(anchor, -2))), ctx)
    assert "in the past" in past.result["message"]
    long = await execute_tool(_call("search_vehicles", **_search_args(anchor, return_at=naive(anchor, 70))), ctx)
    assert "60 days" in long.result["message"]


@pytest.mark.parametrize(
    ("tool", "args", "fragment"),
    [
        ("get_my_reservation", {"confirmation_code": "ABC"}, "confirmation_code"),
        ("get_my_reservation", {"confirmation_code": "DLRAV4", "customer_id": "x"}, "not an accepted argument"),
        ("get_vehicle_details", {}, "Provide 'models'"),
        ("get_vehicle_details", {"models": ["a", "b", "c", "d", "e"]}, "models"),
        ("search_policy", {"query": "hi"}, "query"),
        ("search_policy", {"query": "coverage", "categories": ["MOVIES"]}, "categories"),
        ("list_my_reservations", {"from_date": "2026-10-10", "to_date": "2026-10-01"}, "from_date"),
        ("get_price_quote", {"model_id": 3, "location": "YVR", "pickup_at": "tomorrow", "return_at": "tomorrow"},
         "return_at must be after"),
    ],
)
async def test_other_tool_validation(make_tool_ctx, tool, args, fragment):
    ex = await execute_tool(_call(tool, **args), make_tool_ctx())
    assert ex.result["error_code"] == "VALIDATION_ERROR"
    assert fragment in ex.result["message"]


async def test_unknown_tool(make_tool_ctx):
    ex = await execute_tool(_call("delete_all_reservations"), make_tool_ctx())
    assert ex.result["error_code"] == "VALIDATION_ERROR"
    assert "Unknown tool" in ex.result["message"]


async def test_validation_errors_are_audited(make_tool_ctx):
    ctx = make_tool_ctx()
    await execute_tool(_call("get_my_reservation", confirmation_code="ABC"), ctx)
    (record,) = ctx.audit.events()
    assert record["event"] == "VALIDATION_ERROR"
    assert record["tool"] == "get_my_reservation"
    assert record["customer_id"] == ctx.request.customer_id


# ---------------------------------------------------------------- execute_tool success + upstream errors

async def test_search_vehicles_ok_with_assumptions(make_tool_ctx, anchor):
    ex = await execute_tool(
        _call("search_vehicles", **_search_args(anchor, pickup_at="tomorrow", return_at=naive(anchor, 4))),
        make_tool_ctx(),
    )
    assert ex.status == "ok", ex.result
    assert {o["name"] for o in ex.result["options"]} >= {"Toyota Highlander", "Chrysler Pacifica"}
    assert "Assumed 10:00 local time." in ex.result["assumptions"]
    assert ex.cards[0]["type"] == "vehicle_options"


async def test_tool_call_is_audited_with_redacted_code(make_tool_ctx):
    ctx = make_tool_ctx()
    ex = await execute_tool(_call("get_my_reservation", confirmation_code="dlrav4"), ctx)
    assert ex.status == "ok"
    assert ex.result["confirmation_code"] == "DLRAV4"
    assert [c["type"] for c in ex.cards] == ["reservation", "insurance"]
    (record,) = ctx.audit.events("TOOL_CALL")
    assert record["args_redacted"] == {"confirmation_code": "dl***"}
    assert record["outcome"] == "ok"


async def test_other_customers_code_maps_to_not_found(make_tool_ctx):
    ex = await execute_tool(_call("get_my_reservation", confirmation_code="ELBMW3"), make_tool_ctx("daniel"))
    assert ex.result == {"ok": False, "error_code": "RESERVATION_NOT_FOUND",
                         "message": "No reservation found for that code."}


async def test_unknown_location_returns_supported_list(make_tool_ctx, anchor):
    ex = await execute_tool(_call("search_vehicles", **_search_args(anchor, location="Hogwarts Airport")),
                            make_tool_ctx())
    assert ex.result["error_code"] == "LOCATION_NOT_FOUND"
    assert [l["code"] for l in ex.result["details"]["supported_locations"]] == ["YVR", "YVR-DT", "YYZ"]


async def test_quote_is_cached_in_session_store(make_tool_ctx, anchor):
    ctx = make_tool_ctx()
    ex = await execute_tool(_call("get_price_quote", model_id=3, location="YVR", pickup_at=naive(anchor, 5),
                                  return_at=naive(anchor, 10)), ctx)
    assert ex.result["total"] == 414.4
    assert (await ctx.sessions.get_json(f"quote:{ex.result['quote_id']}"))["total"] == 414.4
    assert ex.cards[0]["data"]["breakdown"]["taxes_fees"] == 44.4


@pytest.mark.parametrize(
    ("handler", "code"),
    [
        (lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow")), "UPSTREAM_TIMEOUT"),
        (lambda request: (_ for _ in ()).throw(httpx.ConnectError("down")), "UPSTREAM_ERROR"),
        (lambda request: httpx.Response(500, text="boom"), "UPSTREAM_ERROR"),
        (lambda request: httpx.Response(504, text="gateway timeout"), "UPSTREAM_TIMEOUT"),
        (lambda request: httpx.Response(200, text="not json"), "UPSTREAM_ERROR"),
    ],
)
async def test_upstream_failures_become_structured_errors(make_tool_ctx, handler, code):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock")
    ex = await execute_tool(_call("get_my_reservation", confirmation_code="DLRAV4"), make_tool_ctx(http=http))
    assert ex.status == "error"
    assert ex.result["error_code"] == code
    assert "boom" not in ex.result["message"]


class _EmptyInput(ToolInput):
    pass


async def test_tool_timeout(make_tool_ctx):
    async def slow(args, ctx):
        await asyncio.sleep(1)
        return ToolOutcome(data={})

    tools = {"slow": Tool("slow", "slow tool", _EmptyInput, slow)}
    ex = await execute_tool(_call("slow"), make_tool_ctx(tool_timeout_seconds=0.05), tools)
    assert ex.result["error_code"] == "UPSTREAM_TIMEOUT"


async def test_unexpected_handler_exception_is_contained(make_tool_ctx):
    async def broken(args, ctx):
        raise RuntimeError("secret internals")

    tools = {"broken": Tool("broken", "broken tool", _EmptyInput, broken)}
    ex = await execute_tool(_call("broken"), make_tool_ctx(), tools)
    assert ex.result["error_code"] == "INTERNAL_ERROR"
    assert "secret" not in ex.result["message"]


async def test_search_policy_min_score(make_tool_ctx):
    strict = await execute_tool(_call("search_policy", query="collision deductible"), make_tool_ctx(kb_min_score=0.99))
    assert strict.result == {"ok": True, "results": [], "note": "no relevant policy found"}
    assert strict.sources == []

    loose = await execute_tool(_call("search_policy", query="collision deductible", plan_code="basic_cdw"),
                               make_tool_ctx(kb_min_score=0.0))
    assert loose.result["results"]
    assert {s["doc_id"] for s in loose.sources} <= {r["doc_id"] for r in loose.result["results"]}
    assert "insurance-premium-protect" not in {r["doc_id"] for r in loose.result["results"]}


# ---------------------------------------------------------------- trip context

def test_context_updates_from_search():
    trip = empty_context()
    apply_tool_result(trip, "search_vehicles", {"passengers": 4, "large_bags": 3}, {
        "search_id": "srch_1", "location": {"code": "YVR"}, "pickup_at": "2026-10-10T10:00:00-07:00",
        "return_at": "2026-10-15T10:00:00-07:00", "options": [{"model_id": 3, "name": "Toyota RAV4"}],
    })
    assert trip["location"] == "YVR"
    assert trip["party"]["passengers"] == 4 and trip["luggage"]["large"] == 3
    assert trip["candidates"] == [{"model_id": 3, "name": "Toyota RAV4"}]
    assert trip["last_search_id"] == "srch_1"


def test_context_updates_from_other_tools():
    trip = empty_context()
    models = {"models": [{"model_id": 3, "name": "Toyota RAV4"}, {"model_id": 2, "name": "Toyota Corolla"}]}
    apply_tool_result(trip, "get_vehicle_details", {}, models)
    apply_tool_result(trip, "get_vehicle_details", {}, models)
    assert [m["model_id"] for m in trip["discussed_models"]] == [3, 2]

    apply_tool_result(trip, "get_price_quote", {}, {"quote_id": "qt_1"})
    assert trip["last_quote_id"] == "qt_1"

    apply_tool_result(trip, "list_my_reservations", {}, {"reservations": [{"confirmation_code": "A"}, {"confirmation_code": "B"}]})
    assert trip["active_reservation_code"] is None
    apply_tool_result(trip, "list_my_reservations", {}, {"reservations": [{"confirmation_code": "DLRAV4"}]})
    assert trip["active_reservation_code"] == "DLRAV4"
    apply_tool_result(trip, "get_my_reservation", {}, {"confirmation_code": "DLCOR1"})
    assert trip["active_reservation_code"] == "DLCOR1"


# ---------------------------------------------------------------- orchestrator loop (both implementations)

@pytest.fixture(params=["custom", "langgraph"])
def orchestrator(request):
    return lambda llm: build_orchestrator(llm, request.param)


async def test_orchestrator_runs_tools_then_answers(orchestrator, make_tool_ctx, anchor):
    llm = FakeLLM([call("search_vehicles", **_search_args(anchor)), say("Here are your options.")])
    trip = empty_context()
    result = await orchestrator(llm).run_turn(user_message="find a car", history=[], trip=trip, ctx=make_tool_ctx())
    assert result.reply == "Here are your options."
    assert [t["tool"] for t in result.tool_trace] == ["search_vehicles"]
    assert result.cards[0]["type"] == "vehicle_options"
    assert trip["location"] == "YVR"
    (tool_msg,) = llm.tool_results()
    assert '"ok": true' in tool_msg["content"]
    assert '"location": "YVR"' in llm.requests[-1]["system"]  # system prompt rebuilt with new context


async def test_validation_error_goes_back_to_llm(orchestrator, make_tool_ctx, anchor):
    llm = FakeLLM([call("search_vehicles", **_search_args(anchor, passengers=0)), say("How many people?")])
    result = await orchestrator(llm).run_turn(user_message="car pls", history=[], trip=empty_context(),
                                              ctx=make_tool_ctx())
    assert result.reply == "How many people?"
    assert result.tool_trace[0]["error_code"] == "VALIDATION_ERROR"
    assert '"VALIDATION_ERROR"' in llm.tool_results()[0]["content"]


async def test_parallel_tool_calls(orchestrator, make_tool_ctx):
    llm = FakeLLM([
        calls(("get_my_reservation", {"confirmation_code": "DLRAV4"}),
              ("search_policy", {"query": "premium protection coverage", "plan_code": "PREMIUM_PROTECT"})),
        say("done"),
    ])
    result = await orchestrator(llm).run_turn(user_message="q", history=[], trip=empty_context(),
                                              ctx=make_tool_ctx(kb_min_score=0.0))
    assert [t["tool"] for t in result.tool_trace] == ["get_my_reservation", "search_policy"]
    assert result.sources


async def test_llm_failure_returns_fallback_and_is_audited(orchestrator, make_tool_ctx):
    ctx = make_tool_ctx()
    result = await orchestrator(FakeLLM([LLMError("provider down")])).run_turn(
        user_message="hi", history=[], trip=empty_context(), ctx=ctx)
    assert result.reply == LLM_ERROR_REPLY
    assert ctx.audit.events("LLM_ERROR")


async def test_llm_failure_after_tools_keeps_partial_results(orchestrator, make_tool_ctx):
    llm = FakeLLM([call("get_my_reservation", confirmation_code="DLRAV4"), LLMError("provider down")])
    trip = empty_context()
    result = await orchestrator(llm).run_turn(user_message="hi", history=[], trip=trip, ctx=make_tool_ctx())
    assert result.reply == LLM_ERROR_REPLY
    assert [t["tool"] for t in result.tool_trace] == ["get_my_reservation"]
    assert trip["active_reservation_code"] == "DLRAV4"


async def test_max_steps(orchestrator, make_tool_ctx):
    llm = FakeLLM([call("get_my_reservation", confirmation_code="DLRAV4") for _ in range(6)])
    result = await orchestrator(llm).run_turn(user_message="loop", history=[], trip=empty_context(),
                                              ctx=make_tool_ctx())
    assert result.reply == MAX_STEPS_REPLY
    assert len(result.tool_trace) == 6
    assert len(llm.requests) == 6


class _SlowLLM:
    async def complete(self, system, messages, tools):
        await asyncio.sleep(1)
        return say("too late")


async def test_turn_timeout(orchestrator, make_tool_ctx):
    result = await orchestrator(_SlowLLM()).run_turn(
        user_message="hi", history=[], trip=empty_context(), ctx=make_tool_ctx(turn_timeout_seconds=0.05))
    assert result.reply == TIMEOUT_REPLY


async def test_history_is_sent_to_llm(orchestrator, make_tool_ctx):
    llm = FakeLLM([say("ok")])
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"}]
    await orchestrator(llm).run_turn(user_message="now", history=history, trip=empty_context(), ctx=make_tool_ctx())
    assert [m["content"] for m in llm.requests[0]["messages"]] == ["earlier", "answer", "now"]
