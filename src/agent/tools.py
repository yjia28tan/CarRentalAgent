"""
Tools exposed to the LLM. None of them accepts a customer id: identity comes from the
RequestContext (verified JWT), which the business API client forwards.

execute_tool() validates arguments, runs the tool with a timeout, audit-logs it, and never raises:
every failure becomes {"ok": false, "error_code", "message"} for the LLM.
"""
import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from src.agent.dates import DateParseError, resolve_date, resolve_datetime
from src.agent.pending import create_pending_action, public_pending_action
from src.clients.business_api import BusinessApiClient, UpstreamError
from src.clients.kb_client import KbClient
from src.deps import RequestContext
from src.llm.gateway import ToolCall, ToolSpec
from src.shared.errors import ErrorCode
from src.storage.audit import AuditLog
from src.storage.session_store import SessionStore
from src.utils.config import Settings
from src.utils.logger import get_logger
from src.utils.utils import redact_code

logger = get_logger(__name__)

MAX_RENTAL_DAYS = 60
PICKUP_GRACE = timedelta(minutes=15)

Category = Literal["ECONOMY", "COMPACT", "MIDSIZE_SUV", "FULLSIZE_SUV", "MINIVAN", "PREMIUM"]
ReservationStatus = Literal["CONFIRMED", "PICKED_UP", "RETURNED", "CANCELLED"]
PolicyCategory = Literal["INSURANCE", "ACCIDENT", "CANCELLATION", "ROADSIDE", "TERMS"]


# ---------------------------------------------------------------- context


@dataclass
class ToolContext:
    request: RequestContext
    session_id: str
    api: BusinessApiClient
    kb: KbClient
    sessions: SessionStore
    audit: AuditLog
    settings: Settings
    now: datetime  # aware, in settings.default_timezone

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.settings.default_timezone)


@dataclass
class DateNotes:
    """Validation context: reference 'now' plus assumptions collected while parsing dates."""

    now: datetime
    notes: list[str] = field(default_factory=list)


def _date_ctx(info: ValidationInfo) -> DateNotes:
    ctx = (info.context or {}).get("dates")
    return ctx if isinstance(ctx, DateNotes) else DateNotes(now=datetime.now().astimezone())


def _parse_local_datetime(value: Any, info: ValidationInfo) -> Any:
    if value is None:
        return None
    ctx = _date_ctx(info)
    try:
        resolved, notes = resolve_datetime(value, ctx.now)
    except DateParseError as exc:
        raise ValueError(str(exc)) from None
    ctx.notes.extend(notes)
    return resolved


def _parse_local_date(value: Any, info: ValidationInfo) -> Any:
    if value is None or isinstance(value, date):
        return value
    ctx = _date_ctx(info)
    if not isinstance(value, str):
        raise ValueError("Expected a date string.")
    try:
        resolved, notes = resolve_date(value, ctx.now.date())
    except DateParseError as exc:
        raise ValueError(str(exc)) from None
    ctx.notes.extend(notes)
    return resolved


LocalDateTime = Annotated[
    datetime,
    BeforeValidator(_parse_local_datetime),
    WithJsonSchema({"type": "string", "description": "Local date/time, e.g. '2026-10-10T10:00', 'October 10', "
                    "'tomorrow 3pm'. 10:00 is assumed if no time is given."}),
]
LocalDate = Annotated[
    date,
    BeforeValidator(_parse_local_date),
    WithJsonSchema({"type": "string", "description": "Date, e.g. '2026-10-10', 'tomorrow', 'October 10'."}),
]


def _aware(dt: datetime, now: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=now.tzinfo)


def check_rental_window(pickup_at: datetime, return_at: datetime, now: datetime) -> None:
    pickup, ret = _aware(pickup_at, now), _aware(return_at, now)
    if ret <= pickup:
        raise ValueError("return_at must be after pickup_at.")
    if pickup < now - PICKUP_GRACE:
        raise ValueError(f"pickup_at ({pickup:%Y-%m-%d %H:%M}) is in the past; today is {now:%Y-%m-%d}.")
    if ret - pickup > timedelta(days=MAX_RENTAL_DAYS):
        raise ValueError(f"Rentals can be at most {MAX_RENTAL_DAYS} days long.")


# ---------------------------------------------------------------- inputs


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, protected_namespaces=())


class SearchVehiclesInput(ToolInput):
    location: Annotated[str, Field(min_length=1, max_length=100, description="Location code or name, e.g. 'YVR' or 'Vancouver Airport'.")]
    pickup_at: LocalDateTime
    return_at: LocalDateTime
    passengers: Annotated[int, Field(ge=1, le=9, description="Total people travelling (1-9). Larger groups need several vehicles.")]
    large_bags: Annotated[int, Field(ge=0, le=10, description="Number of large suitcases.")] = 0
    category: Category | None = None
    max_daily_rate: Annotated[Decimal | None, Field(gt=0, le=10000)] = None

    @model_validator(mode="after")
    def _window(self, info: ValidationInfo) -> "SearchVehiclesInput":
        check_rental_window(self.pickup_at, self.return_at, _date_ctx(info).now)
        return self


class VehicleDetailsInput(ToolInput):
    models: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=50)]] | None,
        Field(min_length=1, max_length=4, description="Model names, e.g. ['RAV4', 'Corolla'].")
    ] = None
    model_ids: Annotated[list[Annotated[int, Field(ge=1)]] | None, Field(min_length=1, max_length=4)] = None

    @model_validator(mode="after")
    def _one_of(self) -> "VehicleDetailsInput":
        if not self.models and not self.model_ids:
            raise ValueError("Provide 'models' (names) or 'model_ids'.")
        return self


class PriceQuoteInput(ToolInput):
    model_id: Annotated[int, Field(ge=1, description="model_id from search_vehicles or get_vehicle_details.")]
    location: Annotated[str, Field(min_length=1, max_length=100)]
    pickup_at: LocalDateTime
    return_at: LocalDateTime
    insurance_codes: Annotated[
        list[Annotated[str, Field(pattern=r"^[A-Za-z_]{2,40}$")]],
        Field(max_length=3, description="Optional protection plans, e.g. ['BASIC_CDW'] or ['PREMIUM_PROTECT'].")
    ] = []

    @field_validator("insurance_codes")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        return [c.upper() for c in v]

    @model_validator(mode="after")
    def _window(self, info: ValidationInfo) -> "PriceQuoteInput":
        check_rental_window(self.pickup_at, self.return_at, _date_ctx(info).now)
        return self


class ListReservationsInput(ToolInput):
    from_date: Annotated[LocalDate | None, Field(description="Earliest pickup date (inclusive).")] = None
    to_date: Annotated[LocalDate | None, Field(description="Latest pickup date (inclusive).")] = None
    status: Annotated[list[ReservationStatus] | None, Field(max_length=4)] = None

    @model_validator(mode="after")
    def _order(self) -> "ListReservationsInput":
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date must be on or before to_date.")
        return self


class GetReservationInput(ToolInput):
    confirmation_code: Annotated[str, Field(pattern=r"^[A-Za-z0-9]{6}$", description="6-character confirmation code.")]

    @field_validator("confirmation_code")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class SearchPolicyInput(ToolInput):
    query: Annotated[str, Field(min_length=3, max_length=300)]
    categories: Annotated[list[PolicyCategory] | None, Field(max_length=5)] = None
    plan_code: Annotated[str | None, Field(pattern=r"^[A-Za-z_]{2,40}$", description="Filter to a purchased plan, e.g. 'PREMIUM_PROTECT'.")] = None
    top_k: Annotated[int, Field(ge=1, le=10)] = 5

    @field_validator("plan_code")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v


class ProposeChangeInput(ToolInput):
    confirmation_code: Annotated[str, Field(pattern=r"^[A-Za-z0-9]{6}$", description="6-character confirmation code.")]
    new_return_at: LocalDateTime

    @field_validator("confirmation_code")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


# ---------------------------------------------------------------- outcomes


@dataclass
class ToolOutcome:
    data: dict[str, Any]
    cards: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None


@dataclass
class ToolExecution:
    call: ToolCall
    status: Literal["ok", "error"]
    error_code: str | None
    latency_ms: float
    result: dict[str, Any]
    args: dict[str, Any] | None = None
    cards: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None

    def trace(self) -> dict[str, Any]:
        return {"tool": self.call.name, "status": self.status, "latency_ms": self.latency_ms,
                "error_code": self.error_code}

    def message(self) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": self.call.id, "name": self.call.name,
                "content": json.dumps(self.result, default=str)}


# ---------------------------------------------------------------- handlers


def _reservation_card(r: dict[str, Any]) -> dict[str, Any]:
    return {"type": "reservation", "data": {
        "confirmation_code": r["confirmation_code"], "status": r["status"], "vehicle": r["vehicle"]["name"],
        "pickup_location": r["pickup_location"]["name"], "pickup_at": r["pickup_at"], "return_at": r["return_at"],
        "payment_status": r["payment_status"], "total_amount": r["total_amount"], "currency": r["currency"],
        "insurance": [p["name"] for p in r["insurance"]],
    }}


async def _search_vehicles(args: SearchVehiclesInput, ctx: ToolContext) -> ToolOutcome:
    resp = await ctx.api.search_availability(
        location=args.location, pickup_at=args.pickup_at, return_at=args.return_at, passengers=args.passengers,
        large_bags=args.large_bags, category=args.category, max_daily_rate=args.max_daily_rate,
    )
    options = [
        {k: o[k] for k in ("model_id", "name", "category", "seats", "large_bags", "small_bags",
                           "available_count", "daily_rate", "currency")}
        for o in resp["options"]
    ]
    data = {k: resp[k] for k in ("search_id", "location", "pickup_at", "return_at", "days")} | {"options": options}
    if not options:
        data["note"] = "No vehicle matches all requirements at this location for these dates."
    return ToolOutcome(data=data, cards=[{"type": "vehicle_options", "data": data}])


async def _get_vehicle_details(args: VehicleDetailsInput, ctx: ToolContext) -> ToolOutcome:
    resp = await ctx.api.vehicle_models(ids=args.model_ids, names=args.models)
    data = {"models": resp["models"], "not_found": resp["not_found"]}
    cards = [{"type": "vehicle_comparison", "data": {"models": resp["models"]}}] if resp["models"] else []
    return ToolOutcome(data=data, cards=cards)


async def _get_price_quote(args: PriceQuoteInput, ctx: ToolContext) -> ToolOutcome:
    quote = await ctx.api.create_quote({
        "model_id": args.model_id, "location": args.location, "pickup_at": args.pickup_at,
        "return_at": args.return_at, "insurance_codes": args.insurance_codes,
    })
    await ctx.sessions.set_json(f"quote:{quote['quote_id']}", quote, ctx.settings.quote_ttl_seconds)
    breakdown = {k: quote[k] for k in ("days", "daily_rate", "base", "insurance", "insurance_total",
                                        "subtotal", "taxes_fees")}
    card = {"type": "price_quote", "data": {
        "quote_id": quote["quote_id"], "model": quote["model"]["name"], "pickup_at": quote["pickup_at"],
        "return_at": quote["return_at"], "total": quote["total"], "currency": quote["currency"],
        "breakdown": breakdown, "expires_at": quote["expires_at"],
    }}
    return ToolOutcome(data=quote, cards=[card])


async def _list_my_reservations(args: ListReservationsInput, ctx: ToolContext) -> ToolOutcome:
    resp = await ctx.api.list_reservations(args.from_date, args.to_date, args.status)
    reservations = resp["reservations"]
    data = {"count": len(reservations), "reservations": reservations}
    return ToolOutcome(data=data, cards=[_reservation_card(r) for r in reservations])


async def _get_my_reservation(args: GetReservationInput, ctx: ToolContext) -> ToolOutcome:
    r = await ctx.api.get_reservation(args.confirmation_code)
    cards = [_reservation_card(r)]
    if r["insurance"]:
        cards.append({"type": "insurance", "data": {"plans": [
            {"code": p["code"], "name": p["name"], "deductible": p["deductible"], "currency": p["currency"]}
            for p in r["insurance"]
        ]}})
    return ToolOutcome(data=r, cards=cards)


async def _search_policy(args: SearchPolicyInput, ctx: ToolContext) -> ToolOutcome:
    results = await ctx.kb.search(args.query, args.top_k, args.categories, args.plan_code)
    relevant = [r for r in results if r["score"] >= ctx.settings.kb_min_score]
    if not relevant:
        return ToolOutcome(data={"results": [], "note": "no relevant policy found"})
    return ToolOutcome(
        data={"results": relevant},
        sources=[{"doc_id": r["doc_id"], "title": r["title"], "section": r["section"]} for r in relevant],
    )


async def _propose_reservation_change(args: ProposeChangeInput, ctx: ToolContext) -> ToolOutcome:
    """Phase one of the two-phase write: price the change and park it. Nothing is modified here."""
    preview = await ctx.api.preview_change(args.confirmation_code, args.new_return_at)
    pending = await create_pending_action(ctx, preview)
    public = public_pending_action(pending)
    data = {
        "status": "AWAITING_CUSTOMER_CONFIRMATION",
        **{k: public[k] for k in ("action_id", "summary", "old_total", "new_total", "price_delta", "currency",
                                  "expires_at")},
        "vehicle_changed": preview.get("vehicle_changed", False),
        "instructions": "Nothing has been changed yet. Ask the customer to review the change and press the "
                        "Confirm button. You cannot confirm it for them.",
    }
    return ToolOutcome(data=data, cards=[{"type": "pending_action", "data": public}], pending_action=public)


# ---------------------------------------------------------------- registry


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[ToolInput]
    handler: Callable[[Any, ToolContext], Awaitable[ToolOutcome]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description,
                        input_schema=_strip_titles(self.input_model.model_json_schema()))


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


TOOLS: dict[str, Tool] = {t.name: t for t in (
    Tool("search_vehicles",
         "Find vehicle models available at a location for a rental period that fit the party size and large "
         "suitcases. Returns options with daily rates.",
         SearchVehiclesInput, _search_vehicles),
    Tool("get_vehicle_details",
         "Get specifications (seats, luggage capacity, features) for 1-4 vehicle models, for comparisons.",
         VehicleDetailsInput, _get_vehicle_details),
    Tool("get_price_quote",
         "Get an exact price quote (base, insurance, taxes, total) for one model, location and rental period.",
         PriceQuoteInput, _get_price_quote),
    Tool("list_my_reservations",
         "List the signed-in customer's own reservations, optionally filtered by pickup date range and status.",
         ListReservationsInput, _list_my_reservations),
    Tool("get_my_reservation",
         "Get full details, including purchased insurance, of one of the signed-in customer's reservations.",
         GetReservationInput, _get_my_reservation),
    Tool("search_policy",
         "Search rental policies: insurance coverage, accident procedure, cancellation/modification, roadside "
         "assistance, rental terms. Filter by the customer's purchased plan_code when explaining coverage.",
         SearchPolicyInput, _search_policy),
    Tool("propose_reservation_change",
         "Propose changing the return date of one of the customer's reservations. This only prepares the change "
         "and its price; the customer must press the Confirm button to apply it.",
         ProposeChangeInput, _propose_reservation_change),
)}


def tool_specs() -> list[ToolSpec]:
    return [t.spec for t in TOOLS.values()]


# ---------------------------------------------------------------- execution


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        if err.get("type") == "extra_forbidden":
            msg = f"'{loc}' is not an accepted argument for this tool"
            loc = ""
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        if k == "confirmation_code" and isinstance(v, str):
            out[k] = redact_code(v)
        elif isinstance(v, (str, int, float, bool, list, type(None))):
            out[k] = v if not isinstance(v, str) else v[:200]
        else:
            out[k] = str(v)
    return out


def _error(code: ErrorCode | str, message: str, details: dict | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "error_code": str(code), "message": message}
    if details:
        result["details"] = details
    return result


async def execute_tool(call: ToolCall, ctx: ToolContext, tools: dict[str, Tool] = TOOLS) -> ToolExecution:
    started = time.perf_counter()
    tool = tools.get(call.name)
    validated: dict[str, Any] | None = None
    cards: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    pending_action: dict[str, Any] | None = None

    if tool is None:
        result = _error(ErrorCode.VALIDATION_ERROR, f"Unknown tool '{call.name}'. Available: {', '.join(tools)}.")
    else:
        notes = DateNotes(now=ctx.now)
        try:
            args = tool.input_model.model_validate(call.arguments, context={"dates": notes})
        except ValidationError as exc:
            result = _error(ErrorCode.VALIDATION_ERROR, format_validation_error(exc))
        else:
            validated = args.model_dump(mode="json")
            try:
                async with asyncio.timeout(ctx.settings.tool_timeout_seconds):
                    outcome = await tool.handler(args, ctx)
            except TimeoutError:
                result = _error(ErrorCode.UPSTREAM_TIMEOUT, "The service took too long to respond.")
            except UpstreamError as exc:
                result = _error(exc.code, exc.message, exc.details)
            except Exception:
                logger.exception("tool_failed", tool=call.name)
                result = _error(ErrorCode.INTERNAL_ERROR, "Unexpected error while running this tool.")
            else:
                result = {"ok": True, **outcome.data}
                if notes.notes:
                    result["assumptions"] = list(dict.fromkeys(notes.notes))
                cards, sources, pending_action = outcome.cards, outcome.sources, outcome.pending_action

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    ok = result["ok"]
    execution = ToolExecution(
        call=call, status="ok" if ok else "error", error_code=None if ok else result["error_code"],
        latency_ms=latency_ms, result=result, args=validated, cards=cards, sources=sources,
        pending_action=pending_action,
    )
    log_args = redact_args(call.arguments)
    logger.info("tool_call", tool=call.name, status=execution.status, error_code=execution.error_code,
                latency_ms=latency_ms, args=log_args)
    await ctx.audit.record(
        "VALIDATION_ERROR" if execution.error_code == ErrorCode.VALIDATION_ERROR else "TOOL_CALL",
        session_id=ctx.session_id,
        customer_id=ctx.request.customer_id,
        tool=call.name,
        args_redacted=log_args,
        outcome=execution.error_code or "ok",
        latency_ms=latency_ms,
    )
    return execution
