"""Catalog and availability: /v1/locations, /v1/availability, /v1/vehicle-models."""
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query

from src.mock_services.deps import get_repo
from src.mock_services.repository import Repository
from src.mock_services.serializers import local_iso, location_dict, model_dict
from src.mock_services.services.pricing import rental_days
from src.mock_services.services.rules import localize, match_models, require_location, validate_rental_window
from src.shared.auth import require_principal
from src.shared.errors import AppError, ErrorCode
from src.utils.utils import new_id, utcnow

router = APIRouter(prefix="/v1", tags=["catalog"], dependencies=[Depends(require_principal)])

Category = Literal["ECONOMY", "COMPACT", "MIDSIZE_SUV", "FULLSIZE_SUV", "MINIVAN", "PREMIUM"]
MAX_MODEL_LOOKUPS = 10


@router.get("/locations")
async def list_locations(repo: Repository = Depends(get_repo)):
    return {"locations": [location_dict(l) for l in await repo.list_locations()]}


@router.get("/availability")
async def search_availability(
    location: Annotated[str, Query(min_length=1, max_length=100)],
    pickup_at: datetime,
    return_at: datetime,
    passengers: Annotated[int, Query(ge=1, le=9)] = 1,
    large_bags: Annotated[int, Query(ge=0, le=10)] = 0,
    category: Category | None = None,
    max_daily_rate: Annotated[Decimal | None, Query(gt=0, max_digits=10, decimal_places=2)] = None,
    repo: Repository = Depends(get_repo),
):
    loc = require_location(location, await repo.list_locations())
    pickup_at = localize(pickup_at, loc.timezone)
    return_at = localize(return_at, loc.timezone)
    validate_rental_window(pickup_at, return_at, utcnow())

    rates = await repo.get_rates(loc.id, pickup_at.astimezone(ZoneInfo(loc.timezone)).date())
    counts = await repo.available_counts(loc.id, pickup_at, return_at)

    options = []
    for m in await repo.list_models():
        rate = rates.get(m.category)
        available = counts.get(m.id, 0)
        if rate is None or available == 0:
            continue
        if m.seats < passengers or m.large_bags < large_bags:
            continue
        if category and m.category != category:
            continue
        if max_daily_rate is not None and rate.daily_rate > max_daily_rate:
            continue
        options.append({
            "model_id": m.id, "make": m.make, "model": m.model, "name": m.name, "category": m.category,
            "seats": m.seats, "large_bags": m.large_bags, "small_bags": m.small_bags,
            "available_count": available, "daily_rate": rate.daily_rate, "currency": rate.currency,
        })
    options.sort(key=lambda o: (o["daily_rate"], o["model_id"]))

    return {
        "search_id": new_id("srch"),
        "location": {"code": loc.code, "name": loc.name, "timezone": loc.timezone},
        "pickup_at": local_iso(pickup_at, loc.timezone),
        "return_at": local_iso(return_at, loc.timezone),
        "days": rental_days(pickup_at, return_at),
        "passengers": passengers,
        "large_bags": large_bags,
        "options": options,
    }


@router.get("/vehicle-models")
async def get_vehicle_models(
    ids: Annotated[str | None, Query(max_length=100, description="Comma-separated model ids")] = None,
    names: Annotated[str | None, Query(max_length=300, description="Comma-separated names, e.g. RAV4,Corolla")] = None,
    repo: Repository = Depends(get_repo),
):
    id_list = [s.strip() for s in (ids or "").split(",") if s.strip()]
    name_list = [s.strip() for s in (names or "").split(",") if s.strip()]
    if not id_list and not name_list:
        raise AppError(ErrorCode.VALIDATION_ERROR, "Provide 'ids' or 'names'.")
    if len(id_list) + len(name_list) > MAX_MODEL_LOOKUPS:
        raise AppError(ErrorCode.VALIDATION_ERROR, f"At most {MAX_MODEL_LOOKUPS} models per request.")
    if not all(s.isdigit() for s in id_list):
        raise AppError(ErrorCode.VALIDATION_ERROR, "'ids' must be comma-separated integers.")

    models = await repo.list_models()
    by_id = {m.id: m for m in models}
    found = {int(i): by_id[int(i)] for i in id_list if int(i) in by_id}
    not_found = [i for i in id_list if int(i) not in by_id]
    matched, unmatched = match_models(name_list, models)
    for m in matched:
        found.setdefault(m.id, m)
    return {"models": [model_dict(m) for m in found.values()], "not_found": not_found + unmatched}
