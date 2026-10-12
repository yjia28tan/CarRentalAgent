"""POST /v1/quotes."""
from datetime import datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from src.mock_services.deps import get_repo
from src.mock_services.repository import Repository
from src.mock_services.serializers import local_iso
from src.mock_services.services.pricing import price_rental
from src.mock_services.services.rules import localize, require_location, validate_rental_window
from src.shared.auth import require_principal
from src.shared.errors import AppError, ErrorCode
from src.utils.config import get_settings
from src.utils.utils import new_id, utcnow

router = APIRouter(prefix="/v1", tags=["quotes"], dependencies=[Depends(require_principal)])


class QuoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: Annotated[int, Field(ge=1)]
    location: Annotated[str, Field(min_length=1, max_length=100)]
    pickup_at: datetime
    return_at: datetime
    insurance_codes: Annotated[list[str], Field(max_length=5)] = []


@router.post("/quotes")
async def create_quote(body: QuoteRequest, repo: Repository = Depends(get_repo)):
    loc = require_location(body.location, await repo.list_locations())
    pickup_at = localize(body.pickup_at, loc.timezone)
    return_at = localize(body.return_at, loc.timezone)
    validate_rental_window(pickup_at, return_at, utcnow())

    model = next((m for m in await repo.list_models() if m.id == body.model_id), None)
    if model is None:
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Unknown model_id {body.model_id}.")

    plans = {p.code: p for p in await repo.list_insurance_plans()}
    codes = list(dict.fromkeys(c.strip().upper() for c in body.insurance_codes))
    unknown = [c for c in codes if c not in plans]
    if unknown:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"Unknown insurance code(s): {', '.join(unknown)}. Valid codes: {', '.join(plans)}.",
        )

    rate = (await repo.get_rates(loc.id, pickup_at.astimezone(ZoneInfo(loc.timezone)).date())).get(model.category)
    available = (await repo.available_counts(loc.id, pickup_at, return_at)).get(model.id, 0)
    if rate is None or available == 0:
        raise AppError(ErrorCode.NOT_AVAILABLE, f"The {model.name} is not available at {loc.name} for those dates.")

    price = price_rental(
        daily_rate=rate.daily_rate,
        pickup_at=pickup_at,
        return_at=return_at,
        insurance_daily_prices={c: plans[c].daily_price for c in codes},
        currency=rate.currency,
    )
    return {
        "quote_id": new_id("qt"),
        "model": {"model_id": model.id, "name": model.name, "category": model.category},
        "location": {"code": loc.code, "name": loc.name, "timezone": loc.timezone},
        "pickup_at": local_iso(pickup_at, loc.timezone),
        "return_at": local_iso(return_at, loc.timezone),
        "days": price.days,
        "daily_rate": price.daily_rate,
        "base": price.base,
        "insurance": [
            {"code": line.code, "name": plans[line.code].name, "daily_price": line.daily_price, "amount": line.amount}
            for line in price.insurance
        ],
        "insurance_total": price.insurance_total,
        "subtotal": price.subtotal,
        "taxes_fees": price.taxes_fees,
        "total": price.total,
        "currency": price.currency,
        "expires_at": utcnow() + timedelta(seconds=get_settings().quote_ttl_seconds),
    }
