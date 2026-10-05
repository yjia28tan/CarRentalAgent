"""
/v1/me/reservations: the caller's own reservations only (customer_id = JWT sub).
A code owned by someone else returns the same 404 as an unknown code, and AUTHZ_DENIED is logged.
"""
from datetime import date, datetime, timedelta
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, Path, Query
from pydantic import BaseModel, ConfigDict, Field

from src.mock_services.deps import get_previews, get_repo
from src.mock_services.repository import ACTIVE_STATUSES, Repository, Reservation, ReturnDateChange
from src.mock_services.serializers import local_iso, reservation_dict
from src.mock_services.services.previews import PreviewStore
from src.mock_services.services.pricing import price_rental
from src.mock_services.services.rules import MAX_RENTAL_DAYS, localize
from src.shared.auth import Principal, require_principal
from src.shared.errors import AppError, ErrorCode
from src.utils.logger import get_logger
from src.utils.utils import redact_code, utcnow

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/me/reservations", tags=["reservations"])

Status = Literal["CONFIRMED", "PICKED_UP", "RETURNED", "CANCELLED"]
ConfirmationCode = Annotated[str, Path(pattern=r"^[A-Za-z0-9]{6}$")]
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_\-:.]+$")]


async def get_owned_reservation(repo: Repository, code: str, principal: Principal) -> Reservation:
    reservation = await repo.get_reservation(code.upper())
    if reservation is None:
        raise AppError(ErrorCode.RESERVATION_NOT_FOUND)
    if reservation.customer_id != principal.customer_id:
        logger.warning(
            "AUTHZ_DENIED",
            resource="reservation",
            confirmation_code=redact_code(reservation.confirmation_code),
            customer_id=str(principal.customer_id),
            owner_customer_id=str(reservation.customer_id),
        )
        raise AppError(ErrorCode.RESERVATION_NOT_FOUND)
    return reservation


@router.get("")
async def list_my_reservations(
    from_date: date | None = None,
    to_date: date | None = None,
    status: Annotated[list[Status] | None, Query()] = None,
    principal: Principal = Depends(require_principal),
    repo: Repository = Depends(get_repo),
):
    """from_date/to_date filter on the pickup date (local to the pickup location)."""
    if from_date and to_date and from_date > to_date:
        raise AppError(ErrorCode.VALIDATION_ERROR, "from_date must be on or before to_date.")
    reservations = await repo.list_reservations(principal.customer_id, from_date, to_date, status)
    return {"reservations": [reservation_dict(r) for r in reservations]}


@router.get("/{code}")
async def get_my_reservation(
    code: ConfirmationCode,
    principal: Principal = Depends(require_principal),
    repo: Repository = Depends(get_repo),
):
    return reservation_dict(await get_owned_reservation(repo, code, principal))


class ChangePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    new_return_at: datetime


@router.post("/{code}/change-previews")
async def preview_change(
    code: ConfirmationCode,
    body: ChangePreviewRequest,
    principal: Principal = Depends(require_principal),
    repo: Repository = Depends(get_repo),
    previews: PreviewStore = Depends(get_previews),
):
    res = await get_owned_reservation(repo, code, principal)
    if res.status not in ACTIVE_STATUSES:
        raise AppError(ErrorCode.INVALID_STATE, f"Reservation is {res.status} and can no longer be changed.")

    new_return_at = localize(body.new_return_at, res.return_location.timezone)
    if new_return_at <= res.pickup_at:
        raise AppError(ErrorCode.VALIDATION_ERROR, "The new return time must be after the pickup time.")
    if new_return_at <= utcnow():
        raise AppError(ErrorCode.VALIDATION_ERROR, "The new return time must be in the future.")
    if new_return_at == res.return_at:
        raise AppError(ErrorCode.VALIDATION_ERROR, "That is already the return time on this reservation.")
    if new_return_at - res.pickup_at > timedelta(days=MAX_RENTAL_DAYS):
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Rentals can be at most {MAX_RENTAL_DAYS} days long.")

    # Same unit first; otherwise another ACTIVE unit of the same model at the same location.
    vehicle_id = await repo.find_free_vehicle(
        res.model.id, res.pickup_location.id, res.pickup_at, new_return_at, res.id, res.vehicle_id
    )
    pickup_day = res.pickup_at.astimezone(ZoneInfo(res.pickup_location.timezone)).date()
    rate = (await repo.get_rates(res.pickup_location.id, pickup_day)).get(res.model.category)
    if vehicle_id is None or rate is None:
        raise AppError(
            ErrorCode.NOT_AVAILABLE,
            f"No {res.model.name} is available at {res.pickup_location.name} for the new dates.",
        )

    price = price_rental(
        daily_rate=rate.daily_rate,
        pickup_at=res.pickup_at,
        return_at=new_return_at,
        insurance_daily_prices={p.code: p.daily_price for p in res.insurance},
        currency=res.currency,
    )
    preview = previews.create(
        reservation_id=res.id,
        customer_id=principal.customer_id,
        confirmation_code=res.confirmation_code,
        expected_version=res.version,
        vehicle_id=vehicle_id,
        old_return_at=res.return_at,
        new_return_at=new_return_at,
        old_total=res.total_amount,
        new_total=price.total,
        price_delta=price.total - res.total_amount,
        currency=res.currency,
        insurance_prices={line.code: line.amount for line in price.insurance},
    )
    tz = res.return_location.timezone
    return {
        "preview_id": preview.preview_id,
        "available": True,
        "confirmation_code": res.confirmation_code,
        "old_return_at": local_iso(res.return_at, tz),
        "new_return_at": local_iso(new_return_at, tz),
        "days": price.days,
        "old_total": preview.old_total,
        "new_total": preview.new_total,
        "price_delta": preview.price_delta,
        "currency": preview.currency,
        "expected_version": preview.expected_version,
        "vehicle_changed": vehicle_id != res.vehicle_id,
        "expires_at": preview.expires_at,
    }


class ChangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_id: Annotated[str, Field(pattern=r"^pv_[0-9A-Z]{26}$")]
    expected_version: Annotated[int, Field(ge=1)]


@router.post("/{code}/changes")
async def apply_change(
    code: ConfirmationCode,
    body: ChangeRequest,
    idempotency_key: IdempotencyKey,
    principal: Principal = Depends(require_principal),
    repo: Repository = Depends(get_repo),
    previews: PreviewStore = Depends(get_previews),
):
    res = await get_owned_reservation(repo, code, principal)

    replayed_for = await repo.get_change_reservation_id(idempotency_key)
    if replayed_for is not None:
        if replayed_for != res.id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "This Idempotency-Key was already used for another request.")
        return {"status": "APPLIED", "reservation": reservation_dict(res)}

    preview = previews.get(body.preview_id)
    if preview is None or preview.reservation_id != res.id or preview.customer_id != principal.customer_id:
        raise AppError(ErrorCode.PREVIEW_EXPIRED)
    if body.expected_version != preview.expected_version or res.version != preview.expected_version:
        raise AppError(ErrorCode.VERSION_CONFLICT)

    await repo.apply_return_date_change(
        ReturnDateChange(
            reservation_id=res.id,
            customer_id=principal.customer_id,
            expected_version=preview.expected_version,
            vehicle_id=preview.vehicle_id,
            new_return_at=preview.new_return_at,
            new_total=preview.new_total,
            insurance_prices=preview.insurance_prices,
            price_delta=preview.price_delta,
            old_values={
                "return_at": res.return_at.isoformat(),
                "total_amount": str(res.total_amount),
                "vehicle_id": res.vehicle_id,
            },
            new_values={
                "return_at": preview.new_return_at.isoformat(),
                "total_amount": str(preview.new_total),
                "vehicle_id": preview.vehicle_id,
            },
            idempotency_key=idempotency_key,
        )
    )
    previews.discard(preview.preview_id)
    updated = await get_owned_reservation(repo, res.confirmation_code, principal)
    logger.info(
        "reservation_changed",
        confirmation_code=redact_code(updated.confirmation_code),
        customer_id=str(principal.customer_id),
        version=updated.version,
    )
    return {"status": "APPLIED", "reservation": reservation_dict(updated)}
