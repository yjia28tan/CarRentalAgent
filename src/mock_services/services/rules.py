"""Deterministic business rules shared by the mock-service routes."""
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.mock_services.repository import Location, VehicleModel
from src.shared.errors import AppError, ErrorCode

MAX_RENTAL_DAYS = 60
PICKUP_GRACE = timedelta(minutes=15)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def resolve_location(query: str, locations: list[Location]) -> Location | None:
    """Match a location code, name, or alias (case/punctuation-insensitive)."""
    q = _normalize(query)
    if not q:
        return None
    for loc in locations:
        names = {_normalize(loc.code), _normalize(loc.name), *(_normalize(a) for a in loc.aliases)}
        if q in names:
            return loc
    return None


def location_not_found(locations: list[Location]) -> AppError:
    supported = [{"code": l.code, "name": l.name} for l in locations]
    names = ", ".join(f"{l.name} ({l.code})" for l in locations)
    return AppError(
        ErrorCode.LOCATION_NOT_FOUND,
        f"We don't operate at that location. Supported locations: {names}.",
        details={"supported_locations": supported},
    )


def require_location(query: str, locations: list[Location]) -> Location:
    loc = resolve_location(query, locations)
    if loc is None:
        raise location_not_found(locations)
    return loc


def localize(dt: datetime, tz_name: str) -> datetime:
    """Treat naive datetimes as local time at the location."""
    return dt.replace(tzinfo=ZoneInfo(tz_name)) if dt.tzinfo is None else dt


def validate_rental_window(pickup_at: datetime, return_at: datetime, now: datetime) -> None:
    if return_at <= pickup_at:
        raise AppError(ErrorCode.VALIDATION_ERROR, "The return time must be after the pickup time.")
    if pickup_at < now - PICKUP_GRACE:
        raise AppError(ErrorCode.VALIDATION_ERROR, "The pickup time is in the past.")
    if return_at - pickup_at > timedelta(days=MAX_RENTAL_DAYS):
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Rentals can be at most {MAX_RENTAL_DAYS} days long.")


def match_models(names: list[str], models: list[VehicleModel]) -> tuple[list[VehicleModel], list[str]]:
    """Match names like 'RAV4', 'Toyota RAV4', '3 series'. Returns (matched models, unmatched names)."""
    found: dict[int, VehicleModel] = {}
    not_found: list[str] = []
    for name in names:
        q = _compact(name)
        exact = [m for m in models if q and q in (_compact(m.model), _compact(m.name))]
        partial = [m for m in models if len(q) >= 3 and q in _compact(m.name)]
        matches = exact or partial
        if not matches:
            not_found.append(name)
        for m in matches:
            found.setdefault(m.id, m)
    return list(found.values()), not_found
