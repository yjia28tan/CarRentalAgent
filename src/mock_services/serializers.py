"""JSON shapes returned by the mock services (Decimal/datetime/UUID are encoded by FastAPI)."""
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.mock_services.repository import InsurancePlan, Location, Reservation, VehicleModel


def local_iso(dt: datetime, tz_name: str) -> str:
    return dt.astimezone(ZoneInfo(tz_name)).isoformat()


def location_dict(loc: Location) -> dict[str, Any]:
    return {"code": loc.code, "name": loc.name, "city": loc.city, "country": loc.country, "timezone": loc.timezone}


def model_dict(m: VehicleModel) -> dict[str, Any]:
    return {
        "model_id": m.id, "make": m.make, "model": m.model, "name": m.name, "category": m.category,
        "seats": m.seats, "large_bags": m.large_bags, "small_bags": m.small_bags,
        "transmission": m.transmission, "fuel_type": m.fuel_type, "features": m.features,
    }


def plan_dict(p: InsurancePlan) -> dict[str, Any]:
    return {
        "code": p.code, "name": p.name, "daily_price": p.daily_price, "deductible": p.deductible,
        "currency": p.currency, "kb_doc_id": p.kb_doc_id,
    }


def reservation_dict(r: Reservation) -> dict[str, Any]:
    tz = r.pickup_location.timezone
    return {
        "confirmation_code": r.confirmation_code,
        "status": r.status,
        "payment_status": r.payment_status,
        "vehicle": {
            "model_id": r.model.id, "make": r.model.make, "model": r.model.model, "name": r.model.name,
            "category": r.model.category, "seats": r.model.seats, "large_bags": r.model.large_bags,
            "plate": r.plate,
        },
        "pickup_location": location_dict(r.pickup_location),
        "return_location": location_dict(r.return_location),
        "pickup_at": local_iso(r.pickup_at, tz),
        "return_at": local_iso(r.return_at, r.return_location.timezone),
        "total_amount": r.total_amount,
        "currency": r.currency,
        "version": r.version,
        "insurance": [
            {"code": p.code, "name": p.name, "deductible": p.deductible, "currency": p.currency,
             "kb_doc_id": p.kb_doc_id}
            for p in r.insurance
        ],
    }
