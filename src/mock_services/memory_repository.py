"""
In-memory Repository with the same semantics as PgRepository, so mock-service and agent tests run
without Docker. Built from the seed rows produced by scripts/seed_data.build_seed().
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import numpy as np

from src.mock_services.repository import (
    ACTIVE_STATUSES,
    Customer,
    InsurancePlan,
    KbHit,
    Location,
    Rate,
    Reservation,
    ReturnDateChange,
    VehicleModel,
)
from src.shared.errors import AppError, ErrorCode


@dataclass
class KbRow:
    doc_id: str
    title: str
    category: str
    section: str
    content: str
    plan_codes: list[str]
    vector: np.ndarray


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


class InMemoryRepository:
    def __init__(self, seed: Any, kb: list[KbRow] | None = None) -> None:
        self.locations = {l["id"]: Location(**{**l, "aliases": tuple(l["aliases"])}) for l in seed.locations}
        self.models = {m["id"]: VehicleModel(**m) for m in seed.vehicle_models}
        self.vehicles = {v["id"]: dict(v) for v in seed.vehicles}
        self.rates = [dict(r) for r in seed.rates]
        self.plans = {p["code"]: InsurancePlan(**p) for p in seed.insurance_plans}
        self.customers = {c["id"]: dict(c) for c in seed.customers}
        self.reservations = {r["id"]: dict(r) for r in seed.reservations}
        self.reservation_insurance = [dict(ri) for ri in seed.reservation_insurance]
        self.changes: list[dict] = []
        self.kb = kb or []

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_customer_by_email(self, email: str) -> Customer | None:
        for c in self.customers.values():
            if c["email"].lower() == email.lower():
                return Customer(c["id"], c["email"], c["full_name"], c["password_hash"])
        return None

    async def list_locations(self) -> list[Location]:
        return list(self.locations.values())

    async def list_models(self) -> list[VehicleModel]:
        return list(self.models.values())

    async def get_rates(self, location_id: int, on: date) -> dict[str, Rate]:
        return {
            r["category"]: Rate(r["category"], r["daily_rate"], r["currency"])
            for r in self.rates
            if r["location_id"] == location_id and r["valid_from"] <= on <= r["valid_to"]
        }

    def _vehicle_is_free(self, vehicle_id: int, start: datetime, end: datetime, exclude: UUID | None) -> bool:
        return not any(
            r["vehicle_id"] == vehicle_id
            and r["id"] != exclude
            and r["status"] in ACTIVE_STATUSES
            and _overlaps(r["pickup_at"], r["return_at"], start, end)
            for r in self.reservations.values()
        )

    async def available_counts(self, location_id: int, start: datetime, end: datetime) -> dict[int, int]:
        counts: dict[int, int] = {}
        for v in self.vehicles.values():
            if v["home_location_id"] == location_id and v["status"] == "ACTIVE" and self._vehicle_is_free(
                v["id"], start, end, None
            ):
                counts[v["model_id"]] = counts.get(v["model_id"], 0) + 1
        return counts

    async def list_insurance_plans(self) -> list[InsurancePlan]:
        return sorted(self.plans.values(), key=lambda p: p.code)

    def _to_reservation(self, r: dict) -> Reservation:
        vehicle = self.vehicles[r["vehicle_id"]]
        plans = sorted(
            (self.plans[ri["plan_code"]] for ri in self.reservation_insurance if ri["reservation_id"] == r["id"]),
            key=lambda p: p.code,
        )
        return Reservation(
            id=r["id"], confirmation_code=r["confirmation_code"], customer_id=r["customer_id"],
            vehicle_id=r["vehicle_id"], plate=vehicle["plate"], model=self.models[vehicle["model_id"]],
            pickup_location=self.locations[r["pickup_location_id"]],
            return_location=self.locations[r["return_location_id"]],
            pickup_at=r["pickup_at"], return_at=r["return_at"], status=r["status"],
            payment_status=r["payment_status"], total_amount=r["total_amount"], currency=r["currency"],
            version=r["version"], insurance=tuple(plans),
        )

    async def list_reservations(
        self, customer_id: UUID, from_date: date | None, to_date: date | None, statuses: list[str] | None
    ) -> list[Reservation]:
        result = []
        for r in self.reservations.values():
            if r["customer_id"] != customer_id:
                continue
            tz = ZoneInfo(self.locations[r["pickup_location_id"]].timezone)
            pickup_day = r["pickup_at"].astimezone(tz).date()
            if from_date and pickup_day < from_date:
                continue
            if to_date and pickup_day > to_date:
                continue
            if statuses and r["status"] not in statuses:
                continue
            result.append(r)
        return [self._to_reservation(r) for r in sorted(result, key=lambda r: r["pickup_at"])]

    async def get_reservation(self, confirmation_code: str) -> Reservation | None:
        for r in self.reservations.values():
            if r["confirmation_code"] == confirmation_code:
                return self._to_reservation(r)
        return None

    async def find_free_vehicle(
        self,
        model_id: int,
        location_id: int,
        start: datetime,
        end: datetime,
        exclude_reservation_id: UUID | None,
        prefer_vehicle_id: int | None,
    ) -> int | None:
        candidates = sorted(
            (v for v in self.vehicles.values()
             if v["model_id"] == model_id and v["home_location_id"] == location_id and v["status"] == "ACTIVE"),
            key=lambda v: (v["id"] != prefer_vehicle_id, v["id"]),
        )
        for v in candidates:
            if self._vehicle_is_free(v["id"], start, end, exclude_reservation_id):
                return v["id"]
        return None

    async def get_change_reservation_id(self, idempotency_key: str) -> UUID | None:
        return next((c["reservation_id"] for c in self.changes if c["idempotency_key"] == idempotency_key), None)

    async def apply_return_date_change(self, change: ReturnDateChange) -> None:
        # No awaits below, so this block is atomic within the event loop (like the Postgres transaction).
        r = self.reservations.get(change.reservation_id)
        if r is None or r["customer_id"] != change.customer_id:
            raise AppError(ErrorCode.RESERVATION_NOT_FOUND)
        if r["version"] != change.expected_version:
            raise AppError(ErrorCode.VERSION_CONFLICT)
        if r["status"] not in ACTIVE_STATUSES:
            raise AppError(ErrorCode.INVALID_STATE)
        if not self._vehicle_is_free(change.vehicle_id, r["pickup_at"], change.new_return_at, r["id"]):
            raise AppError(ErrorCode.NOT_AVAILABLE)
        if any(c["idempotency_key"] == change.idempotency_key for c in self.changes):
            raise AppError(ErrorCode.VERSION_CONFLICT)

        r.update(
            return_at=change.new_return_at,
            vehicle_id=change.vehicle_id,
            total_amount=change.new_total,
            version=r["version"] + 1,
        )
        for ri in self.reservation_insurance:
            if ri["reservation_id"] == r["id"] and ri["plan_code"] in change.insurance_prices:
                ri["price"] = change.insurance_prices[ri["plan_code"]]
        self.changes.append({
            "id": uuid4(), "reservation_id": r["id"], "customer_id": change.customer_id,
            "change_type": "MODIFY_RETURN_DATE", "old_values": change.old_values, "new_values": change.new_values,
            "price_delta": change.price_delta, "idempotency_key": change.idempotency_key, "status": "APPLIED",
        })

    async def kb_search(
        self, vector: np.ndarray, top_k: int, categories: list[str] | None, plan_code: str | None
    ) -> list[KbHit]:
        hits = [
            KbHit(row.doc_id, row.title, row.section, row.content, float(np.dot(row.vector, vector)))
            for row in self.kb
            if (not categories or row.category in categories)
            and (plan_code is None or not row.plan_codes or plan_code in row.plan_codes)
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
