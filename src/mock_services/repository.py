"""Data access for the mock services: domain records and the Repository protocol."""
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID

import numpy as np

ACTIVE_STATUSES = ("CONFIRMED", "PICKED_UP")
RESERVATION_STATUSES = ("CONFIRMED", "PICKED_UP", "RETURNED", "CANCELLED")


@dataclass(frozen=True)
class Location:
    id: int
    code: str
    name: str
    city: str
    country: str
    timezone: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class VehicleModel:
    id: int
    make: str
    model: str
    category: str
    seats: int
    large_bags: int
    small_bags: int
    transmission: str
    fuel_type: str
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.make} {self.model}"


@dataclass(frozen=True)
class Rate:
    category: str
    daily_rate: Decimal
    currency: str


@dataclass(frozen=True)
class InsurancePlan:
    code: str
    name: str
    daily_price: Decimal
    deductible: Decimal
    currency: str
    kb_doc_id: str


@dataclass(frozen=True)
class Reservation:
    id: UUID
    confirmation_code: str
    customer_id: UUID
    vehicle_id: int
    plate: str
    model: VehicleModel
    pickup_location: Location
    return_location: Location
    pickup_at: datetime
    return_at: datetime
    status: str
    payment_status: str
    total_amount: Decimal
    currency: str
    version: int
    insurance: tuple[InsurancePlan, ...] = ()


@dataclass(frozen=True)
class KbHit:
    doc_id: str
    title: str
    section: str
    content: str
    score: float


@dataclass(frozen=True)
class ReturnDateChange:
    reservation_id: UUID
    customer_id: UUID
    expected_version: int
    vehicle_id: int
    new_return_at: datetime
    new_total: Decimal
    insurance_prices: dict[str, Decimal]
    price_delta: Decimal
    old_values: dict[str, Any]
    new_values: dict[str, Any]
    idempotency_key: str


class Repository(Protocol):
    async def ping(self) -> None: ...

    async def close(self) -> None: ...

    async def list_locations(self) -> list[Location]: ...

    async def list_models(self) -> list[VehicleModel]: ...

    async def get_rates(self, location_id: int, on: date) -> dict[str, Rate]: ...

    async def available_counts(self, location_id: int, start: datetime, end: datetime) -> dict[int, int]:
        """model_id -> number of ACTIVE units at the location with no active reservation overlapping [start, end)."""
        ...

    async def list_insurance_plans(self) -> list[InsurancePlan]: ...

    async def list_reservations(
        self,
        customer_id: UUID,
        from_date: date | None,
        to_date: date | None,
        statuses: list[str] | None,
    ) -> list[Reservation]:
        """Filter by pickup date (in the pickup location's timezone), ordered by pickup_at."""
        ...

    async def get_reservation(self, confirmation_code: str) -> Reservation | None:
        """Unscoped lookup; callers MUST check ownership."""
        ...

    async def find_free_vehicle(
        self,
        model_id: int,
        location_id: int,
        start: datetime,
        end: datetime,
        exclude_reservation_id: UUID | None,
        prefer_vehicle_id: int | None,
    ) -> int | None: ...

    async def get_change_reservation_id(self, idempotency_key: str) -> UUID | None: ...

    async def apply_return_date_change(self, change: ReturnDateChange) -> None:
        """Atomically re-check owner/version/state/availability and apply. Raises AppError on conflict."""
        ...

    async def kb_search(
        self, vector: np.ndarray, top_k: int, categories: list[str] | None, plan_code: str | None
    ) -> list[KbHit]: ...
