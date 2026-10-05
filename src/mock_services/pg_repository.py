"""PostgreSQL (asyncpg + pgvector) implementation of the mock-services Repository."""
import json
from datetime import date, datetime
from uuid import UUID

import asyncpg
import numpy as np

from src.mock_services.db import create_pool
from src.mock_services.repository import (
    InsurancePlan,
    KbHit,
    Location,
    Rate,
    Reservation,
    ReturnDateChange,
    VehicleModel,
)
from src.shared.errors import AppError, ErrorCode

_RESERVATION_SELECT = """
SELECT r.id, r.confirmation_code, r.customer_id, r.vehicle_id, v.plate,
       r.pickup_at, r.return_at, r.status::text AS status, r.payment_status::text AS payment_status,
       r.total_amount, r.currency, r.version,
       m.id AS m_id, m.make AS m_make, m.model AS m_model, m.category AS m_category, m.seats AS m_seats,
       m.large_bags AS m_large_bags, m.small_bags AS m_small_bags, m.transmission AS m_transmission,
       m.fuel_type AS m_fuel_type, m.features AS m_features,
       pl.id AS pl_id, pl.code AS pl_code, pl.name AS pl_name, pl.city AS pl_city,
       pl.country AS pl_country, pl.timezone AS pl_timezone, pl.aliases AS pl_aliases,
       rl.id AS rl_id, rl.code AS rl_code, rl.name AS rl_name, rl.city AS rl_city,
       rl.country AS rl_country, rl.timezone AS rl_timezone, rl.aliases AS rl_aliases
FROM reservations r
JOIN vehicles v        ON v.id = r.vehicle_id
JOIN vehicle_models m  ON m.id = v.model_id
JOIN locations pl      ON pl.id = r.pickup_location_id
JOIN locations rl      ON rl.id = r.return_location_id
"""

_ACTIVE = "('CONFIRMED','PICKED_UP')"


def _location(row: asyncpg.Record, prefix: str = "") -> Location:
    return Location(
        id=row[f"{prefix}id"], code=row[f"{prefix}code"], name=row[f"{prefix}name"], city=row[f"{prefix}city"],
        country=row[f"{prefix}country"], timezone=row[f"{prefix}timezone"], aliases=tuple(row[f"{prefix}aliases"]),
    )


def _model(row: asyncpg.Record, prefix: str = "") -> VehicleModel:
    features = row[f"{prefix}features"]
    return VehicleModel(
        id=row[f"{prefix}id"], make=row[f"{prefix}make"], model=row[f"{prefix}model"],
        category=row[f"{prefix}category"], seats=row[f"{prefix}seats"], large_bags=row[f"{prefix}large_bags"],
        small_bags=row[f"{prefix}small_bags"], transmission=row[f"{prefix}transmission"],
        fuel_type=row[f"{prefix}fuel_type"],
        features=json.loads(features) if isinstance(features, str) else dict(features),
    )


def _plan(row: asyncpg.Record) -> InsurancePlan:
    return InsurancePlan(
        code=row["code"], name=row["name"], daily_price=row["daily_price"], deductible=row["deductible"],
        currency=row["currency"].strip(), kb_doc_id=row["kb_doc_id"],
    )


class PgRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, dsn: str | None = None) -> "PgRepository":
        return cls(await create_pool(dsn))

    async def ping(self) -> None:
        await self._pool.fetchval("SELECT 1")

    async def close(self) -> None:
        await self._pool.close()

    async def list_locations(self) -> list[Location]:
        rows = await self._pool.fetch("SELECT * FROM locations ORDER BY id")
        return [_location(r) for r in rows]

    async def list_models(self) -> list[VehicleModel]:
        rows = await self._pool.fetch("SELECT * FROM vehicle_models ORDER BY id")
        return [_model(r) for r in rows]

    async def get_rates(self, location_id: int, on: date) -> dict[str, Rate]:
        rows = await self._pool.fetch(
            "SELECT category, daily_rate, currency FROM rates "
            "WHERE location_id = $1 AND $2 BETWEEN valid_from AND valid_to ORDER BY valid_from",
            location_id, on,
        )
        return {r["category"]: Rate(r["category"], r["daily_rate"], r["currency"].strip()) for r in rows}

    async def available_counts(self, location_id: int, start: datetime, end: datetime) -> dict[int, int]:
        rows = await self._pool.fetch(
            f"""
            SELECT v.model_id, COUNT(*) AS available
            FROM vehicles v
            WHERE v.home_location_id = $1 AND v.status = 'ACTIVE'
              AND NOT EXISTS (
                SELECT 1 FROM reservations r
                WHERE r.vehicle_id = v.id AND r.status IN {_ACTIVE}
                  AND r.rental_period && tstzrange($2, $3))
            GROUP BY v.model_id
            """,
            location_id, start, end,
        )
        return {r["model_id"]: r["available"] for r in rows}

    async def list_insurance_plans(self) -> list[InsurancePlan]:
        rows = await self._pool.fetch("SELECT * FROM insurance_plans ORDER BY code")
        return [_plan(r) for r in rows]

    async def _with_insurance(self, rows: list[asyncpg.Record]) -> list[Reservation]:
        if not rows:
            return []
        ins_rows = await self._pool.fetch(
            "SELECT ri.reservation_id, p.* FROM reservation_insurance ri "
            "JOIN insurance_plans p ON p.code = ri.plan_code "
            "WHERE ri.reservation_id = ANY($1::uuid[]) ORDER BY p.code",
            [r["id"] for r in rows],
        )
        plans: dict[UUID, list[InsurancePlan]] = {}
        for ir in ins_rows:
            plans.setdefault(ir["reservation_id"], []).append(_plan(ir))
        return [
            Reservation(
                id=r["id"], confirmation_code=r["confirmation_code"], customer_id=r["customer_id"],
                vehicle_id=r["vehicle_id"], plate=r["plate"], model=_model(r, "m_"),
                pickup_location=_location(r, "pl_"), return_location=_location(r, "rl_"),
                pickup_at=r["pickup_at"], return_at=r["return_at"], status=r["status"],
                payment_status=r["payment_status"], total_amount=r["total_amount"],
                currency=r["currency"].strip(), version=r["version"],
                insurance=tuple(plans.get(r["id"], [])),
            )
            for r in rows
        ]

    async def list_reservations(
        self, customer_id: UUID, from_date: date | None, to_date: date | None, statuses: list[str] | None
    ) -> list[Reservation]:
        rows = await self._pool.fetch(
            _RESERVATION_SELECT
            + """
            WHERE r.customer_id = $1
              AND ($2::date IS NULL OR (r.pickup_at AT TIME ZONE pl.timezone)::date >= $2)
              AND ($3::date IS NULL OR (r.pickup_at AT TIME ZONE pl.timezone)::date <= $3)
              AND ($4::text[] IS NULL OR r.status::text = ANY($4::text[]))
            ORDER BY r.pickup_at
            """,
            customer_id, from_date, to_date, statuses,
        )
        return await self._with_insurance(rows)

    async def get_reservation(self, confirmation_code: str) -> Reservation | None:
        rows = await self._pool.fetch(_RESERVATION_SELECT + " WHERE r.confirmation_code = $1", confirmation_code)
        found = await self._with_insurance(rows)
        return found[0] if found else None

    async def find_free_vehicle(
        self,
        model_id: int,
        location_id: int,
        start: datetime,
        end: datetime,
        exclude_reservation_id: UUID | None,
        prefer_vehicle_id: int | None,
    ) -> int | None:
        return await self._pool.fetchval(
            f"""
            SELECT v.id FROM vehicles v
            WHERE v.model_id = $1 AND v.home_location_id = $2 AND v.status = 'ACTIVE'
              AND NOT EXISTS (
                SELECT 1 FROM reservations r
                WHERE r.vehicle_id = v.id AND r.status IN {_ACTIVE}
                  AND ($5::uuid IS NULL OR r.id <> $5)
                  AND r.rental_period && tstzrange($3, $4))
            ORDER BY COALESCE(v.id = $6, false) DESC, v.id
            LIMIT 1
            """,
            model_id, location_id, start, end, exclude_reservation_id, prefer_vehicle_id,
        )

    async def get_change_reservation_id(self, idempotency_key: str) -> UUID | None:
        return await self._pool.fetchval(
            "SELECT reservation_id FROM reservation_changes WHERE idempotency_key = $1", idempotency_key
        )

    async def apply_return_date_change(self, change: ReturnDateChange) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT customer_id, version, status::text AS status FROM reservations WHERE id = $1 FOR UPDATE",
                change.reservation_id,
            )
            if row is None or row["customer_id"] != change.customer_id:
                raise AppError(ErrorCode.RESERVATION_NOT_FOUND)
            if row["version"] != change.expected_version:
                raise AppError(ErrorCode.VERSION_CONFLICT)
            if row["status"] not in ("CONFIRMED", "PICKED_UP"):
                raise AppError(ErrorCode.INVALID_STATE)
            try:
                await conn.execute(
                    "UPDATE reservations SET return_at = $2, vehicle_id = $3, total_amount = $4, "
                    "version = version + 1, updated_at = now() WHERE id = $1",
                    change.reservation_id, change.new_return_at, change.vehicle_id, change.new_total,
                )
            except asyncpg.exceptions.ExclusionViolationError:
                raise AppError(ErrorCode.NOT_AVAILABLE) from None
            await conn.executemany(
                "UPDATE reservation_insurance SET price = $3 WHERE reservation_id = $1 AND plan_code = $2",
                [(change.reservation_id, code, price) for code, price in change.insurance_prices.items()],
            )
            try:
                await conn.execute(
                    "INSERT INTO reservation_changes (reservation_id, customer_id, change_type, old_values, "
                    "new_values, price_delta, idempotency_key, status) "
                    "VALUES ($1, $2, 'MODIFY_RETURN_DATE', $3::jsonb, $4::jsonb, $5, $6, 'APPLIED')",
                    change.reservation_id, change.customer_id, change.old_values, change.new_values,
                    change.price_delta, change.idempotency_key,
                )
            except asyncpg.exceptions.UniqueViolationError:
                # A concurrent request with the same key won; roll back this one.
                raise AppError(ErrorCode.VERSION_CONFLICT) from None

    async def kb_search(
        self, vector: np.ndarray, top_k: int, categories: list[str] | None, plan_code: str | None
    ) -> list[KbHit]:
        rows = await self._pool.fetch(
            """
            SELECT c.doc_id, d.title, c.section, c.content, 1 - (c.embedding <=> $1) AS score
            FROM kb_chunks c JOIN kb_documents d ON d.doc_id = c.doc_id
            WHERE ($2::text[] IS NULL OR d.category = ANY($2::text[]))
              AND ($3::text IS NULL OR c.plan_codes = '{}' OR $3 = ANY(c.plan_codes))
            ORDER BY c.embedding <=> $1
            LIMIT $4
            """,
            vector, categories, plan_code, top_k,
        )
        return [KbHit(r["doc_id"], r["title"], r["section"], r["content"], float(r["score"])) for r in rows]
