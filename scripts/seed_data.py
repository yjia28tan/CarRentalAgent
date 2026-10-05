"""
Seed demo data. Idempotent: truncates the business tables and re-inserts.

    python scripts/seed_data.py [--anchor-date YYYY-MM-DD]

Reservation dates are relative to the anchor date (default: today in DEFAULT_TIMEZONE).
KB tables are not touched; use ingest_policies.py for those.
"""
import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg  # noqa: E402
import bcrypt  # noqa: E402

from src.mock_services.services.pricing import price_rental  # noqa: E402
from src.utils.config import get_settings  # noqa: E402
from src.utils.logger import configure_logging, get_logger  # noqa: E402

logger = get_logger("seed_data")

CURRENCY = "CAD"
DEMO_PASSWORD = "demo123"
# Fixed namespace so customer/reservation UUIDs are stable across re-seeds.
_ID_NAMESPACE = uuid.UUID("6f1c2b8e-3d4a-4f5b-9c7d-0e1f2a3b4c5d")

LOCATIONS = [
    {
        "id": 1, "code": "YVR", "name": "Vancouver International Airport", "city": "Vancouver",
        "country": "CA", "timezone": "America/Vancouver",
        "aliases": ["vancouver international airport", "vancouver airport", "vancouver", "yvr"],
    },
    {
        "id": 2, "code": "YVR-DT", "name": "Vancouver Downtown", "city": "Vancouver",
        "country": "CA", "timezone": "America/Vancouver",
        "aliases": ["vancouver downtown", "downtown vancouver", "yvr-dt", "yvr dt"],
    },
    {
        "id": 3, "code": "YYZ", "name": "Toronto Pearson International Airport", "city": "Toronto",
        "country": "CA", "timezone": "America/Toronto",
        "aliases": ["toronto pearson", "pearson", "toronto airport", "toronto", "yyz"],
    },
]

VEHICLE_MODELS = [
    {
        "id": 1, "make": "Kia", "model": "Rio", "category": "ECONOMY", "seats": 5, "large_bags": 1,
        "small_bags": 1, "transmission": "AUTOMATIC", "fuel_type": "GASOLINE", "plate": "RIO",
        "features": {"awd": False, "isofix_points": 2, "cargo_litres": 388,
                     "fuel_economy_l_per_100km": 6.6, "apple_carplay": True},
    },
    {
        "id": 2, "make": "Toyota", "model": "Corolla", "category": "COMPACT", "seats": 5, "large_bags": 2,
        "small_bags": 1, "transmission": "AUTOMATIC", "fuel_type": "GASOLINE", "plate": "COR",
        "features": {"awd": False, "isofix_points": 2, "cargo_litres": 371,
                     "fuel_economy_l_per_100km": 6.8, "apple_carplay": True, "adaptive_cruise": True},
    },
    {
        "id": 3, "make": "Toyota", "model": "RAV4", "category": "MIDSIZE_SUV", "seats": 5, "large_bags": 3,
        "small_bags": 2, "transmission": "AUTOMATIC", "fuel_type": "HYBRID", "plate": "RAV",
        "features": {"awd": True, "isofix_points": 2, "cargo_litres": 1059,
                     "fuel_economy_l_per_100km": 6.0, "apple_carplay": True, "adaptive_cruise": True},
    },
    {
        "id": 4, "make": "Toyota", "model": "Highlander", "category": "FULLSIZE_SUV", "seats": 7,
        "large_bags": 4, "small_bags": 2, "transmission": "AUTOMATIC", "fuel_type": "GASOLINE",
        "plate": "HIG",
        "features": {"awd": True, "isofix_points": 3, "cargo_litres": 456, "third_row": True,
                     "fuel_economy_l_per_100km": 10.2, "apple_carplay": True, "adaptive_cruise": True},
    },
    {
        "id": 5, "make": "Chrysler", "model": "Pacifica", "category": "MINIVAN", "seats": 7, "large_bags": 5,
        "small_bags": 3, "transmission": "AUTOMATIC", "fuel_type": "GASOLINE", "plate": "PAC",
        "features": {"awd": False, "isofix_points": 3, "cargo_litres": 915, "sliding_doors": True,
                     "fuel_economy_l_per_100km": 10.7, "apple_carplay": True, "rear_entertainment": True},
    },
    {
        "id": 6, "make": "BMW", "model": "3 Series", "category": "PREMIUM", "seats": 5, "large_bags": 2,
        "small_bags": 1, "transmission": "AUTOMATIC", "fuel_type": "GASOLINE", "plate": "BMW",
        "features": {"awd": True, "isofix_points": 2, "cargo_litres": 480, "leather_seats": True,
                     "fuel_economy_l_per_100km": 7.6, "apple_carplay": True, "adaptive_cruise": True},
    },
]

DAILY_RATES = {
    "ECONOMY": Decimal("45.00"),
    "COMPACT": Decimal("55.00"),
    "MIDSIZE_SUV": Decimal("74.00"),
    "FULLSIZE_SUV": Decimal("98.00"),
    "MINIVAN": Decimal("105.00"),
    "PREMIUM": Decimal("120.00"),
}

INSURANCE_PLANS = [
    {"code": "BASIC_CDW", "name": "Basic Collision Damage Waiver", "daily_price": Decimal("18.00"),
     "deductible": Decimal("1500.00"), "currency": CURRENCY, "kb_doc_id": "insurance-basic-cdw"},
    {"code": "PREMIUM_PROTECT", "name": "Premium Protection Plan", "daily_price": Decimal("32.00"),
     "deductible": Decimal("0.00"), "currency": CURRENCY, "kb_doc_id": "insurance-premium-protect"},
]

UNITS_PER_LOCATION = {"YVR": 3, "YVR-DT": 2, "YYZ": 2}

CUSTOMERS = [
    {"email": "daniel@example.com", "full_name": "Daniel Lee", "phone": "+1-604-555-0101"},
    {"email": "emily@example.com", "full_name": "Emily Lee", "phone": "+1-604-555-0102"},
    {"email": "guest@example.com", "full_name": "Guest User", "phone": None},
]

# (code, customer email, model id, pickup day offset, return day offset, status, payment, insurance)
RESERVATIONS = [
    ("DLCOR1", "daniel@example.com", 2, -2, 3, "PICKED_UP", "PAID", ["BASIC_CDW"]),
    ("DLRAV4", "daniel@example.com", 3, 1, 6, "CONFIRMED", "AUTHORIZED", ["PREMIUM_PROTECT"]),
    ("ELBMW3", "emily@example.com", 6, -1, 4, "PICKED_UP", "PAID", ["PREMIUM_PROTECT"]),
]
DEFAULT_HOUR = 10


def stable_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(_ID_NAMESPACE, name)


@dataclass
class Seed:
    customers: list[dict] = field(default_factory=list)
    locations: list[dict] = field(default_factory=list)
    vehicle_models: list[dict] = field(default_factory=list)
    vehicles: list[dict] = field(default_factory=list)
    rates: list[dict] = field(default_factory=list)
    insurance_plans: list[dict] = field(default_factory=list)
    reservations: list[dict] = field(default_factory=list)
    reservation_insurance: list[dict] = field(default_factory=list)


def build_seed(anchor: date) -> Seed:
    """Pure function: all rows to insert for the given anchor date."""
    seed = Seed(locations=LOCATIONS, insurance_plans=INSURANCE_PLANS)
    seed.vehicle_models = [{k: v for k, v in m.items() if k != "plate"} for m in VEHICLE_MODELS]

    password_hash = bcrypt.hashpw(DEMO_PASSWORD.encode(), bcrypt.gensalt()).decode()
    seed.customers = [{**c, "id": stable_uuid(c["email"]), "password_hash": password_hash} for c in CUSTOMERS]

    vehicle_id = 0
    for loc in LOCATIONS:
        for model in VEHICLE_MODELS:
            for n in range(1, UNITS_PER_LOCATION[loc["code"]] + 1):
                vehicle_id += 1
                seed.vehicles.append({
                    "id": vehicle_id,
                    "model_id": model["id"],
                    "home_location_id": loc["id"],
                    "plate": f"{loc['code'].replace('-', '')}-{model['plate']}{n}",
                    "status": "ACTIVE",
                })

    rate_id = 0
    for loc in LOCATIONS:
        for category, daily_rate in DAILY_RATES.items():
            rate_id += 1
            seed.rates.append({
                "id": rate_id, "location_id": loc["id"], "category": category, "daily_rate": daily_rate,
                "currency": CURRENCY,
                # Also cover next year so late-year demos can still search/quote.
                "valid_from": date(anchor.year, 1, 1), "valid_to": date(anchor.year + 1, 12, 31),
            })

    yvr = LOCATIONS[0]
    tz = ZoneInfo(yvr["timezone"])
    models_by_id = {m["id"]: m for m in VEHICLE_MODELS}
    plans_by_code = {p["code"]: p for p in INSURANCE_PLANS}
    customer_ids = {c["email"]: c["id"] for c in seed.customers}

    for code, email, model_id, pickup_off, return_off, status, payment, plan_codes in RESERVATIONS:
        vehicle = next(
            v for v in seed.vehicles if v["model_id"] == model_id and v["home_location_id"] == yvr["id"]
        )
        pickup_at = datetime.combine(anchor + timedelta(days=pickup_off), time(DEFAULT_HOUR), tzinfo=tz)
        return_at = datetime.combine(anchor + timedelta(days=return_off), time(DEFAULT_HOUR), tzinfo=tz)
        price = price_rental(
            daily_rate=DAILY_RATES[models_by_id[model_id]["category"]],
            pickup_at=pickup_at,
            return_at=return_at,
            insurance_daily_prices={c: plans_by_code[c]["daily_price"] for c in plan_codes},
            currency=CURRENCY,
        )
        reservation_id = stable_uuid(code)
        seed.reservations.append({
            "id": reservation_id, "confirmation_code": code, "customer_id": customer_ids[email],
            "vehicle_id": vehicle["id"], "pickup_location_id": yvr["id"], "return_location_id": yvr["id"],
            "pickup_at": pickup_at, "return_at": return_at, "status": status, "payment_status": payment,
            "total_amount": price.total, "currency": CURRENCY, "version": 1,
        })
        seed.reservation_insurance.extend(
            {"reservation_id": reservation_id, "plan_code": line.code, "price": line.amount}
            for line in price.insurance
        )
    return seed


async def write_seed(conn: asyncpg.Connection, seed: Seed) -> None:
    async with conn.transaction():
        await conn.execute(
            "TRUNCATE reservation_changes, reservation_insurance, reservations, rates, vehicles, "
            "vehicle_models, insurance_plans, locations, customers RESTART IDENTITY CASCADE"
        )
        await conn.executemany(
            "INSERT INTO customers (id, email, full_name, phone, password_hash) VALUES ($1,$2,$3,$4,$5)",
            [(c["id"], c["email"], c["full_name"], c["phone"], c["password_hash"]) for c in seed.customers],
        )
        await conn.executemany(
            "INSERT INTO locations (id, code, name, city, country, timezone, aliases) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7)",
            [(l["id"], l["code"], l["name"], l["city"], l["country"], l["timezone"], l["aliases"])
             for l in seed.locations],
        )
        await conn.executemany(
            "INSERT INTO vehicle_models (id, make, model, category, seats, large_bags, small_bags, "
            "transmission, fuel_type, features) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb)",
            [(m["id"], m["make"], m["model"], m["category"], m["seats"], m["large_bags"], m["small_bags"],
              m["transmission"], m["fuel_type"], json.dumps(m["features"])) for m in seed.vehicle_models],
        )
        await conn.executemany(
            "INSERT INTO vehicles (id, model_id, home_location_id, plate, status) VALUES ($1,$2,$3,$4,$5)",
            [(v["id"], v["model_id"], v["home_location_id"], v["plate"], v["status"]) for v in seed.vehicles],
        )
        await conn.executemany(
            "INSERT INTO rates (id, location_id, category, daily_rate, currency, valid_from, valid_to) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7)",
            [(r["id"], r["location_id"], r["category"], r["daily_rate"], r["currency"], r["valid_from"],
              r["valid_to"]) for r in seed.rates],
        )
        await conn.executemany(
            "INSERT INTO insurance_plans (code, name, daily_price, deductible, currency, kb_doc_id) "
            "VALUES ($1,$2,$3,$4,$5,$6)",
            [(p["code"], p["name"], p["daily_price"], p["deductible"], p["currency"], p["kb_doc_id"])
             for p in seed.insurance_plans],
        )
        await conn.executemany(
            "INSERT INTO reservations (id, confirmation_code, customer_id, vehicle_id, pickup_location_id, "
            "return_location_id, pickup_at, return_at, status, payment_status, total_amount, currency, version) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
            [(r["id"], r["confirmation_code"], r["customer_id"], r["vehicle_id"], r["pickup_location_id"],
              r["return_location_id"], r["pickup_at"], r["return_at"], r["status"], r["payment_status"],
              r["total_amount"], r["currency"], r["version"]) for r in seed.reservations],
        )
        await conn.executemany(
            "INSERT INTO reservation_insurance (reservation_id, plan_code, price) VALUES ($1,$2,$3)",
            [(ri["reservation_id"], ri["plan_code"], ri["price"]) for ri in seed.reservation_insurance],
        )
        # Explicit ids were inserted, so move SERIAL sequences past them.
        for table in ("locations", "vehicle_models", "vehicles", "rates"):
            await conn.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), (SELECT MAX(id) FROM {table}))"
            )


async def _run(anchor: date) -> None:
    seed = build_seed(anchor)
    conn = await asyncpg.connect(get_settings().postgres_dsn.get_secret_value())
    try:
        await write_seed(conn, seed)
    finally:
        await conn.close()
    logger.info(
        "seed_complete",
        anchor_date=anchor.isoformat(),
        customers=len(seed.customers),
        vehicles=len(seed.vehicles),
        reservations=[r["confirmation_code"] for r in seed.reservations],
    )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--anchor-date",
        type=date.fromisoformat,
        default=datetime.now(ZoneInfo(settings.default_timezone)).date(),
        help="Demo 'today' (YYYY-MM-DD). Default: today in DEFAULT_TIMEZONE.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.anchor_date))


if __name__ == "__main__":
    main()
