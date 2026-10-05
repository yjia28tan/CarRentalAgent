"""Pure-logic checks for seed_data.py and ingest_policies.py (no database or model download)."""
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import bcrypt
import pytest

from scripts.ingest_policies import (
    DEFAULT_POLICIES_DIR,
    VALID_CATEGORIES,
    _split_long,
    chunk_policy,
    load_policies,
    parse_policy,
)
from scripts.seed_data import build_seed

ANCHOR = date(2026, 10, 5)
TZ = ZoneInfo("America/Vancouver")


@pytest.fixture(scope="module")
def seed():
    return build_seed(ANCHOR)


def _res(seed, code):
    return next(r for r in seed.reservations if r["confirmation_code"] == code)


# ---------- seed ----------

def test_fleet_sizes(seed):
    loc_codes = {l["id"]: l["code"] for l in seed.locations}
    counts = Counter((loc_codes[v["home_location_id"]], v["model_id"]) for v in seed.vehicles)
    for model in seed.vehicle_models:
        assert counts[("YVR", model["id"])] >= 3
        assert counts[("YVR-DT", model["id"])] == 2
        assert counts[("YYZ", model["id"])] == 2
    assert len({v["plate"] for v in seed.vehicles}) == len(seed.vehicles)


def test_model_capacities_match_spec(seed):
    caps = {f"{m['make']} {m['model']}": (m["seats"], m["large_bags"], m["small_bags"]) for m in seed.vehicle_models}
    assert caps == {
        "Kia Rio": (5, 1, 1),
        "Toyota Corolla": (5, 2, 1),
        "Toyota RAV4": (5, 3, 2),
        "Toyota Highlander": (7, 4, 2),
        "Chrysler Pacifica": (7, 5, 3),
        "BMW 3 Series": (5, 2, 1),
    }


def test_rates_cover_every_location_and_category(seed):
    assert len(seed.rates) == 3 * 6
    assert all(r["currency"] == "CAD" and isinstance(r["daily_rate"], Decimal) for r in seed.rates)
    assert all(r["valid_from"] <= ANCHOR <= r["valid_to"] for r in seed.rates)


def test_reservation_dates_follow_anchor(seed):
    rav = _res(seed, "DLRAV4")
    assert rav["pickup_at"] == datetime(2026, 10, 6, 10, tzinfo=TZ)
    assert rav["return_at"] == datetime(2026, 10, 11, 10, tzinfo=TZ)
    assert rav["status"] == "CONFIRMED" and rav["payment_status"] == "AUTHORIZED"

    cor = _res(seed, "DLCOR1")
    assert cor["pickup_at"].date() == date(2026, 10, 3)
    assert cor["return_at"].date() == date(2026, 10, 8)
    assert cor["status"] == "PICKED_UP" and cor["payment_status"] == "PAID"

    bmw = _res(seed, "ELBMW3")
    assert bmw["pickup_at"].date() == date(2026, 10, 4)
    assert bmw["status"] == "PICKED_UP"


def test_reservation_totals_and_insurance(seed):
    assert _res(seed, "DLCOR1")["total_amount"] == Decimal("408.80")
    assert _res(seed, "DLRAV4")["total_amount"] == Decimal("593.60")
    assert _res(seed, "ELBMW3")["total_amount"] == Decimal("851.20")
    plans = {(ri["reservation_id"], ri["plan_code"]): ri["price"] for ri in seed.reservation_insurance}
    assert plans[(_res(seed, "DLCOR1")["id"], "BASIC_CDW")] == Decimal("90.00")
    assert plans[(_res(seed, "DLRAV4")["id"], "PREMIUM_PROTECT")] == Decimal("160.00")
    assert plans[(_res(seed, "ELBMW3")["id"], "PREMIUM_PROTECT")] == Decimal("160.00")


def test_ownership(seed):
    owners = {c["id"]: c["email"] for c in seed.customers}
    assert owners[_res(seed, "DLCOR1")["customer_id"]] == "daniel@example.com"
    assert owners[_res(seed, "DLRAV4")["customer_id"]] == "daniel@example.com"
    assert owners[_res(seed, "ELBMW3")["customer_id"]] == "emily@example.com"
    guest_id = next(c["id"] for c in seed.customers if c["email"] == "guest@example.com")
    assert all(r["customer_id"] != guest_id for r in seed.reservations)


def test_rav4_unit_free_for_extension_window(seed):
    rav = _res(seed, "DLRAV4")
    others = [r for r in seed.reservations if r["vehicle_id"] == rav["vehicle_id"] and r is not rav]
    assert others == []


def test_no_overlapping_reservations_per_vehicle(seed):
    by_vehicle: dict[int, list] = {}
    for r in seed.reservations:
        by_vehicle.setdefault(r["vehicle_id"], []).append((r["pickup_at"], r["return_at"]))
    for spans in by_vehicle.values():
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_ids_are_stable_across_runs(seed):
    again = build_seed(ANCHOR)
    assert [c["id"] for c in again.customers] == [c["id"] for c in seed.customers]
    assert [r["id"] for r in again.reservations] == [r["id"] for r in seed.reservations]


def test_demo_password_hash(seed):
    assert all(bcrypt.checkpw(b"demo123", c["password_hash"].encode()) for c in seed.customers)


# ---------- policies ----------

@pytest.fixture(scope="module")
def policies():
    return load_policies(DEFAULT_POLICIES_DIR)


def test_all_required_policies_present(policies):
    docs, _ = policies
    assert {d.doc_id for d in docs} == {
        "insurance-basic-cdw",
        "insurance-premium-protect",
        "insurance-third-party-liability",
        "accident-procedure",
        "cancellation-and-modification",
        "roadside-assistance",
        "rental-terms",
    }
    assert all(d.category in VALID_CATEGORIES for d in docs)


def test_policy_word_counts(policies):
    docs, _ = policies
    for d in docs:
        assert 300 <= len(d.body.split()) <= 700, d.doc_id


def test_plan_codes(policies):
    docs, chunks = policies
    codes = {d.doc_id: d.plan_codes for d in docs}
    assert codes["insurance-basic-cdw"] == ["BASIC_CDW"]
    assert codes["insurance-premium-protect"] == ["PREMIUM_PROTECT"]
    assert codes["insurance-third-party-liability"] == []
    assert all(c.plan_codes == codes[c.doc_id] for c in chunks)


def test_key_facts_present(policies):
    _, chunks = policies
    text = {doc_id: " ".join(c.content for c in chunks if c.doc_id == doc_id) for doc_id in {c.doc_id for c in chunks}}
    assert "CAD 1,500" in text["insurance-basic-cdw"]
    assert "CAD 0" in text["insurance-premium-protect"]
    assert "24 hours" in text["insurance-premium-protect"]
    assert "1-800-555-0199" in text["accident-procedure"]
    assert "911" in text["accident-procedure"]
    assert "48 hours" in text["cancellation-and-modification"]
    assert "29 minutes" in text["cancellation-and-modification"]
    assert "Child seats" in {c.section for c in chunks if c.doc_id == "rental-terms"}


def test_one_chunk_per_section(policies):
    docs, chunks = policies
    cdw = next(d for d in docs if d.doc_id == "insurance-basic-cdw")
    sections = [c.section for c in chunk_policy(cdw)]
    assert sections == ["Overview", "What is covered", "Deductible", "Exclusions", "Conditions", "Claims assessment"]
    assert all(c.content and not c.content.startswith("#") for c in chunks)


def test_chunking_is_deterministic(policies):
    _, chunks = policies
    _, again = load_policies(DEFAULT_POLICIES_DIR)
    assert again == chunks


def test_long_section_is_split():
    text = "\n\n".join(" ".join(["word"] * 300) for _ in range(3))
    parts = _split_long(text, max_words=600)
    assert len(parts) == 2
    assert all(len(p.split()) <= 600 for p in parts)
    oversized = _split_long(" ".join(["w"] * 1300), max_words=600)
    assert [len(p.split()) for p in oversized] == [600, 600, 100]


def test_missing_front_matter_rejected(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("# No front matter\n\n## Section\ntext", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_policy(bad)
