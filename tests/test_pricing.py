from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from src.mock_services.services.pricing import price_extension, price_rental, rental_days

TZ = ZoneInfo("America/Vancouver")
D = Decimal


def at(day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TZ)


# ---------- rental_days ----------

@pytest.mark.parametrize(
    ("pickup", "ret", "days"),
    [
        (at(10), at(15), 5),                       # exactly 5 x 24h
        (at(10), at(15, 10, 1), 6),                # 1 minute over -> extra day
        (at(10), at(10, 10, 30), 1),               # 30 minutes -> minimum 1
        (at(10), at(11), 1),
        (at(10), at(10) + timedelta(hours=47), 2),
    ],
)
def test_rental_days(pickup, ret, days):
    assert rental_days(pickup, ret) == days


@pytest.mark.parametrize("ret", [at(10), at(9)])
def test_rental_days_rejects_non_positive_range(ret):
    with pytest.raises(ValueError):
        rental_days(at(10), ret)


def test_rental_days_rejects_naive_datetimes():
    with pytest.raises(ValueError):
        rental_days(datetime(2026, 10, 10), datetime(2026, 10, 15))


# ---------- price_rental ----------

def test_rav4_five_days_no_insurance():
    p = price_rental(daily_rate=D("74"), pickup_at=at(10), return_at=at(15), currency="CAD")
    assert p.days == 5
    assert p.base == D("370.00")
    assert p.insurance == ()
    assert p.subtotal == D("370.00")
    assert p.taxes_fees == D("44.40")
    assert p.total == D("414.40")


def test_rav4_five_days_premium_protect():
    p = price_rental(
        daily_rate=D("74.00"), pickup_at=at(10), return_at=at(15),
        insurance_daily_prices={"PREMIUM_PROTECT": D("32.00")}, currency="CAD",
    )
    assert p.base == D("370.00")
    assert p.insurance_total == D("160.00")
    assert p.subtotal == D("530.00")
    assert p.taxes_fees == D("63.60")
    assert p.total == D("593.60")


def test_corolla_five_days_basic_cdw():
    p = price_rental(
        daily_rate=D("55.00"), pickup_at=at(3), return_at=at(8),
        insurance_daily_prices={"BASIC_CDW": D("18.00")}, currency="CAD",
    )
    assert p.subtotal == D("365.00")
    assert p.taxes_fees == D("43.80")
    assert p.total == D("408.80")


def test_multiple_insurance_lines_and_tax_rounding():
    p = price_rental(
        daily_rate=D("45.05"), pickup_at=at(10), return_at=at(11),
        insurance_daily_prices={"PREMIUM_PROTECT": D("32.00"), "BASIC_CDW": D("18.00")}, currency="CAD",
    )
    assert [line.code for line in p.insurance] == ["BASIC_CDW", "PREMIUM_PROTECT"]
    assert p.subtotal == D("95.05")
    assert p.taxes_fees == D("11.41")  # 11.406 -> 11.41
    assert p.total == D("106.46")


def test_all_amounts_are_decimal_with_two_places():
    p = price_rental(daily_rate=D("98"), pickup_at=at(10), return_at=at(13), currency="CAD")
    for value in (p.daily_rate, p.base, p.subtotal, p.taxes_fees, p.total):
        assert isinstance(value, Decimal)
        assert value.as_tuple().exponent == -2


@pytest.mark.parametrize("bad", [74.0, 74, "74"])
def test_non_decimal_rate_rejected(bad):
    with pytest.raises(TypeError):
        price_rental(daily_rate=bad, pickup_at=at(10), return_at=at(15), currency="CAD")


def test_non_decimal_insurance_price_rejected():
    with pytest.raises(TypeError):
        price_rental(
            daily_rate=D("74"), pickup_at=at(10), return_at=at(15),
            insurance_daily_prices={"BASIC_CDW": 18.0}, currency="CAD",
        )


def test_negative_rate_rejected():
    with pytest.raises(ValueError):
        price_rental(daily_rate=D("-1"), pickup_at=at(10), return_at=at(15), currency="CAD")


def test_to_dict_contains_breakdown():
    p = price_rental(
        daily_rate=D("74"), pickup_at=at(10), return_at=at(15),
        insurance_daily_prices={"PREMIUM_PROTECT": D("32")}, currency="CAD",
    )
    d = p.to_dict()
    assert d["total"] == D("593.60")
    assert d["insurance"] == [{"code": "PREMIUM_PROTECT", "daily_price": D("32"), "amount": D("160.00")}]


# ---------- price_extension ----------

def test_extension_reprices_full_reservation():
    # DLRAV4 with anchor 2026-10-05: Oct 6 -> Oct 11, extended to Oct 17.
    ext = price_extension(
        daily_rate=D("74.00"), pickup_at=at(6), old_return_at=at(11), new_return_at=at(17),
        insurance_daily_prices={"PREMIUM_PROTECT": D("32.00")}, currency="CAD",
    )
    assert ext.old.days == 5 and ext.new.days == 11
    assert ext.old.total == D("593.60")
    assert ext.new.subtotal == D("1166.00")
    assert ext.new.taxes_fees == D("139.92")
    assert ext.new.total == D("1305.92")
    assert ext.price_delta == D("712.32")


def test_shortening_gives_negative_delta():
    ext = price_extension(
        daily_rate=D("55.00"), pickup_at=at(6), old_return_at=at(11), new_return_at=at(9),
        currency="CAD",
    )
    assert ext.price_delta == D("-123.20")  # 2 days x 55 x 1.12


def test_extension_rejects_return_before_pickup():
    with pytest.raises(ValueError):
        price_extension(
            daily_rate=D("55.00"), pickup_at=at(6), old_return_at=at(11), new_return_at=at(5),
            currency="CAD",
        )
