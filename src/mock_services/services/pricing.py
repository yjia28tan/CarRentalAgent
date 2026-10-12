"""
Deterministic rental pricing (spec section 10). Decimal throughout; floats are rejected.

days       = ceil(hours / 24), minimum 1
subtotal   = daily_rate * days + sum(insurance daily_price * days)
taxes_fees = round(subtotal * 0.12, 2)
total      = subtotal + taxes_fees
"""
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

TAX_RATE = Decimal("0.12")
_CENT = Decimal("0.01")
_ONE_DAY = timedelta(days=1)


def _money(value: Decimal) -> Decimal:
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def _require_decimal(name: str, value: object) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be Decimal, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def rental_days(pickup_at: datetime, return_at: datetime) -> int:
    if pickup_at.tzinfo is None or return_at.tzinfo is None:
        raise ValueError("pickup_at and return_at must be timezone-aware")
    if return_at <= pickup_at:
        raise ValueError("return_at must be after pickup_at")
    # Ceil division on timedelta keeps this exact (no float hours).
    return max(1, -(-(return_at - pickup_at) // _ONE_DAY))


@dataclass(frozen=True)
class InsuranceLine:
    code: str
    daily_price: Decimal
    amount: Decimal


@dataclass(frozen=True)
class PriceBreakdown:
    days: int
    daily_rate: Decimal
    base: Decimal
    insurance: tuple[InsuranceLine, ...]
    insurance_total: Decimal
    subtotal: Decimal
    taxes_fees: Decimal
    total: Decimal
    currency: str

    def to_dict(self) -> dict:
        return {
            "days": self.days,
            "daily_rate": self.daily_rate,
            "base": self.base,
            "insurance": [
                {"code": i.code, "daily_price": i.daily_price, "amount": i.amount} for i in self.insurance
            ],
            "insurance_total": self.insurance_total,
            "subtotal": self.subtotal,
            "taxes_fees": self.taxes_fees,
            "total": self.total,
            "currency": self.currency,
        }


def price_rental(
    *,
    daily_rate: Decimal,
    pickup_at: datetime,
    return_at: datetime,
    insurance_daily_prices: Mapping[str, Decimal] | None = None,
    currency: str,
) -> PriceBreakdown:
    _require_decimal("daily_rate", daily_rate)
    days = rental_days(pickup_at, return_at)

    base = _money(daily_rate * days)
    lines = tuple(
        InsuranceLine(code=code, daily_price=price, amount=_money(_require_decimal(code, price) * days))
        for code, price in sorted((insurance_daily_prices or {}).items())
    )
    insurance_total = sum((line.amount for line in lines), Decimal("0.00"))
    subtotal = base + insurance_total
    taxes_fees = _money(subtotal * TAX_RATE)

    return PriceBreakdown(
        days=days,
        daily_rate=_money(daily_rate),
        base=base,
        insurance=lines,
        insurance_total=insurance_total,
        subtotal=subtotal,
        taxes_fees=taxes_fees,
        total=subtotal + taxes_fees,
        currency=currency,
    )


@dataclass(frozen=True)
class ExtensionPrice:
    old: PriceBreakdown
    new: PriceBreakdown
    price_delta: Decimal


def price_extension(
    *,
    daily_rate: Decimal,
    pickup_at: datetime,
    old_return_at: datetime,
    new_return_at: datetime,
    insurance_daily_prices: Mapping[str, Decimal] | None = None,
    currency: str,
) -> ExtensionPrice:
    """Reprice the full reservation at the new length and return the delta (may be negative)."""
    common = dict(
        daily_rate=daily_rate,
        pickup_at=pickup_at,
        insurance_daily_prices=insurance_daily_prices,
        currency=currency,
    )
    old = price_rental(return_at=old_return_at, **common)
    new = price_rental(return_at=new_return_at, **common)
    return ExtensionPrice(old=old, new=new, price_delta=new.total - old.total)
