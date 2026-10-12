"""Short-lived change previews. In-process only: a restart simply makes previews expire."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from src.utils.utils import new_id, utcnow

PREVIEW_TTL = timedelta(minutes=10)


@dataclass
class ChangePreview:
    preview_id: str
    reservation_id: UUID
    customer_id: UUID
    confirmation_code: str
    expected_version: int
    vehicle_id: int
    old_return_at: datetime
    new_return_at: datetime
    old_total: Decimal
    new_total: Decimal
    price_delta: Decimal
    currency: str
    insurance_prices: dict[str, Decimal]
    expires_at: datetime


class PreviewStore:
    def __init__(self, ttl: timedelta = PREVIEW_TTL) -> None:
        self.ttl = ttl
        self._items: dict[str, ChangePreview] = {}

    def create(self, **fields: Any) -> ChangePreview:
        now = utcnow()
        self._items = {k: v for k, v in self._items.items() if v.expires_at > now}
        preview = ChangePreview(preview_id=new_id("pv"), expires_at=now + self.ttl, **fields)
        self._items[preview.preview_id] = preview
        return preview

    def get(self, preview_id: str) -> ChangePreview | None:
        preview = self._items.get(preview_id)
        if preview is None or preview.expires_at <= utcnow():
            return None
        return preview

    def discard(self, preview_id: str) -> None:
        self._items.pop(preview_id, None)
