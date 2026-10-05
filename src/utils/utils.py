from datetime import datetime, timezone

from ulid import ULID


def new_id(prefix: str) -> str:
    """Prefixed, time-sortable id, e.g. ``req_01J9...``."""
    return f"{prefix}_{ULID()}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
