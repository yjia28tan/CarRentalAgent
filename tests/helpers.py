import hashlib
import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

VANCOUVER = ZoneInfo("America/Vancouver")


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic hashed bag-of-words, L2-normalised, 384-dim. No model download."""
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            out[i, int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % 384] += 1.0
        out[i, 383] += 1e-3
        out[i] /= np.linalg.norm(out[i])
    return out


def local(anchor: date, days: int, hour: int = 10) -> datetime:
    """anchor + days at hour:00 Vancouver time."""
    return datetime.combine(anchor + timedelta(days=days), time(hour), tzinfo=VANCOUVER)
