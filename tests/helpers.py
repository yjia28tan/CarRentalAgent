import hashlib
import json
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


def naive(anchor: date, days: int, hour: int = 10) -> str:
    """Local ISO string without offset, the way the LLM is told to pass dates."""
    return local(anchor, days, hour).replace(tzinfo=None).isoformat(timespec="minutes")


def month_day(anchor: date, days: int) -> str:
    """Natural date without a year, e.g. 'October 10' (resolved to the next occurrence by the agent)."""
    d = anchor + timedelta(days=days)
    return f"{d:%B} {d.day}"


def tool_results(messages: list[dict]) -> list[dict]:
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def reply_from(compose):
    """FakeLLM step whose final text is built from this turn's tool results, like a grounded model would."""
    from src.llm.fake import say

    def step(system, messages, tools):
        return say(compose(tool_results(messages)))

    return step


class AgentHarness:
    """Agent API TestClient wired to in-memory stores, a FakeLLM, and the in-process mock services."""

    def __init__(self, client, services, llm, repo) -> None:
        self.client = client
        self.services = services
        self.llm = llm
        self.repo = repo

    def login(self, who: str) -> dict[str, str]:
        resp = self.client.post("/auth/login", json={"email": f"{who}@example.com", "password": "demo123"})
        assert resp.status_code == 200, resp.text
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    def new_session(self, headers: dict[str, str]) -> str:
        resp = self.client.post("/chat/sessions", headers=headers)
        assert resp.status_code == 200, resp.text
        return resp.json()["session_id"]

    def send(self, headers: dict[str, str], sid: str, message: str):
        return self.client.post(f"/chat/sessions/{sid}/messages", json={"message": message}, headers=headers)
