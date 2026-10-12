"""
Deterministic date resolution for tool arguments ("tomorrow", "next Friday", "October 10", ISO).

Datetimes without an explicit UTC offset are returned naive = local time at the rental location;
the business API localizes them with the location's timezone. Every assumption made (default
10:00, assumed year, interpretation of "next <weekday>") is returned as a note for the LLM to state.
"""
import re
from datetime import date, datetime, time, timedelta

DEFAULT_TIME = time(10, 0)

_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_TIME_WORDS = {
    "noon": time(12), "midday": time(12), "midnight": time(0),
    "morning": time(9), "afternoon": time(14), "evening": time(18), "tonight": time(19),
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_RE = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

_TIME_12H = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?=\s|$)")
_TIME_24H = re.compile(r"\b(?:at\s+)?([01]?\d|2[0-3]):([0-5]\d)\b")
_TIME_WORD = re.compile(r"\b(?:this\s+|in\s+the\s+|at\s+)?(" + "|".join(_TIME_WORDS) + r")\b")


class DateParseError(ValueError):
    pass


def _fmt(d: date) -> str:
    return f"{d:%A}, {d:%B} {d.day}, {d.year}"


def _parse_time(text: str) -> tuple[time | None, str]:
    """Extract a time-of-day from text; return (time or None, text without it)."""
    m = _TIME_12H.search(text)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if not 1 <= hour <= 12 or minute > 59:
            raise DateParseError(f"'{m.group(0).strip()}' is not a valid time.")
        pm = m.group(3).startswith("p")
        hour = hour % 12 + (12 if pm else 0)
        return time(hour, minute), (text[: m.start()] + text[m.end():]).strip()
    m = _TIME_24H.search(text)
    if m:
        return time(int(m.group(1)), int(m.group(2))), (text[: m.start()] + text[m.end():]).strip()
    m = _TIME_WORD.search(text)
    if m:
        word = m.group(1)
        rest = (text[: m.start()] + text[m.end():]).strip()
        if word == "tonight" or m.group(0).startswith("this"):
            rest = rest or "today"
        return _TIME_WORDS[word], rest
    return None, text


def resolve_date(text: str, today: date) -> tuple[date, list[str]]:
    original = text
    s = text.strip().lower()
    try:
        return date.fromisoformat(s), []
    except ValueError:
        pass

    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"[,.]", " ", s)
    s = re.sub(r"\b(on|the|of)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if s in ("", "today", "now"):
        return today, []
    if s == "tomorrow":
        return today + timedelta(days=1), []
    if s == "yesterday":
        return today - timedelta(days=1), []
    if s in ("day after tomorrow", "overmorrow"):
        return today + timedelta(days=2), []
    m = re.fullmatch(r"in (\d{1,3}) days?|(\d{1,3}) days? from (?:now|today)", s)
    if m:
        return today + timedelta(days=int(m.group(1) or m.group(2))), []
    if s in ("in a week", "a week from today", "a week from now"):
        return today + timedelta(days=7), []

    m = re.fullmatch(rf"(this |next |coming )?({_WEEKDAY_RE})", s)
    if m:
        prefix = (m.group(1) or "").strip()
        delta = (_WEEKDAYS[m.group(2)] - today.weekday()) % 7
        if prefix != "this" and delta == 0:
            delta = 7
        resolved = today + timedelta(days=delta)
        notes = [f"Interpreted '{original.strip()}' as {_fmt(resolved)}."] if prefix == "next" else []
        return resolved, notes

    m = re.fullmatch(rf"({_MONTH_RE}) (\d{{1,2}})(?: (\d{{4}}))?", s) or re.fullmatch(
        rf"(\d{{1,2}}) ({_MONTH_RE})(?: (\d{{4}}))?", s
    )
    if m:
        a, b, year = m.groups()
        month, day = (_MONTHS[a], int(b)) if a in _MONTHS else (_MONTHS[b], int(a))
        try:
            if year:
                return date(int(year), month, day), []
            candidate = date(today.year, month, day)
            if candidate < today:
                candidate = date(today.year + 1, month, day)
        except ValueError:
            raise DateParseError(f"'{original.strip()}' is not a valid date.") from None
        return candidate, [f"Assumed '{original.strip()}' means {_fmt(candidate)} (next occurrence)."]

    raise DateParseError(
        f"Couldn't understand the date '{original.strip()}'. Use a date like 2026-10-10 or 'October 10'."
    )


def resolve_datetime(value: str | datetime | date, now: datetime) -> tuple[datetime, list[str]]:
    """`now` is the current aware time in the reference timezone. Returns (datetime, assumption notes)."""
    if isinstance(value, datetime):
        return value, []
    if isinstance(value, date):
        return datetime.combine(value, DEFAULT_TIME), [f"Assumed {DEFAULT_TIME:%H:%M} local time."]
    if not isinstance(value, str) or not value.strip():
        raise DateParseError("A date is required.")

    s = value.strip()
    if "t" in s.lower() or ":" in s:
        try:
            return datetime.fromisoformat(s.upper() if s[:4].isdigit() else s), []
        except ValueError:
            pass

    when, rest = _parse_time(s.lower())
    day, notes = resolve_date(rest, now.date())
    if when is None:
        notes.append(f"Assumed {DEFAULT_TIME:%H:%M} local time.")
        when = DEFAULT_TIME
    return datetime.combine(day, when), notes
