"""
Logging via structlog, in one of two formats (LOG_FORMAT):

text (default), readable for local work:
    [2026-10-06 15:21:18.479 : INFO]
     | src.routers.chat: Line 140 -> Function send_message()
     | Chat turn completed
     | request_id=req_... customer_id=... tools=2 latency_ms=812.4

json, one JSON object per line for log collectors (spec section 17).

Context bound with ``bind_log_context`` (request_id, session_id, customer_id) is added to every line.
Stdlib loggers (uvicorn, httpx, ...) go through the same formatter.
"""
import logging
import sys
from typing import Any, Literal

import structlog
from structlog.typing import EventDict, WrappedLogger

LogFormat = Literal["text", "json"]

_CALLSITE = structlog.processors.CallsiteParameterAdder(
    {
        structlog.processors.CallsiteParameter.MODULE,
        structlog.processors.CallsiteParameter.FUNC_NAME,
        structlog.processors.CallsiteParameter.LINENO,
    }
)
# Shown first on the context line, in this order; any other fields follow.
_CONTEXT_FIRST = ("request_id", "session_id", "customer_id")


def _ensure_context_keys(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("request_id", None)
    return event_dict


def render_text(_: WrappedLogger, __: str, event_dict: EventDict) -> str:
    ed = dict(event_dict)
    ts = str(ed.pop("ts", ""))[:23]  # trim microseconds to milliseconds
    level = str(ed.pop("level", "info")).upper()
    name = ed.pop("logger", None) or ed.get("module") or "root"
    ed.pop("module", None)
    lineno = ed.pop("lineno", "?")
    func = ed.pop("func_name", "?")
    event = ed.pop("event", "")
    exception = ed.pop("exception", None)
    ed.pop("stack", None)

    context = [f"{k}={ed.pop(k)}" for k in _CONTEXT_FIRST if ed.get(k) is not None]
    context += [f"{k}={v}" for k, v in ed.items() if v is not None and not k.startswith("_")]

    lines = [f"[{ts} : {level}]", f" | {name}: Line {lineno} -> Function {func}()", f" | {event}"]
    if context:
        lines.append(" | " + " ".join(context))
    if exception:
        lines.append(str(exception))
    return "\n".join(lines)


def configure_logging(level: str = "INFO", fmt: LogFormat = "text") -> None:
    timestamper = (
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts")
        if fmt == "json"
        else structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f", utc=False, key="ts")
    )
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.processors.add_log_level,
        timestamper,
        _CALLSITE,
        _ensure_context_keys,
    ]
    renderer = structlog.processors.JSONRenderer(default=str) if fmt == "json" else render_text

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    # RequestContextMiddleware already logs every request ("HTTP request handled").
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
    # httpx logs full URLs (incl. query strings) at INFO; keep it quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_log_context(**values: Any) -> None:
    """Bind request-scoped fields (e.g. session_id, customer_id) to all following log lines."""
    structlog.contextvars.bind_contextvars(**{k: v for k, v in values.items() if v is not None})


def get_request_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("request_id")
