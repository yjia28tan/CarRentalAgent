"""
Structured JSON logging via structlog.

Every line is JSON with ``ts``, ``level``, ``event`` and ``request_id``; ``session_id`` and
``customer_id`` are added once bound with ``bind_log_context``. Stdlib loggers (uvicorn,
httpx, ...) are routed through the same renderer.
"""
import logging
import sys
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger


def _ensure_context_keys(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("request_id", None)
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        _ensure_context_keys,
    ]

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
        foreign_pre_chain=[structlog.stdlib.add_logger_name, *shared_processors],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(default=str),
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
    # RequestContextMiddleware already logs every request as "http_request".
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
