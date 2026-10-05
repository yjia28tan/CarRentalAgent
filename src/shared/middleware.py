"""ASGI middleware: request id propagation, access logging, last-resort error handling."""
import re
import time

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.shared.errors import ErrorCode, error_response
from src.utils.logger import get_logger
from src.utils.utils import new_id

logger = get_logger(__name__)

REQUEST_ID_HEADER = "x-request-id"
# Accept upstream ids (e.g. app -> mock_services) only if they are short and log-safe.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class RequestContextMiddleware:
    """
    Pure ASGI middleware (not BaseHTTPMiddleware) so contextvars bound here are visible to
    handlers and so unhandled exceptions still get a JSON body and an X-Request-ID header.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = ""
        for name, value in scope.get("headers", []):
            if name == REQUEST_ID_HEADER.encode():
                incoming = value.decode("latin-1")
                break
        request_id = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else new_id("req")

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        scope.setdefault("state", {})["request_id"] = request_id

        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("unhandled_exception", path=scope.get("path"))
            if response_started:
                raise
            response = error_response(ErrorCode.INTERNAL_ERROR, request_id=request_id)
            await response(scope, receive, send_wrapper)
        finally:
            logger.info(
                "http_request",
                method=scope.get("method"),
                path=scope.get("path"),
                status=status_code,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            structlog.contextvars.clear_contextvars()
