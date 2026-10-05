"""Error codes and the uniform error response format shared by the agent app and mock services."""
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.utils.logger import get_logger, get_request_id

logger = get_logger(__name__)


class ErrorCode(StrEnum):
    UNAUTHENTICATED = "UNAUTHENTICATED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    LOCATION_NOT_FOUND = "LOCATION_NOT_FOUND"
    RESERVATION_NOT_FOUND = "RESERVATION_NOT_FOUND"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    INVALID_STATE = "INVALID_STATE"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    QUOTE_EXPIRED = "QUOTE_EXPIRED"
    PREVIEW_EXPIRED = "PREVIEW_EXPIRED"
    ACTION_NOT_FOUND = "ACTION_NOT_FOUND"
    ACTION_EXPIRED = "ACTION_EXPIRED"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # Generic fallback for framework-level HTTP errors (unknown route, wrong method, ...)
    HTTP_ERROR = "HTTP_ERROR"


HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.LOCATION_NOT_FOUND: 404,
    ErrorCode.RESERVATION_NOT_FOUND: 404,
    ErrorCode.NOT_AVAILABLE: 409,
    ErrorCode.INVALID_STATE: 409,
    ErrorCode.VERSION_CONFLICT: 409,
    ErrorCode.QUOTE_EXPIRED: 410,
    ErrorCode.PREVIEW_EXPIRED: 410,
    ErrorCode.ACTION_NOT_FOUND: 404,
    ErrorCode.ACTION_EXPIRED: 410,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.UPSTREAM_TIMEOUT: 504,
    ErrorCode.UPSTREAM_ERROR: 502,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.HTTP_ERROR: 400,
}

DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNAUTHENTICATED: "Authentication required. Please sign in again.",
    ErrorCode.VALIDATION_ERROR: "The request contains invalid input.",
    ErrorCode.LOCATION_NOT_FOUND: "We don't operate at that location.",
    ErrorCode.RESERVATION_NOT_FOUND: "No reservation found for that code.",
    ErrorCode.NOT_AVAILABLE: "No vehicle is available for the requested dates.",
    ErrorCode.INVALID_STATE: "This reservation can't be changed in its current state.",
    ErrorCode.VERSION_CONFLICT: "The reservation changed since the preview. Please request a new preview.",
    ErrorCode.QUOTE_EXPIRED: "This quote has expired. Please request a new one.",
    ErrorCode.PREVIEW_EXPIRED: "This change preview has expired. Please request a new one.",
    ErrorCode.ACTION_NOT_FOUND: "This action was not found.",
    ErrorCode.ACTION_EXPIRED: "This action has expired. Please ask for the change again.",
    ErrorCode.RATE_LIMITED: "Too many requests. Please wait a moment and try again.",
    ErrorCode.UPSTREAM_TIMEOUT: "A backend service took too long to respond. Please try again.",
    ErrorCode.UPSTREAM_ERROR: "A backend service is temporarily unavailable. Please try again.",
    ErrorCode.INTERNAL_ERROR: "Something went wrong on our side. Please try again.",
    ErrorCode.HTTP_ERROR: "The request could not be processed.",
}


class AppError(Exception):
    """Raise anywhere to return a structured error response."""

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        *,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message or DEFAULT_MESSAGES[code]
        self.status_code = status_code or HTTP_STATUS[code]
        self.details = details
        super().__init__(f"{code}: {self.message}")


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


def error_response(
    code: ErrorCode | str,
    message: str | None = None,
    *,
    status_code: int | None = None,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    known = code if isinstance(code, ErrorCode) else ErrorCode.__members__.get(code)
    body = ErrorResponse(
        error=ErrorBody(
            code=str(code),
            message=message or (DEFAULT_MESSAGES[known] if known else "Request failed."),
            request_id=request_id or get_request_id(),
            details=details,
        )
    )
    status = status_code or (HTTP_STATUS[known] if known else 500)
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True), headers=headers)


def _format_validation_errors(exc: RequestValidationError) -> str:
    # Never echo the raw input back: it may contain secrets or be very large.
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path"))
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts) or DEFAULT_MESSAGES[ErrorCode.VALIDATION_ERROR]


_STATUS_TO_CODE = {401: ErrorCode.UNAUTHENTICATED, 422: ErrorCode.VALIDATION_ERROR, 429: ErrorCode.RATE_LIMITED}


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.info("app_error", error_code=str(exc.code), status=exc.status_code)
        return error_response(exc.code, exc.message, status_code=exc.status_code, details=exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        message = _format_validation_errors(exc)
        logger.info("validation_error", error_code=ErrorCode.VALIDATION_ERROR.value)
        return error_response(ErrorCode.VALIDATION_ERROR, message)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_TO_CODE.get(exc.status_code, ErrorCode.HTTP_ERROR)
        message = exc.detail if isinstance(exc.detail, str) else None
        return error_response(code, message, status_code=exc.status_code, headers=getattr(exc, "headers", None))
