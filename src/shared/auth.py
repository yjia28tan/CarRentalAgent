"""JWT (HS256) issuing and verification, shared by the agent app and the mock services."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.shared.errors import AppError, ErrorCode
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALGORITHM = "HS256"
_REQUIRED_CLAIMS = ["sub", "exp", "iat", "iss"]

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    customer_id: UUID
    name: str
    token: str


def create_access_token(customer_id: UUID, name: str, *, now: datetime | None = None) -> tuple[str, int]:
    """Return (token, expires_in_seconds)."""
    settings = get_settings()
    issued = now or datetime.now(timezone.utc)
    ttl = settings.jwt_ttl_seconds
    claims = {
        "sub": str(customer_id),
        "name": name,
        "iat": issued,
        "exp": issued + timedelta(seconds=ttl),
        "iss": settings.jwt_issuer,
    }
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=ALGORITHM), ttl


def decode_access_token(token: str) -> Principal:
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[ALGORITHM],
            issuer=settings.jwt_issuer,
            options={"require": _REQUIRED_CLAIMS},
        )
        customer_id = UUID(str(claims["sub"]))
    except (jwt.InvalidTokenError, ValueError) as exc:
        logger.info("auth_failed", reason=type(exc).__name__)
        raise AppError(ErrorCode.UNAUTHENTICATED) from None
    return Principal(customer_id=customer_id, name=str(claims.get("name", "")), token=token)


async def require_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise AppError(ErrorCode.UNAUTHENTICATED)
    return decode_access_token(credentials.credentials)
