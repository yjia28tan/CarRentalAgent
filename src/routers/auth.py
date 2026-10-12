"""POST /auth/login: verify credentials with the (mock) enterprise auth service, then issue our JWT."""
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from src.clients.business_api import BusinessApiClient, UpstreamError
from src.deps import Services, get_services
from src.rate_limit import enforce_login_rate_limit
from src.shared.auth import create_access_token
from src.shared.errors import AppError, ErrorCode
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

INVALID_CREDENTIALS = "Invalid email or password."


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: Annotated[str, Field(min_length=3, max_length=254)]
    password: Annotated[str, Field(min_length=1, max_length=128)]


@router.post("/login")
async def login(body: LoginRequest, request: Request, services: Services = Depends(get_services)):
    await enforce_login_rate_limit(services, body.email)
    api = BusinessApiClient(services.http, token=None, request_id=request.state.request_id)
    try:
        customer = await api.verify_credentials(body.email.strip(), body.password)
    except UpstreamError as exc:
        if exc.code in (ErrorCode.UNAUTHENTICATED, ErrorCode.VALIDATION_ERROR):
            logger.info("Login rejected: invalid credentials")
            raise AppError(ErrorCode.UNAUTHENTICATED, INVALID_CREDENTIALS) from None
        logger.warning("Login failed: auth service unavailable", error_code=str(exc.code))
        raise AppError(exc.code, exc.message) from None

    token, expires_in = create_access_token(customer["customer_id"], customer["name"])
    logger.info("Login succeeded", customer_id=customer["customer_id"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "customer": {"id": customer["customer_id"], "name": customer["name"]},
    }
