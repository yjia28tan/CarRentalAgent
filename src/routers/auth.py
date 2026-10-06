"""POST /auth/login: verify credentials with the (mock) enterprise auth service, then issue our JWT."""
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from src.clients.business_api import BusinessApiClient, UpstreamError
from src.deps import Services, get_services
from src.shared.auth import create_access_token
from src.shared.errors import AppError, ErrorCode

router = APIRouter(prefix="/auth", tags=["auth"])

INVALID_CREDENTIALS = "Invalid email or password."


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: Annotated[str, Field(min_length=3, max_length=254)]
    password: Annotated[str, Field(min_length=1, max_length=128)]


@router.post("/login")
async def login(body: LoginRequest, request: Request, services: Services = Depends(get_services)):
    api = BusinessApiClient(services.http, token=None, request_id=request.state.request_id)
    try:
        customer = await api.verify_credentials(body.email.strip(), body.password)
    except UpstreamError as exc:
        if exc.code in (ErrorCode.UNAUTHENTICATED, ErrorCode.VALIDATION_ERROR):
            raise AppError(ErrorCode.UNAUTHENTICATED, INVALID_CREDENTIALS) from None
        raise AppError(exc.code, exc.message) from None

    token, expires_in = create_access_token(customer["customer_id"], customer["name"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "customer": {"id": customer["customer_id"], "name": customer["name"]},
    }
