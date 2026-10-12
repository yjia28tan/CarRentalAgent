"""
POST /v1/auth/verify-credentials: stands in for the enterprise auth service.
Only the agent's /auth/login calls this; it never issues tokens itself.
"""
import asyncio
from typing import Annotated

import bcrypt
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from src.mock_services.deps import get_repo
from src.mock_services.repository import Repository
from src.shared.errors import AppError, ErrorCode
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"])

INVALID_CREDENTIALS = "Invalid email or password."
# Compared against when the email is unknown so response time doesn't reveal which emails exist.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt()).decode()


class VerifyCredentialsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: Annotated[str, Field(min_length=3, max_length=254)]
    password: Annotated[str, Field(min_length=1, max_length=128)]


def _check_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


@router.post("/verify-credentials")
async def verify_credentials(body: VerifyCredentialsRequest, repo: Repository = Depends(get_repo)):
    customer = await repo.get_customer_by_email(body.email.strip())
    ok = await asyncio.to_thread(_check_password, body.password, customer.password_hash if customer else _DUMMY_HASH)
    if customer is None or not ok:
        logger.info("credentials_rejected")
        raise AppError(ErrorCode.UNAUTHENTICATED, INVALID_CREDENTIALS)
    return {"customer_id": customer.id, "name": customer.full_name, "email": customer.email}
