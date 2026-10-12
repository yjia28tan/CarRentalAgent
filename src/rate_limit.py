"""Fixed-window rate limiting in Redis: rl:{subject}:{epoch_minute}, INCR with a 60 s TTL."""
import hashlib
import time

from src.deps import RequestContext, Services
from src.shared.errors import AppError, ErrorCode
from src.utils.logger import get_logger

logger = get_logger(__name__)

LOGIN_ATTEMPTS_PER_MINUTE = 10


def current_epoch_minute() -> int:
    return int(time.time() // 60)


def _retry_after_seconds() -> str:
    return str(60 - int(time.time()) % 60)


async def enforce_chat_rate_limit(services: Services, ctx: RequestContext, session_id: str) -> None:
    limit = services.settings.rate_limit_per_minute
    count = await services.sessions.incr(f"rl:{ctx.customer_id}:{current_epoch_minute()}", 60)
    if count > limit:
        await services.audit.record("RATE_LIMITED", session_id=session_id, customer_id=ctx.customer_id,
                                    outcome="429", count=count, limit=limit)
        raise AppError(ErrorCode.RATE_LIMITED, headers={"Retry-After": _retry_after_seconds()})


async def enforce_login_rate_limit(services: Services, email: str) -> None:
    """Slows down password guessing against one account. The email is hashed so it isn't stored in Redis."""
    subject = hashlib.sha256(email.strip().lower().encode()).hexdigest()[:32]
    count = await services.sessions.incr(f"rl_login:{subject}:{current_epoch_minute()}", 60)
    if count > LOGIN_ATTEMPTS_PER_MINUTE:
        logger.warning("login_rate_limited")
        raise AppError(ErrorCode.RATE_LIMITED, "Too many sign-in attempts. Please wait a minute and try again.",
                       headers={"Retry-After": _retry_after_seconds()})
