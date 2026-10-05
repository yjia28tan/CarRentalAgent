"""
Car Rental Agent API (customer-facing agent + integration layer).
Run: uvicorn app_main:app --port 8000
"""
import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pymongo import AsyncMongoClient

from src.shared.errors import install_exception_handlers
from src.shared.middleware import RequestContextMiddleware
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger, get_request_id

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

HEALTH_CHECK_TIMEOUT_SECONDS = 2


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # All three clients connect lazily, so the app starts even if a dependency is down;
    # /health reports the actual status.
    app.state.redis = aioredis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
    app.state.mongo = AsyncMongoClient(
        settings.mongo_uri.get_secret_value(), serverSelectionTimeoutMS=2000, tz_aware=True
    )
    app.state.mongo_db = app.state.mongo[settings.mongo_db]
    app.state.http = httpx.AsyncClient(
        base_url=settings.mock_services_url, timeout=settings.tool_timeout_seconds
    )
    logger.info("app_started", env=settings.app_env)
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.mongo.close()
        await app.state.redis.aclose()
        logger.info("app_stopped")


app = FastAPI(
    title="Car Rental Agent API",
    description="Customer-facing AI assistant for car rentals",
    lifespan=lifespan,
    docs_url="/docs" if settings.openapi_docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.openapi_docs else None,
)
install_exception_handlers(app)

# add_middleware prepends, so the last one added is the outermost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Request-ID"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.add_middleware(RequestContextMiddleware)


async def _check(name: str, probe: Awaitable[object]) -> str:
    try:
        async with asyncio.timeout(HEALTH_CHECK_TIMEOUT_SECONDS):
            await probe
        return "ok"
    except Exception as exc:  # health must report, not raise
        logger.warning("health_check_failed", component=name, error=type(exc).__name__)
        return "down"


async def _check_postgres_via_mock_services(http: httpx.AsyncClient) -> None:
    resp = await http.get("/health", headers={"X-Request-ID": get_request_id() or ""})
    resp.raise_for_status()
    if resp.json().get("checks", {}).get("postgres") != "ok":
        raise RuntimeError("postgres not ok")


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    state = request.app.state
    names = ("postgres", "mongo", "redis")
    results = await asyncio.gather(
        _check("postgres", _check_postgres_via_mock_services(state.http)),
        _check("mongo", state.mongo.admin.command("ping")),
        _check("redis", state.redis.ping()),
    )
    checks = dict(zip(names, results))
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "checks": checks},
    )
