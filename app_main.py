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
from pymongo.errors import PyMongoError
from redis.exceptions import RedisError

from src.deps import Services
from src.llm.gateway import create_llm_gateway
from src.routers import actions, auth, chat
from src.shared.errors import ErrorCode, error_response, install_exception_handlers
from src.shared.middleware import RequestContextMiddleware
from src.storage.audit import MongoAuditLog
from src.storage.history_store import MongoHistoryStore
from src.storage.session_store import RedisSessionStore
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger, get_request_id

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

HEALTH_CHECK_TIMEOUT_SECONDS = 2


@asynccontextmanager
async def _real_services() -> AsyncIterator[Services]:
    # All clients connect lazily, so the app starts even if a dependency is down; /health reports it.
    redis = aioredis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
    mongo = AsyncMongoClient(settings.mongo_uri.get_secret_value(), serverSelectionTimeoutMS=2000, tz_aware=True)
    db = mongo[settings.mongo_db]
    http = httpx.AsyncClient(base_url=settings.mock_services_url, timeout=settings.tool_timeout_seconds)
    services = Services(
        sessions=RedisSessionStore(redis),
        history=MongoHistoryStore(db, redis),
        audit=MongoAuditLog(db),
        llm=create_llm_gateway(),
        http=http,
        ping_mongo=lambda: mongo.admin.command("ping"),
    )
    try:
        await services.history.ensure_indexes()
        await services.audit.ensure_indexes()
    except PyMongoError as exc:
        logger.warning("mongo_indexes_not_created", error=type(exc).__name__)
    try:
        yield services
    finally:
        await http.aclose()
        await mongo.close()
        await redis.aclose()


def create_app(services: Services | None = None) -> FastAPI:
    """services given -> used as-is (tests). Otherwise Redis/Mongo/httpx clients are built on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
            yield
            return
        async with _real_services() as real:
            app.state.services = real
            logger.info("app_started", env=settings.app_env, llm=type(real.llm).__name__)
            yield
        logger.info("app_stopped")

    app = FastAPI(
        title="Car Rental Agent API",
        description="Customer-facing AI assistant for car rentals",
        lifespan=lifespan,
        docs_url="/docs" if settings.openapi_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.openapi_docs else None,
    )
    if services is not None:
        app.state.services = services
    install_exception_handlers(app)

    @app.exception_handler(RedisError)
    @app.exception_handler(PyMongoError)
    async def _storage_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error("storage_unavailable", error=type(exc).__name__)
        return error_response(ErrorCode.UPSTREAM_ERROR, status_code=503)

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

    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(actions.router)
    app.add_api_route("/health", health, methods=["GET"])
    return app


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


async def health(request: Request) -> JSONResponse:
    services: Services = request.app.state.services
    names = ("postgres", "mongo", "redis")
    results = await asyncio.gather(
        _check("postgres", _check_postgres_via_mock_services(services.http)),
        _check("mongo", services.ping_mongo()),
        _check("redis", services.sessions.ping()),
    )
    checks = dict(zip(names, results))
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "checks": checks},
    )


app = create_app()
