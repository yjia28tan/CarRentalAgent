"""
Car Rental Agent API (customer-facing agent + integration layer).
Run: uvicorn app_main:app --port 8000
"""
import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from urllib.parse import urlparse

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
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)

HEALTH_CHECK_TIMEOUT_SECONDS = 2


def _host(url: str) -> str:
    """host:port only, so credentials in connection URLs never reach the logs."""
    parsed = urlparse(url)
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else str(parsed.hostname)


@asynccontextmanager
async def _real_services() -> AsyncIterator[Services]:
    # All clients connect lazily, so the app starts even if a dependency is down; /health reports it.
    redis_url = settings.redis_url.get_secret_value()
    redis = aioredis.from_url(redis_url, decode_responses=True)
    logger.info("Redis client created", host=_host(redis_url))
    mongo_uri = settings.mongo_uri.get_secret_value()
    mongo = AsyncMongoClient(mongo_uri, serverSelectionTimeoutMS=2000, tz_aware=True)
    db = mongo[settings.mongo_db]
    logger.info("MongoDB client created", host=_host(mongo_uri), database=settings.mongo_db)
    http = httpx.AsyncClient(base_url=settings.mock_services_url, timeout=settings.tool_timeout_seconds)
    logger.info("HTTP client for business services created", base_url=settings.mock_services_url)
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
        logger.info("MongoDB indexes ensured")
    except PyMongoError as exc:
        logger.warning("MongoDB indexes not created; is MongoDB running?", error=type(exc).__name__)
    try:
        yield services
    finally:
        await http.aclose()
        await mongo.close()
        await redis.aclose()
        logger.info("HTTP, MongoDB and Redis clients closed")


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
            logger.info("Agent API started", env=settings.app_env, llm=type(real.llm).__name__,
                        orchestrator=settings.agent_orchestrator)
            yield
        logger.info("Agent API stopped")

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
