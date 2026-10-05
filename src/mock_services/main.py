"""
Mock enterprise services: business REST APIs and KB search, backed by PostgreSQL + pgvector.
Run: uvicorn src.mock_services.main:app --port 8001
"""
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.mock_services.db import create_pool
from src.shared.errors import install_exception_handlers
from src.shared.middleware import RequestContextMiddleware
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pool = await create_pool()
    logger.info("postgres_pool_ready")
    try:
        yield
    finally:
        await app.state.pool.close()
        logger.info("postgres_pool_closed")


app = FastAPI(
    title="Mock Enterprise Services",
    lifespan=lifespan,
    docs_url="/docs" if settings.openapi_docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.openapi_docs else None,
)
install_exception_handlers(app)
app.add_middleware(RequestContextMiddleware)


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    try:
        async with asyncio.timeout(2):
            await request.app.state.pool.fetchval("SELECT 1")
        postgres = "ok"
    except Exception as exc:  # health must report, not raise
        logger.warning("health_check_failed", component="postgres", error=type(exc).__name__)
        postgres = "down"
    ok = postgres == "ok"
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "checks": {"postgres": postgres}},
    )
