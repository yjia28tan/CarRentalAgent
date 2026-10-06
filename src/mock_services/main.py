"""
Mock enterprise services: business REST APIs and KB search, backed by PostgreSQL + pgvector.
Run: uvicorn src.mock_services.main:app --port 8001
"""
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.mock_services.deps import Embedder
from src.mock_services.repository import Repository
from src.mock_services.routers import auth, availability, kb, quotes, reservations
from src.mock_services.services.previews import PreviewStore
from src.shared.errors import install_exception_handlers
from src.shared.middleware import RequestContextMiddleware
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)


def create_app(repo: Repository | None = None, embedder: Embedder | None = None, dsn: str | None = None) -> FastAPI:
    """
    repo given   -> used as-is (tests; no lifespan needed, works with httpx ASGITransport).
    repo omitted -> a PgRepository is created on startup from `dsn` or POSTGRES_DSN.
    """
    warm_up_embedder = embedder is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_repo = app.state.repo is None
        if owns_repo:
            from src.mock_services.pg_repository import PgRepository

            app.state.repo = await PgRepository.create(dsn)
            logger.info("PostgreSQL connection pool ready")
        else:
            logger.info("Using provided repository", repository=type(app.state.repo).__name__)
        if warm_up_embedder:
            # Load the model at startup so the first KB search doesn't hit the tool timeout.
            await asyncio.to_thread(app.state.embedder, ["warm up"])
            logger.info("Embedding model loaded", model=settings.embedding_model)
        logger.info("Mock services started")
        try:
            yield
        finally:
            if owns_repo:
                await app.state.repo.close()
                app.state.repo = None
                logger.info("PostgreSQL connection pool closed")

    app = FastAPI(
        title="Mock Enterprise Services",
        lifespan=lifespan,
        docs_url="/docs" if settings.openapi_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.openapi_docs else None,
    )
    if embedder is None:
        from src.mock_services.services.embeddings import embed_texts

        embedder = embed_texts
    app.state.repo = repo
    app.state.embedder = embedder
    app.state.previews = PreviewStore()

    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)
    for module in (auth, availability, quotes, reservations, kb):
        app.include_router(module.router)

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        try:
            async with asyncio.timeout(2):
                await request.app.state.repo.ping()
            postgres = "ok"
        except Exception as exc:  # health must report, not raise
            logger.warning("health_check_failed", component="postgres", error=type(exc).__name__)
            postgres = "down"
        ok = postgres == "ok"
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"status": "ok" if ok else "degraded", "checks": {"postgres": postgres}},
        )

    return app


app = create_app()
