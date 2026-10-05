import os

# Must be set before any src.* import so cached settings never read real secrets.
os.environ["APP_ENV"] = "test"
os.environ["JWT_SECRET"] = "test-secret-not-for-production-0123456789"
os.environ["MOCK_SERVICES_URL"] = "http://mock-services.test"
os.environ["ALLOWED_HOSTS"] = '["*"]'

import asyncio  # noqa: E402
from datetime import date, datetime  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from scripts.ingest_policies import DEFAULT_POLICIES_DIR, load_policies  # noqa: E402
from scripts.seed_data import build_seed  # noqa: E402
from src.mock_services.memory_repository import InMemoryRepository, KbRow  # noqa: E402
from src.shared.auth import create_access_token  # noqa: E402
from tests.helpers import VANCOUVER, fake_embed  # noqa: E402

# Opt-in: tests TRUNCATE the target database, so never point this at real data.
TEST_POSTGRES_DSN = os.environ.get("TEST_POSTGRES_DSN")


@pytest.fixture(scope="session")
def anchor() -> date:
    return datetime.now(VANCOUVER).date()


@pytest.fixture(scope="session")
def seed(anchor):
    return build_seed(anchor)


@pytest.fixture(scope="session")
def kb_rows() -> list[KbRow]:
    docs, chunks = load_policies(DEFAULT_POLICIES_DIR)
    categories = {d.doc_id: d.category for d in docs}
    vectors = fake_embed([c.embedding_text for c in chunks])
    return [
        KbRow(c.doc_id, c.title, categories[c.doc_id], c.section, c.content, c.plan_codes, v)
        for c, v in zip(chunks, vectors)
    ]


@pytest.fixture(scope="session")
def customers(seed) -> dict[str, dict]:
    return {c["email"].split("@")[0]: c for c in seed.customers}


@pytest.fixture(scope="session")
def auth(customers):
    """auth('daniel') -> Authorization header for that demo customer."""

    def _headers(who: str) -> dict[str, str]:
        c = customers[who]
        token, _ = create_access_token(c["id"], c["full_name"])
        return {"Authorization": f"Bearer {token}"}

    return _headers


@pytest.fixture
def memory_repo(seed, kb_rows) -> InMemoryRepository:
    return InMemoryRepository(seed, kb_rows)


def _reset_postgres(seed, kb_rows) -> None:
    import asyncpg

    from scripts.ingest_policies import Chunk, PolicyDoc, write_kb
    from scripts.seed_data import write_seed

    async def run() -> None:
        conn = await asyncpg.connect(TEST_POSTGRES_DSN)
        try:
            await write_seed(conn, seed)
            docs = {
                r.doc_id: PolicyDoc(r.doc_id, r.title, r.category, "test", r.plan_codes, "", "") for r in kb_rows
            }
            chunks = [Chunk(r.doc_id, r.title, r.section, r.content, r.plan_codes) for r in kb_rows]
            await write_kb(conn, list(docs.values()), chunks, np.stack([r.vector for r in kb_rows]))
        finally:
            await conn.close()

    asyncio.run(run())


@pytest.fixture(params=["memory", "postgres"])
def mock_client(request, seed, kb_rows, memory_repo):
    """TestClient for the mock services, against the in-memory repo and (opt-in) real Postgres."""
    from src.mock_services.main import create_app

    if request.param == "memory":
        app = create_app(repo=memory_repo, embedder=fake_embed)
    else:
        if not TEST_POSTGRES_DSN:
            pytest.skip("set TEST_POSTGRES_DSN to run against Postgres")
        _reset_postgres(seed, kb_rows)
        app = create_app(dsn=TEST_POSTGRES_DSN, embedder=fake_embed)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def memory_client(memory_repo):
    """In-memory only: for tests that manipulate repository state directly."""
    from src.mock_services.main import create_app

    app = create_app(repo=memory_repo, embedder=fake_embed)
    with TestClient(app) as client:
        yield client, memory_repo, app
