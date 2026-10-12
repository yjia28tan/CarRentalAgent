"""
Run the whole backend in one process without Docker, for local UI testing:
agent API + mock services (in-memory repository seeded like seed_data.py, policies embedded with
MiniLM) + in-memory stand-ins for Redis and MongoDB. Data resets on every restart.

    python scripts/dev_server.py [--port 8000] [--anchor-date YYYY-MM-DD]

The LLM is whatever LLM_* in secrets.env configures. Not for production use.
"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import uvicorn

from app_main import create_app as create_agent_app
from scripts.ingest_policies import DEFAULT_POLICIES_DIR, load_policies
from scripts.seed_data import build_seed
from src.deps import Services
from src.llm.gateway import create_llm_gateway
from src.mock_services.main import create_app as create_mock_app
from src.mock_services.memory_repository import InMemoryRepository, KbRow
from src.mock_services.services.embeddings import embed_texts
from src.storage.memory import (
    InMemoryAuditLog,
    InMemoryHistoryStore,
    InMemorySessionStore,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("dev_server")


async def _no_mongo() -> None:
    return None


def build_app(anchor: date):
    seed = build_seed(anchor)
    docs, chunks = load_policies(DEFAULT_POLICIES_DIR)
    categories = {d.doc_id: d.category for d in docs}
    vectors = embed_texts([c.embedding_text for c in chunks])
    kb = [KbRow(c.doc_id, c.title, categories[c.doc_id], c.section, c.content, c.plan_codes, v)
          for c, v in zip(chunks, vectors)]

    mock_app = create_mock_app(repo=InMemoryRepository(seed, kb), embedder=embed_texts)
    services = Services(
        sessions=InMemorySessionStore(),
        history=InMemoryHistoryStore(),
        audit=InMemoryAuditLog(),
        llm=create_llm_gateway(),
        http=httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app), base_url="http://mock-services.local"),
        ping_mongo=_no_mongo,
    )
    logger.info("Dev server ready", anchor_date=anchor.isoformat(), kb_chunks=len(kb),
                llm=type(services.llm).__name__, orchestrator=get_settings().agent_orchestrator)
    return create_agent_app(services)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--anchor-date", type=date.fromisoformat,
                        default=datetime.now(ZoneInfo(settings.default_timezone)).date())
    args = parser.parse_args()
    uvicorn.run(build_app(args.anchor_date), host=args.host, port=args.port, access_log=False, log_config=None)


if __name__ == "__main__":
    main()
