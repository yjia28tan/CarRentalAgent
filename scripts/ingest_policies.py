"""
Chunk, embed and load policies/*.md into kb_documents / kb_chunks. Idempotent: the KB is
replaced in one transaction, so re-running yields the same content.

    python scripts/ingest_policies.py [--policies-dir policies]
"""
import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from pgvector.asyncpg import register_vector  # noqa: E402

from src.utils.config import get_settings  # noqa: E402
from src.utils.logger import configure_logging, get_logger  # noqa: E402

logger = get_logger("ingest_policies")

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_DIR = ROOT_DIR / "policies"
VALID_CATEGORIES = {"INSURANCE", "ACCIDENT", "CANCELLATION", "ROADSIDE", "TERMS"}
REQUIRED_KEYS = ("doc_id", "title", "category", "version", "plan_codes")
# ~800 tokens; MiniLM truncates at 256 word pieces anyway, so smaller chunks embed better.
MAX_CHUNK_WORDS = 600

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)
_SECTION_HEADING = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.M)
_TITLE_HEADING = re.compile(r"^#[ \t]+.*$", re.M)


@dataclass(frozen=True)
class PolicyDoc:
    doc_id: str
    title: str
    category: str
    version: str
    plan_codes: list[str]
    source_uri: str
    body: str


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    section: str
    content: str
    plan_codes: list[str]

    @property
    def embedding_text(self) -> str:
        return f"{self.title} - {self.section}\n{self.content}"


def parse_policy(path: Path) -> PolicyDoc:
    match = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{path.name}: missing YAML front matter")
    meta = yaml.safe_load(match.group(1)) or {}
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{path.name}: front matter missing {missing}")
    if meta["category"] not in VALID_CATEGORIES:
        raise ValueError(f"{path.name}: invalid category {meta['category']!r}")
    return PolicyDoc(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        category=str(meta["category"]),
        version=str(meta["version"]),
        plan_codes=[str(c) for c in meta["plan_codes"] or []],
        source_uri=f"policies/{path.name}",
        body=match.group(2),
    )


def _split_long(text: str, max_words: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        words = len(para.split())
        if current and count + words > max_words:
            parts.append("\n\n".join(current))
            current, count = [], 0
        if words > max_words:
            tokens = para.split()
            parts.extend(" ".join(tokens[i:i + max_words]) for i in range(0, len(tokens), max_words))
            continue
        current.append(para)
        count += words
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_policy(doc: PolicyDoc, max_words: int = MAX_CHUNK_WORDS) -> list[Chunk]:
    """One chunk per '##' section; long sections are split further on paragraph boundaries."""
    pieces = _SECTION_HEADING.split(doc.body)
    preamble = _TITLE_HEADING.sub("", pieces[0]).strip()
    sections = [("Overview", preamble)] if preamble else []
    sections += list(zip(pieces[1::2], pieces[2::2]))

    chunks: list[Chunk] = []
    for heading, content in sections:
        parts = _split_long(content.strip(), max_words)
        for i, part in enumerate(parts):
            section = heading if len(parts) == 1 else f"{heading} (part {i + 1})"
            chunks.append(Chunk(doc.doc_id, doc.title, section, part, doc.plan_codes))
    return chunks


def load_policies(policies_dir: Path) -> tuple[list[PolicyDoc], list[Chunk]]:
    docs = [parse_policy(p) for p in sorted(policies_dir.glob("*.md"))]
    if not docs:
        raise ValueError(f"No policy files found in {policies_dir}")
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate doc_id in policies")
    return docs, [c for d in docs for c in chunk_policy(d)]


async def write_kb(conn: asyncpg.Connection, docs: list[PolicyDoc], chunks: list[Chunk], vectors: np.ndarray) -> None:
    await register_vector(conn)
    async with conn.transaction():
        await conn.execute("DELETE FROM kb_chunks")
        await conn.execute("DELETE FROM kb_documents WHERE doc_id <> ALL($1::text[])", [d.doc_id for d in docs])
        await conn.executemany(
            "INSERT INTO kb_documents (doc_id, title, category, version, source_uri) VALUES ($1,$2,$3,$4,$5) "
            "ON CONFLICT (doc_id) DO UPDATE SET title = EXCLUDED.title, category = EXCLUDED.category, "
            "version = EXCLUDED.version, source_uri = EXCLUDED.source_uri",
            [(d.doc_id, d.title, d.category, d.version, d.source_uri) for d in docs],
        )
        await conn.executemany(
            "INSERT INTO kb_chunks (doc_id, section, content, plan_codes, embedding) VALUES ($1,$2,$3,$4,$5)",
            [(c.doc_id, c.section, c.content, c.plan_codes, v) for c, v in zip(chunks, vectors)],
        )


async def _write(docs: list[PolicyDoc], chunks: list[Chunk], vectors: np.ndarray) -> None:
    conn = await asyncpg.connect(get_settings().postgres_dsn.get_secret_value())
    try:
        await write_kb(conn, docs, chunks, vectors)
    finally:
        await conn.close()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies-dir", type=Path, default=DEFAULT_POLICIES_DIR)
    args = parser.parse_args()

    from src.mock_services.services.embeddings import EMBEDDING_DIM, embed_texts

    docs, chunks = load_policies(args.policies_dir)
    vectors = embed_texts([c.embedding_text for c in chunks])
    if vectors.shape != (len(chunks), EMBEDDING_DIM):
        raise RuntimeError(f"Unexpected embedding shape {vectors.shape}")
    asyncio.run(_write(docs, chunks, vectors))
    logger.info("Policies ingested", documents=len(docs), chunks=len(chunks))


if __name__ == "__main__":
    main()
