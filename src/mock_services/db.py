"""asyncpg pool for the mock enterprise services."""
import json

import asyncpg
from pgvector.asyncpg import register_vector

from src.utils.config import get_settings


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def create_pool(dsn: str | None = None) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=dsn or get_settings().postgres_dsn.get_secret_value(),
        min_size=1,
        max_size=10,
        command_timeout=10,
        init=_init_connection,
    )
