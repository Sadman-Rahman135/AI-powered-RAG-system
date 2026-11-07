"""PostgreSQL-only schema extras, shared by the Alembic migration and ``init_pg_schema`` (tests, bootstrap).

``Base.metadata`` describes the portable tables. Search additionally needs things only PostgreSQL has: the
``vector`` and ``pg_trgm`` extensions, a generated full-text column on ``chunks`` and the indexes behind keyword,
semantic and filtered search. They are plain DDL strings, applied around table creation::

    apply_pre(connection)    # extensions, before any table that uses them
    ...create tables...
    apply_post(connection)   # generated column and indexes, after the tables exist

Every statement is idempotent (``IF NOT EXISTS``), so applying the extras twice is harmless, and both helpers
do nothing on other databases (SQLite uses a Python fallback for search).
"""

from __future__ import annotations

from sqlalchemy import Engine, text
from sqlalchemy.engine import Connection

PRE_DDL: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
)

POST_DDL: tuple[str, ...] = (
    # english full-text vector of the chunk text, maintained by PostgreSQL itself
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_tsv tsvector "
    "GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED",
    "CREATE INDEX IF NOT EXISTS ix_chunks_search_tsv ON chunks USING gin (search_tsv)",
    "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)",
    # ILIKE '%substring%' filters on the party names
    "CREATE INDEX IF NOT EXISTS ix_invoices_supplier_name_trgm ON invoices USING gin (supplier_name gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS ix_invoices_customer_name_trgm ON invoices USING gin (customer_name gin_trgm_ops)",
)


def _run(connection: Connection, statements: tuple[str, ...]) -> None:
    if connection.dialect.name != "postgresql":
        return
    for statement in statements:
        connection.execute(text(statement))


def apply_pre(connection: Connection) -> None:
    """Create the ``vector`` and ``pg_trgm`` extensions (needs a role allowed to create them)."""
    _run(connection, PRE_DDL)


def apply_post(connection: Connection) -> None:
    """Add ``chunks.search_tsv`` and the GIN, HNSW and trigram indexes; the tables must already exist."""
    _run(connection, POST_DDL)


def init_pg_schema(engine: Engine) -> None:
    """Create the whole PostgreSQL schema in one transaction: extensions, tables, then the extras."""
    from app.db import models  # noqa: F401 - registers every table on Base.metadata
    from app.db.base import Base

    with engine.begin() as connection:
        apply_pre(connection)
        Base.metadata.create_all(connection)
        apply_post(connection)


__all__ = ["POST_DDL", "PRE_DDL", "apply_post", "apply_pre", "init_pg_schema"]
