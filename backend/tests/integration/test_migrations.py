"""The Alembic migration builds the full PostgreSQL schema and matches the models exactly."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from alembic import command
from app.config import reset_settings_cache
from app.db.models import ALL_TABLES
from app.db.session import reset_engine

pytestmark = pytest.mark.integration

BACKEND_DIR = Path(__file__).resolve().parents[2]
PG_ONLY_INDEXES = {
    "ix_chunks_search_tsv",
    "ix_chunks_embedding_hnsw",
    "ix_invoices_supplier_name_trgm",
    "ix_invoices_customer_name_trgm",
}


@pytest.fixture
def fresh_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A throw-away database on the integration server; DATABASE_URL points at it for the test."""
    import os

    base_url = make_url(os.environ["TEST_DATABASE_URL"])
    name = f"docintel_migr_{uuid.uuid4().hex[:8]}"
    admin = create_engine(base_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = base_url.set(database=name).render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", url)
    reset_settings_cache()
    reset_engine()
    try:
        yield url
    finally:
        reset_engine()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return cfg


def test_upgrade_creates_schema_and_extras(fresh_database: str) -> None:
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    engine = create_engine(fresh_database)
    try:
        insp = inspect(engine)
        assert set(ALL_TABLES) <= set(insp.get_table_names())
        chunk_columns = {c["name"] for c in insp.get_columns("chunks")}
        assert {"embedding", "search_tsv"} <= chunk_columns
        with engine.connect() as conn:
            indexes = set(conn.scalars(text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")))
            extensions = set(conn.scalars(text("SELECT extname FROM pg_extension")))
        assert indexes >= PG_ONLY_INDEXES
        assert {"vector", "pg_trgm"} <= extensions
    finally:
        engine.dispose()


def test_models_and_migration_do_not_drift(fresh_database: str) -> None:
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.check(cfg)  # raises if autogenerate would emit any operation


def test_downgrade_then_upgrade_round_trip(fresh_database: str) -> None:
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    engine = create_engine(fresh_database)
    try:
        remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
        assert remaining.isdisjoint(ALL_TABLES)
    finally:
        engine.dispose()
    command.upgrade(cfg, "head")
