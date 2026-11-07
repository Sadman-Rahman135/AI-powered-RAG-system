"""Root test configuration.

Every test runs against an isolated configuration: a fresh SQLite database and storage directory under
``tmp_path``, the inline queue, the deterministic hashing embedder and no LLM. Tests marked ``integration``
use ``TEST_DATABASE_URL`` / ``TEST_REDIS_URL`` (PostgreSQL/Redis from docker compose) and are skipped when unset.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.config import Settings, get_settings, reset_settings_cache
from app.db.session import SessionLocal, init_db, reset_engine

_ISOLATED_ENV = {
    "DOCINTEL_IGNORE_DOTENV": "1",
    "APP_ENV": "test",
    "QUEUE_BACKEND": "inline",
    "EMBEDDING_PROVIDER": "hashing",
    "LLM_PROVIDER": "none",
    "EXTRACTION_MODE": "hybrid",
    "OCR_WORKERS": "0",
    "RATE_LIMIT_ENABLED": "false",
    "API_KEYS": "",
    "LOG_JSON": "false",
}


@pytest.fixture(autouse=True)
def _isolated_settings(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    for key, value in _ISOLATED_ENV.items():
        monkeypatch.setenv(key, value)
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LLM_BASE_URL", "REDIS_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    if request.node.get_closest_marker("integration"):
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("integration test: set TEST_DATABASE_URL (docker compose up db)")
        monkeypatch.setenv("DATABASE_URL", url)
        if os.environ.get("TEST_REDIS_URL"):
            monkeypatch.setenv("REDIS_URL", os.environ["TEST_REDIS_URL"])
    else:
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    reset_settings_cache()
    reset_engine()
    yield
    reset_engine()
    reset_settings_cache()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def db_session() -> Iterator[Session]:
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
