"""Fixtures for the search tests: the deterministic embedder and the shared synthetic corpus."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.orm import Session

from app.config import get_settings
from app.search.embeddings import HashingEmbedder

if TYPE_CHECKING:
    from tests.unit.search.corpus import Corpus


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(get_settings().embedding_dim)


@pytest.fixture
def corpus(db_session: Session, embedder: HashingEmbedder) -> Corpus:
    from tests.unit.search.corpus import build_corpus  # imported here: it needs the service under test

    return build_corpus(db_session, embedder)
