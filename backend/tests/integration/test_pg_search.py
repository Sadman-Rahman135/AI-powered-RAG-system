"""Integration twin of the SQLite search tests: the same scenarios against PostgreSQL + pgvector.

Needs ``TEST_DATABASE_URL`` (a throwaway PostgreSQL with the ``vector`` and ``pg_trgm`` extensions available, e.g.
``postgresql+psycopg://docintel:docintel@localhost:5434/docintel_test`` from the ``pgvector/pgvector`` image);
skipped otherwise. The schema is created with ``init_pg_schema`` and dropped again after every test.

Every portable scenario of ``tests/unit/search/test_service.py`` runs here unchanged, so the SQL fallback and the
PostgreSQL path are held to the same behaviour; the rest of this module covers what only PostgreSQL has.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from sqlalchemy import Engine, insert, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import models
from app.db.base import Base
from app.db.pg_extras import POST_DDL, PRE_DDL, init_pg_schema
from app.db.session import SessionLocal, get_engine
from app.domain.schemas import ReviewStatus
from app.search.embeddings import HashingEmbedder
from app.search.service import SearchFilters, search
from tests.unit.search.corpus import Corpus, build_corpus
from tests.unit.search.test_service import PORTABLE

pytestmark = pytest.mark.integration


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(get_settings().embedding_dim)


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    engine = get_engine()  # DATABASE_URL = TEST_DATABASE_URL, set by the root conftest for integration tests
    Base.metadata.drop_all(engine)  # a clean slate even after an interrupted earlier run
    init_pg_schema(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)


@pytest.fixture
def pg_session(pg_engine: Engine) -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def pg_corpus(pg_session: Session, embedder: HashingEmbedder) -> Corpus:
    return build_corpus(pg_session, embedder)


@pytest.mark.parametrize("scenario", PORTABLE, ids=lambda scenario: scenario.__name__)
def test_scenario_behaves_like_the_sqlite_fallback(
    scenario: Callable[..., None], pg_session: Session, pg_corpus: Corpus, embedder: HashingEmbedder
) -> None:
    scenario(pg_session, pg_corpus, embedder)


# --- the PostgreSQL-only schema extras -----------------------------------------------------------------------


def _scalar(session: Session, sql: str, **params: object) -> object:
    return session.execute(text(sql), params).scalar_one()


def _plan(session: Session, sql: str, **params: object) -> str:
    """EXPLAIN with sequential scans discouraged, so tiny test tables still show which index a query can use."""
    session.execute(text("SET LOCAL enable_seqscan = off"))
    return "\n".join(row[0] for row in session.execute(text("EXPLAIN " + sql), params))


def test_schema_extras_are_idempotent_and_complete(pg_engine: Engine, pg_session: Session) -> None:
    init_pg_schema(pg_engine)  # a second application changes nothing and does not fail
    init_pg_schema(pg_engine)
    assert all("IF NOT EXISTS" in statement for statement in (*PRE_DDL, *POST_DDL))

    extensions = {row[0] for row in pg_session.execute(text("SELECT extname FROM pg_extension"))}
    assert {"vector", "pg_trgm"} <= extensions

    indexes = dict(
        pg_session.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'")).all()
    )
    assert "USING gin (search_tsv)" in indexes["ix_chunks_search_tsv"]
    assert "USING hnsw (embedding vector_cosine_ops)" in indexes["ix_chunks_embedding_hnsw"]
    assert "USING gin (supplier_name gin_trgm_ops)" in indexes["ix_invoices_supplier_name_trgm"]
    assert "USING gin (customer_name gin_trgm_ops)" in indexes["ix_invoices_customer_name_trgm"]

    generated = _scalar(
        pg_session,
        "SELECT is_generated FROM information_schema.columns "
        "WHERE table_name = 'chunks' AND column_name = 'search_tsv'",
    )
    assert generated == "ALWAYS"


def test_generated_tsvector_follows_the_chunk_text(pg_session: Session, pg_corpus: Corpus) -> None:
    sql = "SELECT search_tsv::text FROM chunks WHERE text LIKE :pattern"
    assert "'giraff'" in str(_scalar(pg_session, sql, pattern="%giraffe%"))
    pg_session.execute(text("UPDATE chunks SET text = 'zeppelin hangar' WHERE text LIKE '%giraffe%'"))
    updated = _scalar(pg_session, "SELECT search_tsv::text FROM chunks WHERE text = 'zeppelin hangar'")
    assert "'zeppelin'" in str(updated)
    assert search(pg_session, "giraffe", mode="keyword", embedder=HashingEmbedder()).hits == []


def test_indexes_serve_the_search_queries(pg_session: Session, pg_corpus: Corpus) -> None:
    fulltext = _plan(pg_session, "SELECT id FROM chunks WHERE search_tsv @@ websearch_to_tsquery('english', 'chairs')")
    assert "ix_chunks_search_tsv" in fulltext

    trigram = _plan(pg_session, "SELECT id FROM invoices WHERE supplier_name ILIKE '%acme%'")
    assert "ix_invoices_supplier_name_trgm" in trigram

    vector = "[" + ",".join(["0.1"] * 384) + "]"
    nearest = _plan(pg_session, f"SELECT id FROM chunks ORDER BY embedding <=> CAST('{vector}' AS vector) LIMIT 5")
    assert "ix_chunks_embedding_hnsw" in nearest


def test_stored_embeddings_are_real_pgvector_values(pg_session: Session, pg_corpus: Corpus) -> None:
    assert _scalar(pg_session, "SELECT count(*) FROM chunks WHERE embedding IS NULL") == 0
    assert _scalar(pg_session, "SELECT min(vector_dims(embedding)) FROM chunks") == 384
    assert _scalar(pg_session, "SELECT max(embedding <=> embedding) FROM chunks") == pytest.approx(0.0, abs=1e-6)
    assert _scalar(pg_session, "SELECT max(vector_norm(embedding)) FROM chunks") == pytest.approx(1.0, abs=1e-5)


# --- behaviour that differs from the SQLite fallback by design -----------------------------------------------


def test_keyword_search_stems_words_like_postgres_full_text(
    pg_session: Session, pg_corpus: Corpus, embedder: HashingEmbedder
) -> None:
    for query, expected in (("hosted", "INV-77"), ("waiter dinners", "CAT-0005"), ("croissants", "R-9")):
        page = search(pg_session, query, mode="keyword", embedder=embedder)
        assert [hit.invoice_number for hit in page.hits] == [expected], query
        snippet = page.hits[0].snippet
        assert snippet.count("«") >= 1 and snippet.count("«") == snippet.count("»"), snippet
    (hit,) = search(pg_session, "chair", mode="keyword", embedder=embedder).hits
    assert "«chairs»" in hit.snippet  # the whole matching word is highlighted, not the stem


def test_semantic_search_with_filters_survives_an_hnsw_only_plan(
    pg_session: Session, pg_corpus: Corpus, embedder: HashingEmbedder
) -> None:
    """Force the HNSW index (no sequential scan): post-filtering must not starve the result."""
    pg_session.execute(text("SET LOCAL enable_seqscan = off"))
    filters = SearchFilters(supplier="acme", review_status=ReviewStatus.NEEDS_REVIEW)
    page = search(pg_session, "Acme invoice total", mode="semantic", filters=filters, embedder=embedder)
    assert {hit.invoice_number for hit in page.hits} == {"INV-2024-001", "LED-001"}
    assert _scalar(pg_session, "SELECT current_setting('hnsw.ef_search')") == "1000"
    assert _scalar(pg_session, "SELECT current_setting('hnsw.iterative_scan')") == "relaxed_order"


def test_semantic_search_finds_a_needle_among_thousands(pg_session: Session, embedder: HashingEmbedder) -> None:
    corpus = build_corpus(pg_session, embedder)
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((3000, embedder.dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    needle = np.asarray(embedder.embed_query("needle haystack"), dtype=np.float32)
    vectors[2024] = needle
    for index in range(100, 130):  # a neighbourhood around the needle, as real embeddings have
        nearby = needle + 0.05 * vectors[index]
        vectors[index] = nearby / np.linalg.norm(nearby)
    document_id = corpus.documents["notes"].id
    pg_session.execute(
        insert(models.Chunk),
        [
            {
                "document_id": document_id,
                "page_number": index + 10,
                "chunk_index": index + 10,
                "kind": "page_text",
                "text": f"filler chunk {index}",
                "embedding": vectors[index],
            }
            for index in range(3000)
        ],
    )
    pg_session.commit()

    started = time.perf_counter()
    page = search(pg_session, "needle haystack", mode="semantic", embedder=embedder)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"semantic search took {elapsed:.2f}s"
    assert page.hits[0].filename == "notes.pdf" and page.hits[0].score > 0.95  # the needle or a close neighbour


def test_sql_in_filter_values_and_queries_stays_data(
    pg_session: Session, pg_corpus: Corpus, embedder: HashingEmbedder
) -> None:
    hostile = "x'; DROP TABLE invoices; --"
    filters = SearchFilters(supplier=hostile, customer=hostile)
    page = search(pg_session, hostile, mode="hybrid", filters=filters, embedder=embedder)
    assert page.hits == [] and page.total == 0
    assert _scalar(pg_session, "SELECT count(*) FROM invoices") == 8
    only_filters = SearchFilters(invoice_number=hostile, currency=hostile)
    assert search(pg_session, "", filters=only_filters, embedder=embedder).hits == []
    assert int(str(_scalar(pg_session, "SELECT count(*) FROM chunks"))) > 0
