"""Indexing and search over document chunks: keyword, semantic and hybrid, with structured filters.

``index_document`` embeds a document's chunk drafts and replaces its stored chunks. ``search`` ranks chunks and
reports one hit per ``(document, invoice)`` group, showing the group's best chunk:

* **keyword** - PostgreSQL full text (``websearch_to_tsquery`` + ``ts_rank_cd`` + ``ts_headline``); on SQLite a
  ``LIKE`` fallback with the same query syntax (words are AND-ed, ``"phrases"``, ``-exclusion``, ``or``).
* **semantic** - pgvector cosine distance (HNSW); on SQLite exact cosine with numpy over the filtered chunks.
* **hybrid** - reciprocal rank fusion (k = 60) of the two group rankings.
* **no query text** - structured filtering over invoices only, newest first.

A page-text chunk belongs to the invoice whose page range covers its page; see ``index_document``. Chunks that
cannot be attributed to a single invoice (a spreadsheet sheet listing many rows, a document without records)
are document-level hits: they carry no invoice fields and are dropped by invoice filters.

Scores depend on the mode and are only comparable within one result list: ``keyword`` and ``semantic`` give the
best chunk's relevance (0..1), ``hybrid`` the raw RRF sum, filter-only results 0.

PostgreSQL SQL is written with ``text()`` and bound parameters only: user input never becomes SQL text.
"""

from __future__ import annotations

import heapq
import logging
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal, Protocol
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import (
    BigInteger,
    Date,
    Float,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    bindparam,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import BindParameter

from app.db.models import Chunk, Document, Invoice
from app.db.types import EmbeddingVector
from app.domain.schemas import Money, ReviewStatus
from app.extraction.normalize import normalize_invoice_number
from app.ingestion.types import FileKind
from app.search.chunking import ChunkDraft, invoice_summary
from app.search.embeddings import Embedder

logger = logging.getLogger(__name__)

SearchMode = Literal["keyword", "semantic", "hybrid"]
MODES: tuple[str, ...] = ("keyword", "semantic", "hybrid")
RRF_K = 60  # reciprocal rank fusion constant
CANDIDATE_POOL = 1000  # chunks considered per ranking; also the largest ``total`` a ranked search reports
SNIPPET_MAX_CHARS = 300
MAX_QUERY_CHARS = 1000
START_MARK, STOP_MARK = "«", "»"

_EMBED_BATCH = 128
_INSERT_BATCH = 500
_SNIPPET_LEAD_CHARS = 50
_SNIPPET_WINDOW_CHARS = 240
_HEADLINE_OPTIONS = (
    f"StartSel={START_MARK}, StopSel={STOP_MARK}, MaxWords=35, MinWords=15, ShortWord=3, "
    "HighlightAll=FALSE, MaxFragments=0"
)
_HNSW_MAX_EF_SEARCH = 1000  # pgvector's ceiling for hnsw.ef_search

Group = tuple[UUID, UUID | None]  # (document_id, invoice_id)


# --- public models -------------------------------------------------------------------------------------------


class SearchFilters(BaseModel):
    """Structured filters; every field is optional and they combine with AND.

    ``supplier``/``customer`` are case-insensitive substrings, ``invoice_number`` a substring of the printed
    number or equal to its normalised form, ``currency`` an exact ISO code, dates and totals inclusive ranges.
    Blank strings count as unset.
    """

    model_config = ConfigDict(extra="forbid")

    supplier: str | None = None
    customer: str | None = None
    invoice_number: str | None = None
    currency: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    total_min: Decimal | None = None
    total_max: Decimal | None = None
    review_status: ReviewStatus | None = None
    file_kind: FileKind | None = None
    document_id: UUID | None = None

    @field_validator("supplier", "customer", "invoice_number", "currency", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("currency")
    @classmethod
    def _currency_upper(cls, value: str | None) -> str | None:
        return value.upper() if value else value

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, name) is None for name in type(self).model_fields)


class SearchHit(BaseModel):
    """One result: a document, or one invoice of it, with the best matching chunk's snippet.

    The invoice fields are ``None`` for document-level hits. ``snippet`` is at most 300 characters; matches are
    wrapped in ``«`` and ``»``. ``match_type`` says which ranking found it (``filter`` = structured filtering).
    ``page_number`` is the matching chunk's page, or the invoice's first page for a summary chunk.
    """

    document_id: UUID
    invoice_id: UUID | None
    page_number: int | None
    score: float
    snippet: str
    match_type: Literal["keyword", "semantic", "hybrid", "filter"]
    filename: str
    invoice_number: str | None
    supplier_name: str | None
    customer_name: str | None
    invoice_date: date | None
    total: Money | None
    currency: str | None
    review_status: ReviewStatus | None


class SearchPage(BaseModel):
    """``total`` counts all results before ``limit``/``offset`` (ranked searches: at most ``CANDIDATE_POOL``)."""

    hits: list[SearchHit]
    total: int
    took_ms: int
    mode: SearchMode


# --- indexing ------------------------------------------------------------------------------------------------


def _column_dim() -> int:
    column_type = Chunk.__table__.c.embedding.type
    assert isinstance(column_type, EmbeddingVector)
    return column_type.dim


def _check_dimension(embedder: Embedder) -> None:
    expected = _column_dim()
    if embedder.dim != expected:
        raise ValueError(
            f"embedder {embedder.name!r} produces {embedder.dim}-dimensional vectors but chunks.embedding stores "
            f"{expected} dimensions (EMBEDDING_DIM)"
        )


def _page_owner(spans: Sequence[tuple[UUID, int | None, int | None]], page_number: int | None) -> UUID | None:
    """The one invoice a page belongs to: the document's only invoice, else the single one whose page range
    covers the page. Anything ambiguous (overlapping ranges, a whole-sheet chunk, no range) stays unowned."""
    if page_number is None:
        return None
    if len(spans) == 1:
        return spans[0][0]
    covering = [
        invoice_id
        for invoice_id, first, last in spans
        if first is not None and first <= page_number <= (first if last is None else last)
    ]
    return covering[0] if len(covering) == 1 else None


def index_document(session: Session, document_id: UUID, drafts: list[ChunkDraft], embedder: Embedder) -> int:
    """Embed ``drafts`` and make them the document's chunks, replacing whatever was stored. Returns their count.

    Page-text drafts are attributed to an invoice of the document here, from the stored invoices' page ranges
    (summary drafts already name theirs). Runs inside the caller's transaction and does not commit; pending
    invoices are flushed first so their page ranges are visible. Text without any embeddable word is stored
    without an embedding (it is still found by keyword).
    """
    _check_dimension(embedder)
    session.flush()
    texts = [draft.text for draft in drafts]
    vectors = np.empty((len(texts), embedder.dim), dtype=np.float32)
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = embedder.embed_documents(texts[start : start + _EMBED_BATCH])
        vectors[start : start + len(batch)] = np.asarray(batch, dtype=np.float32)
    has_embedding = np.linalg.norm(vectors, axis=1) > 0

    spans = [
        (row.id, row.page_start, row.page_end)
        for row in session.execute(
            select(Invoice.id, Invoice.page_start, Invoice.page_end).where(Invoice.document_id == document_id)
        )
    ]
    rows = [
        {
            "document_id": document_id,
            "invoice_id": draft.invoice_id or _page_owner(spans, draft.page_number),
            "page_number": draft.page_number,
            "chunk_index": draft.chunk_index,
            "kind": draft.kind,
            "text": draft.text,
            "embedding": vectors[position] if has_embedding[position] else None,
        }
        for position, draft in enumerate(drafts)
    ]
    session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    for start in range(0, len(rows), _INSERT_BATCH):
        session.execute(insert(Chunk), rows[start : start + _INSERT_BATCH])
    return len(rows)


# --- filters -> SQL fragments --------------------------------------------------------------------------------


@dataclass(slots=True)
class _Where:
    """Filter conditions as static SQL fragments (aliases ``c`` chunks, ``i`` invoices, ``d`` documents) plus the
    typed values they bind. Fragments never contain user input; values only travel as bound parameters."""

    clauses: list[str] = field(default_factory=list)
    params: list[tuple[str, Any, Any]] = field(default_factory=list)
    needs_invoice: bool = False
    needs_document: bool = False

    def binds(self) -> list[BindParameter[Any]]:
        return [bindparam(name, value, type_=kind) for name, value, kind in self.params]

    def and_sql(self) -> str:
        return "".join(f" AND {clause}" for clause in self.clauses)

    def add(
        self, clause: str, name: str, value: Any, kind: Any, *, joins: Literal["invoice", "document"] | None
    ) -> None:
        """Add a condition on ``joins`` (the table it reads, so queries on chunks know what to join)."""
        self.clauses.append(clause)
        self.params.append((name, value, kind))
        self.needs_invoice = self.needs_invoice or joins == "invoice"
        self.needs_document = self.needs_document or joins == "document"


def _like_pattern(value: str) -> str:
    """Substring pattern for ``LIKE ... ESCAPE '\\'`` in which ``%``, ``_`` and ``\\`` in the value are literal."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _contains(column: str, name: str, dialect: str) -> str:
    if dialect == "postgresql":
        return f"{column} ILIKE :{name} ESCAPE '\\'"
    return f"lower({column}) LIKE lower(:{name}) ESCAPE '\\'"  # SQLite: case-insensitive for ASCII


def _build_where(filters: SearchFilters, dialect: str, document_column: str) -> _Where:
    """Conditions for ``filters``; ``document_column`` is where the document id lives in the query's base table."""
    where = _Where()
    if filters.supplier:
        clause = _contains("i.supplier_name", "f_supplier", dialect)
        where.add(clause, "f_supplier", _like_pattern(filters.supplier), String, joins="invoice")
    if filters.customer:
        clause = _contains("i.customer_name", "f_customer", dialect)
        where.add(clause, "f_customer", _like_pattern(filters.customer), String, joins="invoice")
    if filters.invoice_number:
        clause = _contains("i.invoice_number", "f_number", dialect)
        normalised = normalize_invoice_number(filters.invoice_number)
        if normalised:
            clause = f"({clause} OR i.invoice_number_norm = :f_number_norm)"
            where.params.append(("f_number_norm", normalised, String))
        where.add(clause, "f_number", _like_pattern(filters.invoice_number), String, joins="invoice")
    if filters.currency:
        where.add("i.currency = :f_currency", "f_currency", filters.currency, String, joins="invoice")
    if filters.date_from is not None:
        where.add("i.invoice_date >= :f_date_from", "f_date_from", filters.date_from, Date, joins="invoice")
    if filters.date_to is not None:
        where.add("i.invoice_date <= :f_date_to", "f_date_to", filters.date_to, Date, joins="invoice")
    if filters.total_min is not None:
        where.add("i.total >= :f_total_min", "f_total_min", filters.total_min, Numeric(18, 4), joins="invoice")
    if filters.total_max is not None:
        where.add("i.total <= :f_total_max", "f_total_max", filters.total_max, Numeric(18, 4), joins="invoice")
    if filters.review_status is not None:
        status = filters.review_status.value
        where.add("i.review_status = :f_review_status", "f_review_status", status, String, joins="invoice")
    if filters.file_kind is not None:
        kind = filters.file_kind.value
        where.add("d.file_kind = :f_file_kind", "f_file_kind", kind, String, joins="document")
    if filters.document_id is not None:
        clause = f"{document_column} = :f_document_id"
        where.add(clause, "f_document_id", filters.document_id, Uuid, joins=None)
    return where


def _chunk_source(where: _Where) -> str:
    source = "chunks AS c"
    if where.needs_invoice:
        source += " JOIN invoices AS i ON i.id = c.invoice_id"
    if where.needs_document:
        source += " JOIN documents AS d ON d.id = c.document_id"
    return source


def _chunk_query(columns: str, where: _Where, condition: str, *, tail: str = "", join: str = "") -> str:
    """``SELECT columns FROM chunks [JOIN ...] WHERE condition AND <filters> tail``. Every argument is a literal
    written in this module and the filter fragments come from ``_Where``: no user input is ever interpolated;
    user values are bound parameters."""
    return f"SELECT {columns} FROM {_chunk_source(where)}{join} WHERE {condition}{where.and_sql()}{tail}"  # noqa: S608


# --- ranking helpers -----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Candidate:
    chunk_id: int
    document_id: UUID
    invoice_id: UUID | None
    page_number: int | None
    score: float

    @property
    def group(self) -> Group:
        return self.document_id, self.invoice_id


@dataclass(frozen=True, slots=True)
class _Ranked:
    chunk: _Candidate  # the chunk whose snippet represents the group
    score: float
    match_type: Literal["keyword", "semantic", "hybrid"]


def _best_per_group(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    """One candidate per group - its best chunk - in rank order."""
    best: dict[Group, _Candidate] = {}
    for candidate in sorted(candidates, key=lambda c: (-c.score, c.chunk_id)):
        best.setdefault(candidate.group, candidate)
    return list(best.values())


def _fuse(mode: str, keyword: list[_Candidate], semantic: list[_Candidate]) -> list[_Ranked]:
    """Order groups for ``mode``. Hybrid is reciprocal rank fusion: a group scores the sum of ``1 / (60 + rank)``
    over the rankings it appears in."""
    if mode == "keyword":
        return [_Ranked(c, c.score, "keyword") for c in keyword]
    if mode == "semantic":
        return [_Ranked(c, c.score, "semantic") for c in semantic]
    rankings: list[tuple[Literal["keyword", "semantic"], list[_Candidate]]] = [
        ("keyword", keyword),
        ("semantic", semantic),
    ]
    scores: dict[Group, float] = {}
    shown: dict[Group, _Candidate] = {}
    found_by: dict[Group, list[Literal["keyword", "semantic"]]] = {}
    for kind, ranking in rankings:
        for rank, candidate in enumerate(ranking, start=1):
            group = candidate.group
            scores[group] = scores.get(group, 0.0) + 1.0 / (RRF_K + rank)
            shown.setdefault(group, candidate)  # the keyword chunk wins: it carries the highlighted match
            found_by.setdefault(group, []).append(kind)
    fused = [
        _Ranked(shown[group], score, "hybrid" if len(found_by[group]) == 2 else found_by[group][0])
        for group, score in scores.items()
    ]
    fused.sort(key=lambda ranked: -ranked.score)  # stable: ties keep keyword-list order first
    return fused


# --- query parsing and snippets (SQLite fallback) ------------------------------------------------------------

_TOKEN = re.compile(r'"([^"]*)"|(\S+)')
_EDGE_PUNCTUATION = ".,;:!?()[]{}<>'\"`“”‘’«»"  # PostgreSQL's parser ignores these around words too


@dataclass(slots=True)
class _Alternative:
    """One AND-group of a query: every ``include`` term present, no ``exclude`` term present."""

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


def _parse_query(query: str) -> list[_Alternative]:
    """websearch-style syntax: words AND-ed, ``"quoted phrases"``, ``-excluded``, and ``or`` between groups."""
    alternatives = [_Alternative()]
    for match in _TOKEN.finditer(query):
        phrase, word = match.groups()
        if phrase is not None:
            term = " ".join(phrase.lower().split())
            if term:
                alternatives[-1].include.append(term)
        elif word.lower() == "or":
            alternatives.append(_Alternative())
        elif word.startswith("-") and len(word) > 1:
            term = word[1:].strip(_EDGE_PUNCTUATION).lower()
            if term:
                alternatives[-1].exclude.append(term)
        else:
            term = word.strip(_EDGE_PUNCTUATION).lower()
            if term:
                alternatives[-1].include.append(term)
    return [alternative for alternative in alternatives if alternative.include]


def _keyword_score(body: str, alternatives: Sequence[_Alternative]) -> float:
    """Relevance in (0, 1) of ``body`` for its best matching alternative; 0 when none matches. Term frequency
    with a mild length penalty and a bonus for the terms appearing together, like ``ts_rank_cd`` does."""
    lowered = body.lower()
    best = 0.0
    for alternative in alternatives:
        if not all(term in lowered for term in alternative.include) or any(t in lowered for t in alternative.exclude):
            continue
        raw = sum(1.0 + math.log(lowered.count(term)) for term in alternative.include)
        raw /= 1.0 + math.log(max(len(lowered), 1))
        if len(alternative.include) > 1 and " ".join(alternative.include) in lowered:
            raw *= 1.5
        best = max(best, raw / (raw + 1.0))
    return best


def _flatten(body: str) -> str:
    """One line of text; characters that look like highlight markers are neutralised first."""
    return " ".join(body.replace(START_MARK, '"').replace(STOP_MARK, '"').split())


def _clip(snippet: str, limit: int = SNIPPET_MAX_CHARS) -> str:
    """At most ``limit`` characters, cut at a word boundary, never leaving a highlight open."""
    if len(snippet) <= limit:
        return snippet
    cut = snippet[: limit - 2]  # room for a closing mark and the ellipsis
    boundary = cut.rfind(" ")
    if boundary > limit // 2:
        cut = cut[:boundary]
    if cut.count(START_MARK) > cut.count(STOP_MARK):
        cut += STOP_MARK
    return cut + "…"


def _snippet(body: str, terms: Sequence[str]) -> str:
    """A window of ``body`` around its first match with every matching word wrapped in ``«»``; the start of the
    text when nothing matches (a semantic-only hit)."""
    flat = _flatten(body)
    if not terms:
        return _clip(flat)
    alternatives = "|".join(re.escape(term) for term in sorted(set(terms), key=len, reverse=True))
    pattern = re.compile(rf"\w*(?:{alternatives})\w*", re.IGNORECASE)  # highlight whole words, as ts_headline does
    first = pattern.search(flat)
    if first is None:
        return _clip(flat)
    start = max(0, first.start() - _SNIPPET_LEAD_CHARS)
    if start:
        space = flat.find(" ", start, first.start())
        start = space + 1 if space != -1 else start
    end = min(len(flat), start + _SNIPPET_WINDOW_CHARS)
    if end < len(flat):
        space = flat.rfind(" ", first.end(), end)
        end = space if space != -1 else end
    marked = pattern.sub(lambda m: f"{START_MARK}{m.group(0)}{STOP_MARK}", flat[start:end])
    return _clip(("…" if start else "") + marked + ("…" if end < len(flat) else ""))


# --- backends ------------------------------------------------------------------------------------------------


class _Backend(Protocol):
    def keyword(self, query: str, where: _Where, pool: int) -> list[_Candidate]: ...

    def semantic(self, vector: np.ndarray, where: _Where, pool: int) -> list[_Candidate]: ...

    def snippets(self, chunk_ids: Sequence[int], query: str) -> dict[int, str]: ...


_CANDIDATE_TYPES = {"chunk_id": BigInteger, "document_id": Uuid, "invoice_id": Uuid, "page_number": Integer}


class _PostgresBackend:
    """Full-text and vector search with PostgreSQL-only SQL (needs ``chunks.search_tsv`` and the HNSW index,
    see ``app.db.pg_extras``)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def keyword(self, query: str, where: _Where, pool: int) -> list[_Candidate]:
        sql = _chunk_query(
            # 33 = divide the rank by 1 + log(length), then rank / (rank + 1), so scores land in [0, 1)
            "c.id AS chunk_id, c.document_id, c.invoice_id, c.page_number, ts_rank_cd(c.search_tsv, tsq, 33) AS score",
            where,
            "c.search_tsv @@ tsq",
            join=" CROSS JOIN websearch_to_tsquery('english', :query) AS tsq",
            tail=" ORDER BY score DESC, c.id LIMIT :pool",
        )
        statement = (
            text(sql)
            .bindparams(bindparam("query", query), bindparam("pool", pool, type_=Integer), *where.binds())
            .columns(score=Float, **_CANDIDATE_TYPES)
        )
        return [
            _Candidate(row.chunk_id, row.document_id, row.invoice_id, row.page_number, float(row.score))
            for row in self.session.execute(statement)
            if row.score > 0  # a rank of 0 means only negated terms "matched"
        ]

    def semantic(self, vector: np.ndarray, where: _Where, pool: int) -> list[_Candidate]:
        self._tune_hnsw(pool)
        literal = "[" + ",".join(f"{x:.9g}" for x in vector.tolist()) + "]"  # pgvector's text input format
        sql = _chunk_query(
            "c.id AS chunk_id, c.document_id, c.invoice_id, c.page_number, "
            "1 - (c.embedding <=> CAST(:qvec AS vector)) AS score",
            where,
            "c.embedding IS NOT NULL",
            tail=" ORDER BY c.embedding <=> CAST(:qvec AS vector) LIMIT :pool",
        )
        statement = (
            text(sql)
            .bindparams(bindparam("qvec", literal), bindparam("pool", pool, type_=Integer), *where.binds())
            .columns(score=Float, **_CANDIDATE_TYPES)
        )
        return [
            _Candidate(row.chunk_id, row.document_id, row.invoice_id, row.page_number, float(row.score))
            for row in self.session.execute(statement)
            if row.score is not None and row.score > 0  # cosine <= 0 is unrelated; NaN compares False and drops
        ]

    def _tune_hnsw(self, pool: int) -> None:
        """Let the HNSW scan return ``pool`` rows and keep scanning when filters discard candidates (both
        settings last for this transaction only). Iterative scans need pgvector 0.8; older builds skip them."""
        self.session.execute(
            text("SELECT set_config('hnsw.ef_search', :ef, true)"), {"ef": str(min(pool, _HNSW_MAX_EF_SEARCH))}
        )
        try:
            with self.session.begin_nested():
                self.session.execute(text("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"))
        except DBAPIError:
            logger.debug("pgvector without iterative index scans: filtered semantic search may return fewer rows")

    def snippets(self, chunk_ids: Sequence[int], query: str) -> dict[int, str]:
        statement = (
            text(
                "SELECT c.id AS chunk_id, ts_headline('english', translate(c.text, :marks, :plain), "
                "websearch_to_tsquery('english', :query), :options) AS snippet "
                "FROM chunks AS c WHERE c.id = ANY(:ids)"
            )
            .bindparams(bindparam("ids", list(chunk_ids), type_=ARRAY(BigInteger)))
            .columns(chunk_id=BigInteger, snippet=Text)
        )
        rows = self.session.execute(
            statement,
            {
                "marks": START_MARK + STOP_MARK,  # a literal « or » in the text must not look like a highlight
                "plain": '""',
                "query": query,
                "options": _HEADLINE_OPTIONS,
            },
        )
        return {row.chunk_id: _clip(" ".join(row.snippet.split())) for row in rows}


class _SqliteBackend:
    """Fallback for tests and local development: ``LIKE`` matching and exact cosine similarity in numpy."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def keyword(self, query: str, where: _Where, pool: int) -> list[_Candidate]:
        alternatives = _parse_query(query)
        if not alternatives:
            return []
        params: list[BindParameter[Any]] = []
        groups: list[str] = []
        for a_index, alternative in enumerate(alternatives):
            terms = [(term, "") for term in alternative.include] + [(term, "NOT ") for term in alternative.exclude]
            conditions = []
            for t_index, (term, negation) in enumerate(terms):
                name = f"t{a_index}_{t_index}"
                conditions.append(f"lower(c.text) {negation}LIKE :{name} ESCAPE '\\'")
                params.append(bindparam(name, _like_pattern(term), type_=String))
            groups.append("(" + " AND ".join(conditions) + ")")
        sql = _chunk_query(
            "c.id AS chunk_id, c.document_id, c.invoice_id, c.page_number, c.text AS body",
            where,
            f"({' OR '.join(groups)})",
        )
        statement = text(sql).bindparams(*params, *where.binds()).columns(body=Text, **_CANDIDATE_TYPES)
        scored: list[_Candidate] = []
        for row in self.session.execute(statement):
            score = _keyword_score(row.body, alternatives)
            if score > 0:
                scored.append(_Candidate(row.chunk_id, row.document_id, row.invoice_id, row.page_number, score))
        return heapq.nlargest(pool, scored, key=lambda c: (c.score, -c.chunk_id))

    def semantic(self, vector: np.ndarray, where: _Where, pool: int) -> list[_Candidate]:
        sql = _chunk_query(
            "c.id AS chunk_id, c.document_id, c.invoice_id, c.page_number, c.embedding AS vector",
            where,
            "c.embedding IS NOT NULL",
            tail=" ORDER BY c.id",
        )
        statement = (
            text(sql).bindparams(*where.binds()).columns(vector=EmbeddingVector(len(vector)), **_CANDIDATE_TYPES)
        )
        rows = self.session.execute(statement).all()
        if not rows:
            return []
        matrix = np.stack([row.vector for row in rows])
        norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(vector))
        scores = np.divide(matrix @ vector, norms, out=np.zeros(len(rows), dtype=np.float32), where=norms > 0)
        keep = np.flatnonzero(scores > 0)  # cosine <= 0 is unrelated; NaN compares False and drops out too
        if keep.size > pool:
            keep = np.sort(keep[np.argpartition(-scores[keep], pool - 1)[:pool]])  # back in chunk-id order
        keep = keep[np.argsort(-scores[keep], kind="stable")]  # best first; ties keep ascending chunk id
        return [
            _Candidate(rows[i].chunk_id, rows[i].document_id, rows[i].invoice_id, rows[i].page_number, float(scores[i]))
            for i in keep.tolist()
        ]

    def snippets(self, chunk_ids: Sequence[int], query: str) -> dict[int, str]:
        terms = [term for alternative in _parse_query(query) for term in alternative.include]
        rows = self.session.execute(select(Chunk.id, Chunk.text).where(Chunk.id.in_(list(chunk_ids))))
        return {row.id: _snippet(row.text, terms) for row in rows}


# --- search --------------------------------------------------------------------------------------------------


def _money(value: Decimal | None) -> Decimal | None:
    """Strip storage padding (``110.0000``) to the printed scale, at least two decimals - like Invoice does."""
    if value is None:
        return None
    normalized = Decimal(value).normalize()
    exponent = normalized.as_tuple().exponent
    return Decimal(value).quantize(Decimal("0.01")) if isinstance(exponent, int) and exponent > -2 else normalized


def _hits(session: Session, ranked: Sequence[_Ranked], snippets: dict[int, str]) -> list[SearchHit]:
    document_ids = list({r.chunk.document_id for r in ranked})
    invoice_ids = list({r.chunk.invoice_id for r in ranked if r.chunk.invoice_id is not None})
    filenames = dict(session.execute(select(Document.id, Document.filename).where(Document.id.in_(document_ids))).all())
    invoices = {
        row.id: row
        for row in session.execute(
            select(
                Invoice.id,
                Invoice.invoice_number,
                Invoice.supplier_name,
                Invoice.customer_name,
                Invoice.invoice_date,
                Invoice.total,
                Invoice.currency,
                Invoice.review_status,
                Invoice.page_start,
            ).where(Invoice.id.in_(invoice_ids))
        )
    }
    hits: list[SearchHit] = []
    for item in ranked:
        chunk = item.chunk
        invoice = invoices.get(chunk.invoice_id) if chunk.invoice_id is not None else None
        page = chunk.page_number if chunk.page_number is not None else (invoice.page_start if invoice else None)
        hits.append(
            SearchHit(
                document_id=chunk.document_id,
                invoice_id=chunk.invoice_id,
                page_number=page,
                score=item.score,
                snippet=snippets.get(chunk.chunk_id, ""),
                match_type=item.match_type,
                filename=filenames[chunk.document_id],
                invoice_number=invoice.invoice_number if invoice else None,
                supplier_name=invoice.supplier_name if invoice else None,
                customer_name=invoice.customer_name if invoice else None,
                invoice_date=invoice.invoice_date if invoice else None,
                total=_money(invoice.total) if invoice else None,
                currency=invoice.currency if invoice else None,
                review_status=ReviewStatus(invoice.review_status) if invoice else None,
            )
        )
    return hits


def _ranked_search(
    session: Session,
    query: str,
    mode: str,
    filters: SearchFilters,
    dialect: str,
    limit: int,
    offset: int,
    embedder: Embedder,
) -> tuple[int, list[SearchHit]]:
    where = _build_where(filters, dialect, "c.document_id")
    backend: _Backend = _PostgresBackend(session) if dialect == "postgresql" else _SqliteBackend(session)
    keyword: list[_Candidate] = []
    semantic: list[_Candidate] = []
    if mode in ("keyword", "hybrid"):
        keyword = _best_per_group(backend.keyword(query, where, CANDIDATE_POOL))
    if mode in ("semantic", "hybrid"):
        _check_dimension(embedder)
        vector = np.asarray(embedder.embed_query(query), dtype=np.float32)
        if np.isfinite(vector).all() and float(np.linalg.norm(vector)) > 0:  # no words, nothing to embed
            semantic = _best_per_group(backend.semantic(vector, where, CANDIDATE_POOL))
    ranked = _fuse(mode, keyword, semantic)
    page = ranked[offset : offset + limit]
    snippets = backend.snippets([item.chunk.chunk_id for item in page], query) if page else {}
    return len(ranked), _hits(session, page, snippets)


def _filter_only(
    session: Session, filters: SearchFilters, dialect: str, limit: int, offset: int
) -> tuple[int, list[SearchHit]]:
    """Invoices matching the filters, newest invoice date first; the snippet is the invoice summary."""
    where = _build_where(filters, dialect, "i.document_id")
    source = "FROM invoices AS i JOIN documents AS d ON d.id = i.document_id WHERE " + " AND ".join(where.clauses)
    total = session.execute(text("SELECT count(*) " + source).bindparams(*where.binds())).scalar_one()
    if not total or not limit:
        return total, []
    statement = (
        text(
            f"SELECT i.id AS invoice_id {source} "
            "ORDER BY i.invoice_date DESC NULLS LAST, i.created_at DESC, i.id LIMIT :limit OFFSET :offset"
        )
        .bindparams(
            *where.binds(), bindparam("limit", limit, type_=Integer), bindparam("offset", offset, type_=Integer)
        )
        .columns(invoice_id=Uuid)
    )
    ids = list(session.execute(statement).scalars())
    rows = {
        invoice.id: (invoice, filename)
        for invoice, filename in session.execute(
            select(Invoice, Document.filename)
            .join(Document, Document.id == Invoice.document_id)
            .where(Invoice.id.in_(ids))
        )
    }
    hits: list[SearchHit] = []
    for invoice_id in ids:
        invoice, filename = rows[invoice_id]
        hits.append(
            SearchHit(
                document_id=invoice.document_id,
                invoice_id=invoice.id,
                page_number=invoice.page_start,
                score=0.0,
                snippet=_clip(_flatten(invoice_summary(invoice.to_invoice_data(), filename))),
                match_type="filter",
                filename=filename,
                invoice_number=invoice.invoice_number,
                supplier_name=invoice.supplier_name,
                customer_name=invoice.customer_name,
                invoice_date=invoice.invoice_date,
                total=_money(invoice.total),
                currency=invoice.currency,
                review_status=ReviewStatus(invoice.review_status),
            )
        )
    return total, hits


def search(
    session: Session,
    query: str,
    *,
    mode: SearchMode = "hybrid",
    filters: SearchFilters | None = None,
    limit: int = 20,
    offset: int = 0,
    embedder: Embedder,
) -> SearchPage:
    """Search chunks and return one page of hits grouped per ``(document, invoice)``.

    With query text the chunks are ranked by ``mode`` (filters narrow the candidates). A blank query switches to
    structured filtering over invoices (no ranking, newest first), and returns nothing when there are no filters
    either. ``total`` is the number of results before ``limit``/``offset`` are applied.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must not be negative, got {offset}")
    started = time.perf_counter()
    filters = filters or SearchFilters()
    cleaned = " ".join(query.replace("\x00", " ").split())[:MAX_QUERY_CHARS]
    dialect = session.get_bind().dialect.name
    if cleaned:
        total, hits = _ranked_search(session, cleaned, mode, filters, dialect, limit, offset, embedder)
    elif filters.is_empty:
        total, hits = 0, []
    else:
        total, hits = _filter_only(session, filters, dialect, limit, offset)
    return SearchPage(hits=hits, total=total, took_ms=int((time.perf_counter() - started) * 1000), mode=mode)


__all__ = [
    "CANDIDATE_POOL",
    "MODES",
    "RRF_K",
    "SNIPPET_MAX_CHARS",
    "SearchFilters",
    "SearchHit",
    "SearchMode",
    "SearchPage",
    "index_document",
    "search",
]
