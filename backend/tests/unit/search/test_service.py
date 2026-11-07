"""Indexing and search (keyword, semantic, hybrid, structured filters) on SQLite.

Every ``test_*`` function that takes ``(db_session, corpus, embedder)`` is portable: the PostgreSQL integration
twin (``tests/integration/test_pg_search.py``) runs the very same bodies against a real pgvector database, so
the SQLite fallback and the PostgreSQL path are held to one behaviour. ``PORTABLE`` lists them.
"""

from __future__ import annotations

import time
import uuid
from datetime import date
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, Invoice
from app.domain.schemas import ReviewStatus
from app.ingestion.types import FileKind
from app.search.chunking import ChunkDraft
from app.search.embeddings import Embedder, HashingEmbedder
from app.search.service import SearchFilters, SearchHit, SearchPage, _page_owner, index_document, search
from tests.unit.search.corpus import Corpus, DocumentSpec, build_corpus

ALL_GROUPS = 10  # (document, invoice) groups in the default corpus, see corpus.py


def _numbers(page: SearchPage) -> list[str | None]:
    return [hit.invoice_number for hit in page.hits]


def _found(session: Session, embedder: Embedder, query: str, **kwargs: object) -> set[str | None]:
    page = search(session, query, embedder=embedder, limit=100, **kwargs)  # type: ignore[arg-type]
    return {hit.invoice_number for hit in page.hits}


def _chunk_count(session: Session, document: Document) -> int:
    return session.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == document.id)) or 0


# --- keyword -------------------------------------------------------------------------------------------------


def test_keyword_search_finds_term_with_snippet(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    page = search(db_session, "chairs", mode="keyword", embedder=embedder)

    assert page.mode == "keyword"
    assert page.total == 1 and len(page.hits) == 1
    assert page.took_ms >= 0
    hit = page.hits[0]
    assert hit.document_id == corpus.documents["acme"].id
    assert hit.invoice_id == corpus.invoices["INV-2024-001"].id
    assert hit.match_type == "keyword" and hit.score > 0
    assert hit.filename == "acme_chairs.pdf"
    assert (hit.invoice_number, hit.supplier_name, hit.customer_name) == ("INV-2024-001", "Acme Corp", "Globex Ltd")
    assert hit.invoice_date == date(2024, 3, 5)
    assert str(hit.total) == "110.00" and hit.currency == "USD"
    assert hit.review_status == "needs_review"
    assert hit.page_number == 1
    assert "«chairs»" in hit.snippet
    assert 0 < len(hit.snippet) <= 300


def test_keyword_search_matches_every_term_case_insensitively(
    db_session: Session, corpus: Corpus, embedder: Embedder
) -> None:
    for query in ("CHAIRS", "ergonomic CHAIRS", "chairs, ergonomic!", "“chairs”", "«chairs»", "  chairs \n"):
        assert _numbers(search(db_session, query, mode="keyword", embedder=embedder)) == ["INV-2024-001"], query
    assert search(db_session, "chairs hosting", mode="keyword", embedder=embedder).hits == []
    assert search(db_session, "no-such-word-anywhere", mode="keyword", embedder=embedder).total == 0


def test_keyword_query_syntax_phrase_and_or(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    cases = {
        '"office chairs"': {"INV-2024-001"},
        '"chairs office"': set(),
        "zebra or giraffe": {"M-100", "M-200"},
        "zebra giraffe": set(),
    }
    for query, expected in cases.items():
        assert _found(db_session, embedder, query, mode="keyword") == expected, query


def test_keyword_query_syntax_exclusion(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    everything = _found(db_session, embedder, "invoice", mode="keyword")
    without = _found(db_session, embedder, "invoice -umbrella", mode="keyword")
    assert {"M-100", "M-200"} <= everything
    assert "M-100" not in without and "M-200" in without


def test_ledger_sheet_text_is_a_document_level_hit(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    page = search(db_session, "LED-002", mode="keyword", embedder=embedder)
    by_invoice = {hit.invoice_id: hit for hit in page.hits}
    assert set(by_invoice) == {corpus.invoices["LED-002"].id, None}  # the row's summary and the whole-sheet text
    sheet = by_invoice[None]
    assert sheet.document_id == corpus.documents["ledger"].id
    assert sheet.filename == "ledger.xlsx" and sheet.page_number == 1
    assert sheet.invoice_number is None and sheet.supplier_name is None and sheet.review_status is None


def test_document_without_invoices_is_findable(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    page = search(db_session, "budget", mode="keyword", embedder=embedder)
    (hit,) = page.hits
    assert hit.document_id == corpus.documents["notes"].id and hit.filename == "notes.pdf"
    assert hit.invoice_id is None and hit.total is None and hit.invoice_date is None
    assert "«budget»" in hit.snippet


# --- attribution of page text to invoices ---------------------------------------------------------------------


def test_page_chunks_are_attributed_to_the_covering_invoice(
    db_session: Session, corpus: Corpus, embedder: Embedder
) -> None:
    expectations = {"giraffe": ("M-200", 3), "zebra": ("M-100", 1), "wire transfer": ("M-100", 2)}
    for query, (number, page_number) in expectations.items():
        page = search(db_session, query, mode="keyword", embedder=embedder)
        (hit,) = page.hits
        assert hit.invoice_id == corpus.invoices[number].id, query
        assert hit.document_id == corpus.documents["multi"].id and hit.page_number == page_number, query
        assert hit.supplier_name == corpus.invoices[number].supplier_name


def test_single_invoice_document_owns_all_its_chunks(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    (hit,) = search(db_session, "croissant", mode="keyword", embedder=embedder).hits  # receipt has no page range
    assert hit.invoice_id == corpus.invoices["R-9"].id and hit.page_number == 1
    assert hit.supplier_name == "Corner Cafe" and hit.customer_name is None and hit.currency == "CAD"


# --- semantic ------------------------------------------------------------------------------------------------


def test_semantic_search_ranks_related_first(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    expectations = {
        "ergonomic office furniture": "INV-2024-001",
        "cloud servers hosting backups": "INV-77",
        "dinner buffet waiters": "CAT-0005",
        "espresso croissant cafe": "R-9",
        "laboratory equipment": "M-100",
    }
    for query, number in expectations.items():
        page = search(db_session, query, mode="semantic", embedder=embedder)
        assert page.mode == "semantic"
        assert page.hits[0].invoice_number == number, query
        assert {hit.match_type for hit in page.hits} == {"semantic"}
        scores = [hit.score for hit in page.hits]
        assert scores == sorted(scores, reverse=True) and 0 < scores[0] <= 1.0001
        assert page.total == len(page.hits) <= ALL_GROUPS


# --- hybrid --------------------------------------------------------------------------------------------------


def test_hybrid_rrf_merges(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    # a document whose only chunk has no embedding can only ever be found by keyword
    legacy = Document(
        filename="legacy.pdf", content_type="application/pdf", file_kind="pdf", size_bytes=1, sha256="1" * 64,
        storage_key="legacy",
    )
    db_session.add(legacy)
    db_session.flush()
    db_session.add(Chunk(document_id=legacy.id, page_number=1, kind="page_text", text="cloud hosting migration plan"))
    db_session.commit()

    query = "cloud hosting"
    keyword = search(db_session, query, mode="keyword", limit=100, embedder=embedder)
    semantic = search(db_session, query, mode="semantic", limit=100, embedder=embedder)
    hybrid = search(db_session, query, mode="hybrid", limit=100, embedder=embedder)

    def key(hit: SearchHit) -> tuple[object, object]:
        return hit.document_id, hit.invoice_id

    keyword_rank = {key(hit): rank for rank, hit in enumerate(keyword.hits, start=1)}
    semantic_rank = {key(hit): rank for rank, hit in enumerate(semantic.hits, start=1)}
    assert len(keyword_rank) == 2  # the CloudCo invoice and the legacy document
    expected = {
        group: sum(1 / (60 + ranks[group]) for ranks in (keyword_rank, semantic_rank) if group in ranks)
        for group in keyword_rank.keys() | semantic_rank.keys()
    }

    assert hybrid.mode == "hybrid" and hybrid.total == len(expected) == len(hybrid.hits)
    assert {key(hit): hit.score for hit in hybrid.hits} == pytest.approx(expected)
    assert [hit.score for hit in hybrid.hits] == sorted((hit.score for hit in hybrid.hits), reverse=True)
    for hit in hybrid.hits:
        in_keyword, in_semantic = key(hit) in keyword_rank, key(hit) in semantic_rank
        assert hit.match_type == ("hybrid" if in_keyword and in_semantic else "keyword" if in_keyword else "semantic")
    assert {hit.match_type for hit in hybrid.hits} == {"hybrid", "keyword", "semantic"}
    # the only group in both lists always beats groups found by one list alone (2/62 > 1/61)
    assert hybrid.hits[0].invoice_number == "INV-77" and hybrid.hits[0].match_type == "hybrid"


def test_hybrid_is_the_default_mode(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    page = search(db_session, "chairs", embedder=embedder)
    assert page.mode == "hybrid"
    assert page.hits[0].invoice_number == "INV-2024-001" and page.hits[0].match_type == "hybrid"


# --- structured filters --------------------------------------------------------------------------------------


def test_filters_supplier_date_amount_currency(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    def numbers(**filters: object) -> set[str | None]:
        return _found(db_session, embedder, "", filters=SearchFilters(**filters))  # type: ignore[arg-type]

    assert numbers(supplier="acme") == {"INV-2024-001", "LED-001"}  # case-insensitive substring
    assert numbers(supplier="CORP") == {"INV-2024-001", "LED-001", "M-100"}
    assert numbers(customer="hooli") == {"LED-002", "M-100", "M-200"}
    assert numbers(currency="USD") == {"INV-2024-001", "LED-001", "LED-002", "M-100", "M-200"}
    assert numbers(currency="usd") == numbers(currency="USD")
    assert numbers(currency="EUR") == {"INV-77"}
    # inclusive date range (both boundaries are present in the corpus)
    assert numbers(date_from=date(2024, 3, 5), date_to=date(2024, 4, 20)) == {"INV-2024-001", "LED-001", "CAT-0005"}
    assert numbers(date_from=date(2024, 5, 1)) == {"R-9"}
    assert numbers(date_to=date(2023, 12, 1)) == {"LED-002"}
    # inclusive amount range
    assert numbers(total_min=Decimal("110.00"), total_max=Decimal("480.50")) == {"INV-2024-001", "LED-001", "CAT-0005"}
    assert numbers(total_min=Decimal("2000")) == {"INV-77", "M-200"}
    assert numbers(total_max=Decimal("12.40")) == {"R-9"}
    # filters combine with AND
    assert numbers(supplier="acme", currency="USD", total_min=Decimal("100")) == {"INV-2024-001", "LED-001"}
    assert numbers(supplier="acme", currency="EUR") == set()


def test_filter_invoice_number_substring_or_normalised_equality(
    db_session: Session, corpus: Corpus, embedder: Embedder
) -> None:
    def numbers(value: str) -> set[str | None]:
        return _found(db_session, embedder, "", filters=SearchFilters(invoice_number=value))

    assert numbers("2024") == {"INV-2024-001"}  # substring of the raw number
    assert numbers("LED-") == {"LED-001", "LED-002"}
    assert numbers("0005") == {"CAT-0005"}
    assert numbers("inv 2024 001") == {"INV-2024-001"}  # equals the normalised number INV2024001
    assert numbers("inv/77") == {"INV-77"}
    assert numbers("inv2024") == set()  # neither a raw substring nor the whole normalised number


def test_filter_review_status_file_kind_and_document(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    def numbers(**filters: object) -> set[str | None]:
        return _found(db_session, embedder, "  ", filters=SearchFilters(**filters))  # type: ignore[arg-type]

    assert numbers(review_status=ReviewStatus.REVIEWED) == {"CAT-0005"}
    assert numbers(review_status=ReviewStatus.AUTO_APPROVED) == {"INV-77", "LED-002", "R-9"}
    assert numbers(review_status=ReviewStatus.NEEDS_REVIEW) == {"INV-2024-001", "LED-001", "M-100", "M-200"}
    assert numbers(file_kind=FileKind.XLSX) == {"LED-001", "LED-002"}
    assert numbers(file_kind=FileKind.IMAGE) == {"R-9"}
    assert numbers(file_kind=FileKind.PDF) == {"INV-2024-001", "INV-77", "CAT-0005", "M-100", "M-200"}
    assert numbers(document_id=corpus.documents["multi"].id) == {"M-100", "M-200"}
    assert numbers(document_id=corpus.documents["multi"].id, currency="EUR") == set()


def test_filters_also_narrow_keyword_and_semantic_results(
    db_session: Session, corpus: Corpus, embedder: Embedder
) -> None:
    acme = SearchFilters(supplier="acme")
    assert _found(db_session, embedder, "invoice", mode="keyword", filters=acme) == {"INV-2024-001", "LED-001"}
    assert _found(db_session, embedder, "invoice", mode="semantic", filters=acme) == {"INV-2024-001", "LED-001"}
    assert _found(db_session, embedder, "invoice", mode="hybrid", filters=SearchFilters(currency="EUR")) == {"INV-77"}
    multi = SearchFilters(document_id=corpus.documents["multi"].id)
    assert _found(db_session, embedder, "invoice", mode="keyword", filters=multi) == {"M-100", "M-200"}
    assert _found(db_session, embedder, "invoice", mode="semantic", filters=multi) == {"M-100", "M-200"}
    # a document-level chunk has no invoice, so invoice filters exclude it; document filters keep it
    def budget(**filters: object) -> list[str]:
        page = search(db_session, "budget", mode="keyword", filters=SearchFilters(**filters), embedder=embedder)  # type: ignore[arg-type]
        return [hit.filename for hit in page.hits]

    assert budget(currency="USD") == []
    assert budget(file_kind=FileKind.PDF) == ["notes.pdf"]
    assert budget(file_kind=FileKind.XLSX) == []


def test_filter_values_with_like_wildcards_are_literal(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    for filters in (
        SearchFilters(supplier="%"),
        SearchFilters(supplier="_cme"),
        SearchFilters(customer="G_obex"),
        SearchFilters(invoice_number="%"),
        SearchFilters(supplier="\\"),
    ):
        assert _found(db_session, embedder, "", filters=filters) == set(), filters
    assert _found(db_session, embedder, "", filters=SearchFilters(supplier="Acme")) == {"INV-2024-001", "LED-001"}
    injection = "x'; DROP TABLE invoices; --"
    hostile = SearchFilters(supplier=injection, invoice_number=injection)
    assert _found(db_session, embedder, "", filters=hostile) == set()
    assert db_session.scalar(select(func.count()).select_from(Invoice)) == 8


def test_search_filters_normalise_blank_and_case() -> None:
    blank = SearchFilters(supplier="  ", customer="", invoice_number=" ", currency="")
    assert blank.is_empty
    filters = SearchFilters(supplier="  Acme ", currency=" usd ")
    assert (filters.supplier, filters.currency) == ("Acme", "USD") and not filters.is_empty
    assert not SearchFilters(total_min=Decimal("0")).is_empty  # zero is a real bound


# --- empty query = structured filtering ----------------------------------------------------------------------


def test_empty_query_filters_only(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    page = search(db_session, "   ", filters=SearchFilters(currency="USD"), embedder=embedder)

    assert page.total == 5 and page.mode == "hybrid"
    # invoice_date desc, then created_at desc: LED-001 was created after INV-2024-001 on the same day
    assert _numbers(page) == ["LED-001", "INV-2024-001", "M-200", "M-100", "LED-002"]
    assert {hit.match_type for hit in page.hits} == {"filter"}
    first = page.hits[1]
    assert first.document_id == corpus.documents["acme"].id and first.invoice_id == corpus.invoices["INV-2024-001"].id
    assert first.snippet.startswith("Invoice INV-2024-001 from Acme Corp to Globex Ltd")
    assert "110.00 USD" in first.snippet and "acme_chairs.pdf" in first.snippet and len(first.snippet) <= 300
    assert first.filename == "acme_chairs.pdf" and first.page_number == 1

    only_semantic = search(db_session, "", mode="semantic", filters=SearchFilters(currency="EUR"), embedder=embedder)
    assert only_semantic.mode == "semantic" and _numbers(only_semantic) == ["INV-77"]


def test_empty_query_without_filters_returns_nothing(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    page = search(db_session, "  \n ", embedder=embedder)
    assert page.hits == [] and page.total == 0


def test_filter_only_results_leave_documents_without_invoices_out(
    db_session: Session, corpus: Corpus, embedder: Embedder
) -> None:
    page = search(db_session, "", filters=SearchFilters(file_kind=FileKind.PDF), embedder=embedder, limit=100)
    assert "notes.pdf" not in {hit.filename for hit in page.hits}
    assert page.total == 5


# --- pagination ----------------------------------------------------------------------------------------------


def test_pagination_total(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    everything = search(db_session, "invoice total payment", mode="semantic", limit=100, embedder=embedder)
    assert everything.total == len(everything.hits) >= 5
    order = [(hit.document_id, hit.invoice_id) for hit in everything.hits]

    paged: list[tuple[object, object]] = []
    for offset in range(0, everything.total + 3, 3):
        page = search(db_session, "invoice total payment", mode="semantic", limit=3, offset=offset, embedder=embedder)
        assert page.total == everything.total  # the total ignores limit and offset
        assert len(page.hits) <= 3
        paged.extend((hit.document_id, hit.invoice_id) for hit in page.hits)
    assert paged == order

    beyond = search(db_session, "invoice", mode="keyword", limit=3, offset=500, embedder=embedder)
    assert beyond.hits == [] and beyond.total > 0
    count_only = search(db_session, "invoice", mode="keyword", limit=0, embedder=embedder)
    assert count_only.hits == [] and count_only.total == beyond.total


def test_pagination_total_for_filter_only_results(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    filters = SearchFilters(date_from=date(2023, 1, 1))  # every invoice: 8 in the corpus
    pages = [
        search(db_session, "", filters=filters, limit=3, offset=offset, embedder=embedder) for offset in (0, 3, 6, 9)
    ]
    assert [page.total for page in pages] == [8, 8, 8, 8]
    assert [len(page.hits) for page in pages] == [3, 3, 2, 0]
    numbers = [hit.invoice_number for page in pages for hit in page.hits]
    assert len(set(numbers)) == 8 and numbers[0] == "R-9" and numbers[-1] == "LED-002"


# --- snippets ------------------------------------------------------------------------------------------------


def test_snippets_are_short_and_markers_balanced(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    long_page = "invoice " * 600
    build_corpus(db_session, embedder, (DocumentSpec("long", "long.pdf", "pdf", (long_page,)),))
    page = search(db_session, "invoice", mode="keyword", limit=100, embedder=embedder)
    long_hits = [hit for hit in page.hits if hit.filename == "long.pdf"]
    assert long_hits
    for hit in page.hits:
        assert len(hit.snippet) <= 300
        assert hit.snippet.count("«") == hit.snippet.count("»") >= 1
    assert "«invoice»" in long_hits[0].snippet


def test_literal_marker_characters_in_text_do_not_fake_highlights(
    db_session: Session, corpus: Corpus, embedder: Embedder
) -> None:
    page_text = "Facture « Acme » du 5 mars: acompte reçu"
    build_corpus(db_session, embedder, (DocumentSpec("fr", "facture.pdf", "pdf", (page_text,)),))
    (hit,) = search(db_session, "facture", mode="keyword", embedder=embedder).hits
    assert hit.snippet.count("«") == 1 and hit.snippet.count("»") == 1
    assert "«Facture»" in hit.snippet


def test_hostile_query_strings_are_inert(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    before = db_session.scalar(select(func.count()).select_from(Chunk))
    hostile = ("'; DROP TABLE chunks; --", '" unbalanced quote', "\\", "%", "_", "a" * 5000, "nul\x00byte")
    for query in (*hostile, "-", "or or or"):
        for mode in ("keyword", "semantic", "hybrid"):
            page = search(db_session, query, mode=mode, embedder=embedder)  # type: ignore[arg-type]
            assert isinstance(page, SearchPage)
    assert db_session.scalar(select(func.count()).select_from(Chunk)) == before


# --- indexing ------------------------------------------------------------------------------------------------


def test_index_document_replaces_existing_chunks(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    document, invoice = corpus.documents["acme"], corpus.invoices["INV-2024-001"]
    assert _chunk_count(db_session, document) == 2

    drafts = [ChunkDraft(kind="page_text", text="replacement text about zeppelins", page_number=1)]
    assert index_document(db_session, document.id, drafts, embedder) == 1
    db_session.commit()

    assert _chunk_count(db_session, document) == 1
    chunk = db_session.scalars(select(Chunk).where(Chunk.document_id == document.id)).one()
    assert chunk.invoice_id == invoice.id  # the document's only invoice owns its page text
    assert chunk.embedding is not None and chunk.embedding.shape == (embedder.dim,)
    assert float(np.linalg.norm(chunk.embedding)) == pytest.approx(1.0, abs=1e-5)
    assert search(db_session, "chairs", mode="keyword", embedder=embedder).hits == []
    assert _numbers(search(db_session, "zeppelins", mode="keyword", embedder=embedder)) == ["INV-2024-001"]

    assert index_document(db_session, document.id, [], embedder) == 0
    assert _chunk_count(db_session, document) == 0
    # other documents are untouched
    assert _chunk_count(db_session, corpus.documents["cloud"]) == 2


def test_page_ownership_rules() -> None:
    first, second, third = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    only = [(first, None, None)]
    assert _page_owner(only, 7) == first  # a document's only invoice owns every page, range or not
    assert _page_owner(only, None) is None  # summaries and other pageless chunks name their own invoice

    ranges = [(first, 1, 2), (second, 3, 3), (third, None, None)]
    assert [_page_owner(ranges, page) for page in (1, 2, 3, 4)] == [first, first, second, None]
    overlapping = [(first, 1, 3), (second, 3, 4)]
    assert _page_owner(overlapping, 2) == first and _page_owner(overlapping, 3) is None  # ambiguous: unowned
    assert _page_owner([(first, 2, None), (second, 5, 6)], 2) == first  # an open end means a single page
    assert _page_owner([(first, 2, None), (second, 5, 6)], 3) is None


def test_text_without_words_is_stored_without_an_embedding(
    db_session: Session, corpus: Corpus, embedder: Embedder
) -> None:
    document = corpus.documents["notes"]
    drafts = [
        ChunkDraft(kind="page_text", text="!!! ???", page_number=1),
        ChunkDraft(kind="page_text", text="real words", chunk_index=1),
    ]
    assert index_document(db_session, document.id, drafts, embedder) == 2
    stored = db_session.scalars(select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)).all()
    assert [chunk.embedding is None for chunk in stored] == [True, False]
    # nothing to embed in the query either: semantic search has no answer, keyword search still works
    assert search(db_session, "!!! ???", mode="semantic", embedder=embedder).total == 0
    assert _numbers(search(db_session, "real", mode="keyword", embedder=embedder)) == [None]


def test_index_document_rejects_an_embedder_of_the_wrong_size(
    db_session: Session, corpus: Corpus, embedder: Embedder
) -> None:
    drafts = [ChunkDraft(kind="page_text", text="x", page_number=1)]
    with pytest.raises(ValueError, match="dimension"):
        index_document(db_session, corpus.documents["acme"].id, drafts, HashingEmbedder(dim=16))
    with pytest.raises(ValueError, match="dimension"):
        search(db_session, "chairs", mode="semantic", embedder=HashingEmbedder(dim=16))
    assert _chunk_count(db_session, corpus.documents["acme"]) == 2  # nothing was deleted


def test_search_rejects_invalid_arguments(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    with pytest.raises(ValueError, match="limit"):
        search(db_session, "x", limit=-1, embedder=embedder)
    with pytest.raises(ValueError, match="offset"):
        search(db_session, "x", offset=-1, embedder=embedder)
    with pytest.raises(ValueError, match="mode"):
        search(db_session, "x", mode="fuzzy", embedder=embedder)  # type: ignore[arg-type]


def test_search_page_serialises_for_the_api(db_session: Session, corpus: Corpus, embedder: Embedder) -> None:
    data = search(db_session, "chairs", mode="keyword", embedder=embedder).model_dump(mode="json")
    assert set(data) == {"hits", "total", "took_ms", "mode"}
    hit = data["hits"][0]
    assert hit["total"] == "110.00" and hit["invoice_date"] == "2024-03-05"
    assert hit["review_status"] == "needs_review" and hit["match_type"] == "keyword"
    assert isinstance(hit["document_id"], str) and isinstance(hit["invoice_id"], str)
    assert set(hit) == {
        "document_id", "invoice_id", "page_number", "score", "snippet", "match_type", "filename", "invoice_number",
        "supplier_name", "customer_name", "invoice_date", "total", "currency", "review_status",
    }


PORTABLE = (
    test_keyword_search_finds_term_with_snippet,
    test_keyword_search_matches_every_term_case_insensitively,
    test_keyword_query_syntax_phrase_and_or,
    test_keyword_query_syntax_exclusion,
    test_ledger_sheet_text_is_a_document_level_hit,
    test_document_without_invoices_is_findable,
    test_page_chunks_are_attributed_to_the_covering_invoice,
    test_single_invoice_document_owns_all_its_chunks,
    test_semantic_search_ranks_related_first,
    test_hybrid_rrf_merges,
    test_hybrid_is_the_default_mode,
    test_filters_supplier_date_amount_currency,
    test_filter_invoice_number_substring_or_normalised_equality,
    test_filter_review_status_file_kind_and_document,
    test_filters_also_narrow_keyword_and_semantic_results,
    test_filter_values_with_like_wildcards_are_literal,
    test_empty_query_filters_only,
    test_empty_query_without_filters_returns_nothing,
    test_filter_only_results_leave_documents_without_invoices_out,
    test_pagination_total,
    test_pagination_total_for_filter_only_results,
    test_snippets_are_short_and_markers_balanced,
    test_literal_marker_characters_in_text_do_not_fake_highlights,
    test_hostile_query_strings_are_inert,
    test_text_without_words_is_stored_without_an_embedding,
    test_index_document_replaces_existing_chunks,
    test_index_document_rejects_an_embedder_of_the_wrong_size,
    test_search_rejects_invalid_arguments,
    test_search_page_serialises_for_the_api,
)


# --- performance (SQLite fallback only) ----------------------------------------------------------------------


def test_semantic_search_over_ten_thousand_chunks_is_fast(db_session: Session, embedder: HashingEmbedder) -> None:
    rng = np.random.default_rng(42)
    documents = []
    for number in range(100):
        document = Document(
            filename=f"bulk_{number}.pdf", content_type="application/pdf", file_kind="pdf", size_bytes=1,
            sha256="0" * 64, storage_key=f"bulk/{number}",
        )
        db_session.add(document)
        documents.append(document)
    db_session.flush()
    invoices = [
        Invoice(document_id=doc.id, invoice_number=f"B-{n}", supplier_name="Bulk Co") for n, doc in enumerate(documents)
    ]
    db_session.add_all(invoices)
    db_session.flush()

    vectors = rng.standard_normal((10_000, embedder.dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    needle = np.asarray(embedder.embed_query("needle haystack"), dtype=np.float32)
    vectors[7_777] = needle  # an exact match somewhere in the middle
    rows = [
        {
            "document_id": documents[i // 100].id,
            "invoice_id": invoices[i // 100].id,
            "page_number": i % 100,
            "chunk_index": i,
            "kind": "page_text",
            "text": f"chunk {i}",
            "embedding": vectors[i],
        }
        for i in range(10_000)
    ]
    db_session.execute(insert(Chunk), rows)
    db_session.commit()

    started = time.perf_counter()
    page = search(db_session, "needle haystack", mode="semantic", embedder=embedder)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"semantic search over 10k chunks took {elapsed:.2f}s"
    assert page.hits[0].invoice_number == "B-77" and page.hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert page.total > 20
