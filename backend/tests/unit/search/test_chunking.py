"""Chunking page text and building the searchable invoice summary."""

from __future__ import annotations

import random
import re
import uuid
from datetime import date
from decimal import Decimal
from itertools import pairwise

import pytest

from app.config import reset_settings_cache
from app.domain.schemas import DocumentType, InvoiceData, LineItem
from app.search.chunking import ChunkDraft, build_chunks, chunk_text, invoice_summary


def _shared_overlap(previous: str, following: str) -> int:
    """Length of the longest suffix of ``previous`` that is also a prefix of ``following``."""
    for size in range(min(len(previous), len(following)), 0, -1):
        if previous[-size:] == following[:size]:
            return size
    return 0


def _invoice() -> InvoiceData:
    return InvoiceData(
        invoice_number="INV-2024-001",
        invoice_date=date(2024, 3, 5),
        due_date=date(2024, 4, 4),
        supplier_name="Acme Corp",
        customer_name="Globex Ltd",
        subtotal=Decimal("100.00"),
        tax=Decimal("10.00"),
        tax_rate=Decimal("10"),
        total=Decimal("110.00"),
        currency="USD",
        document_type=DocumentType.INVOICE,
        line_items=[
            LineItem(
                description="Ergonomic office chair",
                quantity=Decimal("2"),
                unit_price=Decimal("50.00"),
                amount=Decimal("100.00"),
            )
        ],
    )


# --- chunk_text ----------------------------------------------------------------------------------------------


def test_chunk_text_overlap_and_limits() -> None:
    words = [f"word{i:04d}" for i in range(600)]
    text = " ".join(words)  # 5399 characters

    chunks = chunk_text(text, max_chars=1200, overlap=150)

    assert len(chunks) >= 5
    assert all(0 < len(chunk) <= 1200 for chunk in chunks)
    assert {word for chunk in chunks for word in chunk.split()} == set(words)  # nothing lost, no word split
    for previous, following in pairwise(chunks):
        assert 150 <= _shared_overlap(previous, following) < 1200
    assert chunks[0].startswith("word0000") and chunks[-1].endswith("word0599")


def test_chunk_text_short_text_is_one_stripped_chunk() -> None:
    assert chunk_text("  Invoice total 110.00\n ") == ["Invoice total 110.00"]
    assert chunk_text("x" * 1200) == ["x" * 1200]


@pytest.mark.parametrize("text", ["", "   ", "\n\t \r\n"])
def test_chunk_text_blank_is_empty(text: str) -> None:
    assert chunk_text(text) == []


def test_chunk_text_prefers_line_boundaries() -> None:
    lines = [f"Item {i:03d} " + "x" * 40 for i in range(120)]  # 50-char lines
    chunks = chunk_text("\n".join(lines), max_chars=500, overlap=60)
    assert len(chunks) > 5
    for chunk in chunks:
        assert re.fullmatch(r"(?s).*Item \d{3} x{40}", chunk), chunk[-60:]  # every chunk ends on a whole line


def test_chunk_text_splits_an_unbroken_token_without_looping() -> None:
    chunks = chunk_text("a" * 3000, max_chars=1000, overlap=100)
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert len(chunks) >= 3
    assert "".join(chunks).count("a") >= 3000  # overlap duplicates, never drops


def test_chunk_text_handles_overlap_close_to_max() -> None:
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(text, max_chars=100, overlap=90)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[-1].endswith("w399")


def test_chunk_text_normalises_line_endings_and_drops_nul() -> None:
    assert chunk_text("a\r\nb\rc\x00d") == ["a\nb\ncd"]


def test_chunk_text_bounds_whitespace_runs_so_blank_rows_never_fill_windows() -> None:
    sheet = "Invoice No\tTotal\n" + "\t\n" * 20_000 + "INV-9\t12.50"  # a spreadsheet with a long blank stretch
    assert chunk_text(sheet) == ["Invoice No\tTotal\n\nINV-9\t12.50"]
    padded = "Total:" + " " * 500 + "110.00"
    assert chunk_text(padded) == ["Total:  110.00"]


def test_chunk_text_invariants_hold_for_random_documents() -> None:
    rng = random.Random(7)
    vocabulary = ["invoice", "total", "Acme", "2024-03-05", "110.00", "USD", "net", "30", "x" * 40, "INV-2024-001"]
    separators = [" ", " ", " ", "\n", "  ", "\n\n", ". "]
    for _ in range(150):
        text = "".join(rng.choice(vocabulary) + rng.choice(separators) for _ in range(rng.randint(1, 900)))
        max_chars = rng.randint(200, 2000)
        overlap = rng.randint(0, max_chars // 4)
        chunks = chunk_text(text, max_chars, overlap)
        assert chunks and all(0 < len(chunk) <= max_chars for chunk in chunks)
        assert {word for chunk in chunks for word in chunk.split()} == set(text.split())
        for previous, following in pairwise(chunks):
            assert _shared_overlap(previous, following) >= overlap


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 250)])
def test_chunk_text_rejects_invalid_sizes(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError, match=r"max_chars|overlap"):
        chunk_text("some text", max_chars=max_chars, overlap=overlap)


# --- invoice_summary -----------------------------------------------------------------------------------------


def test_invoice_summary_contains_key_fields() -> None:
    summary = invoice_summary(_invoice(), "acme_march.pdf")
    for expected in (
        "Invoice INV-2024-001",
        "Acme Corp",
        "Globex Ltd",
        "2024-03-05",
        "March 2024",
        "2024-04-04",
        "110.00",
        "USD",
        "100.00",
        "10.00",
        "10%",
        "Ergonomic office chair",
        "acme_march.pdf",
    ):
        assert expected in summary, expected
    assert "None" not in summary
    assert summary.endswith(".")


def test_invoice_summary_omits_missing_fields() -> None:
    summary = invoice_summary(InvoiceData(supplier_name="Acme Corp"))
    assert summary == "Document from Acme Corp."
    receipt = InvoiceData(document_type=DocumentType.RECEIPT, total=Decimal("9.50"))
    assert invoice_summary(receipt) == "Receipt. Total 9.50."
    assert invoice_summary(InvoiceData(document_type=DocumentType.CREDIT_NOTE, invoice_number="CN-1")).startswith(
        "Credit note CN-1"
    )


def test_invoice_summary_is_bounded_and_keeps_money_exact() -> None:
    data = InvoiceData(
        supplier_name="Acme Corp",
        total=Decimal("1234567.80"),
        currency="EUR",
        line_items=[LineItem(description=f"Very long product description number {i} " * 4) for i in range(200)],
    )
    summary = invoice_summary(data, "big.xlsx")
    assert len(summary) <= 1000  # always a single chunk
    assert "1234567.80 EUR" in summary
    assert "big.xlsx" in summary  # the filename survives the line-item cut


# --- build_chunks --------------------------------------------------------------------------------------------


def test_build_chunks_emits_summaries_and_page_windows() -> None:
    first, second = uuid.uuid4(), uuid.uuid4()
    long_page = " ".join(f"token{i:04d}" for i in range(500))  # ~4500 chars -> several windows
    pages = [(1, "Page one text"), (2, "  \n "), (3, long_page)]
    drafts = build_chunks(pages, [(first, _invoice()), (second, InvoiceData(supplier_name="Initech"))], "multi.pdf")

    assert all(isinstance(draft, ChunkDraft) for draft in drafts)
    summaries = [d for d in drafts if d.kind == "invoice_summary"]
    assert [d.invoice_id for d in summaries] == [first, second]
    assert all(d.page_number is None for d in summaries)
    assert "INV-2024-001" in summaries[0].text and "multi.pdf" in summaries[1].text

    windows = [d for d in drafts if d.kind == "page_text"]
    assert {d.page_number for d in windows} == {1, 3}  # blank page 2 produces nothing
    assert all(d.invoice_id is None for d in windows)  # attributed to invoices when indexed (page ranges)
    assert len([d for d in windows if d.page_number == 3]) >= 4
    assert all(0 < len(d.text) <= 1200 for d in windows)
    assert [d.chunk_index for d in drafts] == list(range(len(drafts)))


def test_build_chunks_without_invoices_or_text() -> None:
    assert build_chunks([], [], "empty.pdf") == []
    only_pages = build_chunks([(1, "hello")], [], "x.pdf")
    assert [(d.kind, d.page_number, d.text) for d in only_pages] == [("page_text", 1, "hello")]


def test_build_chunks_honours_configured_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    text = " ".join(f"w{i}" for i in range(300))
    monkeypatch.setenv("CHUNK_MAX_CHARS", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    reset_settings_cache()
    windows = build_chunks([(1, text)], [], "x.pdf")
    assert len(windows) > 5 and all(len(d.text) <= 200 for d in windows)

    explicit = build_chunks([(1, text)], [], "x.pdf", max_chars=500, overlap=50)
    assert all(len(d.text) <= 500 for d in explicit) and len(explicit) < len(windows)
