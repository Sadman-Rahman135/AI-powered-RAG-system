"""The extraction service (segments -> rules || LLM -> grounding -> merge; spreadsheets)."""

from __future__ import annotations

import re
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from app.config import Settings, get_settings, reset_settings_cache
from app.db.models import CrossVerification, Document, Invoice, ValidationIssueRow
from app.db.session import SessionLocal
from app.domain.schemas import ExtractionWarning, SourceKind
from app.extraction.llm.base import LLMRefusalError, LLMUnavailableError
from app.extraction.service import extract_records
from app.ingestion.router import parse_document
from app.ingestion.types import FileKind, ParsedDocument
from app.pipeline.processor import process_job
from tests.unit.pipeline.builders import (
    CUSTOMER,
    SUPPLIER,
    invoice_pages,
    ledger_row,
    upload,
    write_ledger_xlsx,
    write_pdf,
)

_NUMBER_IN_PROMPT = re.compile(r"Invoice No:\s+(INV-\d+)")


def _multi_invoice_pdf(path: Path) -> ParsedDocument:
    """Three invoices on four pages: INV-1001 (p1), INV-1002 (p2-3, "Page k of 2"), INV-1003 (p4)."""
    pages = [
        *invoice_pages("INV-1001"),
        *invoice_pages("INV-1002", customer="Initech LLC", split=True),
        *invoice_pages("INV-1003", customer="Hooli Inc", items=(("Service fee", "1", "200.00", "200.00"),)),
    ]
    return parse_document(write_pdf(path, pages), FileKind.PDF)


def _numbers(records: list[Any]) -> list[str | None]:
    return [r.result.data.invoice_number for r in records]


class FakeProvider:
    """Answers like a well-behaved model: copies the invoice's printed values out of the prompt."""

    name = "fake"
    model = "fake-model"

    def __init__(self, *, delay: float = 0.0, error: Exception | None = None, invent_customer: bool = False) -> None:
        self.delay = delay
        self.error = error
        self.invent_customer = invent_customer
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4096) -> Any:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.error is not None:
                raise self.error
            match = _NUMBER_IN_PROMPT.search(user)
            return {
                "invoice_number": match.group(1) if match else None,
                "invoice_date": "2024-03-05",
                "due_date": None,
                "supplier_name": SUPPLIER,
                "customer_name": "Invented Corp" if self.invent_customer else None,
                "subtotal": None,
                "discount": None,
                "shipping": None,
                "tax": None,
                "tax_rate": None,
                "total": None,
                "currency": "USD",
                "document_type": "invoice",
                "line_items": [],
            }
        finally:
            with self._lock:
                self.active -= 1


def test_extract_records_pdf_multi_invoice(tmp_path: Path, settings: Settings) -> None:
    doc = _multi_invoice_pdf(tmp_path / "multi.pdf")
    records = extract_records(doc, mode="rules", provider=None, settings=settings)

    assert _numbers(records) == ["INV-1001", "INV-1002", "INV-1003"]
    assert [(r.page_start, r.page_end) for r in records] == [(1, 1), (2, 3), (4, 4)]
    assert all(r.source_kind is SourceKind.INVOICE_DOCUMENT and r.sheet_name is None for r in records)
    assert [r.result.data.total for r in records] == [Decimal("110.00"), Decimal("110.00"), Decimal("220.00")]
    assert [r.result.data.customer_name for r in records] == [CUSTOMER, "Initech LLC", "Hooli Inc"]
    second = records[1]
    assert len(second.result.data.line_items) == 2  # items of both pages of INV-1002
    assert "\f" in second.text and "INV-1002" in second.text  # the segment's pages, form-feed separated
    assert all(r.result.method == "rules" for r in records)
    assert not any(w.startswith("llm") for r in records for w in r.result.warnings)  # rules mode: no LLM notes


def test_extract_records_hybrid_without_provider_notes_llm_unavailable(tmp_path: Path, settings: Settings) -> None:
    doc = _multi_invoice_pdf(tmp_path / "multi.pdf")
    records = extract_records(doc, mode="hybrid", provider=None, settings=settings)
    assert len(records) == 3
    assert all(ExtractionWarning.LLM_UNAVAILABLE.value in r.result.warnings for r in records)
    assert all(r.result.method == "rules" for r in records)


def test_extract_records_spreadsheet_ledger(tmp_path: Path, settings: Settings) -> None:
    path = write_ledger_xlsx(
        tmp_path / "ledger.xlsx",
        [ledger_row("INV-1001", 100, 10, 110), ledger_row("INV-1002", 50, 5, 55), ledger_row("INV-1003", 200, 20, 225)],
    )
    doc = parse_document(path, FileKind.XLSX)
    provider = FakeProvider()
    records = extract_records(doc, mode="hybrid", provider=provider, settings=settings)

    assert provider.calls == 0  # spreadsheets are mapped deterministically, never sent to the LLM
    assert _numbers(records) == ["INV-1001", "INV-1002", "INV-1003"]
    assert all(r.source_kind is SourceKind.LEDGER_ROW for r in records)
    assert [(r.sheet_name, r.row_number) for r in records] == [("Ledger", 2), ("Ledger", 3), ("Ledger", 4)]
    assert all((r.page_start, r.page_end) == (1, 1) for r in records)  # the sheet's page number
    assert [r.result.data.total for r in records] == [Decimal("110"), Decimal("55"), Decimal("225")]
    assert all(r.result.method == "spreadsheet" and not r.result.warnings for r in records)
    assert "INV-1002" in records[1].text and "INV-1001" not in records[1].text  # the row, not the whole sheet


def test_llm_failure_falls_back_to_rules(tmp_path: Path, settings: Settings) -> None:
    doc = _multi_invoice_pdf(tmp_path / "multi.pdf")
    rules = extract_records(doc, mode="rules", provider=None, settings=settings)

    for error, warning in (
        (LLMUnavailableError("fake connection failed"), ExtractionWarning.LLM_FAILED),
        (LLMRefusalError("fake declined"), ExtractionWarning.LLM_REFUSED),
        (RuntimeError("bug in a provider"), ExtractionWarning.LLM_FAILED),
    ):
        for mode in ("hybrid", "llm"):
            provider = FakeProvider(error=error)
            records = extract_records(doc, mode=mode, provider=provider, settings=settings)  # type: ignore[arg-type]
            assert provider.calls == 3, (error, mode)
            assert [r.result.data for r in records] == [r.result.data for r in rules], (error, mode)
            assert all(r.result.method == "rules" for r in records)
            assert all(warning.value in r.result.warnings for r in records), (error, mode)


def test_hybrid_merges_grounded_llm_result(tmp_path: Path, settings: Settings) -> None:
    doc = _multi_invoice_pdf(tmp_path / "multi.pdf")
    provider = FakeProvider(invent_customer=True)
    records = extract_records(doc, mode="hybrid", provider=provider, settings=settings)

    assert _numbers(records) == ["INV-1001", "INV-1002", "INV-1003"]
    first = records[0].result
    assert first.method == "hybrid" and first.llm_provider == "fake" and first.llm_model == "fake-model"
    assert first.meta["invoice_number"].source.value == "rules+llm"  # both sides agreed
    assert first.data.customer_name == CUSTOMER  # the invented name was rejected by grounding, rules kept
    assert first.rejected.get("customer_name") == "Invented Corp"
    assert first.data.total == Decimal("110.00")  # the LLM returned null; the rules value stands


def test_llm_mode_returns_the_grounded_llm_result(tmp_path: Path, settings: Settings) -> None:
    doc = _multi_invoice_pdf(tmp_path / "multi.pdf")
    records = extract_records(doc, mode="llm", provider=FakeProvider(), settings=settings)
    assert all(r.result.method == "llm" for r in records)
    assert _numbers(records) == ["INV-1001", "INV-1002", "INV-1003"]
    assert records[0].result.data.total is None  # the model gave no total; nothing is filled in from the rules


def test_llm_calls_run_concurrently_in_segment_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_CONCURRENCY", "2")
    reset_settings_cache()
    pages = [page for n in range(1, 5) for page in invoice_pages(f"INV-{2000 + n}")]
    doc = parse_document(write_pdf(tmp_path / "four.pdf", pages), FileKind.PDF)
    provider = FakeProvider(delay=0.2)
    progress: list[tuple[int, int, str]] = []

    started = time.perf_counter()
    records = extract_records(
        doc, mode="hybrid", provider=provider, settings=get_settings(), on_progress=lambda *a: progress.append(a)
    )
    elapsed = time.perf_counter() - started

    assert _numbers(records) == ["INV-2001", "INV-2002", "INV-2003", "INV-2004"]
    assert provider.max_active == 2
    assert elapsed < 4 * 0.2  # two at a time, not one after another
    assert progress[-1] == (4, 4, "extracting")
    assert [done for done, _, _ in progress] == sorted(done for done, _, _ in progress)


def test_blank_segment_yields_one_empty_record_without_llm_call(tmp_path: Path, settings: Settings) -> None:
    doc = parse_document(write_pdf(tmp_path / "blank.pdf", [""]), FileKind.PDF)
    provider = FakeProvider()
    records = extract_records(doc, mode="hybrid", provider=provider, settings=settings)
    assert len(records) == 1
    assert records[0].result.data.invoice_number is None and records[0].result.data.total is None
    assert (records[0].page_start, records[0].page_end) == (1, 1)
    assert provider.calls == 0  # nothing to read: no LLM call, and no LLM warning either
    assert not any(w.startswith("llm") for w in records[0].result.warnings)


def test_end_to_end_pdf_and_ledger_cross_verified(tmp_path: Path) -> None:
    """The real extraction through the whole job: a 3-invoice PDF and a ledger disagreeing on INV-1003."""
    pdf = write_pdf(
        tmp_path / "invoices.pdf",
        [
            *invoice_pages("INV-1001"),
            *invoice_pages("INV-1002", customer="Initech LLC", items=(("Gadget", "4", "12.50", "50.00"),)),
            *invoice_pages("INV-1003", customer="Hooli Inc", items=(("Service fee", "1", "200.00", "200.00"),)),
        ],
    )
    ledger = write_ledger_xlsx(
        tmp_path / "ledger.xlsx",
        [
            ledger_row("INV-1001", 100, 10, 110),
            ledger_row("INV-1002", 50, 5, 55, customer="Initech LLC"),
            ledger_row("INV-1003", 200, 20, 225, customer="Hooli Inc"),
        ],
    )
    pdf_doc, pdf_job = upload(pdf)
    ledger_doc, ledger_job = upload(ledger)
    process_job(pdf_job)
    process_job(ledger_job)

    with SessionLocal() as session:
        statuses = {d.id: d.status for d in session.scalars(select(Document))}
        assert statuses == {pdf_doc: "needs_review", ledger_doc: "needs_review"}
        invoices = {(i.document_id, i.invoice_number): i for i in session.scalars(select(Invoice))}
        assert len(invoices) == 6
        flagged = {
            (row.document_id, row.invoice_id)
            for row in session.scalars(
                select(ValidationIssueRow).where(ValidationIssueRow.code == "CROSS_VERIFICATION_MISMATCH")
            )
        }
        assert flagged == {
            (pdf_doc, invoices[(pdf_doc, "INV-1003")].id),
            (ledger_doc, invoices[(ledger_doc, "INV-1003")].id),
        }
        comparisons = {
            cv.left_invoice_id: (cv.kind, cv.status) for cv in session.scalars(select(CrossVerification))
        }
        assert comparisons == {
            invoices[(pdf_doc, "INV-1001")].id: ("pdf_excel", "match"),
            invoices[(pdf_doc, "INV-1002")].id: ("pdf_excel", "match"),
            invoices[(pdf_doc, "INV-1003")].id: ("pdf_excel", "mismatch"),
        }
        review = {key: inv.review_status for key, inv in invoices.items()}
        assert review[(pdf_doc, "INV-1001")] == "auto_approved"
        assert review[(ledger_doc, "INV-1001")] == "auto_approved"  # a ledger row is not missing "line items"
        assert review[(pdf_doc, "INV-1003")] == review[(ledger_doc, "INV-1003")] == "needs_review"
