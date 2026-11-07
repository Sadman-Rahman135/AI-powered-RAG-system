"""Re-validation after manual corrections and the document status rule (review.py)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, CrossVerification, Document, DocumentStatus, Invoice, ValidationIssueRow
from app.db.session import SessionLocal, session_scope
from app.pipeline.processor import process_job
from app.pipeline.review import refresh_document_status, revalidate_invoice
from app.search.embeddings import get_embedder
from app.search.service import search
from tests.unit.pipeline.builders import (
    canned_ledger_row,
    canned_record,
    invoice_pages,
    ledger_row,
    upload,
    write_ledger_xlsx,
    write_pdf,
)
from tests.unit.pipeline.conftest import CannedExtraction


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    return write_pdf(tmp_path / "invoice.pdf", invoice_pages("INV-1001"))


def _process(path: Path, records: list[Any], canned: CannedExtraction) -> UUID:
    canned(records)
    document_id, job_id = upload(path)
    process_job(job_id)
    return document_id


def _only_invoice(session: Session, document_id: UUID) -> Invoice:
    return session.scalars(select(Invoice).where(Invoice.document_id == document_id)).one()


def _correct(session: Session, invoice: Invoice, **values: Any) -> list[ValidationIssueRow]:
    """What the PATCH endpoint does: apply the corrected values, then re-validate."""
    invoice.apply_invoice_data(invoice.to_invoice_data().model_copy(update=values))
    return revalidate_invoice(session, invoice)


def _status(session: Session, document_id: UUID) -> str:
    document = session.get(Document, document_id)
    assert document is not None
    return document.status


def test_correction_revalidates_the_record_and_refreshes_its_search_summary(
    pdf: Path, canned: CannedExtraction
) -> None:
    document_id = _process(pdf, [canned_record(total="120.00")], canned)
    with session_scope() as session:
        invoice = _only_invoice(session, document_id)
        assert invoice.review_status == "needs_review" and [i.code for i in invoice.issues] == ["TOTAL_MISMATCH"]
        issues = _correct(session, invoice, total=Decimal("110.00"), supplier_name="Initrode Systems")
        assert issues == [] and invoice.issues == []  # the loaded collection is refreshed too
        assert invoice.review_status == "auto_approved"
        assert _status(session, document_id) == "completed"
        invoice_id = invoice.id

    with SessionLocal() as session:
        summaries = session.scalars(
            select(Chunk).where(Chunk.invoice_id == invoice_id, Chunk.kind == "invoice_summary")
        ).all()
        assert len(summaries) == 1 and "Initrode Systems" in summaries[0].text
        assert summaries[0].embedding is not None
        page = search(session, "Initrode", mode="keyword", embedder=get_embedder())
        assert [hit.invoice_id for hit in page.hits] == [invoice_id]
        assert page.hits[0].supplier_name == "Initrode Systems"


def test_reviewed_is_kept_only_while_no_error_or_warning_remains(pdf: Path, canned: CannedExtraction) -> None:
    document_id = _process(pdf, [canned_record()], canned)
    with session_scope() as session:
        invoice = _only_invoice(session, document_id)
        invoice.review_status = "reviewed"
        _correct(session, invoice, customer_name="Globex Corp.")
        assert invoice.review_status == "reviewed"
        _correct(session, invoice, total=Decimal("999.00"))
        assert invoice.review_status == "needs_review"
        assert _status(session, document_id) == "needs_review"


def test_correction_reaches_the_cross_verified_counterpart(
    pdf: Path, tmp_path: Path, canned: CannedExtraction
) -> None:
    pdf_doc = _process(pdf, [canned_record()], canned)
    ledger = write_ledger_xlsx(tmp_path / "ledger.xlsx", [ledger_row("INV-1001", 100, 15, 115)])
    ledger_doc = _process(ledger, [canned_ledger_row("INV-1001", "115", subtotal="100", tax="15", row_number=2)],
                          canned)
    with SessionLocal() as session:
        codes = {i.code for i in session.scalars(select(ValidationIssueRow))}
        assert codes == {"CROSS_VERIFICATION_MISMATCH"}

    with session_scope() as session:  # the ledger row was right after all: the PDF record is corrected
        _correct(session, _only_invoice(session, pdf_doc), tax=Decimal("15.00"), total=Decimal("115.00"))
    with SessionLocal() as session:
        assert session.scalars(select(ValidationIssueRow)).all() == []
        assert session.scalars(select(CrossVerification)).one().status == "match"
        assert _status(session, pdf_doc) == _status(session, ledger_doc) == "completed"
        assert _only_invoice(session, ledger_doc).review_status == "auto_approved"

    with session_scope() as session:  # a renumbered record leaves its old counterpart alone
        _correct(session, _only_invoice(session, ledger_doc), customer_name="Umbrella Corporation")
        assert {i.code for i in session.scalars(select(ValidationIssueRow))} == {"CROSS_VERIFICATION_MISMATCH"}
        _correct(session, _only_invoice(session, pdf_doc), invoice_number="INV-9999")
    with SessionLocal() as session:
        assert session.scalars(select(CrossVerification)).all() == []
        assert [(i.code, i.field) for i in session.scalars(select(ValidationIssueRow))] == []


def test_document_status_rules(pdf: Path, canned: CannedExtraction) -> None:
    document_id = _process(
        pdf, [canned_record(), canned_record("INV-1002", warnings=["llm_unavailable"])], canned
    )
    with session_scope() as session:
        document = session.get(Document, document_id)
        assert document is not None and document.status == "completed"
        session.add(ValidationIssueRow(document_id=document_id, invoice_id=None, code="OCR_LOW_QUALITY",
                                       severity="warning", message="OCR confidence is low", details={}))
        refresh_document_status(session, document)
        assert document.status == "needs_review"  # a document-level warning, records not reviewed yet
        for invoice in session.scalars(select(Invoice).where(Invoice.document_id == document_id)):
            invoice.review_status = "reviewed"
        refresh_document_status(session, document)
        assert document.status == "completed"  # every record was reviewed: the warning has been seen
        for frozen in (DocumentStatus.FAILED, DocumentStatus.PROCESSING, DocumentStatus.QUEUED):
            document.status = frozen
            refresh_document_status(session, document)
            assert document.status == frozen
