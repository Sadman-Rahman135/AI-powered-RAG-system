"""The pipeline on PostgreSQL: the advisory lock, the full persist path and the takeover check at commit time.

Needs ``TEST_DATABASE_URL`` (e.g. ``postgresql+psycopg://docintel:docintel@localhost:5434/docintel_test``); skipped
otherwise. Every test runs in its own throw-away database (created from that server and dropped afterwards), so it
never touches tables other suites use.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url

from app.config import reset_settings_cache
from app.db.models import CrossVerification, Document, Invoice, ProcessingJob, ValidationIssueRow
from app.db.pg_extras import init_pg_schema
from app.db.session import SessionLocal, get_engine, reset_engine, session_scope
from app.pipeline import processor
from app.pipeline.intake import create_reprocess_job
from app.pipeline.processor import process_job
from app.pipeline.review import RECONCILE_LOCK_KEY, serialize_reconciliation
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

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    base = os.environ["TEST_DATABASE_URL"]  # the root conftest skips this test when it is unset
    name = f"docintel_t12_{uuid4().hex[:12]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(base).set(database=name).render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", url)
    reset_settings_cache()
    reset_engine()
    try:
        init_pg_schema(get_engine())
        yield name
    finally:
        reset_engine()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def test_pg_reconciliation_lock_serialises_and_is_reentrant(pg_database: str) -> None:
    holder, other = SessionLocal(), SessionLocal()
    try:
        serialize_reconciliation(holder)
        serialize_reconciliation(holder)  # a caller that already took it (corrections endpoint) may call again
        try_lock = text("SELECT pg_try_advisory_xact_lock(:key)")
        assert other.execute(try_lock, {"key": RECONCILE_LOCK_KEY}).scalar_one() is False
        other.rollback()
        holder.commit()  # transaction-level: released at the end of the transaction
        assert other.execute(try_lock, {"key": RECONCILE_LOCK_KEY}).scalar_one() is True
    finally:
        other.rollback()
        holder.rollback()
        other.close()
        holder.close()


def test_pg_pdf_and_ledger_cross_verified_and_reprocessed(
    pg_database: str, tmp_path: Path, canned: CannedExtraction
) -> None:
    pdf = write_pdf(tmp_path / "invoice.pdf", [*invoice_pages("INV-1001"), *invoice_pages("INV-1002")])
    ledger = write_ledger_xlsx(tmp_path / "ledger.xlsx", [ledger_row("INV-1001", 100, 15, 115)])
    pdf_doc, pdf_job = upload(pdf)
    ledger_doc, ledger_job = upload(ledger)
    canned([canned_ledger_row("INV-1001", "115", subtotal="100", tax="15", row_number=2)])
    process_job(ledger_job)  # the ledger first: the PDF's processing must still flag both sides
    canned([canned_record("INV-1001"), canned_record("INV-1002", page_start=2, page_end=2)])
    process_job(pdf_job)

    def snapshot() -> tuple[Any, ...]:
        with SessionLocal() as session:
            flagged = sorted(
                str(row.document_id)
                for row in session.scalars(
                    select(ValidationIssueRow).where(ValidationIssueRow.code == "CROSS_VERIFICATION_MISMATCH")
                )
            )
            comparison = session.scalars(select(CrossVerification)).one()
            left = session.get(Invoice, comparison.left_invoice_id)
            assert left is not None
            statuses = {d.id: d.status for d in session.scalars(select(Document))}
            invoices = session.scalar(select(func.count()).select_from(Invoice))
            return flagged, comparison.kind, comparison.status, left.document_id, statuses, invoices

    first = snapshot()
    assert first[:4] == (sorted([str(pdf_doc), str(ledger_doc)]), "pdf_excel", "mismatch", pdf_doc)
    assert first[4] == {pdf_doc: "needs_review", ledger_doc: "needs_review"} and first[5] == 3

    with session_scope() as session:
        document = session.get(Document, pdf_doc)
        assert document is not None
        reprocess_job = create_reprocess_job(session, document).id
    process_job(reprocess_job)
    assert snapshot() == first  # replaced in one transaction, nothing doubled


def test_pg_attempt_taken_over_during_persist_discards_its_results(
    pg_database: str, tmp_path: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned([canned_record("INV-1001")])
    document_id, job_id = upload(write_pdf(tmp_path / "invoice.pdf", invoice_pages("INV-1001")))
    real_index = processor.index_document

    def taken_over_meanwhile(session: Any, *args: Any) -> int:
        with session_scope() as other:  # another worker takes the job over while this attempt persists
            job = other.get(ProcessingJob, job_id)
            assert job is not None
            job.attempt = 2
        return real_index(session, *args)

    monkeypatch.setattr(processor, "index_document", taken_over_meanwhile)
    process_job(job_id)
    with SessionLocal() as session:
        job = session.get(ProcessingJob, job_id)
        assert job is not None and (job.status, job.attempt) == ("running", 2)
        assert session.scalar(select(func.count()).select_from(Invoice)) == 0  # rolled back, not committed
        document = session.get(Document, document_id)
        assert document is not None and document.status == "processing"


def test_pg_jobs_table_round_trip(pg_database: str, tmp_path: Path) -> None:
    """Sanity check that the throw-away schema is the real one (timezone-aware columns, UUID keys)."""
    _, job_id = upload(write_pdf(tmp_path / "invoice.pdf", invoice_pages("INV-1")))
    with SessionLocal() as session:
        job = session.get(ProcessingJob, job_id)
        assert job is not None and isinstance(job.id, UUID)
        assert job.created_at.tzinfo is not None
