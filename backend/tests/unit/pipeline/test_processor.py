"""process_job: stages, progress, history, idempotency, duplicates, cross-verification, errors.

Extraction is canned (``canned`` fixture) so these tests depend on ingestion, validation and search only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db.models import (
    Chunk,
    Correction,
    CrossVerification,
    Document,
    DocumentPage,
    Invoice,
    JobStatus,
    ProcessingEvent,
    ProcessingJob,
    ValidationIssueRow,
)
from app.db.session import SessionLocal, session_scope
from app.domain.schemas import ExtractionWarning, SourceKind
from app.pipeline.intake import create_reprocess_job
from app.pipeline.processor import TransientError, process_job, recover_stale_jobs
from app.pipeline.progress import ProgressReporter
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
    return write_pdf(tmp_path / "invoice.pdf", [*invoice_pages("INV-1001"), *invoice_pages("INV-1002")])


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return write_ledger_xlsx(tmp_path / "ledger.xlsx", [ledger_row("INV-1001", 100, 10, 115)])


def _two_records() -> list[Any]:
    return [
        canned_record("INV-1001", page_start=1, page_end=1),
        canned_record("INV-1002", page_start=2, page_end=2, customer="Initech LLC"),
    ]


def _doc(session: Session, document_id: UUID) -> Document:
    document = session.get(Document, document_id)
    assert document is not None
    return document


def _job(session: Session, job_id: UUID) -> ProcessingJob:
    job = session.get(ProcessingJob, job_id)
    assert job is not None
    return job


def _issues(session: Session, document_id: UUID) -> list[ValidationIssueRow]:
    return list(session.scalars(select(ValidationIssueRow).where(ValidationIssueRow.document_id == document_id)))


def _count(session: Session, model: Any, document_id: UUID) -> int:
    return session.scalar(select(func.count()).select_from(model).where(model.document_id == document_id)) or 0


def _reprocess(document_id: UUID) -> UUID:
    with session_scope() as session:
        return create_reprocess_job(session, _doc(session, document_id)).id


# --------------------------------------------------------------------------------------------- outcomes


def test_process_job_digital_pdf_completed(pdf: Path, canned: CannedExtraction) -> None:
    calls = canned(_two_records())
    document_id, job_id = upload(pdf)
    process_job(job_id)

    with SessionLocal() as session:
        document, job = _doc(session, document_id), _job(session, job_id)
        assert document.status == "completed"
        assert (document.page_count, document.ocr_page_count) == (2, 0)
        assert document.processing_ms is not None and document.processing_ms >= 0
        assert document.processed_at is not None and document.extraction_method == "rules"
        assert document.error_code is None
        assert ExtractionWarning.LLM_UNAVAILABLE.value not in document.warnings  # canned records carry none
        assert (job.status, job.stage, job.attempt) == ("succeeded", "done", 1)
        assert job.finished_at is not None and job.error is None

        pages = list(session.scalars(select(DocumentPage).where(DocumentPage.document_id == document_id)))
        assert [(p.page_number, p.method) for p in sorted(pages, key=lambda p: p.page_number)] == [
            (1, "text"),
            (2, "text"),
        ]
        assert all("Acme Supplies Ltd" in p.text and p.char_count == len(p.text) for p in pages)

        invoices = list(session.scalars(select(Invoice).where(Invoice.document_id == document_id)))
        by_index = sorted(invoices, key=lambda i: i.record_index)
        assert [(i.record_index, i.invoice_number, i.page_start, i.page_end) for i in by_index] == [
            (0, "INV-1001", 1, 1),
            (1, "INV-1002", 2, 2),
        ]
        assert all(i.review_status == "auto_approved" and i.field_meta["total"]["confidence"] == 0.9 for i in invoices)
        assert all(i.source_kind == SourceKind.INVOICE_DOCUMENT.value for i in invoices)
        assert _issues(session, document_id) == []

        chunks = list(session.scalars(select(Chunk).where(Chunk.document_id == document_id)))
        summaries = [c for c in chunks if c.kind == "invoice_summary"]
        # every summary chunk points at its invoice (ids exist before chunking, never a None document-level hit)
        assert sorted(str(c.invoice_id) for c in summaries) == sorted(str(i.id) for i in invoices)
        assert {c.invoice_id for c in chunks if c.kind == "page_text"} == {i.id for i in invoices}
        assert all(c.embedding is not None for c in chunks)
    assert calls[0][1]["mode"] == "hybrid"  # the configured EXTRACTION_MODE reaches the service


def test_process_job_records_llm_unavailable_as_document_note(pdf: Path, canned: CannedExtraction) -> None:
    canned([canned_record("INV-1001", warnings=[ExtractionWarning.LLM_UNAVAILABLE.value])])
    document_id, job_id = upload(pdf)
    process_job(job_id)
    with SessionLocal() as session:
        document = _doc(session, document_id)
        notes = [(i.code, i.severity, i.invoice_id) for i in _issues(session, document_id)]
        assert notes == [("LLM_UNAVAILABLE", "info", None)]
        assert document.status == "completed"  # an info note sends nothing to review
        assert "llm_unavailable" in document.warnings


def test_process_job_corrupt_file_failed_with_code(tmp_path: Path, canned: CannedExtraction) -> None:
    calls = canned([canned_record()])
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 20)
    document_id, job_id = upload(path)
    process_job(job_id)  # does not raise: a permanent failure is recorded, not retried

    with SessionLocal() as session:
        document, job = _doc(session, document_id), _job(session, job_id)
        assert (document.status, document.error_code) == ("failed", "corrupt_file")
        assert document.error_message and "Traceback" not in document.error_message
        assert (job.status, job.attempt) == ("failed", 1)
        assert job.error is not None and job.error.startswith("corrupt_file")
        events = list(session.scalars(select(ProcessingEvent).where(ProcessingEvent.document_id == document_id)))
        assert [(e.stage, e.level) for e in events][-1] == ("error", "error")
        assert events[-1].data["error_code"] == "corrupt_file"
        assert _count(session, Invoice, document_id) == 0
    assert calls == []  # extraction never ran
    process_job(job_id)  # a redelivery of a failed job changes nothing
    with SessionLocal() as session:
        assert _job(session, job_id).attempt == 1


def test_process_job_missing_file_fails_cleanly(pdf: Path, canned: CannedExtraction) -> None:
    from app.storage import get_storage

    canned([canned_record()])
    document_id, job_id = upload(pdf)
    with SessionLocal() as session:
        get_storage().delete(_doc(session, document_id).storage_key)
    process_job(job_id)
    with SessionLocal() as session:
        assert (_doc(session, document_id).status, _doc(session, document_id).error_code) == ("failed", "file_missing")


def test_process_job_needs_review_when_issues(pdf: Path, canned: CannedExtraction) -> None:
    canned(
        [
            canned_record("INV-1001", total="120.00"),  # 100 + 10 != 120
            canned_record("INV-1002", page_start=2, page_end=2, supplier=None),
        ]
    )
    document_id, job_id = upload(pdf)
    process_job(job_id)

    with SessionLocal() as session:
        assert _doc(session, document_id).status == "needs_review"
        invoices = {i.invoice_number: i for i in session.scalars(select(Invoice))}
        codes = {
            number: {(r.code, r.severity, r.field) for r in _issues(session, document_id) if r.invoice_id == inv.id}
            for number, inv in invoices.items()
        }
        assert ("TOTAL_MISMATCH", "error", "total") in codes["INV-1001"]
        assert ("MISSING_FIELD", "error", "supplier_name") in codes["INV-1002"]
        assert {inv.review_status for inv in invoices.values()} == {"needs_review"}


def test_document_without_records_gets_a_document_level_warning(tmp_path: Path, canned: CannedExtraction) -> None:
    canned([])
    document_id, job_id = upload(write_ledger_xlsx(tmp_path / "empty.xlsx", []))
    process_job(job_id)
    with SessionLocal() as session:
        issues = [(i.code, i.severity, i.invoice_id) for i in _issues(session, document_id)]
        assert issues == [("MISSING_FIELD", "warning", None)]
        assert _doc(session, document_id).status == "needs_review"
        pages = list(session.scalars(select(DocumentPage).where(DocumentPage.document_id == document_id)))
        assert [(p.page_number, p.label, p.method) for p in pages] == [(1, "Ledger", "sheet")]


# ------------------------------------------------------------------------------------------ idempotency


def _snapshot(session: Session, document_id: UUID) -> dict[str, Any]:
    return {
        "pages": _count(session, DocumentPage, document_id),
        "invoices": _count(session, Invoice, document_id),
        "issues": sorted((i.code, i.field or "") for i in _issues(session, document_id)),
        "chunks": _count(session, Chunk, document_id),
        "comparisons": session.scalar(select(func.count()).select_from(CrossVerification)),
        "status": _doc(session, document_id).status,
    }


def test_reprocess_is_idempotent(pdf: Path, ledger: Path, canned: CannedExtraction) -> None:
    canned([canned_ledger_row("INV-1001", "115", subtotal="100", tax="10", row_number=2)])
    _, ledger_job = upload(ledger)
    process_job(ledger_job)
    canned([*_two_records()[:1], canned_record("INV-1002", page_start=2, page_end=2, total="999.00")])
    document_id, job_id = upload(pdf)
    process_job(job_id)

    with session_scope() as session:
        first = _snapshot(session, document_id)
        old_ids = set(session.scalars(select(Invoice.id).where(Invoice.document_id == document_id)))
        target = next(iter(sorted(old_ids, key=str)))
        session.add(Correction(document_id=document_id, invoice_id=target, field="total", old_value="1",
                               new_value="2", corrected_by="tester"))
    assert first["comparisons"] == 1 and ("CROSS_VERIFICATION_MISMATCH", "") in first["issues"]

    process_job(job_id)  # duplicate delivery of a finished job: a no-op
    with SessionLocal() as session:
        assert _snapshot(session, document_id) == first
        assert set(session.scalars(select(Invoice.id).where(Invoice.document_id == document_id))) == old_ids

    second_job = _reprocess(document_id)
    process_job(second_job)
    with SessionLocal() as session:
        assert _snapshot(session, document_id) == first  # replaced, not duplicated
        new_ids = set(session.scalars(select(Invoice.id).where(Invoice.document_id == document_id)))
        assert new_ids.isdisjoint(old_ids)
        correction = session.scalars(select(Correction)).one()  # the audit trail survives, unlinked
        assert correction.invoice_id is None and correction.document_id == document_id
        jobs = list(session.scalars(select(ProcessingJob).where(ProcessingJob.document_id == document_id)))
        assert sorted(j.status for j in jobs) == ["succeeded", "succeeded"]
        ledger_issues = [i for i in session.scalars(select(ValidationIssueRow)) if i.document_id != document_id]
        # the counterpart's mismatch was re-pointed at the new record, not doubled (its own 100 + 10 != 115 stays)
        assert sorted(i.code for i in ledger_issues) == ["CROSS_VERIFICATION_MISMATCH", "TOTAL_MISMATCH"]
        mismatch = next(i for i in ledger_issues if i.code == "CROSS_VERIFICATION_MISMATCH")
        assert mismatch.details["other_invoice_id"] in {str(i) for i in new_ids}


def test_same_file_twice_flags_duplicate_file_once(pdf: Path, canned: CannedExtraction) -> None:
    canned(_two_records())
    first_doc, first_job = upload(pdf, filename="a.pdf")
    second_doc, second_job = upload(pdf, filename="a-copy.pdf")
    process_job(first_job)
    process_job(second_job)
    process_job(_reprocess(second_doc))  # twice processed: still one DUPLICATE_FILE
    process_job(_reprocess(first_doc))  # re-processing the original must not make it the duplicate

    with SessionLocal() as session:
        assert _doc(session, second_doc).duplicate_of_id == first_doc
        assert _doc(session, first_doc).duplicate_of_id is None
        second = _issues(session, second_doc)
        assert [(i.code, i.invoice_id) for i in second if i.code == "DUPLICATE_FILE"] == [("DUPLICATE_FILE", None)]
        duplicates = [i for i in second if i.code == "DUPLICATE_INVOICE"]
        first_invoices = {str(i) for i in session.scalars(select(Invoice.id).where(Invoice.document_id == first_doc))}
        assert len(duplicates) == 2 and {i.details["other_invoice_id"] for i in duplicates} == first_invoices
        assert _issues(session, first_doc) == []
        assert (_doc(session, first_doc).status, _doc(session, second_doc).status) == ("completed", "needs_review")
        comparisons = list(session.scalars(select(CrossVerification)))
        assert {(c.kind, c.status) for c in comparisons} == {("pdf_pdf", "match")} and len(comparisons) == 2
        assert {str(c.left_invoice_id) for c in comparisons} == first_invoices  # left = the earlier upload


@pytest.mark.parametrize("order", ["pdf_first", "ledger_first"])
def test_cross_verification_pdf_vs_ledger_mismatch(
    pdf: Path, ledger: Path, canned: CannedExtraction, order: str
) -> None:
    pdf_doc, pdf_job = upload(pdf)
    ledger_doc, ledger_job = upload(ledger)
    pdf_records = _two_records()
    ledger_records = [canned_ledger_row("INV-1001", "115", subtotal="100", tax="10", row_number=2)]
    steps = [(pdf_job, pdf_records), (ledger_job, ledger_records)]
    for job_id, records in steps if order == "pdf_first" else reversed(steps):
        canned(records)
        process_job(job_id)

    with SessionLocal() as session:
        invoices = {(i.document_id, i.invoice_number): i for i in session.scalars(select(Invoice))}
        pdf_inv, ledger_inv = invoices[(pdf_doc, "INV-1001")], invoices[(ledger_doc, "INV-1001")]
        comparison = session.scalars(select(CrossVerification)).one()
        assert (comparison.left_invoice_id, comparison.right_invoice_id) == (pdf_inv.id, ledger_inv.id)
        assert (comparison.kind, comparison.status, comparison.trigger) == ("pdf_excel", "mismatch", "auto")
        total = next(f for f in comparison.fields if f["field"] == "total")
        assert (total["left"], total["right"], total["status"]) == ("110.00", "115.00", "mismatch")

        mismatches = {
            row.invoice_id: row.details
            for row in session.scalars(
                select(ValidationIssueRow).where(ValidationIssueRow.code == "CROSS_VERIFICATION_MISMATCH")
            )
        }
        assert set(mismatches) == {pdf_inv.id, ledger_inv.id}  # both sides, whichever was processed first
        assert mismatches[pdf_inv.id]["other_invoice_id"] == str(ledger_inv.id)
        assert mismatches[ledger_inv.id]["other_invoice_id"] == str(pdf_inv.id)
        assert mismatches[pdf_inv.id]["mismatched_fields"] == ["total"]
        assert pdf_inv.review_status == ledger_inv.review_status == "needs_review"
        assert invoices[(pdf_doc, "INV-1002")].review_status == "auto_approved"
        assert _doc(session, pdf_doc).status == _doc(session, ledger_doc).status == "needs_review"


def test_cross_verification_clears_when_the_counterpart_is_fixed(
    pdf: Path, ledger: Path, canned: CannedExtraction
) -> None:
    pdf_doc, pdf_job = upload(pdf)
    ledger_doc, ledger_job = upload(ledger)
    canned(_two_records())
    process_job(pdf_job)
    canned([canned_ledger_row("INV-1001", "115", subtotal="100", tax="10", row_number=2)])
    process_job(ledger_job)
    canned([canned_ledger_row("INV-1001", "110", subtotal="100", tax="10", row_number=2)])  # corrected ledger
    process_job(_reprocess(ledger_doc))

    with SessionLocal() as session:
        assert session.scalars(select(ValidationIssueRow)).all() == []
        assert session.scalars(select(CrossVerification)).one().status == "match"
        assert _doc(session, pdf_doc).status == _doc(session, ledger_doc).status == "completed"


# ------------------------------------------------------------------------------------- history, progress


def test_events_recorded_per_stage(pdf: Path, canned: CannedExtraction) -> None:
    canned(_two_records())
    document_id, job_id = upload(pdf)
    process_job(job_id)

    with SessionLocal() as session:
        events = list(
            session.scalars(
                select(ProcessingEvent).where(ProcessingEvent.document_id == document_id).order_by(ProcessingEvent.id)
            )
        )
    assert [e.stage for e in events] == ["upload", "start", "parsing", "extracting", "validating", "indexing", "done"]
    assert all(e.job_id == job_id and e.level == "info" for e in events)
    by_stage = {e.stage: e.data for e in events}
    assert by_stage["parsing"]["pages"] == 2 and by_stage["parsing"]["ocr_pages"] == 0
    assert by_stage["extracting"]["records"] == 2
    assert by_stage["validating"]["issues"] == {"error": 0, "warning": 0, "info": 0}
    assert by_stage["indexing"]["chunks"] >= 4
    assert by_stage["done"]["status"] == "completed"
    assert all(isinstance(by_stage[s]["duration_ms"], int) for s in ("parsing", "extracting", "validating", "indexing"))
    history = json.dumps([[e.message, e.data] for e in events])
    for secret in ("INV-1001", "Acme", "Globex", "110.00"):  # counts and timings only, never document content
        assert secret not in history


def test_progress_writes_are_throttled(pdf: Path) -> None:
    _, job_id = upload(pdf)
    with session_scope() as session:
        job = _job(session, job_id)
        job.status, job.attempt = JobStatus.RUNNING, 1
    clock = [0.0]
    reporter = ProgressReporter(job_id, 1, SessionLocal, min_interval_s=1.0, clock=lambda: clock[0])
    for page in range(1, 1001):  # 1000 pages in 2 seconds
        clock[0] = page * 0.002
        reporter.on_progress(page, 1000, "parsing")
    assert reporter.writes <= 3  # the first page, one per second, the last page
    with SessionLocal() as session:
        job = _job(session, job_id)
        assert (job.stage, job.progress_current, job.progress_total) == ("parsing", 1000, 1000)
        assert job.heartbeat_at is not None


def test_progress_reaches_the_job_during_parsing(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.pipeline import processor

    seen: list[tuple[str | None, int, int]] = []
    real_parse = processor.parse_document

    def spying_parse(path: Path, kind: Any, on_progress: Any = None) -> Any:
        doc = real_parse(path, kind, on_progress)
        with SessionLocal() as session:  # what another connection (the API) sees meanwhile
            job = _job(session, job_id)
            seen.append((job.stage, job.progress_current, job.progress_total))
        return doc

    monkeypatch.setattr(processor, "parse_document", spying_parse)
    canned(_two_records())
    _, job_id = upload(pdf)
    process_job(job_id)
    assert seen == [("parsing", 2, 2)]


# ------------------------------------------------------------------------------------------------ errors


def test_transient_database_error_requeues_then_succeeds(
    pdf: Path, canned: CannedExtraction, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.pipeline import processor

    canned(_two_records())
    real = processor.extract_records
    failures = [OperationalError("SELECT 1", {}, Exception("database is locked"))]

    def flaky(doc: Any, **kwargs: Any) -> Any:
        if failures:
            raise failures.pop()
        return real(doc, **kwargs)

    monkeypatch.setattr(processor, "extract_records", flaky)
    document_id, job_id = upload(pdf)
    with pytest.raises(TransientError):
        process_job(job_id)
    with SessionLocal() as session:
        job, document = _job(session, job_id), _doc(session, document_id)
        assert (job.status, job.attempt, document.status) == ("queued", 1, "queued")
        assert job.error is not None and "database is locked" not in job.error  # no driver text in the DB
        assert session.scalars(select(ProcessingEvent.stage).order_by(ProcessingEvent.id)).all()[-1] == "retry"

    process_job(job_id)  # the queue's retry
    with SessionLocal() as session:
        job = _job(session, job_id)
        assert (job.status, job.attempt) == ("succeeded", 2)
        assert _doc(session, document_id).status == "completed"


def test_transient_error_on_the_last_attempt_fails_the_job(pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.pipeline import processor

    def always_locked(doc: Any, **kwargs: Any) -> Any:
        raise OperationalError("SELECT 1", {}, Exception("database is locked"))

    monkeypatch.setattr(processor, "extract_records", always_locked)
    document_id, job_id = upload(pdf)
    for _ in range(2):
        with pytest.raises(TransientError):
            process_job(job_id)
    process_job(job_id)  # the third and last attempt records the failure instead of raising
    with SessionLocal() as session:
        job, document = _job(session, job_id), _doc(session, document_id)
        assert (job.status, job.attempt) == ("failed", 3)
        assert (document.status, document.error_code) == ("failed", "database_error")


def test_unexpected_error_fails_with_generic_message(
    pdf: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from app.pipeline import processor

    def broken(doc: Any, **kwargs: Any) -> Any:
        raise ValueError("cannot parse 'Globex Corporation owes 110.00'")

    monkeypatch.setattr(processor, "extract_records", broken)
    document_id, job_id = upload(pdf)
    process_job(job_id)  # recorded, not raised, not retried
    with SessionLocal() as session:
        document, job = _doc(session, document_id), _job(session, job_id)
        assert (document.status, document.error_code, job.status) == ("failed", "internal_error", "failed")
        stored = json.dumps([document.error_message, job.error]) + json.dumps(
            [e.data for e in session.scalars(select(ProcessingEvent))]
        )
    assert "Globex" not in stored and "Globex" not in caplog.text  # neither the DB nor the log quote the document
    assert "ValueError" in caplog.text and str(job_id) in caplog.text


def test_running_job_with_fresh_heartbeat_is_not_taken_over(pdf: Path, canned: CannedExtraction) -> None:
    calls = canned(_two_records())
    _, job_id = upload(pdf)
    with session_scope() as session:
        job = _job(session, job_id)
        job.status, job.attempt, job.heartbeat_at = JobStatus.RUNNING, 1, datetime.now(UTC)
    process_job(job_id)  # a duplicate delivery while another worker runs it
    assert calls == []
    with SessionLocal() as session:
        assert (_job(session, job_id).status, _job(session, job_id).attempt) == ("running", 1)


def test_recover_stale_jobs(pdf: Path, canned: CannedExtraction) -> None:
    now = datetime.now(UTC)
    ids = [upload(pdf, filename=f"{n}.pdf") for n in range(4)]
    setups = [
        (JobStatus.RUNNING, 1, now - timedelta(hours=2)),  # stale, attempts left -> queued
        (JobStatus.RUNNING, 3, now - timedelta(hours=2)),  # stale, out of attempts -> failed
        (JobStatus.RUNNING, 1, now - timedelta(minutes=1)),  # alive (stale after 5 min) -> untouched
        (JobStatus.QUEUED, 0, None),  # waiting -> untouched
    ]
    with session_scope() as session:
        for (_, job_id), (status, attempt, heartbeat) in zip(ids, setups, strict=True):
            job = _job(session, job_id)
            job.status, job.attempt, job.heartbeat_at = status, attempt, heartbeat
            job.started_at = heartbeat
    with session_scope() as session:
        assert recover_stale_jobs(session, now=now) == 2
    with SessionLocal() as session:
        jobs = [_job(session, job_id) for _, job_id in ids]
        documents = [_doc(session, document_id) for document_id, _ in ids]
        assert [j.status for j in jobs] == ["queued", "failed", "running", "queued"]
        assert documents[0].status == "queued"
        assert (documents[1].status, documents[1].error_code) == ("failed", "worker_lost")
        assert documents[2].status == documents[3].status == "queued"  # untouched ("queued" from intake)

    canned(_two_records())
    process_job(ids[0][1])  # the re-queued job runs to completion on its next attempt
    with SessionLocal() as session:
        assert (_job(session, ids[0][1]).status, _job(session, ids[0][1]).attempt) == ("succeeded", 2)
