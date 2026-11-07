"""Intake: display filenames, documents and jobs, re-processing requests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db.models import Document, JobStatus, ProcessingEvent, ProcessingJob
from app.db.session import SessionLocal, session_scope
from app.ingestion.types import FileKind
from app.pipeline.intake import AlreadyProcessingError, create_document, create_reprocess_job, sanitize_filename
from app.storage import get_storage
from tests.unit.pipeline.builders import invoice_pages, upload, write_pdf


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("invoice.pdf", "invoice.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\Desktop\\inv 7.pdf", "inv 7.pdf"),
        ("C:inv.pdf", "inv.pdf"),
        ("/var/tmp/a/b.xlsx", "b.xlsx"),
        ("a\x00b\x1f\x7f.pdf", "ab.pdf"),
        ("invoice\u202efdp.exe", "invoicefdp.exe"),  # bidi override removed
        ("Re\u0301sume\u0301.pdf", "Résumé.pdf"),  # NFC
        ("", "document"),
        ("..", "document"),
        ("  ", "document"),
        ("dir/", "document"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_caps_length_keeping_the_extension() -> None:
    name = sanitize_filename("x" * 400 + ".pdf")
    assert len(name) == 255 and name.endswith(".pdf")


def test_create_document_adds_document_job_and_event(tmp_path: Path) -> None:
    data = write_pdf(tmp_path / "a.pdf", invoice_pages("INV-1")).read_bytes()
    stored = get_storage().save_bytes(data, suffix=".pdf")
    with session_scope() as session:
        first = create_document(session, filename="../a.pdf", stored=stored, kind=FileKind.PDF,
                                content_type="application/pdf")
        second = create_document(session, filename="scan.png", stored=stored, kind=FileKind.PDF,
                                 content_type="application/pdf")
    assert first.duplicate_of_id is None and second.duplicate_of_id == first.document.id
    with SessionLocal() as session:
        document = session.get(Document, first.document.id)
        assert document is not None
        assert (document.filename, document.status, document.file_kind) == ("a.pdf", "queued", "pdf")
        assert (document.sha256, document.size_bytes, document.storage_key) == (stored.sha256, stored.size, stored.key)
        assert document.warnings == []
        job = session.get(ProcessingJob, first.job.id)
        assert job is not None and (job.status, job.attempt, job.max_attempts) == ("queued", 0, 3)
        event = session.scalars(select(ProcessingEvent).where(ProcessingEvent.document_id == document.id)).one()
        assert (event.stage, event.job_id) == ("upload", job.id)
        other = session.get(Document, second.document.id)
        assert other is not None and other.warnings == ["extension_mismatch"]  # PDF content named .png


def test_create_reprocess_job_refuses_while_active(tmp_path: Path) -> None:
    document_id, job_id = upload(write_pdf(tmp_path / "a.pdf", invoice_pages("INV-1")))
    with session_scope() as session:
        document = session.get(Document, document_id)
        assert document is not None
        with pytest.raises(AlreadyProcessingError):
            create_reprocess_job(session, document)  # the upload's job is still queued
        job = session.get(ProcessingJob, job_id)
        assert job is not None
        job.status, job.attempt, job.heartbeat_at = JobStatus.RUNNING, 1, datetime.now(UTC)
        session.flush()
        with pytest.raises(AlreadyProcessingError):
            create_reprocess_job(session, document)
        job.heartbeat_at = datetime.now(UTC) - timedelta(hours=2)  # its worker died
        session.flush()
        new_job = create_reprocess_job(session, document)
        assert job.status == "failed" and (new_job.status, document.status) == ("queued", "queued")
    with SessionLocal() as session:
        stages = session.scalars(select(ProcessingEvent.stage).order_by(ProcessingEvent.id)).all()
        assert stages == ["upload", "reprocess"]
