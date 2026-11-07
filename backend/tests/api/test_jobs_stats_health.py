"""Jobs (list, bulk reprocess of failed documents), stats, and the health / ready / config endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, ProcessingJob, ValidationIssueRow
from tests.api.conftest import ClientFactory, RecordingQueue
from tests.api.factories import invoice_data, seed_document, seed_document_issue, seed_invoice

NO_FALLBACKS = {"fallback_documents": 0, "fallback_records": 0}


def seed_llm_note(session: Session, document: Document, details: dict[str, Any]) -> None:
    session.add(
        ValidationIssueRow(
            document_id=document.id,
            invoice_id=None,
            code="LLM_UNAVAILABLE",
            severity="info",
            message="LLM extraction was unavailable; the rules-based result was used",
            details=details,
        )
    )
    session.commit()


def test_list_jobs(client: TestClient, db_session: Session) -> None:
    seed_document(db_session, filename="a.pdf", job_status="succeeded", created_offset_s=1)
    seed_document(db_session, filename="b.pdf", job_status="failed", status="failed", created_offset_s=2)
    seed_document(db_session, filename="c.pdf", job_status="queued", status="queued", created_offset_s=3)

    body = client.get("/api/v1/jobs").json()
    assert set(body) == {"items", "total"}
    assert body["total"] == 3
    assert [job["status"] for job in body["items"]] == ["queued", "failed", "succeeded"]  # newest first

    failed = client.get("/api/v1/jobs", params={"status": "failed"}).json()
    assert failed["total"] == 1 and failed["items"][0]["error"] == "boom"
    assert client.get("/api/v1/jobs", params={"limit": 1, "offset": 2}).json()["items"][0]["status"] == "succeeded"
    assert client.get("/api/v1/jobs", params={"status": "exploded"}).status_code == 422


def test_reprocess_failed_bulk(client: TestClient, queue: RecordingQueue, db_session: Session) -> None:
    failed_a = seed_document(db_session, filename="a.pdf", status="failed", job_status="failed", created_offset_s=1)
    failed_b = seed_document(db_session, filename="b.csv", file_kind="csv", status="failed", job_status="failed",
                             created_offset_s=2)  # fmt: skip
    seed_document(db_session, filename="ok.pdf", status="completed", created_offset_s=3)

    response = client.post("/api/v1/jobs/reprocess-failed")

    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"requeued", "job_ids"}
    assert body["requeued"] == 2
    job_ids = [UUID(j) for j in body["job_ids"]]
    assert queue.job_ids == job_ids
    db_session.expire_all()
    jobs = db_session.scalars(select(ProcessingJob).where(ProcessingJob.id.in_(job_ids))).all()
    assert {job.document_id for job in jobs} == {failed_a.id, failed_b.id}
    assert all(job.status == "queued" for job in jobs)
    assert {db_session.get(Document, d.id).status for d in (failed_a, failed_b)} == {"queued"}

    again = client.post("/api/v1/jobs/reprocess-failed").json()
    assert again == {"requeued": 0, "job_ids": []}


def test_stats(client: TestClient, db_session: Session) -> None:
    timings = [100, 200, 300, 400, 1000]
    for i, ms in enumerate(timings):
        document = seed_document(db_session, filename=f"d{i}.pdf", status="needs_review" if i else "completed",
                                 processing_ms=ms, page_count=2, created_offset_s=i)  # fmt: skip
        seed_invoice(db_session, document, review_status="needs_review" if i else "auto_approved",
                     issues=(("TOTAL_MISMATCH", "error", "total"),) if i else ())  # fmt: skip
    seed_document_issue(db_session, document, "DUPLICATE_FILE", "warning")
    seed_document(db_session, filename="new.pdf", status="queued", job_status="queued", created_offset_s=9)

    body = client.get("/api/v1/stats").json()

    assert body["documents"]["total"] == 6
    assert body["documents"]["by_status"] == {
        "queued": 1,
        "processing": 0,
        "completed": 1,
        "needs_review": 4,
        "failed": 0,
    }
    assert body["invoices"] == {"total": 5, "by_review_status": {"auto_approved": 1, "needs_review": 4, "reviewed": 0}}
    assert body["issues"]["by_code"] == {"TOTAL_MISMATCH": 4, "DUPLICATE_FILE": 1}
    assert body["issues"]["by_severity"] == {"error": 4, "warning": 1, "info": 0}
    assert body["processing"] == {"avg_ms": 400.0, "p95_ms": 1000, "pages_total": 10}
    assert body["llm"] == {"provider": "none", "model": None, "fallback_documents": 0, "fallback_records": 0}


def test_stats_count_llm_fallbacks(client: TestClient, db_session: Session) -> None:
    for offset, records in ((0, 3), (1, 1)):
        document = seed_document(db_session, filename=f"f{offset}.pdf", created_offset_s=offset)
        seed_llm_note(db_session, document, {"reasons": ["llm_failed"], "records": records})
    legacy = seed_document(db_session, filename="old.pdf", created_offset_s=5)  # a note from before the count
    seed_invoice(db_session, legacy, record_index=0)
    seed_invoice(db_session, legacy, invoice_data(invoice_number="INV-2"), record_index=1)
    seed_llm_note(db_session, legacy, {"reasons": ["llm_unavailable"]})
    clean = seed_document(db_session, filename="llm.pdf", created_offset_s=6)
    seed_invoice(db_session, clean)  # no note: the LLM was used

    llm = client.get("/api/v1/stats").json()["llm"]

    assert (llm["fallback_documents"], llm["fallback_records"]) == (3, 3 + 1 + 2)


def test_stats_on_empty_database(client: TestClient) -> None:
    body = client.get("/api/v1/stats").json()

    assert body["documents"]["total"] == 0
    assert body["processing"] == {"avg_ms": None, "p95_ms": None, "pages_total": 0}


def test_health_ready_config(client: TestClient, queue: RecordingQueue) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/ready").json() == {"status": "ok", "database": "ok", "queue": "ok", "llm": "none"}

    config = client.get("/api/v1/config").json()
    assert config["max_upload_mb"] == 200 and config["max_files_per_request"] == 50
    assert config["max_request_mb"] == 1000  # 5 full-size files (plus multipart framing) per request
    assert {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".xlsx", ".xls", ".csv"} <= set(
        config["allowed_extensions"]
    )
    assert config["allowed_extensions"] == sorted(config["allowed_extensions"])
    assert (config["llm_provider"], config["llm_model"], config["auth_required"]) == ("none", None, False)

    queue.healthy = False
    not_ready = client.get("/api/v1/ready")
    assert not_ready.status_code == 503
    assert not_ready.json() == {"status": "unavailable", "database": "ok", "queue": "error", "llm": "none"}


def test_config_reports_local_llm_without_calling_it(make_client: ClientFactory) -> None:
    client = make_client(
        env={"LLM_PROVIDER": "local", "LLM_BASE_URL": "http://127.0.0.1:9/v1", "LLM_MODEL": "qwen3-4b"}
    )

    config = client.get("/api/v1/config").json()
    assert (config["llm_provider"], config["llm_model"]) == ("local", "qwen3-4b")
    assert client.get("/api/v1/stats").json()["llm"] == {"provider": "local", "model": "qwen3-4b", **NO_FALLBACKS}
    assert client.get("/api/v1/ready").json()["llm"] == "configured"


def test_misconfigured_llm_is_reported_not_raised(make_client: ClientFactory) -> None:
    client = make_client(env={"LLM_PROVIDER": "anthropic"})  # no ANTHROPIC_API_KEY

    assert client.get("/api/v1/stats").json()["llm"] == {"provider": "anthropic", "model": None, **NO_FALLBACKS}
    assert client.get("/api/v1/ready").json()["llm"] == "misconfigured"
