"""Document endpoints other than upload: list, detail, status, results, pages, events, file, reprocess, delete."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import CrossVerification, Document, Invoice, ProcessingJob
from app.domain.schemas import SourceKind
from app.pipeline.review import reconcile_invoices, refresh_document_status
from tests.api.conftest import RecordingQueue
from tests.api.factories import (
    invoice_data,
    pdf_bytes,
    seed_document,
    seed_document_issue,
    seed_event,
    seed_invoice,
    seed_pages,
)

DOCS = "/api/v1/documents"
DOCUMENT_KEYS = {
    "id", "filename", "file_kind", "content_type", "size_bytes", "sha256", "status", "page_count", "ocr_page_count",
    "error_code", "error_message", "processing_ms", "duplicate_of_id", "created_at", "updated_at", "processed_at",
    "invoice_count", "issue_counts", "job",
}  # fmt: skip
JOB_KEYS = {
    "id", "document_id", "status", "stage", "progress_current", "progress_total", "attempt", "max_attempts", "error",
    "created_at", "started_at", "finished_at",
}  # fmt: skip
INVOICE_KEYS = {
    "id", "document_id", "filename", "record_index", "source_kind", "page_start", "page_end", "sheet_name",
    "row_number", "invoice_number", "invoice_date", "due_date", "supplier_name", "customer_name", "subtotal",
    "discount", "shipping", "tax", "tax_rate", "total", "currency", "document_type", "line_items", "field_meta",
    "extraction_method", "overall_confidence", "review_status", "issues", "created_at", "updated_at", "reviewed_at",
    "reviewed_by",
}  # fmt: skip
ISSUE_KEYS = {"id", "invoice_id", "document_id", "code", "severity", "field", "message", "details"}


def _error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code
    return body["error"]


@pytest.fixture
def corpus(db_session: Session) -> dict[str, Document]:
    """Three documents: a reviewed PDF with an invoice and issues, a failed CSV, a queued image."""
    pdf = seed_document(db_session, filename="acme_march.pdf", status="needs_review", created_offset_s=10,
                        page_count=1, ocr_page_count=0, processing_ms=1500, extraction_method="hybrid",
                        llm_provider="local", llm_model="qwen3.5-9b", prompt_version="v3")  # fmt: skip
    seed_invoice(db_session, pdf, review_status="needs_review",
                 issues=(("TOTAL_MISMATCH", "error", "total"), ("LOW_CONFIDENCE", "warning", "supplier_name"),
                         ("CURRENCY_INFERRED", "info", "currency")))  # fmt: skip
    seed_document_issue(db_session, pdf, "DUPLICATE_FILE", "warning")
    failed = seed_document(db_session, filename="ledger.csv", file_kind="csv", status="failed", created_offset_s=20,
                           job_status="failed", error_code="corrupt_file", error_message="bad file")  # fmt: skip
    queued = seed_document(db_session, filename="scan.png", file_kind="image", status="queued", created_offset_s=30,
                           job_status="queued")  # fmt: skip
    return {"pdf": pdf, "failed": failed, "queued": queued}


def test_list_documents_shape_and_default_order(client: TestClient, corpus: dict[str, Document]) -> None:
    body = client.get(DOCS).json()

    assert set(body) == {"items", "total", "limit", "offset"}
    assert (body["total"], body["limit"], body["offset"]) == (3, 20, 0)
    assert [d["filename"] for d in body["items"]] == ["scan.png", "ledger.csv", "acme_march.pdf"]
    pdf = body["items"][2]
    assert set(pdf) == DOCUMENT_KEYS
    assert set(pdf["job"]) == JOB_KEYS
    assert pdf["invoice_count"] == 1
    assert pdf["issue_counts"] == {"error": 1, "warning": 2, "info": 1}
    assert pdf["job"]["status"] == "succeeded"
    assert pdf["created_at"].endswith("Z") or pdf["created_at"].endswith("+00:00")


def test_list_documents_filters(client: TestClient, corpus: dict[str, Document]) -> None:
    def names(**params) -> list[str]:
        response = client.get(DOCS, params=params)
        assert response.status_code == 200, response.text
        return [d["filename"] for d in response.json()["items"]]

    assert names(status="failed") == ["ledger.csv"]
    assert names(status=["failed", "queued"]) == ["scan.png", "ledger.csv"]
    assert names(file_kind="image") == ["scan.png"]
    assert names(q="ACME") == ["acme_march.pdf"]
    assert names(q="INV-2024") == ["acme_march.pdf"]  # also matches the invoice number of a record
    assert names(q="100%") == []  # LIKE wildcards are literal
    assert names(needs_review="true") == ["acme_march.pdf"]
    assert names(needs_review="false") == ["scan.png", "ledger.csv"]
    assert names(sort="filename") == ["acme_march.pdf", "ledger.csv", "scan.png"]
    assert names(sort="-filename", limit=1, offset=1) == ["ledger.csv"]


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"sort": "password"}, {"status": "lost"}, {"file_kind": "exe"},
     {"needs_review": "maybe"}],
)  # fmt: skip
def test_list_documents_rejects_bad_parameters(client: TestClient, params: dict) -> None:
    error = _error(client.get(DOCS, params=params), 422, "validation_error")
    assert set(error["details"]["fields"]) == set(params)


def test_get_document_detail(client: TestClient, corpus: dict[str, Document]) -> None:
    body = client.get(f"{DOCS}/{corpus['pdf'].id}").json()

    assert set(body) == DOCUMENT_KEYS | {"invoices", "document_issues", "extraction"}
    assert body["extraction"] == {"method": "hybrid", "llm_provider": "local", "llm_model": "qwen3.5-9b",
                                  "prompt_version": "v3"}  # fmt: skip
    assert [i["code"] for i in body["document_issues"]] == ["DUPLICATE_FILE"]
    assert body["document_issues"][0]["invoice_id"] is None
    (invoice,) = body["invoices"]
    assert set(invoice) == INVOICE_KEYS
    assert invoice["filename"] == "acme_march.pdf"
    assert (invoice["subtotal"], invoice["tax"], invoice["total"], invoice["tax_rate"]) == (
        "100.00",
        "10.00",
        "110.00",
        "10.00",
    )
    assert invoice["discount"] is None
    assert invoice["invoice_date"] == "2024-03-05"
    assert invoice["line_items"] == [{"description": "Office chairs", "quantity": "2", "unit_price": "50.00",
                                      "amount": "100.00"}]  # fmt: skip
    assert invoice["field_meta"]["total"] == {"confidence": 0.9, "source": "rules", "evidence": "total evidence",
                                              "page": 1, "notes": []}  # fmt: skip
    assert [i["code"] for i in invoice["issues"]] == ["TOTAL_MISMATCH", "LOW_CONFIDENCE", "CURRENCY_INFERRED"]
    assert all(set(issue) == ISSUE_KEYS for issue in invoice["issues"])
    assert invoice["issues"][0]["details"] == {"expected": "100.00", "actual": "120.00"}


def test_document_status_and_results(client: TestClient, corpus: dict[str, Document]) -> None:
    status = client.get(f"{DOCS}/{corpus['queued'].id}/status").json()
    assert set(status) == {"id", "status", "job", "updated_at"}
    assert status["status"] == "queued" and status["job"]["status"] == "queued"

    results = client.get(f"{DOCS}/{corpus['pdf'].id}/results").json()
    assert set(results) == {"document_id", "status", "invoices"}
    assert results["status"] == "needs_review"
    assert [i["invoice_number"] for i in results["invoices"]] == ["INV-2024-001"]


def test_document_pages_are_paginated(client: TestClient, db_session: Session, corpus: dict[str, Document]) -> None:
    seed_pages(db_session, corpus["pdf"], ["page one", "page two", "page three"], method="ocr")

    body = client.get(f"{DOCS}/{corpus['pdf'].id}/pages", params={"offset": 1, "limit": 1}).json()

    assert body["total"] == 3
    assert body["items"] == [
        {"page_number": 2, "label": None, "method": "ocr", "text": "page two", "ocr_confidence": 0.91}
    ]
    _error(client.get(f"{DOCS}/{corpus['pdf'].id}/pages", params={"limit": 500}), 422, "validation_error")


def test_document_events(client: TestClient, db_session: Session, corpus: dict[str, Document]) -> None:
    seed_event(db_session, corpus["pdf"], "parsing", "Parsing started")
    seed_event(db_session, corpus["pdf"], "indexing", "Indexed", level="warning")

    items = client.get(f"{DOCS}/{corpus['pdf'].id}/events").json()["items"]

    assert [(e["stage"], e["level"], e["message"]) for e in items] == [("parsing", "info", "Parsing started"),
                                                                     ("indexing", "warning", "Indexed")]  # fmt: skip
    assert set(items[0]) == {"id", "job_id", "stage", "level", "message", "data", "created_at"}
    assert items[0]["data"] == {"n": 1}


@pytest.mark.parametrize("suffix", ["", "/status", "/results", "/pages", "/events", "/file"])
def test_unknown_document_is_404(client: TestClient, suffix: str) -> None:
    _error(client.get(f"{DOCS}/{uuid.uuid4()}{suffix}"), 404, "not_found")


def test_unknown_document_write_operations_are_404(client: TestClient) -> None:
    _error(client.post(f"{DOCS}/{uuid.uuid4()}/reprocess"), 404, "not_found")
    _error(client.delete(f"{DOCS}/{uuid.uuid4()}"), 404, "not_found")


def test_malformed_document_id_is_422(client: TestClient) -> None:
    error = _error(client.get(f"{DOCS}/not-a-uuid"), 422, "validation_error")
    assert "document_id" in error["details"]["fields"]


def test_download_original_file(client: TestClient) -> None:
    data = pdf_bytes("Download me")
    item = client.post(DOCS, files=[("files", ("my invoice.pdf", data, "application/pdf"))]).json()["items"][0]

    response = client.get(f"{DOCS}/{item['document_id']}/file")

    assert response.status_code == 200
    assert response.content == data
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == (
        "attachment; filename=\"my invoice.pdf\"; filename*=UTF-8''my%20invoice.pdf"
    )
    assert response.headers["x-content-type-options"] == "nosniff"


def test_download_missing_file_is_404(client: TestClient, corpus: dict[str, Document]) -> None:
    _error(client.get(f"{DOCS}/{corpus['pdf'].id}/file"), 404, "not_found")


def test_reprocess_document(client: TestClient, queue: RecordingQueue, db_session: Session,
                            corpus: dict[str, Document]) -> None:  # fmt: skip
    response = client.post(f"{DOCS}/{corpus['failed'].id}/reprocess")

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == str(corpus["failed"].id) and body["status"] == "queued"
    assert queue.job_ids == [UUID(body["job_id"])]
    db_session.expire_all()
    assert db_session.get(Document, corpus["failed"].id).status == "queued"

    _error(client.post(f"{DOCS}/{corpus['failed'].id}/reprocess"), 409, "already_processing")
    assert len(queue.job_ids) == 1


def test_delete_document(client: TestClient, db_session: Session, settings: Settings) -> None:
    item = client.post(DOCS, files=[("files", ("gone.pdf", pdf_bytes("Gone"), "application/pdf"))]).json()["items"][0]
    document_id = UUID(item["document_id"])
    stored = Path(settings.storage_dir) / db_session.get(Document, document_id).storage_key
    assert stored.exists()

    response = client.delete(f"{DOCS}/{item['document_id']}")

    assert response.status_code == 204
    assert response.content == b""
    _error(client.get(f"{DOCS}/{item['document_id']}"), 404, "not_found")
    assert not stored.exists()
    db_session.expire_all()
    assert db_session.scalars(select(ProcessingJob).where(ProcessingJob.document_id == document_id)).all() == []


def test_delete_while_running_is_409(client: TestClient, db_session: Session) -> None:
    running = seed_document(db_session, filename="busy.pdf", status="processing", job_status="running")
    job = db_session.scalars(select(ProcessingJob).where(ProcessingJob.document_id == running.id)).one()
    job.heartbeat_at = datetime.now(UTC)
    db_session.commit()

    _error(client.delete(f"{DOCS}/{running.id}"), 409, "already_processing")
    assert client.get(f"{DOCS}/{running.id}").status_code == 200

    job.heartbeat_at = datetime.now(UTC) - timedelta(days=1)  # the worker died: the job no longer blocks
    db_session.commit()
    assert client.delete(f"{DOCS}/{running.id}").status_code == 204


# ------------------------------------------------------------------ delete: counterparts are re-derived (I1)


def _reconciled(session: Session, *invoices: Invoice) -> None:
    """Derive the cross-document issues and document statuses exactly as processing does."""
    summary = reconcile_invoices(session, list(invoices))
    for document_id in summary.document_ids:
        document = session.get(Document, document_id)
        assert document is not None
        refresh_document_status(session, document)
    session.commit()


def _codes(client: TestClient, invoice: Invoice) -> list[str]:
    return sorted(issue["code"] for issue in client.get(f"/api/v1/invoices/{invoice.id}").json()["issues"])


def test_delete_clears_duplicate_issues_on_the_later_upload(client: TestClient, db_session: Session) -> None:
    first = seed_document(db_session, filename="a.pdf", created_offset_s=0)
    original = seed_invoice(db_session, first)
    later = seed_document(db_session, filename="b.pdf", created_offset_s=10)
    copy = seed_invoice(db_session, later)
    _reconciled(db_session, original, copy)
    assert _codes(client, copy) == ["DUPLICATE_INVOICE"]
    assert client.get(f"{DOCS}/{later.id}").json()["status"] == "needs_review"

    assert client.delete(f"{DOCS}/{first.id}").status_code == 204

    assert _codes(client, copy) == []  # the flag pointed at a document that no longer exists
    assert client.get(f"/api/v1/invoices/{copy.id}").json()["review_status"] == "auto_approved"
    assert client.get(f"{DOCS}/{later.id}").json()["status"] == "completed"


def test_delete_keeps_duplicates_among_the_survivors(client: TestClient, db_session: Session) -> None:
    documents = [seed_document(db_session, filename=f"{n}.pdf", created_offset_s=n * 10) for n in range(3)]
    first, second, third = (seed_invoice(db_session, document) for document in documents)
    _reconciled(db_session, first, second, third)
    assert _codes(client, third) == ["DUPLICATE_INVOICE", "DUPLICATE_INVOICE"]

    assert client.delete(f"{DOCS}/{documents[0].id}").status_code == 204

    assert _codes(client, second) == []  # now the earliest upload of the number
    issues = client.get(f"/api/v1/invoices/{third.id}").json()["issues"]
    assert [(i["code"], i["details"]["other_document_id"]) for i in issues] == [
        ("DUPLICATE_INVOICE", str(documents[1].id))
    ]
    assert client.get(f"{DOCS}/{documents[2].id}").json()["status"] == "needs_review"


def test_delete_clears_cross_verification_mismatch_on_the_ledger(client: TestClient, db_session: Session) -> None:
    pdf = seed_document(db_session, filename="inv.pdf", created_offset_s=0)
    record = seed_invoice(db_session, pdf)
    ledger = seed_document(db_session, filename="ledger.csv", file_kind="csv", created_offset_s=10)
    row = seed_invoice(  # the ledger books another date for the same invoice
        db_session,
        ledger,
        invoice_data(invoice_date=date(2024, 3, 9), line_items=[]),
        source_kind=SourceKind.LEDGER_ROW.value,
    )
    _reconciled(db_session, record, row)
    assert "CROSS_VERIFICATION_MISMATCH" in _codes(client, row)

    assert client.delete(f"{DOCS}/{pdf.id}").status_code == 204

    assert _codes(client, row) == []
    assert client.get(f"{DOCS}/{ledger.id}").json()["status"] == "completed"
    db_session.expire_all()
    assert db_session.scalars(select(CrossVerification)).all() == []


def test_delete_clears_duplicate_file_on_reuploads(client: TestClient, db_session: Session) -> None:
    original = seed_document(db_session, filename="a.pdf", created_offset_s=0)
    reupload = seed_document(db_session, filename="a-again.pdf", created_offset_s=10, duplicate_of_id=original.id)
    seed_invoice(db_session, reupload, invoice_data(invoice_number="INV-OTHER"))
    seed_document_issue(db_session, reupload, "DUPLICATE_FILE", "warning")
    refresh_document_status(db_session, reupload)
    db_session.commit()
    assert client.get(f"{DOCS}/{reupload.id}").json()["status"] == "needs_review"

    assert client.delete(f"{DOCS}/{original.id}").status_code == 204

    detail = client.get(f"{DOCS}/{reupload.id}").json()
    assert detail["duplicate_of_id"] is None
    assert detail["status"] == "completed"
    assert detail["document_issues"] == []
