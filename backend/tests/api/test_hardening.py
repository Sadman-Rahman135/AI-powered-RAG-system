"""Hardening: no data in error logs, ASCII-only number/date parsing, control characters, bounded paging and
amounts, startup validation of rate limits, review writes refused while processing, row locks and lock order."""

from __future__ import annotations

import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api import corrections
from app.pipeline import intake
from tests.api.conftest import ClientFactory
from tests.api.factories import pdf_bytes, seed_document, seed_invoice

SECRET_SUPPLIER = "Secret Supplier GmbH"
SECRET_NOTE = "private note"
SECRET_FILENAME = "Confidential Payroll 2026.pdf"


def _db_error(*values: str) -> OperationalError:
    """A driver error whose text carries the bound values, as SQLAlchemy builds it without ``hide_parameters``."""
    try:
        raise ValueError(f"value {values[0]!r} rejected")  # a chained message carrying data too
    except ValueError as cause:
        error = OperationalError("UPDATE invoices SET supplier_name=?, comment=?", values, cause)
        error.__cause__ = cause
        return error


# ------------------------------------------------------------------------------- 1. no data in error logs


def test_db_error_during_patch_logs_no_submitted_values(
    make_client: ClientFactory, db_session: Session, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = make_client(raise_server_exceptions=False)
    invoice = seed_invoice(db_session, seed_document(db_session))

    def failing_revalidate(_session: Session, _invoice: object) -> None:
        raise _db_error(SECRET_SUPPLIER, SECRET_NOTE)

    monkeypatch.setattr(corrections, "revalidate_invoice", failing_revalidate)
    with caplog.at_level(logging.INFO):
        response = client.patch(
            f"/api/v1/invoices/{invoice.id}",
            json={"fields": {"supplier_name": SECRET_SUPPLIER}, "comment": SECRET_NOTE},
            headers={"X-Request-ID": "req-db-patch"},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert SECRET_SUPPLIER not in caplog.text and SECRET_NOTE not in caplog.text
    assert "OperationalError" in caplog.text and "req-db-patch" in caplog.text  # still diagnosable


def test_db_error_during_upload_logs_no_filename_or_values(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_create(*_args: object, **_kwargs: object) -> None:
        raise _db_error(SECRET_FILENAME, SECRET_NOTE)

    monkeypatch.setattr(intake, "create_document", failing_create)
    with caplog.at_level(logging.INFO):
        response = client.post(
            "/api/v1/documents", files=[("files", (SECRET_FILENAME, pdf_bytes("x"), "application/pdf"))]
        )

    item = response.json()["items"][0]
    assert (item["status"], item["error"]["code"]) == ("rejected", "internal_error")
    assert SECRET_FILENAME not in caplog.text and SECRET_NOTE not in caplog.text
    assert "OperationalError" in caplog.text


# ------------------------------------------------------------------------ 2. ASCII digits, 3. control chars


@pytest.fixture
def record(db_session: Session) -> str:
    return str(seed_invoice(db_session, seed_document(db_session)).id)


@pytest.mark.parametrize(
    ("fields", "name"),
    [
        ({"total": "١٢٣.٠٠"}, "total"),  # Arabic-Indic digits
        ({"total": "１２３"}, "total"),  # full-width digits
        ({"invoice_date": "２０２４-０３-０５"}, "invoice_date"),
        ({"due_date": "٢٠٢٤-٠٣-٠٥"}, "due_date"),
        ({"invoice_number": "A B"}, "invoice_number"),
        ({"supplier_name": "Acme\u0085Corp"}, "supplier_name"),  # C1 (NEL)
        ({"customer_name": "Glo\tbex"}, "customer_name"),
    ],
)
def test_patch_rejects_non_ascii_digits_and_control_characters(
    client: TestClient, record: str, fields: dict, name: str
) -> None:
    response = client.patch(f"/api/v1/invoices/{record}", json={"fields": fields})

    assert response.status_code == 422, response.text
    assert name in response.json()["error"]["details"]["fields"]


@pytest.mark.parametrize("comment", ["a\x00b", "x\x01", "line break", "c1\x9b"])
def test_comments_reject_control_characters(client: TestClient, record: str, comment: str) -> None:
    patch = client.patch(f"/api/v1/invoices/{record}", json={"fields": {"total": "111.00"}, "comment": comment})
    approve = client.post(f"/api/v1/invoices/{record}/approve", json={"comment": comment})

    for response in (patch, approve):
        assert response.status_code == 422, response.text
        assert "comment" in response.json()["error"]["details"]["fields"]
    assert client.get(f"/api/v1/invoices/{record}").json()["review_status"] == "auto_approved"


def test_comments_keep_line_breaks(client: TestClient, record: str) -> None:
    response = client.patch(f"/api/v1/invoices/{record}", json={"fields": {"total": "111.00"}, "comment": "two\nlines"})

    assert response.status_code == 200
    assert client.get(f"/api/v1/invoices/{record}/corrections").json()["items"][0]["comment"] == "two\nlines"


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/api/v1/documents", {"q": "a\x00b"}),
        ("/api/v1/invoices", {"q": "a\x00b"}),
        ("/api/v1/invoices", {"supplier": "x\x00"}),
        ("/api/v1/invoices", {"invoice_number": "\x1b[31m"}),
        ("/api/v1/search", {"q": "a\x00b"}),
        ("/api/v1/search", {"customer": "\x00"}),
        ("/api/v1/search", {"currency": "U\x00D"}),
    ],
)
def test_query_strings_reject_control_characters(client: TestClient, path: str, params: dict) -> None:
    response = client.get(path, params=params)

    assert response.status_code == 422, response.text
    assert set(response.json()["error"]["details"]["fields"]) == set(params)


# ------------------------------------------------------------------------------- 4. bounded paging and totals


@pytest.mark.parametrize("path", ["/api/v1/documents", "/api/v1/invoices", "/api/v1/jobs", "/api/v1/search?q=x",
                                  "/api/v1/cross-verifications"])  # fmt: skip
def test_offset_is_bounded(client: TestClient, path: str) -> None:
    assert client.get(path, params={"offset": 10**20}).status_code == 422
    assert client.get(path, params={"offset": 1_000_001}).status_code == 422
    assert client.get(path, params={"offset": 1_000_000}).status_code == 200


@pytest.mark.parametrize(
    "value", ["1e999999", "1e3", "10000000000000000", "-10000000000000000", "١٠٠", "NaN", "Infinity"]
)
@pytest.mark.parametrize("path", ["/api/v1/invoices", "/api/v1/search"])
def test_total_filters_are_finite_plain_decimals(client: TestClient, path: str, value: str) -> None:
    for name in ("total_min", "total_max"):
        response = client.get(path, params={name: value})
        assert response.status_code == 422, (name, value)
        assert name in response.json()["error"]["details"]["fields"]
    assert (
        client.get(path, params={"total_min": "-1000000000000000", "total_max": "1000000000000000"}).status_code == 200
    )


# ------------------------------------------------------------------------------ 5. rate limits checked at start


@pytest.mark.parametrize("name", ["RATE_LIMIT_DEFAULT", "RATE_LIMIT_UPLOAD", "RATE_LIMIT_SEARCH"])
def test_invalid_rate_limit_settings_fail_at_startup(make_client: ClientFactory, name: str) -> None:
    with pytest.raises(RuntimeError, match=name):
        make_client(env={name: "lots per minute"})


# ------------------------------------------------------------- 6. no review writes while processing, row lock


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_review_writes_refused_while_the_document_is_processed(
    client: TestClient, db_session: Session, status: str
) -> None:
    invoice = seed_invoice(db_session, seed_document(db_session, status=status, job_status="running"))

    for response in (
        client.patch(f"/api/v1/invoices/{invoice.id}", json={"fields": {"total": "1.00"}}),
        client.post(f"/api/v1/invoices/{invoice.id}/approve", json={}),
    ):
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "already_processing"
    assert client.get(f"/api/v1/invoices/{invoice.id}/corrections").json()["items"] == []
    assert client.get(f"/api/v1/invoices/{invoice.id}").json()["total"] == "110.00"


def test_review_writes_lock_the_invoice_row() -> None:
    statement = corrections.locked_invoice(uuid.uuid4())

    assert "FOR UPDATE" in str(statement.compile(dialect=postgresql.dialect()))


# ---------------------------------------------------------------------------------------- 8. lock order


def test_review_writes_take_the_reconciliation_lock_first(
    client: TestClient, record: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    began_before_lock: list[bool] = []
    real = corrections.serialize_reconciliation

    def spy(session: Session) -> None:
        began_before_lock.append(session.in_transaction())
        real(session)

    monkeypatch.setattr(corrections, "serialize_reconciliation", spy)
    assert client.patch(f"/api/v1/invoices/{record}", json={"fields": {"total": "111.00"}}).status_code == 200
    assert client.post(f"/api/v1/invoices/{record}/approve", json={}).status_code == 200

    assert began_before_lock == [False, False]  # the lock is the first statement of each transaction


# ------------------------------------------------------------------------------------------- 7. bad JSON


def test_malformed_json_is_reported_on_the_body(client: TestClient, record: str) -> None:
    response = client.patch(
        f"/api/v1/invoices/{record}", content=b'{"fields": {"total": "1"', headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert list(response.json()["error"]["details"]["fields"]) == ["body"]
