"""Invoice endpoints: list/filter, get, PATCH corrections (audited and re-validated), approve, correction history."""

from __future__ import annotations

import hashlib
import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Correction, Document, Invoice
from app.domain.schemas import LineItem
from tests.api.conftest import ClientFactory
from tests.api.factories import invoice_data, seed_document, seed_invoice

INVOICES = "/api/v1/invoices"


def _error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    return error


@pytest.fixture
def records(db_session: Session) -> dict[str, Invoice]:
    pdf = seed_document(db_session, filename="acme.pdf", status="needs_review")
    ledger = seed_document(db_session, filename="ledger.xlsx", file_kind="xlsx", created_offset_s=5)
    acme = seed_invoice(db_session, pdf, review_status="needs_review", created_offset_s=1,
                        issues=(("TOTAL_MISMATCH", "error", "total"),))  # fmt: skip
    globex = seed_invoice(
        db_session, ledger,
        invoice_data(invoice_number="LED-7", supplier_name="Initech", customer_name="Hooli", total=Decimal("75.25"),
                     subtotal=Decimal("75.25"), tax=None, tax_rate=None, currency="EUR", line_items=[],
                     invoice_date=invoice_data().invoice_date.replace(year=2023)),
        source_kind="ledger_row", created_offset_s=2, issues=(("LOW_CONFIDENCE", "warning", "tax"),),
    )  # fmt: skip
    eur = seed_invoice(
        db_session, ledger,
        invoice_data(invoice_number="LED-8", supplier_name="Umbrella Corp", total=Decimal("2500.00"),
                     subtotal=Decimal("2500.00"), tax=None, tax_rate=None, currency="EUR", line_items=[]),
        record_index=1, source_kind="ledger_row", created_offset_s=3, review_status="reviewed",
    )  # fmt: skip
    return {"acme": acme, "initech": globex, "umbrella": eur}


def test_list_invoices_shape(client: TestClient, records: dict[str, Invoice]) -> None:
    body = client.get(INVOICES).json()

    assert set(body) == {"items", "total", "limit", "offset"}
    assert body["total"] == 3
    assert [i["invoice_number"] for i in body["items"]] == ["LED-8", "LED-7", "INV-2024-001"]
    assert body["items"][2]["filename"] == "acme.pdf"
    assert [i["code"] for i in body["items"][2]["issues"]] == ["TOTAL_MISMATCH"]


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"q": "umbrella"}, ["LED-8"]),
        ({"q": "ledger"}, ["LED-8", "LED-7"]),
        ({"supplier": "acme"}, ["INV-2024-001"]),
        ({"customer": "hooli"}, ["LED-7"]),
        ({"invoice_number": "led-7"}, ["LED-7"]),
        ({"invoice_number": "LED7"}, ["LED-7"]),
        ({"currency": "eur"}, ["LED-8", "LED-7"]),
        ({"date_from": "2024-01-01"}, ["LED-8", "INV-2024-001"]),
        ({"date_to": "2023-12-31"}, ["LED-7"]),
        ({"total_min": "100", "total_max": "200"}, ["INV-2024-001"]),
        ({"review_status": "reviewed"}, ["LED-8"]),
        ({"has_errors": "true"}, ["INV-2024-001"]),
        ({"has_errors": "false"}, ["LED-8", "LED-7"]),
        ({"sort": "total"}, ["LED-7", "INV-2024-001", "LED-8"]),
        ({"sort": "-total", "limit": 1, "offset": 1}, ["INV-2024-001"]),
    ],
)
def test_list_invoices_filters(
    client: TestClient, records: dict[str, Invoice], params: dict, expected: list[str]
) -> None:
    response = client.get(INVOICES, params=params)

    assert response.status_code == 200, response.text
    assert [i["invoice_number"] for i in response.json()["items"]] == expected


@pytest.mark.parametrize(
    "params",
    [{"total_min": "abc"}, {"date_from": "05/03/2024"}, {"review_status": "maybe"}, {"has_errors": "perhaps"},
     {"sort": "secret"}, {"limit": 1000}],
)  # fmt: skip
def test_list_invoices_rejects_bad_parameters(client: TestClient, params: dict) -> None:
    error = _error(client.get(INVOICES, params=params), 422, "validation_error")
    assert set(error["details"]["fields"]) == set(params)


def test_get_invoice_and_404(client: TestClient, records: dict[str, Invoice]) -> None:
    body = client.get(f"{INVOICES}/{records['initech'].id}").json()
    assert body["source_kind"] == "ledger_row"
    assert body["total"] == "75.25" and body["tax"] is None and body["line_items"] == []

    for path in ("", "/corrections"):
        _error(client.get(f"{INVOICES}/{uuid.uuid4()}{path}"), 404, "not_found")
    _error(client.patch(f"{INVOICES}/{uuid.uuid4()}", json={"fields": {"total": "1.00"}}), 404, "not_found")
    _error(client.post(f"{INVOICES}/{uuid.uuid4()}/approve", json={}), 404, "not_found")


def test_correction_audited_and_revalidated(client: TestClient, db_session: Session) -> None:
    document = seed_document(db_session, filename="wrong_total.pdf", status="needs_review")
    invoice = seed_invoice(
        db_session, document, invoice_data(total=Decimal("120.00")), review_status="needs_review",
        issues=(("TOTAL_MISMATCH", "error", "total"),),
    )  # fmt: skip

    response = client.patch(
        f"{INVOICES}/{invoice.id}",
        json={"fields": {"total": "110.00", "supplier_name": "  ACME Corporation  "}, "comment": "Fixed from the PDF"},
        headers={"X-Actor": " alice\x07 "},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == "110.00" and body["supplier_name"] == "ACME Corporation"
    assert body["field_meta"]["total"] == {"confidence": 1.0, "source": "manual", "evidence": None, "page": None,
                                           "notes": []}  # fmt: skip
    assert body["field_meta"]["supplier_name"]["source"] == "manual"
    assert body["field_meta"]["invoice_number"]["source"] == "rules"  # untouched fields keep their metadata
    assert "TOTAL_MISMATCH" not in {issue["code"] for issue in body["issues"]}
    assert body["review_status"] == "auto_approved"

    history = client.get(f"{INVOICES}/{invoice.id}/corrections").json()["items"]
    assert [(c["field"], c["old_value"], c["new_value"]) for c in history] == [
        ("supplier_name", "Acme Corp", "ACME Corporation"),
        ("total", "120.00", "110.00"),
    ]
    assert all(c["corrected_by"] == "alice" and c["comment"] == "Fixed from the PDF" for c in history)
    assert set(history[0]) == {"id", "field", "old_value", "new_value", "comment", "corrected_by", "created_at"}

    db_session.expire_all()
    stored = db_session.get(Invoice, invoice.id)
    assert stored.total == Decimal("110.00")
    assert stored.supplier_name_norm  # normalized lookup keys follow the correction
    assert len(db_session.scalars(select(Correction).where(Correction.invoice_id == invoice.id)).all()) == 2
    assert db_session.get(Document, document.id).status == "completed"


def test_patch_line_items_replaces_them(client: TestClient, db_session: Session) -> None:
    document = seed_document(db_session)
    invoice = seed_invoice(db_session, document)
    new_items = [
        {"description": "Chair", "quantity": "1", "unit_price": "60.00", "amount": "60.00"},
        {"description": "Desk", "quantity": "1", "unit_price": "40.00", "amount": "40.00"},
    ]

    body = client.patch(f"{INVOICES}/{invoice.id}", json={"fields": {}, "line_items": new_items}).json()

    assert body["line_items"] == new_items
    assert body["field_meta"]["line_items"]["source"] == "manual"
    (correction,) = client.get(f"{INVOICES}/{invoice.id}/corrections").json()["items"]
    assert correction["field"] == "line_items"
    old_item = LineItem(
        description="Office chairs", quantity=Decimal("2"), unit_price=Decimal("50.00"), amount=Decimal("100.00")
    )
    assert correction["old_value"] == [old_item.model_dump(mode="json")]
    assert correction["new_value"] == new_items
    assert correction["corrected_by"] == "anonymous"


def test_patch_without_real_changes_writes_no_correction(client: TestClient, db_session: Session) -> None:
    document = seed_document(db_session)
    invoice = seed_invoice(db_session, document)

    body = client.patch(f"{INVOICES}/{invoice.id}", json={"fields": {"total": "110.0", "currency": "USD"}}).json()

    assert body["total"] == "110.00"
    assert client.get(f"{INVOICES}/{invoice.id}/corrections").json()["items"] == []


def test_patch_can_clear_a_field(client: TestClient, db_session: Session) -> None:
    document = seed_document(db_session)
    invoice = seed_invoice(db_session, document)

    body = client.patch(f"{INVOICES}/{invoice.id}", json={"fields": {"due_date": None, "document_type": None}}).json()

    assert body["due_date"] is None and body["document_type"] == "unknown"
    history = client.get(f"{INVOICES}/{invoice.id}/corrections").json()["items"]
    assert [(c["field"], c["old_value"], c["new_value"]) for c in history] == [
        ("due_date", "2024-04-04", None),
        ("document_type", "invoice", "unknown"),
    ]


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"fields": {"total": "1,234.50"}}, "total"),
        ({"fields": {"total": 12.5}}, "total"),
        ({"fields": {"total": "1e3"}}, "total"),
        ({"fields": {"subtotal": "10.12345"}}, "subtotal"),
        ({"fields": {"tax": "123456789012345.00"}}, "tax"),
        ({"fields": {"invoice_date": "05/03/2024"}}, "invoice_date"),
        ({"fields": {"due_date": "2024-02-30"}}, "due_date"),
        ({"fields": {"currency": "usd"}}, "currency"),
        ({"fields": {"currency": "US Dollar"}}, "currency"),
        ({"fields": {"document_type": "bill"}}, "document_type"),
        ({"fields": {"supplier_name": "x" * 300}}, "supplier_name"),
        ({"fields": {"invoice_number": "A\x00B"}}, "invoice_number"),
        ({"fields": {"storage_key": "../x"}}, "storage_key"),
        ({"fields": {"total": "1.00"}, "status": "approved"}, "status"),
        ({"line_items": [{"description": "x", "amount": "ten"}]}, "line_items.0.amount"),
        ({"line_items": [{"description": "x", "colour": "red"}]}, "line_items.0.colour"),
    ],
)
def test_patch_rejects_bad_values(client: TestClient, db_session: Session, payload: dict, field: str) -> None:
    document = seed_document(db_session)
    invoice = seed_invoice(db_session, document)

    error = _error(client.patch(f"{INVOICES}/{invoice.id}", json=payload), 422, "validation_error")

    assert field in error["details"]["fields"], error
    assert client.get(f"{INVOICES}/{invoice.id}").json()["total"] == "110.00"
    assert client.get(f"{INVOICES}/{invoice.id}/corrections").json()["items"] == []


def test_corrected_by_uses_api_key_fingerprint(make_client: ClientFactory, db_session: Session) -> None:
    key = "k" * 32
    client = make_client(env={"API_KEYS": key})
    document = seed_document(db_session)
    invoice = seed_invoice(db_session, document)

    client.patch(f"{INVOICES}/{invoice.id}", json={"fields": {"total": "111.00"}}, headers={"X-API-Key": key})

    (correction,) = client.get(f"{INVOICES}/{invoice.id}/corrections", headers={"X-API-Key": key}).json()["items"]
    assert correction["corrected_by"] == "api-key:" + hashlib.sha256(key.encode()).hexdigest()[:8]


def test_approve_invoice(client: TestClient, db_session: Session) -> None:
    document = seed_document(db_session, status="needs_review")
    invoice = seed_invoice(db_session, document, review_status="needs_review",
                           issues=(("LOW_CONFIDENCE", "warning", "supplier_name"),))  # fmt: skip

    response = client.post(f"{INVOICES}/{invoice.id}/approve", json={"comment": "Checked"}, headers={"X-Actor": "bob"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["review_status"] == "reviewed"
    assert body["reviewed_by"] == "bob" and body["reviewed_at"]
    assert [i["code"] for i in body["issues"]] == ["LOW_CONFIDENCE"]  # approving keeps the issues visible
    (entry,) = client.get(f"{INVOICES}/{invoice.id}/corrections").json()["items"]
    assert (entry["field"], entry["old_value"], entry["new_value"], entry["comment"], entry["corrected_by"]) == (
        "review_status", "needs_review", "reviewed", "Checked", "bob")  # fmt: skip
    db_session.expire_all()
    assert db_session.get(Document, document.id).status == "completed"

    again = client.post(f"{INVOICES}/{invoice.id}/approve")  # body is optional; approving twice changes nothing
    assert again.status_code == 200 and again.json()["reviewed_by"] == "bob"
    assert len(client.get(f"{INVOICES}/{invoice.id}/corrections").json()["items"]) == 1


@pytest.mark.parametrize("comment", [None, "   "])
def test_a_null_or_blank_comment_is_no_comment(client: TestClient, db_session: Session, comment: str | None) -> None:
    # The UI sends {"comment": null} when the reviewer leaves the box empty; the length limit must not see None.
    document = seed_document(db_session, status="needs_review")
    invoice = seed_invoice(db_session, document, review_status="needs_review")

    patched = client.patch(f"{INVOICES}/{invoice.id}", json={"fields": {"supplier_name": "Acme"}, "comment": comment})
    assert patched.status_code == 200, patched.text
    approved = client.post(f"{INVOICES}/{invoice.id}/approve", json={"comment": comment})
    assert approved.status_code == 200, approved.text

    entries = client.get(f"{INVOICES}/{invoice.id}/corrections").json()["items"]
    assert [entry["comment"] for entry in entries] == [None, None]
    too_long = client.post(f"{INVOICES}/{invoice.id}/approve", json={"comment": "x" * 2001})
    _error(too_long, 422, "validation_error")
