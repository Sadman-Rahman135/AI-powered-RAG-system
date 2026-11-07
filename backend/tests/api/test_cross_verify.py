"""POST /api/v1/cross-verify (manual comparison) and GET /api/v1/cross-verifications."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import CrossVerification, Invoice
from tests.api.factories import invoice_data, seed_cross_verification, seed_document, seed_invoice

VERIFY = "/api/v1/cross-verify"
LIST = "/api/v1/cross-verifications"
CV_KEYS = {
    "id", "left_invoice_id", "right_invoice_id", "left", "right", "kind", "status", "fields", "line_items", "trigger",
    "created_at",
}  # fmt: skip


@pytest.fixture
def pair(db_session: Session) -> dict[str, Invoice]:
    pdf = seed_document(db_session, filename="acme.pdf")
    ledger = seed_document(db_session, filename="ledger.xlsx", file_kind="xlsx", created_offset_s=5)
    other = seed_document(db_session, filename="other.pdf", created_offset_s=9)
    left = seed_invoice(db_session, pdf)
    right = seed_invoice(
        db_session, ledger, invoice_data(total=Decimal("120.00"), line_items=[]), source_kind="ledger_row"
    )
    unrelated = seed_invoice(db_session, ledger, invoice_data(invoice_number="LED-9", line_items=[]), record_index=1,
                             source_kind="ledger_row")  # fmt: skip
    lone = seed_invoice(db_session, other, invoice_data(invoice_number="Z-1"))
    return {"left": left, "right": right, "unrelated": unrelated, "lone": lone}


def test_cross_verify_by_invoice_ids(client: TestClient, pair: dict[str, Invoice], db_session: Session) -> None:
    response = client.post(
        VERIFY, json={"left_invoice_id": str(pair["left"].id), "right_invoice_id": str(pair["right"].id)}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == CV_KEYS
    assert (body["kind"], body["status"], body["trigger"]) == ("pdf_excel", "mismatch", "manual")
    assert body["left"] == {"document_id": str(pair["left"].document_id), "filename": "acme.pdf",
                            "invoice_number": "INV-2024-001"}  # fmt: skip
    assert body["right"]["filename"] == "ledger.xlsx"
    total = next(f for f in body["fields"] if f["field"] == "total")
    assert total == {
        "field": "total",
        "left": "110.00",
        "right": "120.00",
        "status": "mismatch",
        "note": "difference 10.00",
    }
    assert body["line_items"] == {"left_count": 1, "right_count": 0, "matched": 0, "status": "missing"}

    again = client.post(
        VERIFY, json={"left_invoice_id": str(pair["right"].id), "right_invoice_id": str(pair["left"].id)}
    ).json()
    assert again["id"] == body["id"]  # the pair's row is reused, whichever way round it is asked
    assert again["left_invoice_id"] == str(pair["left"].id)  # left = the earlier upload, as the pipeline stores it
    assert db_session.scalar(select(func.count()).select_from(CrossVerification)) == 1


def test_cross_verify_by_document_ids_pairs_by_invoice_number(client: TestClient, pair: dict[str, Invoice]) -> None:
    body = client.post(
        VERIFY,
        json={"left_document_id": str(pair["left"].document_id), "right_document_id": str(pair["right"].document_id)},
    ).json()

    assert (body["left_invoice_id"], body["right_invoice_id"]) == (str(pair["left"].id), str(pair["right"].id))


def test_cross_verify_single_records_compare_directly(client: TestClient, pair: dict[str, Invoice]) -> None:
    body = client.post(
        VERIFY,
        json={"left_document_id": str(pair["left"].document_id), "right_document_id": str(pair["lone"].document_id)},
    ).json()

    assert body["kind"] == "pdf_pdf"
    assert next(f for f in body["fields"] if f["field"] == "invoice_number")["status"] == "mismatch"


def test_cross_verify_documents_without_a_pair_is_422(client: TestClient, pair: dict[str, Invoice]) -> None:
    response = client.post(
        VERIFY,
        json={"left_document_id": str(pair["right"].document_id), "right_document_id": str(pair["lone"].document_id)},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "invoice number" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"left_invoice_id": str(uuid.uuid4())},
        {"left_invoice_id": "x", "right_invoice_id": "y"},
        {
            key: str(uuid.uuid4())
            for key in ("left_invoice_id", "right_invoice_id", "left_document_id", "right_document_id")
        },
    ],
)
def test_cross_verify_rejects_bad_requests(client: TestClient, payload: dict) -> None:
    response = client.post(VERIFY, json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_cross_verify_same_record_is_422(client: TestClient, pair: dict[str, Invoice]) -> None:
    same = str(pair["left"].id)
    assert client.post(VERIFY, json={"left_invoice_id": same, "right_invoice_id": same}).status_code == 422


def test_cross_verify_unknown_ids_are_404(client: TestClient, pair: dict[str, Invoice]) -> None:
    for payload in (
        {"left_invoice_id": str(pair["left"].id), "right_invoice_id": str(uuid.uuid4())},
        {"left_document_id": str(uuid.uuid4()), "right_document_id": str(pair["left"].document_id)},
    ):
        response = client.post(VERIFY, json=payload)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


def test_list_cross_verifications(client: TestClient, db_session: Session, pair: dict[str, Invoice]) -> None:
    seed_cross_verification(db_session, pair["left"], pair["right"], status="mismatch")
    seed_cross_verification(db_session, pair["lone"], pair["unrelated"], status="match")

    def ids(**params) -> list[tuple[str, str]]:
        response = client.get(LIST, params=params)
        assert response.status_code == 200, response.text
        return [(cv["left"]["invoice_number"], cv["status"]) for cv in response.json()["items"]]

    assert sorted(ids()) == [("INV-2024-001", "mismatch"), ("Z-1", "match")]
    assert ids(invoice_id=str(pair["right"].id)) == [("INV-2024-001", "mismatch")]
    assert ids(document_id=str(pair["lone"].document_id)) == [("Z-1", "match")]
    assert ids(document_id=str(pair["right"].document_id), status="match") == [("Z-1", "match")]
    assert set(client.get(LIST).json()["items"][0]) == CV_KEYS
    assert client.get(LIST, params={"status": "same"}).status_code == 422


def test_cross_verify_documents_sharing_several_numbers_stores_every_pair(
    client: TestClient, db_session: Session
) -> None:
    pdf = seed_document(db_session, filename="two_invoices.pdf")
    ledger = seed_document(db_session, filename="ledger.csv", file_kind="csv", created_offset_s=3)
    first = seed_invoice(db_session, pdf, invoice_data(invoice_number="A-1"))
    second = seed_invoice(db_session, pdf, invoice_data(invoice_number="A-2"), record_index=1)
    seed_invoice(db_session, ledger, invoice_data(invoice_number="A-2", line_items=[]), source_kind="ledger_row")
    seed_invoice(db_session, ledger, invoice_data(invoice_number="a1", line_items=[]), record_index=1)

    body = client.post(VERIFY, json={"left_document_id": str(pdf.id), "right_document_id": str(ledger.id)}).json()

    assert body["left_invoice_id"] == str(first.id)  # the first pair in record order
    listed = client.get(LIST, params={"document_id": str(pdf.id)}).json()["items"]
    assert {cv["left_invoice_id"] for cv in listed} == {str(first.id), str(second.id)}
    assert all(cv["trigger"] == "manual" for cv in listed)


def test_cross_verify_reuses_the_automatic_row(
    client: TestClient, db_session: Session, pair: dict[str, Invoice]
) -> None:
    auto = seed_cross_verification(db_session, pair["left"], pair["right"], status="match")

    body = client.post(
        VERIFY, json={"left_invoice_id": str(pair["right"].id), "right_invoice_id": str(pair["left"].id)}
    ).json()

    assert body["id"] == str(auto.id)
    assert (body["left_invoice_id"], body["trigger"], body["status"]) == (str(pair["left"].id), "manual", "mismatch")
    assert db_session.scalar(select(func.count()).select_from(CrossVerification)) == 1


def test_cross_verify_orients_new_pairs_by_upload_order(client: TestClient, pair: dict[str, Invoice]) -> None:
    body = client.post(
        VERIFY, json={"left_invoice_id": str(pair["lone"].id), "right_invoice_id": str(pair["left"].id)}
    ).json()

    assert (body["left_invoice_id"], body["right_invoice_id"]) == (str(pair["left"].id), str(pair["lone"].id))
