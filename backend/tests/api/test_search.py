"""GET /api/v1/search: keyword / semantic / hybrid ranking and filter-only results over indexed documents."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.api.factories import index, invoice_data, seed_document, seed_invoice

SEARCH = "/api/v1/search"
HIT_KEYS = {
    "document_id", "invoice_id", "page_number", "score", "snippet", "match_type", "filename", "invoice_number",
    "supplier_name", "customer_name", "invoice_date", "total", "currency", "review_status",
}  # fmt: skip


@pytest.fixture
def indexed(db_session: Session) -> dict[str, str]:
    chairs = seed_document(db_session, filename="acme_chairs.pdf")
    seed_invoice(db_session, chairs)
    index(db_session, chairs, ["Acme Corp invoice INV-2024-001 for ergonomic office chairs. Total 110.00 USD"])
    hosting = seed_document(db_session, filename="cloud.pdf", created_offset_s=5)
    seed_invoice(db_session, hosting, invoice_data(invoice_number="INV-77", supplier_name="CloudCo Hosting",
                                                   total=Decimal("2500.00"), currency="EUR"))  # fmt: skip
    index(db_session, hosting, ["CloudCo Hosting monthly server subscription with daily backups. Total 2500.00 EUR"])
    return {"chairs": str(chairs.id), "hosting": str(hosting.id)}


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_search_modes_find_the_matching_document(client: TestClient, indexed: dict[str, str], mode: str) -> None:
    response = client.get(SEARCH, params={"q": "ergonomic chairs", "mode": mode})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"items", "total", "took_ms", "mode"}
    assert body["mode"] == mode and body["total"] >= 1
    top = body["items"][0]
    assert set(top) == HIT_KEYS
    assert top["document_id"] == indexed["chairs"]
    assert top["invoice_number"] == "INV-2024-001" and top["total"] == "110.00"
    assert top["match_type"] in {"keyword", "semantic", "hybrid"}


def test_search_keyword_snippet_highlights(client: TestClient, indexed: dict[str, str]) -> None:
    top = client.get(SEARCH, params={"q": "backups", "mode": "keyword"}).json()["items"][0]

    assert top["document_id"] == indexed["hosting"]
    assert "«backups»" in top["snippet"]


def test_search_filters_apply(client: TestClient, indexed: dict[str, str]) -> None:
    body = client.get(SEARCH, params={"q": "total", "mode": "keyword", "currency": "eur"}).json()

    assert {hit["document_id"] for hit in body["items"]} == {indexed["hosting"]}


def test_search_blank_query_is_filter_only(client: TestClient, indexed: dict[str, str]) -> None:
    body = client.get(SEARCH, params={"supplier": "acme", "total_min": "100"}).json()

    assert body["total"] == 1
    assert body["items"][0]["match_type"] == "filter"
    assert body["items"][0]["document_id"] == indexed["chairs"]
    empty = client.get(SEARCH).json()  # no text and no filters: nothing to match
    assert (empty["items"], empty["total"], empty["mode"]) == ([], 0, "hybrid")


def test_search_pagination(client: TestClient, indexed: dict[str, str]) -> None:
    body = client.get(SEARCH, params={"q": "total", "mode": "keyword", "limit": 1, "offset": 1}).json()

    assert body["total"] == 2 and len(body["items"]) == 1


@pytest.mark.parametrize(
    "params",
    [{"mode": "fuzzy"}, {"limit": 0}, {"limit": 101}, {"offset": -1}, {"total_min": "lots"}, {"date_to": "yesterday"},
     {"file_kind": "doc"}, {"review_status": "done"}, {"q": "x" * 1001}],
)  # fmt: skip
def test_search_rejects_bad_parameters(client: TestClient, params: dict) -> None:
    response = client.get(SEARCH, params=params)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert set(response.json()["error"]["details"]["fields"]) == set(params)
