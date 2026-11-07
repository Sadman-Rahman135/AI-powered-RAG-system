"""End to end through the API with the real inline queue: upload -> process -> results -> correct -> search."""

from __future__ import annotations

from tests.api.conftest import ClientFactory
from tests.api.factories import pdf_bytes

INVOICE_LINES = (
    "Northwind Traders Ltd",
    "12 Harbour Road, Bristol",
    "INVOICE",
    "Invoice No: NW-2024-0042",
    "Invoice Date: 2024-03-05",
    "Bill To: Contoso Pharmaceuticals",
    "Description              Qty    Unit Price    Amount",
    "Lavender soap bars       10     5.00          50.00",
    "Subtotal                                      50.00",
    "VAT 20%                                       10.00",
    "Total                                     GBP 60.00",
)


def test_full_flow_upload_to_search(make_client: ClientFactory) -> None:
    client = make_client(real_queue=True)  # QUEUE_BACKEND=inline: the upload request processes the document

    upload = client.post(
        "/api/v1/documents", files=[("files", ("northwind.pdf", pdf_bytes(*INVOICE_LINES), "application/pdf"))]
    )
    assert upload.status_code == 202
    item = upload.json()["items"][0]
    assert item["status"] == "queued"
    document_id = item["document_id"]

    status = client.get(f"/api/v1/documents/{document_id}/status").json()
    assert status["status"] in ("completed", "needs_review"), status
    assert status["job"]["status"] == "succeeded"

    detail = client.get(f"/api/v1/documents/{document_id}").json()
    assert detail["page_count"] == 1 and detail["processing_ms"] is not None
    assert detail["invoice_count"] >= 1
    invoice = detail["invoices"][0]
    assert invoice["invoice_number"] == "NW-2024-0042"
    assert invoice["total"] == "60.00"

    pages = client.get(f"/api/v1/documents/{document_id}/pages").json()
    assert pages["total"] == 1 and "Lavender" in pages["items"][0]["text"]
    events = client.get(f"/api/v1/documents/{document_id}/events").json()["items"]
    assert events[0]["stage"] == "upload"

    corrected = client.patch(f"/api/v1/invoices/{invoice['id']}", json={"fields": {"customer_name": "Contoso Pharma"}},
                             headers={"X-Actor": "e2e"})  # fmt: skip
    assert corrected.status_code == 200
    assert corrected.json()["customer_name"] == "Contoso Pharma"

    for mode in ("keyword", "semantic", "hybrid"):
        hits = client.get("/api/v1/search", params={"q": "lavender soap", "mode": mode}).json()["items"]
        assert hits and hits[0]["document_id"] == document_id, mode
    filtered = client.get("/api/v1/search", params={"invoice_number": "NW-2024-0042"}).json()["items"]
    assert [hit["match_type"] for hit in filtered] == ["filter"]

    stats = client.get("/api/v1/stats").json()
    assert stats["documents"]["total"] == 1 and stats["processing"]["pages_total"] == 1
