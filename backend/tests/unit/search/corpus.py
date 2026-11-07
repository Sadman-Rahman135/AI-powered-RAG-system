"""A small synthetic corpus shared by the SQLite unit tests and the PostgreSQL integration twin.

Rows are created by assigning columns directly (never through ``Invoice.apply_invoice_data``), so these tests
do not depend on the extraction normalisers. Documents are indexed through the real ``build_chunks`` and
``index_document`` path, exactly as the pipeline does it.

Layout (invoice number -> what it exercises):

* ``INV-2024-001`` acme_chairs.pdf        single page, summary + page chunk, USD, needs_review
* ``INV-77``       cloudco_hosting.pdf    EUR, auto_approved
* ``CAT-0005``     catering.pdf           GBP, reviewed, a boundary amount (480.50) and date (2024-04-20)
* ``LED-001/002``  ledger.xlsx            two ledger rows on one sheet -> the sheet text is a document-level chunk
* ``M-100/M-200``  multi.pdf              one PDF, two invoices (pages 1-2 and 3) -> page chunks attributed by range
* ``R-9``          receipt_scan.png       image receipt, CAD, no page range (single invoice owns every chunk)
* (no invoice)     notes.pdf              a document without any extracted record
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.models import Document, Invoice
from app.domain.schemas import DocumentType, InvoiceData, LineItem, ReviewStatus
from app.search.chunking import build_chunks
from app.search.embeddings import Embedder
from app.search.service import index_document

BASE_TIME = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class InvoiceSpec:
    number: str
    supplier: str
    customer: str | None
    issued: date
    total: Decimal
    currency: str
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    pages: tuple[int, int] | None = None
    items: tuple[str, ...] = ()
    document_type: DocumentType = DocumentType.INVOICE


@dataclass(frozen=True)
class DocumentSpec:
    key: str
    filename: str
    file_kind: str
    pages: tuple[str, ...]
    invoices: tuple[InvoiceSpec, ...] = ()


SPECS: tuple[DocumentSpec, ...] = (
    DocumentSpec(
        key="acme",
        filename="acme_chairs.pdf",
        file_kind="pdf",
        pages=(
            "Acme Corp\nInvoice INV-2024-001\nDate: 2024-03-05\nBill to: Globex Ltd\n"
            "Description Qty Price Amount\nErgonomic office chairs 2 50.00 100.00\nSubtotal 100.00\n"
            "Tax 10% 10.00\nTotal 110.00 USD\nPayment terms: net 30 days",
        ),
        invoices=(
            InvoiceSpec(
                "INV-2024-001", "Acme Corp", "Globex Ltd", date(2024, 3, 5), Decimal("110.00"), "USD",
                pages=(1, 1), items=("Ergonomic office chairs",),
            ),
        ),
    ),
    DocumentSpec(
        key="cloud",
        filename="cloudco_hosting.pdf",
        file_kind="pdf",
        pages=(
            "CloudCo Hosting GmbH\nInvoice INV-77\nMonthly cloud hosting subscription: dedicated servers, "
            "bandwidth and daily backups\nTotal 2500.00 EUR",
        ),
        invoices=(
            InvoiceSpec(
                "INV-77", "CloudCo Hosting GmbH", "Globex Ltd", date(2024, 2, 10), Decimal("2500.00"), "EUR",
                status=ReviewStatus.AUTO_APPROVED, pages=(1, 1), items=("Cloud hosting subscription",),
            ),
        ),
    ),
    DocumentSpec(
        key="catering",
        filename="catering.pdf",
        file_kind="pdf",
        pages=(
            "Sunrise Catering Ltd\nInvoice CAT-0005\nAnnual dinner buffet with beverages and waiters for forty "
            "guests\nTotal 480.50 GBP",
        ),
        invoices=(
            InvoiceSpec(
                "CAT-0005", "Sunrise Catering Ltd", "Initech", date(2024, 4, 20), Decimal("480.50"), "GBP",
                status=ReviewStatus.REVIEWED, pages=(1, 1), items=("Annual dinner buffet",),
            ),
        ),
    ),
    DocumentSpec(
        key="ledger",
        filename="ledger.xlsx",
        file_kind="xlsx",
        pages=("Invoice No\tSupplier\tTotal\nLED-001\tAcme Corp\t110.00\nLED-002\tInitech\t75.25",),
        invoices=(
            InvoiceSpec("LED-001", "Acme Corp", "Globex Ltd", date(2024, 3, 5), Decimal("110.00"), "USD", pages=(1, 1)),
            InvoiceSpec(
                "LED-002", "Initech", "Hooli", date(2023, 12, 1), Decimal("75.25"), "USD",
                status=ReviewStatus.AUTO_APPROVED, pages=(1, 1),
            ),
        ),
    ),
    DocumentSpec(
        key="multi",
        filename="multi.pdf",
        file_kind="pdf",
        pages=(
            "Umbrella Corp invoice M-100 laboratory equipment zebra",
            "Continued: shipping and handling, payment by wire transfer",
            "Stark Industries invoice M-200 arc reactor maintenance giraffe",
        ),
        invoices=(
            InvoiceSpec(
                "M-100", "Umbrella Corp", "Hooli", date(2024, 1, 15), Decimal("1000.00"), "USD", pages=(1, 2)
            ),
            InvoiceSpec(
                "M-200", "Stark Industries", "Hooli", date(2024, 1, 20), Decimal("2000.00"), "USD", pages=(3, 3)
            ),
        ),
    ),
    DocumentSpec(
        key="receipt",
        filename="receipt_scan.png",
        file_kind="image",
        pages=("Corner Cafe receipt\nEspresso and croissant\nTotal 12.40 CAD",),
        invoices=(
            InvoiceSpec(
                "R-9", "Corner Cafe", None, date(2024, 5, 2), Decimal("12.40"), "CAD",
                status=ReviewStatus.AUTO_APPROVED, document_type=DocumentType.RECEIPT,
            ),
        ),
    ),
    DocumentSpec(
        key="notes",
        filename="notes.pdf",
        file_kind="pdf",
        pages=("Quarterly budget meeting notes: travel expenses and headcount planning",),
    ),
)


@dataclass
class Corpus:
    documents: dict[str, Document]  # by DocumentSpec.key
    invoices: dict[str, Invoice]  # by invoice number


def _norm(number: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", number).upper()


def invoice_data(spec: InvoiceSpec) -> InvoiceData:
    return InvoiceData(
        invoice_number=spec.number,
        invoice_date=spec.issued,
        supplier_name=spec.supplier,
        customer_name=spec.customer,
        total=spec.total,
        currency=spec.currency,
        document_type=spec.document_type,
        line_items=[LineItem(description=item) for item in spec.items],
    )


def build_corpus(session: Session, embedder: Embedder, specs: tuple[DocumentSpec, ...] = SPECS) -> Corpus:
    """Insert and index every spec'd document; creation times increase in spec order."""
    documents: dict[str, Document] = {}
    invoices: dict[str, Invoice] = {}
    tick = 0
    for spec in specs:
        tick += 1
        document = Document(
            filename=spec.filename,
            content_type="application/octet-stream",
            file_kind=spec.file_kind,
            size_bytes=1024,
            sha256=uuid.uuid4().hex * 2,
            storage_key=f"test/{spec.key}",
            status="completed",
            page_count=len(spec.pages),
            created_at=BASE_TIME + timedelta(minutes=tick),
        )
        session.add(document)
        session.flush()
        documents[spec.key] = document

        rows: list[tuple[Invoice, InvoiceSpec]] = []
        for index, inv in enumerate(spec.invoices):
            tick += 1
            row = Invoice(
                document_id=document.id,
                record_index=index,
                page_start=inv.pages[0] if inv.pages else None,
                page_end=inv.pages[1] if inv.pages else None,
                invoice_number=inv.number,
                invoice_number_norm=_norm(inv.number),
                invoice_date=inv.issued,
                supplier_name=inv.supplier,
                customer_name=inv.customer,
                total=inv.total,
                currency=inv.currency,
                document_type=inv.document_type.value,
                line_items=[{"description": item} for item in inv.items],
                review_status=inv.status.value,
                created_at=BASE_TIME + timedelta(minutes=tick),
            )
            session.add(row)
            rows.append((row, inv))
            invoices[inv.number] = row
        session.flush()

        drafts = build_chunks(
            list(enumerate(spec.pages, start=1)), [(row.id, invoice_data(inv)) for row, inv in rows], spec.filename
        )
        index_document(session, document.id, drafts, embedder)
    session.commit()
    return Corpus(documents=documents, invoices=invoices)
