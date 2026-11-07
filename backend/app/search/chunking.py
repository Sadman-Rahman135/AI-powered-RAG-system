"""Turn extracted documents into searchable chunks.

Two kinds of chunk exist per document: one ``invoice_summary`` per extracted record (a short sentence block with
the key fields, so structured facts are findable by keyword and by meaning) and ``page_text`` windows over the
raw page text. Chunks are plain drafts here; ``service.index_document`` embeds and stores them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from app.config import get_settings
from app.domain.schemas import DocumentType, InvoiceData

KIND_PAGE_TEXT = "page_text"
KIND_INVOICE_SUMMARY = "invoice_summary"

_BREAKS = ("\n\n", "\n", ". ", " ")  # preferred split points, strongest first
_HORIZONTAL_RUN = re.compile(r"[ \t\f\v]{3,}")
_BLANK_LINES = re.compile(r"\n(?:[ \t\f\v]*\n){2,}")
_DOCUMENT_LABELS = {
    DocumentType.INVOICE: "Invoice",
    DocumentType.RECEIPT: "Receipt",
    DocumentType.CREDIT_NOTE: "Credit note",
    DocumentType.UNKNOWN: "Document",
}
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_ITEMS_MAX_CHARS = 300


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk before embedding. ``invoice_id`` is set for summaries; page windows are attributed to an invoice
    when stored (``index_document``), because only the stored invoices know their page ranges."""

    kind: str
    text: str
    page_number: int | None = None
    invoice_id: UUID | None = None
    chunk_index: int = 0


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into windows of at most ``max_chars`` characters.

    Windows end at the strongest break available in their second half (blank line, line end, sentence end,
    space) and only a token longer than the window is cut mid-word. Consecutive windows share at least
    ``overlap`` characters and start on a word boundary, so a phrase straddling a cut stays whole in one of them
    (a token longer than the window is the one case where the next window starts inside it).
    Blank text gives no chunks; text that fits gives exactly one. Whitespace runs are bounded first
    (see ``_normalize``).
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if not 0 <= overlap < max_chars:
        raise ValueError(f"overlap must be between 0 and max_chars - 1, got overlap={overlap}, max_chars={max_chars}")
    text = _normalize(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start + max_chars < len(text):
        end = _break_point(text, start + max(max_chars // 2, overlap + 1), start + max_chars)
        window = text[start:end]
        if window.strip():
            chunks.append(window.strip())
        start = _next_start(text, start, start + len(window.rstrip()) - overlap) if overlap else end
    chunks.append(text[start:].strip())
    return chunks


def _normalize(text: str) -> str:
    """Unix newlines, no NUL (invalid in PostgreSQL text) and bounded whitespace runs: layout padding and blank
    spreadsheet rows carry no meaning for search but would otherwise fill whole windows."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return _BLANK_LINES.sub("\n\n", _HORIZONTAL_RUN.sub("  ", text)).strip()


def _break_point(text: str, lowest: int, limit: int) -> int:
    """End (exclusive) of the window: just after the strongest separator found in ``[lowest, limit)``."""
    for separator in _BREAKS:
        index = text.rfind(separator, lowest, limit)
        if index != -1:
            return index + len(separator)
    return limit


def _next_start(text: str, start: int, latest: int) -> int:
    """Start of the following window: the last word start at or before ``latest``, always after ``start`` so the
    scan makes progress. Inside a token longer than the window there is no word start and ``latest`` is used."""
    latest = max(latest, start + 1)
    for index in range(latest, start, -1):
        if text[index - 1].isspace() and not text[index].isspace():
            return index
    return latest


def invoice_summary(data: InvoiceData, filename: str | None = None) -> str:
    """One readable block of the record's key fields, e.g. ``Invoice INV-1 from Acme to Globex, dated
    2024-03-05 (March 2024). Total 110.00 USD (subtotal 100.00, tax 10.00 at 10%). Items: Chair. File: a.pdf.``

    It is both the searchable summary chunk and the snippet shown for filter-only results, so absent fields
    are left out entirely rather than printed as placeholders.
    """
    heading = _DOCUMENT_LABELS[data.document_type]
    if data.invoice_number:
        heading += f" {data.invoice_number}"
    if data.supplier_name:
        heading += f" from {data.supplier_name}"
    if data.customer_name:
        heading += f" to {data.customer_name}"
    if data.invoice_date:
        issued = data.invoice_date
        heading += f", dated {issued.isoformat()} ({_MONTHS[issued.month - 1]} {issued.year})"
    if data.due_date:
        heading += f", due {data.due_date.isoformat()}"
    sentences = [heading, _amounts(data), _items(data)]
    if filename:
        sentences.append(f"File: {filename}")
    return " ".join(f"{sentence.rstrip('.')}." for sentence in sentences if sentence)


def _money(value: Decimal) -> str:
    return format(value, "f")


def _amounts(data: InvoiceData) -> str:
    currency = f" {data.currency}" if data.currency else ""
    details: list[str] = []
    if data.subtotal is not None:
        details.append(f"subtotal {_money(data.subtotal)}")
    if data.discount is not None:
        details.append(f"discount {_money(data.discount)}")
    if data.shipping is not None:
        details.append(f"shipping {_money(data.shipping)}")
    if data.tax is not None:
        tax = f"tax {_money(data.tax)}"
        if data.tax_rate is not None:
            tax += f" at {format(data.tax_rate.normalize(), 'f')}%"
        details.append(tax)
    elif data.tax_rate is not None:
        details.append(f"tax rate {format(data.tax_rate.normalize(), 'f')}%")
    if data.total is not None:
        return f"Total {_money(data.total)}{currency}" + (f" ({', '.join(details)})" if details else "")
    if details:
        return f"Amounts{f' ({data.currency})' if data.currency else ''}: {', '.join(details)}"
    return ""


def _items(data: InvoiceData) -> str:
    """Line-item descriptions, cut to a fixed budget so a 200-line invoice still yields one small summary."""
    descriptions = [i.description.strip() for i in data.line_items if i.description and i.description.strip()]
    shown: list[str] = []
    used = 0
    for description in descriptions:
        if shown and used + len(description) > _ITEMS_MAX_CHARS:
            break
        shown.append(description if len(description) <= _ITEMS_MAX_CHARS else description[: _ITEMS_MAX_CHARS - 1] + "…")
        used += len(description) + 2
    if not shown:
        return ""
    omitted = len(descriptions) - len(shown)
    return "Items: " + "; ".join(shown) + (f"; and {omitted} more" if omitted else "")


def build_chunks(
    pages: Sequence[tuple[int, str]],
    invoices: Sequence[tuple[UUID, InvoiceData]],
    filename: str,
    *,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> list[ChunkDraft]:
    """Drafts for one document: a summary per invoice, then page-text windows in page order.

    ``pages`` are ``(page_number, text)``; window sizes default to ``CHUNK_MAX_CHARS`` / ``CHUNK_OVERLAP``.
    ``chunk_index`` numbers the drafts in the returned order.
    """
    settings = get_settings()
    max_chars = settings.chunk_max_chars if max_chars is None else max_chars
    overlap = settings.chunk_overlap if overlap is None else overlap
    drafts: list[ChunkDraft] = []
    for invoice_id, data in invoices:
        drafts.append(
            ChunkDraft(
                kind=KIND_INVOICE_SUMMARY,
                text=invoice_summary(data, filename),
                invoice_id=invoice_id,
                chunk_index=len(drafts),
            )
        )
    for page_number, text in pages:
        for window in chunk_text(text, max_chars, overlap):
            drafts.append(
                ChunkDraft(kind=KIND_PAGE_TEXT, text=window, page_number=page_number, chunk_index=len(drafts))
            )
    return drafts


__all__ = ["KIND_INVOICE_SUMMARY", "KIND_PAGE_TEXT", "ChunkDraft", "build_chunks", "chunk_text", "invoice_summary"]
