"""Ground-truth labels of the synthetic evaluation dataset: model, loader and the labeling conventions.

``labels.json`` is ``{"version": 1, "seed": <int>, "files": [LabeledFile, ...]}``; ``load_labels`` returns the files.
Files are listed in **ingestion order** (see "Groups" below).

Labeling conventions (the ground truth is *exactly what is printed*, normalized):

Values
    Money is a ``Decimal`` with two decimals (JSON string ``"1234.50"``); dates are ISO 8601; ``currency`` is an
    upper-case ISO 4217 code; ``quantity`` and ``tax_rate`` are plain decimals without trailing zeros (``"7.5"``,
    ``"8.25"``, ``"20"``). Text (names, invoice numbers, descriptions) is copied as printed, including case and
    punctuation; a description that wraps over several printed lines is one string with single spaces.
Null
    A value that is not printed anywhere in the document is ``null`` — never inferred, never computed. A receipt
    without a receipt number has ``invoice_number: null``; an invoice without a "Bill To" block has
    ``customer_name: null``; a document that prints no tax line has ``tax: null``; one that prints no currency
    symbol or code has ``currency: null``.
Amounts
    ``subtotal`` is the printed subtotal: the sum of the line amounts **before** the discount. ``discount`` is the
    printed deduction as a positive number; ``shipping`` is a positive number. ``tax`` is the **total** tax printed
    — for the Indian template CGST + SGST, for a two-line sales tax the sum of both lines — whether or not a combined
    line is printed. Tax is levied on ``subtotal - discount`` (shipping is not taxed). ``total`` is the grand total as
    printed. For a clean document ``subtotal - discount + shipping + tax == total`` (within 0.02), the line amounts
    add up to ``subtotal`` and ``quantity * unit_price == amount`` for every line. When a document carries an
    intentional arithmetic error the *printed* (wrong) numbers are the labels and the matching issue code is listed.
``tax_rate``
    Only when one overall percentage is printed (``VAT 20 %``, ``GST @ 18%``, ``Sales Tax (8.25%)``). When a document
    prints several rates (``CGST 9%`` + ``SGST 9%``, state + county tax, 7 % and 19 % VAT) ``tax_rate`` is ``null``.
Dates
    ``invoice_date`` is the issue date, never the due date; the time of day on a receipt is ignored. Numeric formats
    follow the template (US ``MM/DD/YYYY``, EU ``DD.MM.YYYY``, Indian ``DD/MM/YYYY``) so some dates are genuinely
    ambiguous without that context.
Currency
    The ISO code the printed symbol or code implies. A bare ``$`` is only used (and labelled ``USD``) on US documents;
    every CAD and AUD document prints ``CAD``/``AUD`` (or ``C$``/``A$``); ``₹`` and ``Rs.`` are INR.
Document type
    ``invoice`` / ``receipt`` / ``credit_note`` as the printed title says; ledger rows are ``invoice``.

Expected issue codes (derived from the labels, never from the generator's intent)
    ``MISSING_FIELD`` when any required or recommended field is null (for ledger rows ``line_items`` is not expected);
    ``TOTAL_MISMATCH``, ``LINE_ITEMS_SUM_MISMATCH``, ``LINE_ITEM_MATH`` when the printed numbers violate the identities
    above; ``DUPLICATE_INVOICE`` on the later file of a ``dup-N`` pair; ``CROSS_VERIFICATION_MISMATCH`` on both sides of
    a deliberately inconsistent ``xv`` pair. Quality- or system-dependent codes (``LOW_CONFIDENCE``,
    ``OCR_LOW_QUALITY``, ``EXTRACTION_DISAGREEMENT``, ``AMBIGUOUS_DATE``, ``CURRENCY_INFERRED``,
    ``DUPLICATE_FILE``, ``TAX_RATE_*`` ...) are not ground truth; ``EVALUATED_ISSUE_CODES`` lists the codes the labels
    speak about. ``record_issues`` holds the codes per record (parallel to ``records``); ``expected_issue_codes`` is
    their union.
Groups
    ``dup-N``: the same invoice as a digital PDF (listed first, the original) and as a scan/photo (listed second, the
    copy that must be flagged ``DUPLICATE_INVOICE``) — ingest in listed order. ``xv``: one two-sheet XLSX ledger and
    the PDF invoices its rows describe; three rows deliberately disagree with their PDF (different total or date) and
    both sides carry ``CROSS_VERIFICATION_MISMATCH``.
Locations
    ``page_start``/``page_end``: 1-based, inclusive, over the pages (PDF), frames (TIFF) of the file;
    ``sheet_name``/``row_number`` (1-based row in the sheet, or line in a CSV): ledger rows and sheet invoices;
    ``source_kind``: ``invoice_document`` or ``ledger_row``.
Files
    ``kind`` is the true content type (``pdf png jpg tiff xlsx xls csv``) or ``corrupt`` for files that must fail; a
    PNG renamed ``.pdf`` keeps ``kind: "png"`` and ``expect: "success"`` because the type is sniffed from bytes.
    ``scale/…`` files (only with ``--scale``) mix templates (``template: "mixed"``, split ``na``) and exist to
    measure throughput; their labels carry page ranges like any other multi-invoice PDF.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.schemas import (
    RECOMMENDED_FIELDS,
    REQUIRED_FIELDS,
    InvoiceData,
    IssueCode,
    SourceKind,
    field_is_present,
)

LABELS_VERSION = 1
AMOUNT_TOLERANCE = Decimal("0.02")

LOCATION_KEYS: tuple[str, ...] = ("page_start", "page_end", "sheet_name", "row_number", "source_kind")
QUALITIES: tuple[str, ...] = ("clean", "scan_light", "scan_heavy", "photo", "lowres")
ERROR_CODES: tuple[str, ...] = ("empty_file", "corrupt_file", "encrypted_file", "unsupported_file_type")

# The issue codes the dataset has ground truth for.
EVALUATED_ISSUE_CODES: tuple[IssueCode, ...] = (
    IssueCode.MISSING_FIELD,
    IssueCode.TOTAL_MISMATCH,
    IssueCode.LINE_ITEMS_SUM_MISMATCH,
    IssueCode.LINE_ITEM_MATH,
    IssueCode.DUPLICATE_INVOICE,
    IssueCode.CROSS_VERIFICATION_MISMATCH,
)
ARITHMETIC_CODES: tuple[IssueCode, ...] = (
    IssueCode.TOTAL_MISMATCH,
    IssueCode.LINE_ITEMS_SUM_MISMATCH,
    IssueCode.LINE_ITEM_MATH,
)


class LabeledRecord(InvoiceData):
    """One expected extraction result: the invoice fields plus where in the file the record lives."""

    page_start: int | None = None
    page_end: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    source_kind: SourceKind = SourceKind.INVOICE_DOCUMENT


class LabeledFile(BaseModel):
    """One dataset file with everything the evaluation needs to score it."""

    model_config = ConfigDict(extra="forbid")

    file: str  # posix path relative to the dataset root
    kind: str  # pdf | png | jpg | tiff | xlsx | xls | csv | corrupt
    template: str | None  # T1..T6, "ledger", "mixed", or None (corrupt)
    split: Literal["dev", "heldout", "na"]
    quality: Literal["clean", "scan_light", "scan_heavy", "photo", "lowres"]
    expect: Literal["success", "failure"]
    expected_error_code: Literal["empty_file", "corrupt_file", "encrypted_file", "unsupported_file_type"] | None = None
    records: list[LabeledRecord] = Field(default_factory=list)
    expected_issue_codes: list[IssueCode] = Field(default_factory=list)
    group: str | None = None
    pages: int = 0  # pages (PDF), frames (TIFF/PNG/JPG = 1), sheets (workbook / CSV = 1)
    record_issues: list[list[IssueCode]] = Field(default_factory=list)  # parallel to ``records``


def arithmetic_issue_codes(data: InvoiceData) -> set[IssueCode]:
    """The arithmetic identities the printed numbers violate (tolerance 0.02)."""
    codes: set[IssueCode] = set()
    if data.subtotal is not None and data.total is not None:
        expected = data.subtotal - (data.discount or 0) + (data.shipping or 0) + (data.tax or 0)
        if abs(expected - data.total) > AMOUNT_TOLERANCE:
            codes.add(IssueCode.TOTAL_MISMATCH)
    amounts = [item.amount for item in data.line_items if item.amount is not None]
    complete = bool(amounts) and len(amounts) == len(data.line_items)
    if data.subtotal is not None and complete and abs(sum(amounts, Decimal(0)) - data.subtotal) > AMOUNT_TOLERANCE:
        codes.add(IssueCode.LINE_ITEMS_SUM_MISMATCH)
    for item in data.line_items:
        if item.quantity is None or item.unit_price is None or item.amount is None:
            continue
        if abs(item.quantity * item.unit_price - item.amount) > AMOUNT_TOLERANCE:
            codes.add(IssueCode.LINE_ITEM_MATH)
    return codes


def missing_field_applies(data: InvoiceData, source_kind: SourceKind) -> bool:
    """True when a required or recommended field is null (ledger rows are not expected to carry line items)."""
    names = [*REQUIRED_FIELDS, *RECOMMENDED_FIELDS]
    if source_kind is SourceKind.LEDGER_ROW:
        names = [n for n in names if n != "line_items"]
    return any(not field_is_present(data, name) for name in names)


def issue_codes_for(data: InvoiceData, source_kind: SourceKind) -> list[IssueCode]:
    """The expected issue codes of one record that follow from its labels alone, sorted by code."""
    codes = arithmetic_issue_codes(data)
    if missing_field_applies(data, source_kind):
        codes.add(IssueCode.MISSING_FIELD)
    return sorted(codes, key=lambda c: c.value)


def labels_payload(seed: int, files: list[LabeledFile]) -> dict[str, Any]:
    return {"version": LABELS_VERSION, "seed": seed, "files": [f.model_dump(mode="json") for f in files]}


def write_labels(path: Path, seed: int, files: list[LabeledFile]) -> None:
    """Write ``labels.json`` deterministically (UTF-8, stable key order)."""
    text = json.dumps(labels_payload(seed, files), ensure_ascii=False, indent=1)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def load_labels(path: Path | str) -> list[LabeledFile]:
    """Load and validate ``labels.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != LABELS_VERSION:
        raise ValueError(f"unsupported labels version: {data.get('version')!r}")
    return [LabeledFile.model_validate(entry) for entry in data["files"]]
