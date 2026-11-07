"""Shared domain models: the structured invoice record, per-field metadata and validation issues.

Rules for every producer of these models:
- a value that is not printed in the document is ``None`` — never a guess;
- money is ``Decimal`` (serialized to JSON as a plain-notation string, e.g. ``"1234.50"``);
- dates are ``datetime.date``; currency is an upper-case ISO 4217 code;
- confidence is a float in [0, 1].
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer


def _decimal_to_str(value: Decimal) -> str:
    return format(value, "f")


# Decimal that always serializes to JSON as plain notation ("100", never "1E+2").
Money = Annotated[Decimal, PlainSerializer(_decimal_to_str, return_type=str, when_used="json")]


class DocumentType(StrEnum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CREDIT_NOTE = "credit_note"
    UNKNOWN = "unknown"


class SourceKind(StrEnum):
    INVOICE_DOCUMENT = "invoice_document"  # an invoice/receipt document (PDF page range, image, invoice-layout sheet)
    LEDGER_ROW = "ledger_row"  # one row of a spreadsheet ledger listing many invoices


class ExtractionSource(StrEnum):
    RULES = "rules"
    LLM = "llm"
    HYBRID = "rules+llm"
    SPREADSHEET = "spreadsheet"
    MANUAL = "manual"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ReviewStatus(StrEnum):
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"


class IssueCode(StrEnum):
    # errors
    MISSING_FIELD = "MISSING_FIELD"  # error for REQUIRED_FIELDS, warning for RECOMMENDED_FIELDS
    INVALID_AMOUNT = "INVALID_AMOUNT"
    INVALID_DATE = "INVALID_DATE"
    TOTAL_MISMATCH = "TOTAL_MISMATCH"
    LINE_ITEMS_SUM_MISMATCH = "LINE_ITEMS_SUM_MISMATCH"
    LINE_ITEM_MATH = "LINE_ITEM_MATH"
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    CROSS_VERIFICATION_MISMATCH = "CROSS_VERIFICATION_MISMATCH"
    # warnings
    NEGATIVE_AMOUNT = "NEGATIVE_AMOUNT"
    FUTURE_DATE = "FUTURE_DATE"
    TAX_RATE_MISMATCH = "TAX_RATE_MISMATCH"
    TAX_RATE_UNUSUAL = "TAX_RATE_UNUSUAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    EXTRACTION_DISAGREEMENT = "EXTRACTION_DISAGREEMENT"
    DUPLICATE_FILE = "DUPLICATE_FILE"
    OCR_LOW_QUALITY = "OCR_LOW_QUALITY"
    POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"
    # info
    AMBIGUOUS_DATE = "AMBIGUOUS_DATE"
    CURRENCY_INFERRED = "CURRENCY_INFERRED"
    UNGROUNDED_VALUE_REJECTED = "UNGROUNDED_VALUE_REJECTED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"


# Default severity per code (MISSING_FIELD varies by field; see REQUIRED_FIELDS / RECOMMENDED_FIELDS).
DEFAULT_SEVERITY: dict[IssueCode, Severity] = {
    IssueCode.MISSING_FIELD: Severity.ERROR,
    IssueCode.INVALID_AMOUNT: Severity.ERROR,
    IssueCode.INVALID_DATE: Severity.ERROR,
    IssueCode.TOTAL_MISMATCH: Severity.ERROR,
    IssueCode.LINE_ITEMS_SUM_MISMATCH: Severity.ERROR,
    IssueCode.LINE_ITEM_MATH: Severity.ERROR,
    IssueCode.DUPLICATE_INVOICE: Severity.ERROR,
    IssueCode.CROSS_VERIFICATION_MISMATCH: Severity.ERROR,
    IssueCode.NEGATIVE_AMOUNT: Severity.WARNING,
    IssueCode.FUTURE_DATE: Severity.WARNING,
    IssueCode.TAX_RATE_MISMATCH: Severity.WARNING,
    IssueCode.TAX_RATE_UNUSUAL: Severity.WARNING,
    IssueCode.LOW_CONFIDENCE: Severity.WARNING,
    IssueCode.EXTRACTION_DISAGREEMENT: Severity.WARNING,
    IssueCode.DUPLICATE_FILE: Severity.WARNING,
    IssueCode.OCR_LOW_QUALITY: Severity.WARNING,
    IssueCode.POSSIBLE_DUPLICATE: Severity.WARNING,
    IssueCode.AMBIGUOUS_DATE: Severity.INFO,
    IssueCode.CURRENCY_INFERRED: Severity.INFO,
    IssueCode.UNGROUNDED_VALUE_REJECTED: Severity.INFO,
    IssueCode.LLM_UNAVAILABLE: Severity.INFO,
}


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    quantity: Money | None = None
    unit_price: Money | None = None
    amount: Money | None = None


class InvoiceData(BaseModel):
    """The structured record extracted from one invoice (or one ledger row)."""

    model_config = ConfigDict(extra="forbid")

    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    supplier_name: str | None = None
    customer_name: str | None = None
    subtotal: Money | None = None
    discount: Money | None = None
    shipping: Money | None = None
    tax: Money | None = None
    tax_rate: Money | None = None  # percent as printed, e.g. Decimal("20") for "VAT 20%"
    total: Money | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    document_type: DocumentType = DocumentType.UNKNOWN
    line_items: list[LineItem] = Field(default_factory=list)


# Every field of InvoiceData, in display order.
INVOICE_FIELDS: tuple[str, ...] = (
    "invoice_number",
    "invoice_date",
    "due_date",
    "supplier_name",
    "customer_name",
    "subtotal",
    "discount",
    "shipping",
    "tax",
    "tax_rate",
    "total",
    "currency",
    "document_type",
    "line_items",
)
# The nine fields the brief requires.
CORE_FIELDS: tuple[str, ...] = (
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "customer_name",
    "subtotal",
    "tax",
    "total",
    "currency",
    "line_items",
)
# Missing => MISSING_FIELD error.
REQUIRED_FIELDS: tuple[str, ...] = ("invoice_number", "invoice_date", "supplier_name", "total")
# Missing => MISSING_FIELD warning.
RECOMMENDED_FIELDS: tuple[str, ...] = ("customer_name", "subtotal", "tax", "currency", "line_items")
MONEY_FIELDS: tuple[str, ...] = ("subtotal", "discount", "shipping", "tax", "total")
DATE_FIELDS: tuple[str, ...] = ("invoice_date", "due_date")
TEXT_FIELDS: tuple[str, ...] = ("invoice_number", "supplier_name", "customer_name")


class MetaNote(StrEnum):
    """Notes extractors attach to ``FieldMeta.notes``; validation turns them into issues."""

    AMBIGUOUS_DATE_ORDER = "ambiguous_date_order"  # -> AMBIGUOUS_DATE (info)
    CURRENCY_INFERRED = "currency_inferred_from_symbol"  # -> CURRENCY_INFERRED (info)
    UNGROUNDED_VALUE_REJECTED = "ungrounded_value_rejected"  # -> UNGROUNDED_VALUE_REJECTED (info)
    EXTRACTION_DISAGREEMENT = "extraction_disagreement"  # -> EXTRACTION_DISAGREEMENT (warning)
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"  # value read from a low-confidence OCR line


class ExtractionWarning(StrEnum):
    """Codes for ``ExtractionResult.warnings`` (never free text containing document content)."""

    LLM_UNAVAILABLE = "llm_unavailable"  # no provider configured / reachable -> LLM_UNAVAILABLE (info)
    LLM_FAILED = "llm_failed"  # provider error; rules result used -> LLM_UNAVAILABLE (info)
    LLM_REFUSED = "llm_refused"  # provider declined; rules result used -> LLM_UNAVAILABLE (info)
    LLM_INPUT_TRUNCATED = "llm_input_truncated"  # document longer than LLM_MAX_INPUT_CHARS
    NO_LINE_ITEMS_TABLE = "no_line_items_table"


class FieldMeta(BaseModel):
    """How a field value was obtained and how much to trust it."""

    confidence: float = Field(ge=0.0, le=1.0)
    source: ExtractionSource
    evidence: str | None = None  # verbatim snippet of the source text supporting the value
    page: int | None = None
    notes: list[str] = Field(default_factory=list)  # e.g. "ambiguous_date_order", "currency_inferred_from_symbol"


def field_is_present(data: InvoiceData, name: str) -> bool:
    value = getattr(data, name)
    if name == "line_items":
        return bool(value)
    if name == "document_type":
        return value is not DocumentType.UNKNOWN
    return value is not None


class ExtractionResult(BaseModel):
    """Output of any extractor (rules, LLM, spreadsheet mapping, merge)."""

    data: InvoiceData = Field(default_factory=InvoiceData)
    meta: dict[str, FieldMeta] = Field(default_factory=dict)
    method: str = "rules"  # rules | llm | hybrid | spreadsheet | manual
    overall_confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    rejected: dict[str, Any] = Field(default_factory=dict)  # values removed by grounding, kept for transparency
    llm_provider: str | None = None
    llm_model: str | None = None
    prompt_version: str | None = None

    def compute_overall_confidence(self) -> float:
        """Mean confidence over the nine core fields; a missing required field counts as 0, a missing
        recommended field as 0.5 (absence of an optional value is not evidence of a bad read)."""
        scores: list[float] = []
        for name in CORE_FIELDS:
            if field_is_present(self.data, name):
                meta = self.meta.get(name)
                scores.append(meta.confidence if meta else 0.5)
            elif name in REQUIRED_FIELDS:
                scores.append(0.0)
            else:
                scores.append(0.5)
        self.overall_confidence = round(sum(scores) / len(scores), 4)
        return self.overall_confidence


class ValidationIssue(BaseModel):
    code: IssueCode
    severity: Severity
    field: str | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
