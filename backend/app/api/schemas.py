"""Request and response models of the REST API (the JSON contract the web UI and external clients rely on).

Money and other decimals are serialized as plain-notation strings (``"1234.50"``) through the domain ``Money``
type; dates are ISO ``YYYY-MM-DD``; timestamps are ISO 8601 in UTC. Absent values are ``null``.

The invoice PATCH body is parsed strictly: amounts must be plain decimal strings, dates ``YYYY-MM-DD`` (ASCII
digits only; a Unicode digit class would also take Arabic-Indic or full-width ones), currency an upper-case ISO
code, ``document_type`` one of the enum values, and unknown fields are refused - every failure is a 422
``validation_error`` naming the field. Text values and query strings refuse control characters (C0, DEL, C1,
U+2028/U+2029; comments may keep tab and line breaks), which PostgreSQL would reject (NUL) or which would garble
logs and displays.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from app.domain.schemas import DocumentType, Money


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; every stored timestamp is UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


UtcDateTime = Annotated[datetime, AfterValidator(_as_utc)]
DocumentStatusValue = Literal["queued", "processing", "completed", "needs_review", "failed"]
JobStatusValue = Literal["queued", "running", "succeeded", "failed"]
FileKindValue = Literal["pdf", "image", "xlsx", "xls", "csv"]
ReviewStatusValue = Literal["auto_approved", "needs_review", "reviewed"]
ComparisonStatusValue = Literal["match", "mismatch", "partial"]


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------------------------ errors


class ErrorBody(ApiModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(ApiModel):
    error: ErrorBody


# ------------------------------------------------------------------------------------------------ documents


class JobOut(ApiModel):
    id: UUID
    document_id: UUID
    status: str
    stage: str | None
    progress_current: int
    progress_total: int
    attempt: int
    max_attempts: int
    error: str | None
    created_at: UtcDateTime
    started_at: UtcDateTime | None
    finished_at: UtcDateTime | None


class IssueCounts(ApiModel):
    error: int = 0
    warning: int = 0
    info: int = 0


class DocumentOut(ApiModel):
    id: UUID
    filename: str
    file_kind: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    page_count: int | None
    ocr_page_count: int | None
    error_code: str | None
    error_message: str | None
    processing_ms: int | None
    duplicate_of_id: UUID | None
    created_at: UtcDateTime
    updated_at: UtcDateTime
    processed_at: UtcDateTime | None
    invoice_count: int
    issue_counts: IssueCounts
    job: JobOut | None


class IssueOut(ApiModel):
    id: int
    invoice_id: UUID | None
    document_id: UUID
    code: str
    severity: str
    field: str | None
    message: str
    details: dict[str, Any] | None


class LineItemOut(ApiModel):
    description: str | None = None
    quantity: Money | None = None
    unit_price: Money | None = None
    amount: Money | None = None


class FieldMetaOut(ApiModel):
    confidence: float
    source: str
    evidence: str | None = None
    page: int | None = None
    notes: list[str] = Field(default_factory=list)


class InvoiceOut(ApiModel):
    id: UUID
    document_id: UUID
    filename: str
    record_index: int
    source_kind: str
    page_start: int | None
    page_end: int | None
    sheet_name: str | None
    row_number: int | None
    invoice_number: str | None
    invoice_date: date | None
    due_date: date | None
    supplier_name: str | None
    customer_name: str | None
    subtotal: Money | None
    discount: Money | None
    shipping: Money | None
    tax: Money | None
    tax_rate: Money | None
    total: Money | None
    currency: str | None
    document_type: str
    line_items: list[LineItemOut]
    field_meta: dict[str, FieldMetaOut]
    extraction_method: str
    overall_confidence: float
    review_status: str
    issues: list[IssueOut]
    created_at: UtcDateTime
    updated_at: UtcDateTime
    reviewed_at: UtcDateTime | None
    reviewed_by: str | None


class ExtractionInfo(ApiModel):
    method: str | None
    llm_provider: str | None
    llm_model: str | None
    prompt_version: str | None


class DocumentDetailOut(DocumentOut):
    invoices: list[InvoiceOut]
    document_issues: list[IssueOut]
    extraction: ExtractionInfo


class DocumentList(ApiModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class DocumentStatusOut(ApiModel):
    id: UUID
    status: str
    job: JobOut | None
    updated_at: UtcDateTime


class DocumentResultsOut(ApiModel):
    document_id: UUID
    status: str
    invoices: list[InvoiceOut]


class PageOut(ApiModel):
    page_number: int
    label: str | None
    method: str
    text: str
    ocr_confidence: float | None


class PageList(ApiModel):
    items: list[PageOut]
    total: int


class EventOut(ApiModel):
    id: int
    job_id: UUID | None
    stage: str
    level: str
    message: str
    data: dict[str, Any] | None
    created_at: UtcDateTime


class EventList(ApiModel):
    items: list[EventOut]


class UploadError(ApiModel):
    code: str
    message: str


class UploadItem(ApiModel):
    filename: str
    status: Literal["queued", "rejected"]
    document_id: UUID | None = None
    job_id: UUID | None = None
    duplicate_of_id: UUID | None = None
    error: UploadError | None = None


class UploadResponse(ApiModel):
    items: list[UploadItem]


class ReprocessResponse(ApiModel):
    document_id: UUID
    job_id: UUID
    status: Literal["queued"] = "queued"


# ------------------------------------------------------------------------------------------------ jobs


class JobList(ApiModel):
    items: list[JobOut]
    total: int


class ReprocessFailedResponse(ApiModel):
    requeued: int
    job_ids: list[UUID]


# ------------------------------------------------------------------------------------------------ invoices


class InvoiceList(ApiModel):
    items: list[InvoiceOut]
    total: int
    limit: int
    offset: int


class CorrectionOut(ApiModel):
    id: int
    field: str
    old_value: Any
    new_value: Any
    comment: str | None
    corrected_by: str
    created_at: UtcDateTime


class CorrectionList(ApiModel):
    items: list[CorrectionOut]


_DECIMAL = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")  # C0, DEL, C1, line and paragraph separators
_COMMENT_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f  ]")  # tab, LF and CR are kept
MAX_QUERY_AMOUNT = Decimal(10) ** 15
_MAX_FRACTION_DIGITS = 4  # the database stores Numeric(18, 4) / Numeric(9, 4)


def _decimal_parser(max_integer_digits: int) -> Any:
    def parse(value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
            raise PydanticCustomError("decimal_string", 'must be a decimal string such as "1234.50"')
        integer, _, fraction = value.lstrip("-").partition(".")
        if len(fraction) > _MAX_FRACTION_DIGITS:
            raise PydanticCustomError("decimal_places", "must have at most 4 decimal places")
        if len(integer.lstrip("0")) > max_integer_digits:
            raise PydanticCustomError("decimal_range", "is too large")
        return Decimal(value)

    return parse


def _iso_date(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise PydanticCustomError("iso_date", "must be a date in YYYY-MM-DD format")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise PydanticCustomError("iso_date", "is not a valid calendar date") from None


def _currency(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str) or not _CURRENCY.fullmatch(value):
        raise PydanticCustomError("currency", "must be an upper-case ISO 4217 code such as USD")
    return value


def _text_parser(max_length: int) -> Any:
    def parse(value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError("text", "must be a string")
        if _CONTROL.search(value):
            raise PydanticCustomError("text_control", "must not contain control characters")
        stripped = value.strip()
        if len(stripped) > max_length:
            raise PydanticCustomError(
                "text_length", "must be at most {max_length} characters", {"max_length": max_length}
            )
        return stripped or None

    return parse


def _comment(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PydanticCustomError("text", "must be a string")
    if _COMMENT_CONTROL.search(value):
        raise PydanticCustomError("text_control", "must not contain control characters")
    return value.strip() or None


def _query_text(value: Any) -> Any:
    if isinstance(value, str) and _CONTROL.search(value):
        raise PydanticCustomError("text_control", "must not contain control characters")
    return value


def _query_amount(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
        raise PydanticCustomError("decimal_string", "must be a plain decimal number such as 1234.50")
    amount = Decimal(value)
    if abs(amount) > MAX_QUERY_AMOUNT:
        raise PydanticCustomError("decimal_range", "must be between -10^15 and 10^15")
    return amount


def _nullable_string_schema(**extra: Any) -> WithJsonSchema:
    return WithJsonSchema({"anyOf": [{"type": "string", **extra}, {"type": "null"}]})


Amount = Annotated[
    Decimal | None,
    BeforeValidator(_decimal_parser(14)),
    _nullable_string_schema(pattern=_DECIMAL.pattern, examples=["1234.50"]),
]
Rate = Annotated[
    Decimal | None,
    BeforeValidator(_decimal_parser(5)),
    _nullable_string_schema(pattern=_DECIMAL.pattern, examples=["20"]),
]
IsoDate = Annotated[date | None, BeforeValidator(_iso_date), _nullable_string_schema(format="date")]
Currency = Annotated[str | None, BeforeValidator(_currency), _nullable_string_schema(pattern=_CURRENCY.pattern)]
Text100 = Annotated[str | None, BeforeValidator(_text_parser(100)), _nullable_string_schema(maxLength=100)]
Text255 = Annotated[str | None, BeforeValidator(_text_parser(255)), _nullable_string_schema(maxLength=255)]
Text1000 = Annotated[str | None, BeforeValidator(_text_parser(1000)), _nullable_string_schema(maxLength=1000)]
# The length limit sits on the string branch: on the union it would also run for None (an empty comment) and raise
# a TypeError, i.e. a 500 for {"comment": null}.
Comment = Annotated[Annotated[str, Field(max_length=2000)] | None, BeforeValidator(_comment)]
QueryText = Annotated[str | None, AfterValidator(_query_text)]
"""A query-string text parameter: control characters (NUL included) are a 422."""
QueryAmount = Annotated[
    Decimal | None,
    BeforeValidator(_query_amount),
    _nullable_string_schema(pattern=_DECIMAL.pattern, examples=["1234.50"]),
]
"""A query-string amount: a plain ASCII decimal within +-10^15 (no exponent, NaN or infinity)."""


class InvoiceFieldChanges(BaseModel):
    """The editable fields; a field present in the body is set (``null`` clears it), absent fields are kept."""

    model_config = ConfigDict(extra="forbid")

    invoice_number: Text100 = None
    invoice_date: IsoDate = None
    due_date: IsoDate = None
    supplier_name: Text255 = None
    customer_name: Text255 = None
    subtotal: Amount = None
    discount: Amount = None
    shipping: Amount = None
    tax: Amount = None
    tax_rate: Rate = None
    total: Amount = None
    currency: Currency = None
    document_type: DocumentType | None = None


EDITABLE_FIELDS: tuple[str, ...] = tuple(InvoiceFieldChanges.model_fields)


class LineItemIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Text1000 = None
    quantity: Amount = None
    unit_price: Amount = None
    amount: Amount = None


class InvoicePatch(BaseModel):
    """``fields`` to change, an optional full replacement of ``line_items`` and a comment for the audit trail."""

    model_config = ConfigDict(extra="forbid")

    fields: InvoiceFieldChanges = Field(default_factory=InvoiceFieldChanges)
    line_items: list[LineItemIn] | None = Field(default=None, max_length=1000)
    comment: Comment = None


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment: Comment = None


# ------------------------------------------------------------------------------------------------ search


class SearchHitOut(ApiModel):
    document_id: UUID
    invoice_id: UUID | None
    page_number: int | None
    score: float
    snippet: str
    match_type: Literal["keyword", "semantic", "hybrid", "filter"]
    filename: str
    invoice_number: str | None
    supplier_name: str | None
    customer_name: str | None
    invoice_date: date | None
    total: Money | None
    currency: str | None
    review_status: str | None


class SearchResponse(ApiModel):
    items: list[SearchHitOut]
    total: int
    took_ms: int
    mode: Literal["keyword", "semantic", "hybrid"]


# ------------------------------------------------------------------------------------------------ cross-verification


class CrossVerifyRequest(BaseModel):
    """Two invoice ids, or two document ids whose records are paired by invoice number."""

    model_config = ConfigDict(extra="forbid")

    left_invoice_id: UUID | None = None
    right_invoice_id: UUID | None = None
    left_document_id: UUID | None = None
    right_document_id: UUID | None = None

    @model_validator(mode="after")
    def _one_pair(self) -> CrossVerifyRequest:
        invoices = (self.left_invoice_id, self.right_invoice_id)
        documents = (self.left_document_id, self.right_document_id)
        by_invoice = all(v is not None for v in invoices) and all(v is None for v in documents)
        by_document = all(v is not None for v in documents) and all(v is None for v in invoices)
        if not (by_invoice or by_document):
            raise PydanticCustomError(
                "cross_verify_pair",
                "give either left_invoice_id and right_invoice_id, or left_document_id and right_document_id",
            )
        return self


class CrossVerifySide(ApiModel):
    document_id: UUID
    filename: str
    invoice_number: str | None


class FieldComparisonOut(ApiModel):
    field: str
    left: str | None
    right: str | None
    status: str
    note: str | None


class LineItemComparisonOut(ApiModel):
    left_count: int
    right_count: int
    matched: int
    status: str


class CrossVerificationOut(ApiModel):
    id: UUID
    left_invoice_id: UUID
    right_invoice_id: UUID
    left: CrossVerifySide
    right: CrossVerifySide
    kind: str
    status: str
    fields: list[FieldComparisonOut]
    line_items: LineItemComparisonOut
    trigger: str
    created_at: UtcDateTime


class CrossVerificationList(ApiModel):
    items: list[CrossVerificationOut]


# ------------------------------------------------------------------------------------------------ stats, health


class DocumentStats(ApiModel):
    total: int
    by_status: dict[str, int]


class InvoiceStats(ApiModel):
    total: int
    by_review_status: dict[str, int]


class IssueStats(ApiModel):
    by_code: dict[str, int]
    by_severity: dict[str, int]


class ProcessingStats(ApiModel):
    avg_ms: float | None
    p95_ms: int | None
    pages_total: int


class LlmInfo(ApiModel):
    provider: str | None
    model: str | None
    fallback_documents: int = Field(
        description="Documents for which the LLM was unavailable, failed or refused, so the rules result was used"
    )
    fallback_records: int = Field(description="Records (of those documents) that used the rules result")


class StatsOut(ApiModel):
    documents: DocumentStats
    invoices: InvoiceStats
    issues: IssueStats
    processing: ProcessingStats
    llm: LlmInfo


class HealthOut(ApiModel):
    status: Literal["ok"] = "ok"


class ReadyOut(ApiModel):
    status: Literal["ok", "unavailable"]
    database: Literal["ok", "error"]
    queue: Literal["ok", "error"]
    llm: Literal["none", "configured", "misconfigured"]


class ConfigOut(ApiModel):
    max_upload_mb: int
    max_files_per_request: int
    max_request_mb: int = Field(description="Cap of one upload request body (all its files together), in MiB")
    allowed_extensions: list[str]
    llm_provider: str | None
    llm_model: str | None
    auth_required: bool
