"""Record issues, review status and document status - after processing and after manual corrections.

A record's issues are a function of the database, never of processing order:

* its own fields: ``validate_invoice`` with the record's source kind (a ledger row needs no line items);
* duplicates (invoice documents only): ``duplicate_issues`` against the records with the same normalised invoice
  number in documents **uploaded before** its own document - the later upload is the duplicate, whichever of the
  two was processed (or re-processed) first;
* cross-verification: every pair of records sharing a normalised invoice number across two documents is compared
  once and stored as one ``CrossVerification`` (``left`` = the record of the earlier-uploaded document, trigger
  ``auto``); a ``pdf_excel`` mismatch puts ``CROSS_VERIFICATION_MISMATCH`` on **both** records. PDF<->PDF pairs are
  stored too (their conflicts are the duplicate rules' business).

``reconcile_invoices`` recomputes this for a set of changed records and every record that shares (or shared) an
invoice number with them, so counterparts in other documents gain or lose their issues and review status. On
PostgreSQL the whole step is serialised with a transaction-level advisory lock, so two documents processed at
the same time cannot miss each other (on SQLite the single writer lock does the same).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import delete, func, insert, or_, select, text
from sqlalchemy.orm import Session

from app.db.models import (
    Chunk,
    CrossVerification,
    Document,
    DocumentStatus,
    Invoice,
    ValidationIssueRow,
)
from app.domain.schemas import ReviewStatus, Severity, SourceKind, ValidationIssue
from app.ingestion.types import FileKind
from app.pipeline.progress import as_utc
from app.search.chunking import KIND_INVOICE_SUMMARY, invoice_summary
from app.search.embeddings import get_embedder
from app.validation.cross_verify import (
    ComparisonResult,
    FieldComparison,
    LineItemComparison,
    compare_invoices,
    comparison_kind,
    cross_verification_issue,
)
from app.validation.duplicates import duplicate_issues
from app.validation.review import review_status_for
from app.validation.rules import validate_invoice

TRIGGER_AUTO = "auto"
RECONCILE_LOCK_KEY = 0x446F63496E74656C  # "DocIntel": pg_advisory_xact_lock key of the reconciliation step
_BATCH = 500  # ids / numbers per IN (...) list
_FLAGGING = (Severity.ERROR.value, Severity.WARNING.value)
_FROZEN_STATUSES = (DocumentStatus.FAILED, DocumentStatus.QUEUED, DocumentStatus.PROCESSING)

__all__ = [
    "ReconcileSummary",
    "own_record_issues",
    "reconcile_invoices",
    "record_issue_rows",
    "refresh_document_status",
    "refresh_invoice_summary_chunk",
    "revalidate_invoice",
    "serialize_reconciliation",
]


@dataclass(slots=True)
class ReconcileSummary:
    """What a reconciliation did: records re-validated, pairs compared, and the documents it touched."""

    records: int = 0
    comparisons: int = 0
    mismatches: int = 0
    duplicates: int = 0
    issues: dict[str, int] = field(default_factory=lambda: {"error": 0, "warning": 0, "info": 0})
    document_ids: set[UUID] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _Member:
    """A record of a number group with the facts that order and classify it."""

    invoice: Invoice
    uploaded: tuple[datetime, str]  # (document created_at, document id): upload order
    kind: FileKind

    @property
    def order(self) -> tuple[datetime, str, int, str]:
        return (*self.uploaded, self.invoice.record_index, str(self.invoice.id))


def _batches(values: Iterable[Any]) -> Iterable[list[Any]]:
    items = list(values)
    for start in range(0, len(items), _BATCH):
        yield items[start : start + _BATCH]


def serialize_reconciliation(session: Session) -> None:
    """Take the reconciliation lock for the rest of the transaction (PostgreSQL ``pg_advisory_xact_lock``; SQLite
    serialises writers anyway).

    Lock order matters: a caller that writes rows of its own before reconciling (the corrections endpoint updates
    the invoice, then calls ``revalidate_invoice``) must call this **first** in its transaction. Otherwise it holds
    row locks while waiting for this lock, and a processing job holding this lock and waiting for those rows
    deadlocks with it. Calling it again later in the same transaction is harmless (the lock is re-entrant).
    """
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": RECONCILE_LOCK_KEY})


# ----------------------------------------------------------------------------------------------- issues


def own_record_issues(invoice: Invoice) -> list[ValidationIssue]:
    """The issues of the record's own fields (no other document involved): ``validate_invoice`` with its source
    kind. Needs no database, so the processor runs it before its transaction."""
    return validate_invoice(
        invoice.to_invoice_data(), invoice.field_meta_models(), source_kind=SourceKind(invoice.source_kind)
    )


def record_issue_rows(issues: Sequence[ValidationIssue], *, document_id: UUID, invoice_id: UUID | None) -> list[dict]:
    """Insert parameters of ``validation_issues`` rows for ``issues``."""
    return [
        {
            "document_id": document_id,
            "invoice_id": invoice_id,
            "code": issue.code.value,
            "severity": issue.severity.value,
            "field": issue.field,
            "message": issue.message,
            "details": issue.model_dump(mode="json")["details"],
        }
        for issue in issues
    ]


def _stored_result(row: CrossVerification) -> ComparisonResult:
    return ComparisonResult(
        status=row.status,  # type: ignore[arg-type]
        fields=[FieldComparison(**item) for item in row.fields or []],
        line_items=LineItemComparison(**row.line_items),
    )


def _load_members(session: Session, numbers: set[str]) -> dict[str, list[_Member]]:
    groups: dict[str, list[_Member]] = defaultdict(list)
    for batch in _batches(sorted(numbers)):
        rows = session.execute(
            select(Invoice, Document.created_at, Document.file_kind)
            .join(Document, Document.id == Invoice.document_id)
            .where(Invoice.invoice_number_norm.in_(batch))
        )
        for invoice, created_at, file_kind in rows:
            number = invoice.invoice_number_norm
            if number is None:  # cannot happen: the query matches on the number
                continue
            uploaded = (as_utc(created_at), str(invoice.document_id))
            groups[number].append(_Member(invoice, uploaded, FileKind(file_kind)))  # type: ignore[arg-type]
    for members in groups.values():
        members.sort(key=lambda m: m.order)
    return groups


def _rows_involving(session: Session, invoice_ids: set[UUID]) -> list[CrossVerification]:
    rows: dict[UUID, CrossVerification] = {}
    for batch in _batches(invoice_ids):
        statement = select(CrossVerification).where(
            or_(CrossVerification.left_invoice_id.in_(batch), CrossVerification.right_invoice_id.in_(batch))
        )
        rows.update((row.id, row) for row in session.scalars(statement))
    return list(rows.values())


def _compare(session: Session, left: _Member, right: _Member, row: CrossVerification | None) -> ComparisonResult:
    """Compare ``left`` (earlier) with ``right``; store the result (a manual row keeps its own orientation)."""
    kind = comparison_kind(left.kind, right.kind)
    if row is not None and row.left_invoice_id != left.invoice.id:
        left, right = right, left
    result = compare_invoices(left.invoice.to_invoice_data(), right.invoice.to_invoice_data())
    payload = result.to_dict()
    if row is None:
        session.add(
            CrossVerification(
                left_invoice_id=left.invoice.id,
                right_invoice_id=right.invoice.id,
                kind=kind,
                status=result.status,
                fields=payload["fields"],
                line_items=payload["line_items"],
                trigger=TRIGGER_AUTO,
            )
        )
    else:
        row.kind, row.status, row.fields, row.line_items = kind, result.status, payload["fields"], payload["line_items"]
    return result


def reconcile_invoices(
    session: Session,
    changed: Sequence[Invoice],
    *,
    extra_numbers: Iterable[str] = (),
    own_issues: Mapping[UUID, list[ValidationIssue]] | None = None,
) -> ReconcileSummary:
    """Re-derive pairs, issues and review status after ``changed`` records were added or edited.

    ``own_issues`` may hold ``own_record_issues`` already computed for some records (by id).

    ``extra_numbers`` are invoice numbers that records of ``changed``'s documents carried before (records deleted
    by re-processing): their counterparts lose the issues that pointed at them. Every record sharing a number with
    a changed record (now or before) is re-validated. Runs in the caller's transaction; does not commit and does
    not refresh document statuses (``summary.document_ids`` lists the documents to refresh).
    """
    summary = ReconcileSummary()
    session.flush()
    changed_ids = {invoice.id for invoice in changed}

    stale = _rows_involving(session, changed_ids)
    partner_ids = {
        other
        for row in stale
        if row.trigger == TRIGGER_AUTO
        for other in (row.left_invoice_id, row.right_invoice_id)
        if other not in changed_ids
    }
    for stale_row in stale:
        if stale_row.trigger == TRIGGER_AUTO:
            session.delete(stale_row)
    numbers = {invoice.invoice_number_norm for invoice in changed if invoice.invoice_number_norm}
    numbers.update(number for number in extra_numbers if number)
    for batch in _batches(partner_ids):
        partner_numbers = session.scalars(select(Invoice.invoice_number_norm).where(Invoice.id.in_(batch)))
        numbers.update(number for number in partner_numbers if number)
    session.flush()

    groups = _load_members(session, numbers)
    member_ids = {m.invoice.id for members in groups.values() for m in members}
    stored: dict[frozenset[UUID], CrossVerification] = {}
    for existing in _rows_involving(session, member_ids):
        if existing.left_invoice_id in member_ids and existing.right_invoice_id in member_ids:
            stored[frozenset((existing.left_invoice_id, existing.right_invoice_id))] = existing

    results: dict[frozenset[UUID], tuple[ComparisonResult, str]] = {}
    for members in groups.values():
        for left, right in combinations(members, 2):  # members are sorted: left is the earlier upload
            if left.invoice.document_id == right.invoice.document_id:
                continue
            key = frozenset((left.invoice.id, right.invoice.id))
            row = stored.get(key)
            if row is not None and not key & changed_ids:
                results[key] = (_stored_result(row), row.kind)
                continue
            result = _compare(session, left, right, row)
            results[key] = (result, comparison_kind(left.kind, right.kind))
            summary.comparisons += 1
            summary.mismatches += result.status == "mismatch"

    targets: dict[UUID, Invoice] = {invoice.id: invoice for invoice in changed}
    for members in groups.values():
        targets.update((m.invoice.id, m.invoice) for m in members)
    member_of = {m.invoice.id: m for members in groups.values() for m in members}

    rows: list[dict] = []
    for invoice in targets.values():
        known = own_issues.get(invoice.id) if own_issues is not None else None
        issues = list(known) if known is not None else own_record_issues(invoice)
        me = member_of.get(invoice.id)
        group = groups.get(invoice.invoice_number_norm or "", []) if me is not None else []
        others = [m for m in group if m.invoice.document_id != invoice.document_id]
        if me is not None and invoice.source_kind == SourceKind.INVOICE_DOCUMENT.value:
            earlier = [m.invoice for m in others if m.uploaded < me.uploaded]
            found = duplicate_issues(invoice.to_invoice_data(), earlier, source_kind=SourceKind(invoice.source_kind))
            summary.duplicates += len(found)
            issues.extend(found)
        for other in others:
            result, kind = results[frozenset((invoice.id, other.invoice.id))]
            if kind != "pdf_excel":
                continue
            issue = cross_verification_issue(
                result, other_invoice_id=other.invoice.id, other_document_id=other.invoice.document_id
            )
            if issue is not None:
                issues.append(issue)
        computed = review_status_for(issues)
        if not (invoice.review_status == ReviewStatus.REVIEWED.value and computed is ReviewStatus.AUTO_APPROVED):
            invoice.review_status = computed.value
        for issue in issues:
            summary.issues[issue.severity.value] += 1
        rows.extend(record_issue_rows(issues, document_id=invoice.document_id, invoice_id=invoice.id))
        summary.document_ids.add(invoice.document_id)

    for batch in _batches(targets):
        session.execute(
            delete(ValidationIssueRow)
            .where(ValidationIssueRow.invoice_id.in_(batch))
            .execution_options(synchronize_session=False)
        )
    for batch in _batches(rows):
        session.execute(insert(ValidationIssueRow), batch)
    for invoice in targets.values():
        if "issues" in invoice.__dict__:
            session.expire(invoice, ["issues"])  # loaded collections would show the replaced rows
    summary.records = len(targets)
    session.flush()
    return summary


# ------------------------------------------------------------------------------------- document status


def refresh_document_status(session: Session, document: Document) -> None:
    """``needs_review`` while a record with an error or warning is not ``reviewed``, or while a document-level
    error/warning stands and not every record has been reviewed; else ``completed``.

    ``failed`` stays failed, and a document that is ``queued``/``processing`` is left to its job. Does not commit.
    """
    if document.status in _FROZEN_STATUSES:
        return
    session.flush()
    flagged = set(
        session.scalars(
            select(ValidationIssueRow.invoice_id)
            .where(ValidationIssueRow.document_id == document.id, ValidationIssueRow.severity.in_(_FLAGGING))
            .distinct()
        )
    )
    records = session.execute(
        select(Invoice.id, Invoice.review_status).where(Invoice.document_id == document.id)
    ).all()
    reviewed = ReviewStatus.REVIEWED.value
    pending = any(status != reviewed and invoice_id in flagged for invoice_id, status in records)
    document_level = None in flagged
    all_reviewed = bool(records) and all(status == reviewed for _, status in records)
    needs_review = pending or (document_level and not all_reviewed)
    document.status = DocumentStatus.NEEDS_REVIEW if needs_review else DocumentStatus.COMPLETED


# ------------------------------------------------------------------------------------ manual corrections


def refresh_invoice_summary_chunk(session: Session, invoice: Invoice, filename: str) -> None:
    """Replace the record's ``invoice_summary`` chunk (text and embedding) after its values changed, without
    re-indexing the whole document. Does not commit."""
    session.execute(
        delete(Chunk)
        .where(Chunk.invoice_id == invoice.id, Chunk.kind == KIND_INVOICE_SUMMARY)
        .execution_options(synchronize_session=False)
    )
    summary = invoice_summary(invoice.to_invoice_data(), filename)
    vector = np.asarray(get_embedder().embed_documents([summary])[0], dtype=np.float32)
    last_index = session.scalar(select(func.max(Chunk.chunk_index)).where(Chunk.document_id == invoice.document_id))
    session.add(
        Chunk(
            document_id=invoice.document_id,
            invoice_id=invoice.id,
            page_number=invoice.page_start,
            chunk_index=0 if last_index is None else last_index + 1,
            kind=KIND_INVOICE_SUMMARY,
            text=summary,
            embedding=vector if float(np.linalg.norm(vector)) > 0 else None,
        )
    )
    session.flush()


def revalidate_invoice(session: Session, invoice: Invoice) -> list[ValidationIssueRow]:
    """Recompute ``invoice``'s issues after a manual correction (own fields, duplicates, cross-verification against
    its counterparts - whose issues and review status follow), replace its issue rows, set its review status
    (``reviewed`` is kept only if no error or warning remains), refresh its search summary and the status of every
    document involved. Does not commit. Returns the record's new issue rows.

    Call ``serialize_reconciliation(session)`` before writing the correction itself (see its docstring): taking the
    lock here, after the caller's own row updates, can deadlock with a processing job on PostgreSQL.
    """
    serialize_reconciliation(session)
    summary = reconcile_invoices(session, [invoice])
    document = session.get(Document, invoice.document_id)
    if document is not None:
        refresh_invoice_summary_chunk(session, invoice, document.filename)
    for document_id in sorted(summary.document_ids | {invoice.document_id}, key=str):
        affected = session.get(Document, document_id)
        if affected is not None:
            refresh_document_status(session, affected)
    session.flush()
    statement = (
        select(ValidationIssueRow).where(ValidationIssueRow.invoice_id == invoice.id).order_by(ValidationIssueRow.id)
    )
    return list(session.scalars(statement))
