"""Duplicate detection against the database (SQLite session fixture)."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.config import reset_settings_cache
from app.db.models import Document, Invoice
from app.domain.schemas import InvoiceData, IssueCode, Severity, SourceKind
from app.extraction.normalize import normalize_invoice_number
from app.validation.duplicates import duplicate_issues, find_duplicate_candidates

MakeInvoice = Callable[..., InvoiceData]


def add_document(session: Session) -> Document:
    document = Document(
        filename="invoice.pdf",
        content_type="application/pdf",
        file_kind="pdf",
        size_bytes=1,
        sha256="0" * 64,
        storage_key=f"key-{uuid4()}",
    )
    session.add(document)
    session.flush()
    return document


def add_invoice(
    session: Session,
    document: Document,
    data: InvoiceData,
    *,
    source_kind: SourceKind = SourceKind.INVOICE_DOCUMENT,
) -> Invoice:
    row = Invoice(document_id=document.id, source_kind=source_kind.value)
    row.apply_invoice_data(data)
    session.add(row)
    session.flush()
    return row


def reload(session: Session) -> None:
    """Commit and expire, so that queries return what the database stored (Numeric round trip included)."""
    session.commit()
    session.expire_all()


def candidates_for(session: Session, data: InvoiceData, document: Document) -> list[Invoice]:
    assert data.invoice_number is not None
    return find_duplicate_candidates(
        session,
        invoice_number_norm=normalize_invoice_number(data.invoice_number),
        exclude_document_id=document.id,
    )


# ------------------------------------------------------------------------------ find_duplicate_candidates


def test_find_candidates_matches_normalized_number_in_other_documents(
    db_session: Session, make_invoice: MakeInvoice
) -> None:
    mine, other, unrelated = add_document(db_session), add_document(db_session), add_document(db_session)
    add_invoice(db_session, mine, make_invoice(invoice_number="INV-001"))
    match = add_invoice(db_session, other, make_invoice(invoice_number="inv 001"))
    add_invoice(db_session, unrelated, make_invoice(invoice_number="INV-002"))
    reload(db_session)

    found = find_duplicate_candidates(db_session, invoice_number_norm="INV001", exclude_document_id=mine.id)

    assert [row.id for row in found] == [match.id]


def test_same_document_excluded(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    data = make_invoice()
    add_invoice(db_session, mine, data)
    add_invoice(db_session, mine, data)  # a second record with the same number inside the same document
    reload(db_session)
    assert candidates_for(db_session, data, mine) == []

    elsewhere = add_invoice(db_session, other, data)
    reload(db_session)
    assert [row.id for row in candidates_for(db_session, data, mine)] == [elsewhere.id]


def test_blank_number_matches_nothing(db_session: Session, make_invoice: MakeInvoice) -> None:
    other = add_document(db_session)
    add_invoice(db_session, other, make_invoice(invoice_number="---"))  # normalizes to ""
    reload(db_session)

    assert find_duplicate_candidates(db_session, invoice_number_norm="", exclude_document_id=uuid4()) == []


def test_ledger_rows_are_candidates_but_never_duplicates(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, ledger_document = add_document(db_session), add_document(db_session)
    data = make_invoice()
    ledger_row = add_invoice(db_session, ledger_document, data, source_kind=SourceKind.LEDGER_ROW)
    reload(db_session)

    candidates = candidates_for(db_session, data, mine)

    assert [row.id for row in candidates] == [ledger_row.id]  # the caller decides what to do with them ...
    assert duplicate_issues(data, candidates) == []  # ... and a ledger row is a counterpart, not a duplicate


def test_ledger_row_not_duplicate(db_session: Session, make_invoice: MakeInvoice) -> None:
    ledger_document = add_document(db_session)
    ledger_row = add_invoice(db_session, ledger_document, make_invoice(), source_kind=SourceKind.LEDGER_ROW)
    reload(db_session)

    assert duplicate_issues(make_invoice(), [ledger_row]) == []
    assert duplicate_issues(make_invoice(total=Decimal("999.00")), [ledger_row]) == []


# ---------------------------------------------------------------------------------------- duplicate_issues


def test_duplicate_same_total_is_error(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    original = add_invoice(db_session, other, make_invoice())
    reload(db_session)
    data = make_invoice()

    [issue] = duplicate_issues(data, candidates_for(db_session, data, mine))

    assert issue.code is IssueCode.DUPLICATE_INVOICE
    assert issue.severity is Severity.ERROR
    assert issue.field == "invoice_number"
    assert issue.details == {"other_invoice_id": str(original.id), "other_document_id": str(other.id)}


def test_possible_duplicate_different_total(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    original = add_invoice(db_session, other, make_invoice(total=Decimal("120.00")))
    reload(db_session)
    data = make_invoice()

    [issue] = duplicate_issues(data, candidates_for(db_session, data, mine))

    assert issue.code is IssueCode.POSSIBLE_DUPLICATE
    assert issue.severity is Severity.WARNING
    assert issue.field == "invoice_number"
    assert issue.details == {"other_invoice_id": str(original.id), "other_document_id": str(other.id)}


@pytest.mark.parametrize(
    ("other_total", "code"),
    [
        ("110.00", IssueCode.DUPLICATE_INVOICE),
        ("110.02", IssueCode.DUPLICATE_INVOICE),
        ("109.98", IssueCode.DUPLICATE_INVOICE),
        ("110.03", IssueCode.POSSIBLE_DUPLICATE),
    ],
)
def test_total_comparison_uses_the_tolerance(
    db_session: Session, make_invoice: MakeInvoice, other_total: str, code: IssueCode
) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    add_invoice(db_session, other, make_invoice(total=Decimal(other_total)))
    reload(db_session)
    data = make_invoice()

    assert [i.code for i in duplicate_issues(data, candidates_for(db_session, data, mine))] == [code]


def test_clearly_different_supplier_is_not_a_duplicate(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    add_invoice(db_session, other, make_invoice(supplier_name="Initech"))
    reload(db_session)
    data = make_invoice()

    assert duplicate_issues(data, candidates_for(db_session, data, mine)) == []


def test_fuzzy_supplier_counts_as_the_same_supplier(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    add_invoice(db_session, other, make_invoice(supplier_name="Acme Corporation"))
    reload(db_session)
    data = make_invoice(supplier_name="ACME Corp.")

    assert [i.code for i in duplicate_issues(data, candidates_for(db_session, data, mine))] == [
        IssueCode.DUPLICATE_INVOICE
    ]


@pytest.mark.parametrize(
    ("mine_supplier", "other_supplier"), [(None, "Acme Corp"), ("Acme Corp", None), (None, None), ("  ", "Acme Corp")]
)
def test_missing_supplier_counts_as_the_same_supplier(
    db_session: Session, make_invoice: MakeInvoice, mine_supplier: str | None, other_supplier: str | None
) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    add_invoice(db_session, other, make_invoice(supplier_name=other_supplier))
    reload(db_session)
    data = make_invoice(supplier_name=mine_supplier)

    assert [i.code for i in duplicate_issues(data, candidates_for(db_session, data, mine))] == [
        IssueCode.DUPLICATE_INVOICE
    ]


@pytest.mark.parametrize(("mine_total", "other_total"), [(None, "110.00"), ("110.00", None), (None, None)])
def test_unknown_total_is_only_a_possible_duplicate(
    db_session: Session, make_invoice: MakeInvoice, mine_total: str | None, other_total: str | None
) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    add_invoice(db_session, other, make_invoice(total=None if other_total is None else Decimal(other_total)))
    reload(db_session)
    data = make_invoice(total=None if mine_total is None else Decimal(mine_total))

    assert [i.code for i in duplicate_issues(data, candidates_for(db_session, data, mine))] == [
        IssueCode.POSSIBLE_DUPLICATE
    ]


def test_one_issue_per_candidate_most_severe_first(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, older, newer, unrelated = (add_document(db_session) for _ in range(4))
    possible = add_invoice(db_session, older, make_invoice(total=Decimal("120.00")))
    certain = add_invoice(db_session, newer, make_invoice())
    add_invoice(db_session, unrelated, make_invoice(supplier_name="Initech"))
    reload(db_session)
    data = make_invoice()

    issues = duplicate_issues(data, candidates_for(db_session, data, mine))

    assert [i.code for i in issues] == [IssueCode.DUPLICATE_INVOICE, IssueCode.POSSIBLE_DUPLICATE]
    assert [i.details["other_invoice_id"] for i in issues] == [str(certain.id), str(possible.id)]


def test_duplicate_tolerance_is_configurable(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    add_invoice(db_session, other, make_invoice(total=Decimal("110.50")))
    reload(db_session)
    data = make_invoice()
    candidates = candidates_for(db_session, data, mine)

    assert [i.code for i in duplicate_issues(data, candidates)] == [IssueCode.POSSIBLE_DUPLICATE]
    assert [i.code for i in duplicate_issues(data, candidates, tolerance=Decimal("0.50"))] == [
        IssueCode.DUPLICATE_INVOICE
    ]


def test_duplicate_tolerance_defaults_to_the_setting(
    db_session: Session, make_invoice: MakeInvoice, monkeypatch: pytest.MonkeyPatch
) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    add_invoice(db_session, other, make_invoice(total=Decimal("110.50")))
    reload(db_session)
    data = make_invoice()
    candidates = candidates_for(db_session, data, mine)
    assert [i.code for i in duplicate_issues(data, candidates)] == [IssueCode.POSSIBLE_DUPLICATE]

    monkeypatch.setenv("AMOUNT_TOLERANCE", "0.50")
    reset_settings_cache()

    assert [i.code for i in duplicate_issues(data, candidates)] == [IssueCode.DUPLICATE_INVOICE]
    explicit = duplicate_issues(data, candidates, tolerance=Decimal("0.02"))
    assert [i.code for i in explicit] == [IssueCode.POSSIBLE_DUPLICATE]  # explicit argument wins


def test_a_ledger_row_record_is_never_a_duplicate(db_session: Session, make_invoice: MakeInvoice) -> None:
    mine, other = add_document(db_session), add_document(db_session)
    add_invoice(db_session, other, make_invoice())  # an invoice document with the same number, supplier and total
    reload(db_session)
    data = make_invoice()
    candidates = candidates_for(db_session, data, mine)

    # As an invoice document the record would duplicate it; as a ledger row it is a cross-verification counterpart.
    assert [i.code for i in duplicate_issues(data, candidates)] == [IssueCode.DUPLICATE_INVOICE]
    assert duplicate_issues(data, candidates, source_kind=SourceKind.INVOICE_DOCUMENT) == duplicate_issues(
        data, candidates
    )
    assert duplicate_issues(data, candidates, source_kind=SourceKind.LEDGER_ROW) == []


def test_no_candidates_no_issues(make_invoice: MakeInvoice) -> None:
    assert duplicate_issues(make_invoice(), []) == []
