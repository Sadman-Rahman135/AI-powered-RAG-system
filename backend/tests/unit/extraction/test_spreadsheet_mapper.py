"""Spreadsheet sheets -> invoice records (ledger rows and invoice-layout sheets)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from app.domain.schemas import ExtractionSource, MetaNote, SourceKind
from app.extraction.spreadsheet_mapper import SheetRecord, records_from_sheets
from app.ingestion.types import Sheet


def _ledger_sheet() -> Sheet:
    return Sheet(
        name="Ledger",
        index=0,
        rows=[
            ["Accounts payable - March 2024", None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            ["Invoice No", "Invoice Date", "Supplier", "Customer", "Net", "Tax", "Total", "Currency"],
            ["INV-001", "05/03/2024", "Acme Corp", "Globex", Decimal("100.00"), Decimal("20.00"),
             Decimal("120.00"), "EUR"],
            ["INV-002", "15/03/2024", "Initech", "Globex", "200.00", "40.00", "240.00", "eur"],
            [None, None, None, None, None, None, None, None],
            ["INV-003", datetime(2024, 3, 20, 0, 0), "Hooli", "Globex", 300, 60, 360, "GBP"],
            ["Total", None, None, None, Decimal("600.00"), Decimal("120.00"), Decimal("720.00"), None],
        ],
    )


def _invoice_sheet() -> Sheet:
    return Sheet(
        name="Invoice",
        index=1,
        rows=[
            ["ACME Corp", None, None, None],
            ["123 Main Street, Springfield", None, None, None],
            ["INVOICE", None, None, None],
            ["Invoice No:", "INV-2024-77", None, None],
            ["Date:", date(2024, 3, 5), None, None],
            ["Bill To:", "Globex Inc", None, None],
            ["Currency:", "USD", None, None],
            [None, None, None, None],
            ["Description", "Qty", "Unit Price", "Amount"],
            ["Widget", 2, Decimal("10.00"), Decimal("20.00")],
            ["Gadget", 1, Decimal("5.50"), Decimal("5.50")],
            [None, None, None, None],
            [None, None, "Subtotal", Decimal("25.50")],
            [None, None, "Tax", Decimal("2.55")],
            [None, None, "Total", Decimal("28.05")],
        ],
    )


def _by_row(records: list[SheetRecord]) -> dict[int | None, SheetRecord]:
    return {r.row_number: r for r in records}


def test_ledger_sheet_rows_to_records() -> None:
    records = records_from_sheets([_ledger_sheet()])
    assert [r.row_number for r in records] == [4, 5, 7]  # blank row skipped, "Total" summary row ignored
    assert all(r.source_kind == SourceKind.LEDGER_ROW and r.sheet_name == "Ledger" for r in records)
    rows = _by_row(records)

    first = rows[4].result
    assert first.method == "spreadsheet"
    assert first.data.invoice_number == "INV-001"
    assert first.data.invoice_date == date(2024, 3, 5)  # column-level inference: "15/03/2024" proves day-first
    assert first.data.supplier_name == "Acme Corp" and first.data.customer_name == "Globex"
    amounts = (first.data.subtotal, first.data.tax, first.data.total)
    assert amounts == (Decimal("100.00"), Decimal("20.00"), Decimal("120.00"))
    assert first.data.currency == "EUR"
    total_meta = first.meta["total"]
    assert total_meta.source == ExtractionSource.SPREADSHEET
    assert total_meta.evidence == "Ledger!G4=120.00"
    assert total_meta.confidence == 0.95  # typed cell used directly
    assert first.meta["invoice_date"].confidence == 0.85  # string cell, parsed
    assert MetaNote.AMBIGUOUS_DATE_ORDER.value not in first.meta["invoice_date"].notes

    second = rows[5].result
    assert second.data.invoice_date == date(2024, 3, 15)
    assert second.data.total == Decimal("240.00") and second.meta["total"].confidence == 0.85
    assert second.data.currency == "EUR"

    third = rows[7].result
    assert third.data.invoice_date == date(2024, 3, 20)
    assert third.data.total == Decimal("360") and third.data.currency == "GBP"
    assert third.meta["invoice_date"].evidence == "Ledger!B7=2024-03-20"
    assert 0.0 < third.overall_confidence <= 1.0
    assert third.data.line_items == []


def test_invoice_layout_sheet() -> None:
    records = records_from_sheets([_invoice_sheet()])
    assert len(records) == 1
    record = records[0]
    assert record.source_kind == SourceKind.INVOICE_DOCUMENT
    assert record.row_number is None and record.sheet_name == "Invoice"
    data = record.result.data
    assert data.invoice_number == "INV-2024-77"
    assert data.invoice_date == date(2024, 3, 5)
    assert data.customer_name == "Globex Inc"
    assert data.supplier_name == "ACME Corp"
    assert (data.subtotal, data.tax, data.total) == (Decimal("25.50"), Decimal("2.55"), Decimal("28.05"))
    assert data.currency == "USD"
    assert [(li.description, li.quantity, li.unit_price, li.amount) for li in data.line_items] == [
        ("Widget", Decimal("2"), Decimal("10.00"), Decimal("20.00")),
        ("Gadget", Decimal("1"), Decimal("5.50"), Decimal("5.50")),
    ]
    meta = record.result.meta
    assert record.result.method == "spreadsheet"
    assert all(m.source == ExtractionSource.SPREADSHEET for m in meta.values())
    assert meta["invoice_number"].evidence == "Invoice!B4=INV-2024-77"
    assert meta["total"].evidence == "Invoice!D15=28.05"
    assert meta["invoice_date"].confidence == 0.95  # typed date cell


def test_workbook_with_ledger_and_invoice_sheets() -> None:
    notes = Sheet(name="Notes", index=2, rows=[["Prepared by finance"], ["Do not edit"]])
    records = records_from_sheets([_ledger_sheet(), _invoice_sheet(), notes])
    kinds = [(r.sheet_name, r.source_kind, r.row_number) for r in records]
    assert kinds == [
        ("Ledger", SourceKind.LEDGER_ROW, 4),
        ("Ledger", SourceKind.LEDGER_ROW, 5),
        ("Ledger", SourceKind.LEDGER_ROW, 7),
        ("Invoice", SourceKind.INVOICE_DOCUMENT, None),
    ]


def test_sheet_without_signals_yields_no_records() -> None:
    products = Sheet(name="Products", index=0, rows=[["Product", "Colour"], ["Widget", "Blue"], ["Gadget", "Red"]])
    assert records_from_sheets([products]) == []
    assert records_from_sheets([Sheet(name="Empty", index=0, rows=[])]) == []


def test_csv_ledger_decimal_comma() -> None:
    sheet = Sheet(
        name="ledger.csv",
        index=0,
        rows=[
            ["Rechnungsnr / Invoice No", "Date", "Vendor", "Net Amount", "VAT", "Total", "Currency"],
            ["R-1", "05.03.2024", "Müller GmbH", "1.000,00", "190,00", "1.190,00", "EUR"],
            ["R-2", "06.03.2024", "Schmidt AG", "250,50", "47,60", "298,10", "EUR"],
        ],
    )
    records = records_from_sheets([sheet])
    assert [r.row_number for r in records] == [2, 3]
    first = records[0].result.data
    assert first.invoice_number == "R-1" and first.supplier_name == "Müller GmbH"
    assert first.invoice_date == date(2024, 3, 5)
    assert (first.subtotal, first.tax, first.total) == (Decimal("1000.00"), Decimal("190.00"), Decimal("1190.00"))
    second = records[1].result.data
    assert (second.subtotal, second.tax, second.total) == (Decimal("250.50"), Decimal("47.60"), Decimal("298.10"))


def test_ledger_currency_from_header_and_never_defaulted() -> None:
    with_header = Sheet(
        name="S",
        index=0,
        rows=[["Invoice #", "Supplier", "Total (EUR)"], ["A-1", "Acme", Decimal("10.00")]],
    )
    assert records_from_sheets([with_header])[0].result.data.currency == "EUR"
    without = Sheet(name="S", index=0, rows=[["Invoice #", "Supplier", "Total"], ["A-1", "Acme", Decimal("10.00")]])
    record = records_from_sheets([without])[0]
    assert record.result.data.currency is None
    assert "currency" not in record.result.meta
    dollars = Sheet(name="S", index=0, rows=[["Invoice #", "Supplier", "Total"], ["A-1", "Acme", "$10.00"]])
    guessed = records_from_sheets([dollars], default_dollar="CAD")[0].result
    assert guessed.data.currency == "CAD" and guessed.data.total == Decimal("10.00")
    assert MetaNote.CURRENCY_INFERRED.value in guessed.meta["currency"].notes


def test_ledger_ambiguous_string_dates_are_flagged() -> None:
    sheet = Sheet(name="S", index=0, rows=[["Invoice #", "Date", "Total"], ["A-1", "04/06/2024", "10.00"]])
    result = records_from_sheets([sheet])[0].result
    assert result.data.invoice_date == date(2024, 4, 6)
    assert MetaNote.AMBIGUOUS_DATE_ORDER.value in result.meta["invoice_date"].notes
    assert result.meta["invoice_date"].confidence <= 0.7
    explicit = records_from_sheets([sheet], date_order="DMY")[0].result
    assert explicit.data.invoice_date == date(2024, 6, 4)


def test_invoice_layout_typed_integer_amounts_and_value_cell_evidence() -> None:
    """Typed cells are used as they are: an integer total of 2000 is an amount, not a year."""
    sheet = Sheet(
        name="Inv",
        index=0,
        rows=[
            ["Invoice Number:", None, "A-55"],
            ["Invoice Date:", None, datetime(2024, 2, 1)],
            ["Supplier:", None, "Blue Yonder Inc"],
            ["VAT 20%", None, Decimal("333.33")],
            ["Total:", None, 2000],
        ],
    )
    result = records_from_sheets([sheet])[0].result
    assert result.data.invoice_number == "A-55" and result.data.supplier_name == "Blue Yonder Inc"
    assert result.data.total == Decimal("2000")
    assert result.meta["total"].evidence == "Inv!C5=2000" and result.meta["total"].confidence == 0.95
    assert result.data.tax == Decimal("333.33")
    assert result.meta["tax"].evidence == "Inv!C4=333.33"  # the value cell, not the caption
    assert result.meta["tax_rate"].evidence == "Inv!A4=VAT 20%"


def test_invoice_layout_item_amounts_from_typed_cells() -> None:
    """Integral amounts arrive as ``int`` from .xls/.xlsx; a 2000 in the Amount column is an amount, not a year."""
    sheet = Sheet(
        name="Inv",
        index=0,
        rows=[
            ["Invoice No:", "INV-9"],
            ["Bill To:", "Globex Inc"],
            ["Description", "Qty", "Unit Price", "Amount"],
            ["Consulting", 10, 200, 2000],
            ["Travel", 1, Decimal("150.50"), Decimal("150.50")],
            [None, None, "Total", Decimal("2150.50")],
        ],
    )
    items = records_from_sheets([sheet])[0].result.data.line_items
    assert [(i.description, i.quantity, i.unit_price, i.amount) for i in items] == [
        ("Consulting", Decimal("10"), Decimal("200"), Decimal("2000")),
        ("Travel", Decimal("1"), Decimal("150.50"), Decimal("150.50")),
    ]


def test_invoice_layout_summed_tax_lines_keep_their_sum() -> None:
    """CGST + SGST in typed cells: the tax is their sum; a typed cell only replaces a value read from that cell."""
    sheet = Sheet(
        name="S",
        index=0,
        rows=[
            ["Invoice No:", "INV-1"],
            ["Bill To:", "Globex"],
            ["Description", "Qty", "Rate", "Amount"],
            ["Steel rods", 10, Decimal("100.00"), Decimal("1000.00")],
            ["Subtotal", None, None, Decimal("1000.00")],
            ["CGST 9%", None, None, Decimal("90.00")],
            ["SGST 9%", None, None, Decimal("90.00")],
            ["Total", None, None, Decimal("1180.00")],
        ],
    )
    result = records_from_sheets([sheet])[0].result
    assert result.data.tax == Decimal("180.00")
    assert result.data.subtotal == Decimal("1000.00") and result.data.total == Decimal("1180.00")
    assert result.meta["tax"].evidence == "S!D6=90.00 | S!D7=90.00"
    assert result.meta["total"].evidence == "S!D8=1180.00" and result.meta["total"].confidence == 0.95
    # the value next to "Bill To:" is the customer; the item-table header below it is not a party block
    assert result.data.customer_name == "Globex"
    assert result.data.supplier_name is None


def test_ledger_rows_without_signals_are_skipped_and_counted() -> None:
    """A row with neither invoice number nor total and < 2 key fields is a note / sub-header, not an invoice:
    it yields no record, and the skip is reported as a count-only diagnostic code (no document text)."""
    sheet = Sheet(
        name="Ledger",
        index=2,
        rows=[
            ["Invoice No", "Date", "Supplier", "Total"],
            ["INV-1", date(2024, 3, 1), "Acme", Decimal("10.00")],
            [None, None, "Region North", None],  # sub-header row
            ["INV-2", date(2024, 3, 2), "Initech", Decimal("20.00")],
            [None, "see note", None, None],  # note row
        ],
    )
    diagnostics: list[str] = []
    records = records_from_sheets([sheet], diagnostics=diagnostics)
    assert [r.row_number for r in records] == [2, 4]
    assert diagnostics == ["ledger_rows_skipped:2:2"]  # sheet index 2, two rows
    assert records_from_sheets([sheet]) == records  # diagnostics are optional
    clean: list[str] = []
    records_from_sheets([_ledger_sheet()], diagnostics=clean)
    assert clean == []  # blank and "Total" summary rows are not skips
