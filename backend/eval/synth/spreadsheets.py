"""Spreadsheet renderings: invoice-layout sheets (XLSX, XLS, CSV) and ledgers (one row per invoice).

Invoice-layout sheets print one document the way the template would: supplier block, title, key/value cells in one of
three arrangements (label | value, ``Label: value`` in one cell, value above its caption), the bill-to block, an item
table and a totals block. *Typed* sheets hold real numbers and dates (and an explicit ``Currency`` cell); *text* sheets
hold the strings the template would print (``1.234,56 €``, ``05.03.2024``).

Ledgers list many invoices, one per row, under a header row that may sit below title rows and may be followed by a
``Total`` row that must not become a record. Workbooks are written deterministically (fixed document properties and zip
timestamps) so the same seed gives the same bytes.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import openpyxl
import xlwt
from openpyxl.styles import Alignment, Font, PatternFill

from app.domain.schemas import DocumentType, SourceKind
from eval.labels import LabeledRecord
from eval.synth.data import InvoiceSpec, q2
from eval.synth.templates import TEMPLATES, Template, fmt_date, fmt_number

FIXED_TIME = datetime(2024, 1, 1)
ARRANGEMENT = {"T1": "side", "T2": "inline", "T3": "side", "T4": "inline", "T5": "side", "T6": "stacked"}
XL_DATE_FORMAT = {
    "us": "mm/dd/yyyy",
    "us_long": "mmmm d, yyyy",
    "us_mon": "mmm d, yyyy",
    "eu_dot": "dd.mm.yyyy",
    "eu_slash": "dd/mm/yyyy",
    "uk_long": "d mmmm yyyy",
    "uk_short": "d mmm yyyy",
    "in_slash": "dd/mm/yyyy",
    "in_dash": "dd-mm-yyyy",
    "in_mon": "dd-mmm-yyyy",
    "iso": "yyyy-mm-dd",
    "au_long": "d mmmm yyyy",
    "au_short": "dd mmm yyyy",
}
MONEY_FORMAT = "#,##0.00"


@dataclass(frozen=True)
class Cell:
    value: Any = None
    kind: str = "text"  # text | title | label | caption | head | money | date | num


# --------------------------------------------------------------------------------------------------------------------
# sheet titles
# --------------------------------------------------------------------------------------------------------------------
def sheet_title(name: str | None, index: int, used: Sequence[str] = ()) -> str:
    """An Excel-legal, unique sheet name (<= 31 characters, none of ``[]:*?/\\``)."""
    base = re.sub(r"[\[\]:*?/\\]", "-", name or f"Invoice {index + 1}").strip("' ")[:31] or f"Sheet{index + 1}"
    title, n = base, 2
    while title in used:
        title = f"{base[: 28 - len(str(n))]}-{n}"
        n += 1
    return title


# --------------------------------------------------------------------------------------------------------------------
# invoice-layout grid
# --------------------------------------------------------------------------------------------------------------------
def _money(spec: InvoiceSpec, tpl: Template, value: Decimal, role: str, typed: bool) -> Cell:
    return Cell(value, "money") if typed else Cell(tpl.money(spec, value, role))


def invoice_grid(spec: InvoiceSpec, *, typed: bool) -> list[list[Cell]]:
    """The cells of an invoice-layout sheet, row by row."""
    tpl = TEMPLATES[spec.template]
    grid: list[list[Cell]] = []

    def row(*cells: Cell) -> None:
        grid.append(list(cells))

    sup = spec.supplier
    if sup.name:
        row(Cell(sup.name, "title"))
    for line in sup.lines:
        row(Cell(line))
    for text in (f"Tel: {sup.phone}" if sup.phone else "", sup.email or ""):
        if text:
            row(Cell(text))
    for label, value in sup.ids:
        row(Cell(f"{label}: {value}"))
    row()
    row(Cell(tpl.title(spec), "title"))

    def shown(value: str | date) -> str:
        return tpl.date(spec, value) if isinstance(value, date) else value

    def meta_cell(value: str | date) -> Cell:
        if isinstance(value, date):
            return Cell(value, "date") if typed else Cell(tpl.date(spec, value))
        return Cell(value)

    meta = tpl.meta_items(spec)
    mode = ARRANGEMENT[spec.template]
    if spec.currency_printed and typed:
        meta = [*meta, ("Currency", spec.currency)]
    if mode == "side":
        for k, v in meta:
            row(Cell(k, "label"), meta_cell(v))
    elif mode == "inline":
        for k, v in meta:
            row(Cell(f"{k}: {shown(v)}"))
    else:
        for i in range(0, len(meta), 4):
            chunk = meta[i : i + 4]
            row(*[meta_cell(v) for _, v in chunk])
            row(*[Cell(k.upper(), "caption") for k, _ in chunk])
    row()
    if spec.customer is not None:
        cust = spec.customer
        row(Cell(f"{tpl.vocab(spec, 'bill_to').rstrip(':')}:", "label"))
        if cust.name:
            row(Cell(cust.name, "head"))
        for line in cust.lines:
            row(Cell(line))
        for label, value in cust.ids:
            row(Cell(f"{label}: {value}"))
        row()
    cols = tpl.sheet_columns(spec)
    if spec.rows:
        row(*[Cell(c.header, "head") for c in cols])
        for index, item in enumerate(spec.rows):
            cells: list[Cell] = []
            for col in cols:
                if col.key == "idx":
                    cells.append(Cell(index + 1, "num"))
                elif col.key == "desc":
                    cells.append(Cell(item.description))
                elif col.key == "hsn":
                    cells.append(Cell(item.hsn or ""))
                elif col.key == "unit":
                    cells.append(Cell(item.unit))
                elif col.key == "vat":
                    rate = spec.tax_rates[item.vat_class] if spec.tax_scheme == "classes" else spec.tax_rates[0]
                    cells.append(Cell(f"{tpl.rate_text(spec, rate)} %"))
                elif col.key == "qty":
                    cells.append(Cell(item.quantity, "num") if typed else Cell(tpl.qty(spec, item.quantity)))
                elif col.key == "unit_price":
                    cells.append(_money(spec, tpl, item.unit_price, "row", typed) if item.price_printed else Cell())
                else:
                    cells.append(_money(spec, tpl, item.amount, "row", typed))
            row(*cells)
        row()
    width = len(cols)
    for label, amount, kind in tpl.totals_items(spec):
        role = "total" if kind in {"total", "balance"} else "sub"
        row(*[Cell()] * max(width - 2, 0), Cell(label, "label"), _money(spec, tpl, amount, role, typed))
    return grid


# --------------------------------------------------------------------------------------------------------------------
# writers for grids
# --------------------------------------------------------------------------------------------------------------------
def _xl_date_format(spec_style: str) -> str:
    return XL_DATE_FORMAT[spec_style]


def _widths(grid: list[list[Cell]]) -> dict[int, int]:
    widths: dict[int, int] = {}
    for row in grid:
        for i, cell in enumerate(row):
            if cell.value is not None and cell.kind not in {"title"}:
                widths[i] = max(widths.get(i, 8), min(len(str(cell.value)) + 2, 58))
    return widths


def _literal(value: Any) -> Any:
    """Every cell is a literal: totals are computed here, never as formulas (openpyxl stores no cached result for a
    formula, so such a cell would read back empty) — and text that starts with ``=`` would silently become one."""
    if isinstance(value, str) and value.startswith("="):
        raise ValueError(f"text {value!r} would be stored as a formula")
    return value


def _style_xlsx(cell: Any, kind: str, date_format: str) -> None:
    if kind == "title":
        cell.font = Font(bold=True, size=14)
    elif kind == "head":
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="E8EDF3")
    elif kind == "label":
        cell.font = Font(color="555555")
    elif kind == "caption":
        cell.font = Font(size=8, color="888888")
    elif kind == "money":
        cell.number_format = MONEY_FORMAT
        cell.alignment = Alignment(horizontal="right")
    elif kind == "date":
        cell.number_format = date_format
        cell.alignment = Alignment(horizontal="left")


def normalise_zip(data: bytes) -> bytes:
    """Rewrite a zip with fixed timestamps so identical content gives identical bytes."""
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        entries = [(info, zin.read(info.filename)) for info in zin.infolist()]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, content in entries:
            fixed = zipfile.ZipInfo(info.filename, date_time=(2024, 1, 1, 0, 0, 0))
            fixed.compress_type = zipfile.ZIP_DEFLATED
            fixed.external_attr = info.external_attr
            zout.writestr(fixed, content)
    return out.getvalue()


def _workbook_bytes(wb: openpyxl.Workbook) -> bytes:
    buffer = io.BytesIO()
    wb.save(buffer)
    return normalise_zip(buffer.getvalue())


def _new_workbook() -> openpyxl.Workbook:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    wb.properties.creator = "eval-synth"
    wb.properties.lastModifiedBy = "eval-synth"
    wb.properties.created = FIXED_TIME
    wb.properties.modified = FIXED_TIME
    return wb


def _fill_sheet_xlsx(ws: Any, grid: list[list[Cell]], date_format: str) -> None:
    for r, row in enumerate(grid, start=1):
        for c, cell in enumerate(row, start=1):
            if cell.value is None or cell.value == "":
                continue
            value = datetime.combine(cell.value, time()) if cell.kind == "date" else _literal(cell.value)
            _style_xlsx(ws.cell(row=r, column=c, value=value), cell.kind, date_format)
    for index, width in _widths(grid).items():
        ws.column_dimensions[openpyxl.utils.get_column_letter(index + 1)].width = width


def invoice_xlsx_bytes(specs: Sequence[InvoiceSpec], titles: Sequence[str], *, typed: bool) -> bytes:
    """One sheet per document (``titles`` are the sheet names)."""
    wb = _new_workbook()
    for spec, title in zip(specs, titles, strict=True):
        ws = wb.create_sheet(title)
        _fill_sheet_xlsx(ws, invoice_grid(spec, typed=typed), _xl_date_format(spec.style["date"]))
    return _workbook_bytes(wb)


def write_invoice_xlsx(path: Path, specs: Sequence[InvoiceSpec], titles: Sequence[str], *, typed: bool) -> None:
    path.write_bytes(invoice_xlsx_bytes(specs, titles, typed=typed))


def _xls_style(kind: str, date_format: str, cache: dict[tuple[str, str], Any]) -> Any:
    key = (kind, date_format)
    if key not in cache:
        spec = {
            "title": "font: bold on, height 280",
            "head": "font: bold on; pattern: pattern solid, fore_colour gray25",
            "label": "font: colour gray50",
            "caption": "font: height 160, colour gray40",
            "money": "align: horiz right",
            "date": "align: horiz left",
        }.get(kind, "")
        num = MONEY_FORMAT if kind == "money" else date_format.upper() if kind == "date" else "General"
        cache[key] = xlwt.easyxf(spec, num_format_str=num)
    return cache[key]


def _fill_sheet_xls(ws: Any, grid: list[list[Cell]], date_format: str, cache: dict[tuple[str, str], Any]) -> None:
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell.value is None or cell.value == "":
                continue
            value = float(cell.value) if isinstance(cell.value, Decimal) else _literal(cell.value)
            ws.write(r, c, value, _xls_style(cell.kind, date_format, cache))
    for index, width in _widths(grid).items():
        ws.col(index).width = min(width, 58) * 270


def write_invoice_xls(path: Path, specs: Sequence[InvoiceSpec], titles: Sequence[str]) -> None:
    wb = xlwt.Workbook(encoding="utf-8")
    cache: dict[tuple[str, str], Any] = {}
    for spec, title in zip(specs, titles, strict=True):
        ws = wb.add_sheet(title)
        _fill_sheet_xls(ws, invoice_grid(spec, typed=True), _xl_date_format(spec.style["date"]), cache)
    wb.save(str(path))


def encode_csv(
    rows: Sequence[Sequence[str]], *, delimiter: str, encoding: str, quote_all: bool = False, newline: str = "\n"
) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(
        buffer, delimiter=delimiter, lineterminator=newline, quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL
    )
    writer.writerows(rows)
    return buffer.getvalue().encode(encoding)


def write_invoice_csv(path: Path, spec: InvoiceSpec, *, delimiter: str, encoding: str) -> None:
    grid = invoice_grid(spec, typed=False)
    rows = [["" if c.value is None else str(c.value) for c in row] for row in grid]
    path.write_bytes(encode_csv(rows, delimiter=delimiter, encoding=encoding))


# --------------------------------------------------------------------------------------------------------------------
# ledgers
# --------------------------------------------------------------------------------------------------------------------
LEDGER_LAYOUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "A": (
        ("invoice_number", "Invoice No"),
        ("invoice_date", "Invoice Date"),
        ("due_date", "Due Date"),
        ("supplier_name", "Supplier"),
        ("customer_name", "Customer"),
        ("currency", "Currency"),
        ("subtotal", "Subtotal"),
        ("tax", "Tax"),
        ("total", "Total"),
    ),
    "B": (
        ("invoice_number", "Inv #"),
        ("invoice_date", "Date"),
        ("supplier_name", "Vendor"),
        ("customer_name", "Bill To"),
        ("subtotal", "Net"),
        ("tax", "VAT"),
        ("total", "Gross"),
        ("currency", "Curr"),
    ),
    "C": (
        ("invoice_number", "Document No."),
        ("invoice_date", "Posting Date"),
        ("supplier_name", "Supplier Name"),
        ("customer_name", "Customer Name"),
        ("total", "Amount"),
        ("currency", "Currency"),
    ),
    "D": (
        ("invoice_number", "Invoice Number"),
        ("invoice_date", "Date"),
        ("due_date", "Payment Due"),
        ("supplier_name", "Supplier Name"),
        ("customer_name", "Customer Name"),
        ("subtotal", "Subtotal"),
        ("discount", "Discount"),
        ("shipping", "Shipping"),
        ("tax", "Tax"),
        ("total", "Total"),
        ("currency", "Currency"),
    ),
}
MONEY_KEYS = ("subtotal", "discount", "shipping", "tax", "total")
DATE_KEYS = ("invoice_date", "due_date")


@dataclass(frozen=True)
class LedgerOptions:
    layout: str = "A"
    typed: bool = True  # real numbers/dates (True) or the strings of a text export (False)
    header_row: int = 1  # 1-based row (or line) of the header; rows above hold a title
    title: str | None = None
    total_row: bool = False  # append a blank row and a "Total" row that must not become a record
    money_fmt: str = "plain"  # plain | plain_comma | us | eu | sym_us | sym_gbp
    date_fmt: str = "iso"  # a ``fmt_date`` style
    delimiter: str = ","
    encoding: str = "utf-8"
    quote_all: bool = False
    newline: str = "\n"


def ledger_record(source: LabeledRecord, layout: str, *, sheet_name: str | None, row_number: int) -> LabeledRecord:
    """The label of a ledger row: only the fields the layout has columns for, the rest null."""
    keys = {key for key, _ in LEDGER_LAYOUTS[layout]}
    values = {
        k: (getattr(source, k) if k in keys else None)
        for k in (
            "invoice_number",
            "invoice_date",
            "due_date",
            "supplier_name",
            "customer_name",
            "subtotal",
            "discount",
            "shipping",
            "tax",
            "total",
            "currency",
        )
    }
    return LabeledRecord(
        **values,
        document_type=DocumentType.INVOICE,
        sheet_name=sheet_name,
        row_number=row_number,
        source_kind=SourceKind.LEDGER_ROW,
    )


def _money_text(value: Decimal, fmt: str) -> str:
    if fmt == "plain":
        return f"{value:.2f}"
    if fmt == "plain_comma":
        return f"{value:.2f}".replace(".", ",")
    if fmt in {"us", "sym_us", "sym_gbp"}:
        text = fmt_number(value)
        return {"us": text, "sym_us": f"${text}", "sym_gbp": f"£{text}"}[fmt]
    return fmt_number(value, group=".", dec=",")


def ledger_cell(record: LabeledRecord, key: str, options: LedgerOptions) -> Any:
    value = getattr(record, key)
    if value is None:
        return None
    if key in DATE_KEYS:
        return value if options.typed else fmt_date(value, options.date_fmt)
    if key in MONEY_KEYS:
        return value if options.typed else _money_text(value, options.money_fmt)
    return value


def _ledger_grid(rows: Sequence[LabeledRecord], options: LedgerOptions) -> list[list[Cell]]:
    columns = LEDGER_LAYOUTS[options.layout]
    grid: list[list[Cell]] = []
    if options.header_row > 1:
        grid.append([Cell(options.title or "Sales ledger", "title")])
        grid.append([Cell("Generated 2025-01-15", "caption")])
        while len(grid) < options.header_row - 1:
            grid.append([])
    grid.append([Cell(header, "head") for _, header in columns])
    for record in rows:
        cells: list[Cell] = []
        for key, _ in columns:
            value = ledger_cell(record, key, options)
            kind = (
                "date"
                if key in DATE_KEYS and options.typed
                else "money"
                if key in MONEY_KEYS and options.typed
                else "text"
            )
            cells.append(Cell(value, kind))
        grid.append(cells)
    if options.total_row and rows:
        grid.append([])
        totals = [Cell("Total", "head")] + [Cell() for _ in columns[1:]]
        for i, (key, _) in enumerate(columns):
            if key in {"subtotal", "tax", "total"}:
                total = sum((getattr(r, key) or Decimal(0) for r in rows), Decimal(0))
                totals[i] = Cell(q2(total), "money") if options.typed else Cell(_money_text(total, options.money_fmt))
        grid.append(totals)
    return grid


def write_ledger_xlsx(
    path: Path,
    sheets: Sequence[tuple[str, Sequence[LabeledRecord]]],
    options: LedgerOptions,
    *,
    extra_sheets: Sequence[tuple[str, list[list[Cell]]]] = (),
) -> None:
    wb = _new_workbook()
    for name, rows in sheets:
        _fill_sheet_xlsx(wb.create_sheet(name), _ledger_grid(rows, options), "yyyy-mm-dd")
    for name, grid in extra_sheets:
        _fill_sheet_xlsx(wb.create_sheet(name), grid, "yyyy-mm-dd")
    path.write_bytes(_workbook_bytes(wb))


def write_ledger_xls(path: Path, sheets: Sequence[tuple[str, Sequence[LabeledRecord]]], options: LedgerOptions) -> None:
    wb = xlwt.Workbook(encoding="utf-8")
    cache: dict[tuple[str, str], Any] = {}
    for name, rows in sheets:
        _fill_sheet_xls(wb.add_sheet(name), _ledger_grid(rows, options), "YYYY-MM-DD", cache)
    wb.save(str(path))


def write_ledger_csv(path: Path, rows: Sequence[LabeledRecord], options: LedgerOptions) -> None:
    lines = [["" if c.value is None else str(c.value) for c in row] for row in _ledger_grid(rows, options)]
    path.write_bytes(
        encode_csv(
            lines,
            delimiter=options.delimiter,
            encoding=options.encoding,
            quote_all=options.quote_all,
            newline=options.newline,
        )
    )


def notes_sheet() -> list[list[Cell]]:
    """A "Notes" sheet of plain sentences: no header row, no figures, so it must yield no records."""
    return [
        [Cell("Notes", "title")],
        [],
        [Cell("Figures exclude credit notes issued after the period end.")],
        [Cell("Reconciled with the general ledger on 2025-01-15 by the finance team.")],
    ]
