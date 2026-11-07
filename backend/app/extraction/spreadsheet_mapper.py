"""Spreadsheet sheets -> invoice records.

- **Ledger sheets**: a header row within the first 15 rows naming >= 2 of {invoice number, total, invoice date,
  supplier, customer} (``labels.LEDGER_COLUMNS``). Each following non-empty row is one record
  (``SourceKind.LEDGER_ROW``, ``row_number`` = 1-based sheet row) until two blank rows in a row; "Total" /
  "Grand Total" rows are summaries, not records. A row with neither an invoice number nor a total and fewer
  than two of the key fields (a note or sub-header row such as "Region North") is not an invoice: it is skipped
  and counted in the optional ``diagnostics`` (``"ledger_rows_skipped:<sheet index>:<n>"``). Typed cells
  (Decimal, int, date, datetime) are used directly (confidence 0.95); string cells are parsed with column-level
  decimal-separator and day/month-order inference (confidence 0.85). Evidence is the cell, e.g.
  ``"Sheet1!D7=1234.50"``.
- **Invoice-layout sheets** (key/value cells and an item table): the sheet is laid out as lines (one span per
  cell, columns aligned) and read by the rules extractor - one ``SourceKind.INVOICE_DOCUMENT`` record per sheet.
- A sheet with neither yields nothing. Currency: a currency column, else a code or symbol in the headers/title
  ("Total (EUR)") or the row's amount cells; never a default.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from app.domain.schemas import (
    ExtractionResult,
    ExtractionSource,
    FieldMeta,
    InvoiceData,
    LineItem,
    MetaNote,
    SourceKind,
)
from app.extraction.labels import COLUMN_ALIASES, LEDGER_COLUMNS, LINE_ITEM_COLUMNS, match_column, normalize_header
from app.extraction.normalize import (
    ISO_CURRENCIES,
    CurrencyGuess,
    clean_text_value,
    date_order_prior,
    detect_currency,
    find_dates,
    find_money_values,
    infer_date_order,
    infer_decimal_separator,
)
from app.extraction.rules import EVIDENCE_LIMIT, FieldHit, run_rules
from app.extraction.rules_parts.values import money_only
from app.extraction.rules_parts.vocab import DATE_FIELDS, MONEY_FIELDS, find_field_labels
from app.ingestion.types import CellValue, Line, Sheet, Span

DateOrderArg = Literal["DMY", "MDY", "auto"]
TYPED = 0.95
PARSED = 0.85
AMBIGUOUS_CAP = 0.7
_KEY_FIELDS = frozenset({"invoice_number", "total", "invoice_date", "supplier_name", "customer_name"})
_TEXT_FIELDS = frozenset({"invoice_number", "supplier_name", "customer_name"})
_HEADER_SCAN = 15
_BLANK_RUN = 2
_SUMMARY = re.compile(r"^\s*(?:grand\s+|sub\s*-?\s*)?totals?\s*:?\s*$|^\s*sum\s*:?\s*$", re.IGNORECASE)
_CHAR_W = 6.0
_ROW_H = 12.0
_COL_GAP = 3


@dataclass(slots=True)
class SheetRecord:
    result: ExtractionResult
    sheet_name: str
    row_number: int | None
    source_kind: SourceKind


def render_cell(value: CellValue) -> str:
    """Cell as text, as ``Sheet.text`` renders it (Decimal plain, midnight datetimes as dates)."""
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def column_letter(index: int) -> str:
    """0 -> "A", 25 -> "Z", 26 -> "AA"."""
    letters = ""
    n = index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _cell_ref(sheet: Sheet, row: int, col: int, value: CellValue) -> str:
    return f"{sheet.name}!{column_letter(col)}{row + 1}={render_cell(value)}"[:EVIDENCE_LIMIT]


def _blank(row: Sequence[CellValue]) -> bool:
    return all(v is None or (isinstance(v, str) and not v.strip()) for v in row)


# ------------------------------------------------------------------------------------------------ ledger


def _alias_rank(cell: str, name: str) -> int:
    aliases = LEDGER_COLUMNS[name]
    h = normalize_header(cell)
    return aliases.index(h) if h in aliases else len(aliases)


def _ledger_header(sheet: Sheet) -> tuple[int, dict[str, int]] | None:
    for r, row in enumerate(sheet.rows[:_HEADER_SCAN]):
        found: dict[str, tuple[int, int]] = {}
        for c, cell in enumerate(row):
            if not isinstance(cell, str) or not cell.strip():
                continue
            name = match_column(cell, LEDGER_COLUMNS)
            if name is None:
                continue
            rank = _alias_rank(cell, name)
            if name not in found or rank < found[name][1]:
                found[name] = (c, rank)
        if len(_KEY_FIELDS & found.keys()) >= 2:
            return r, {name: c for name, (c, _) in found.items()}
    return None


def _header_like(row: Sequence[CellValue]) -> bool:
    hits = sum(1 for v in row if isinstance(v, str) and v.strip() and match_column(v, COLUMN_ALIASES))
    return hits >= 2 and not any(isinstance(v, Decimal | int | date) for v in row)


def _summary(row: Sequence[CellValue], cols: dict[str, int]) -> bool:
    number_col = cols.get("invoice_number")
    for c, v in enumerate(row):
        if isinstance(v, str) and _SUMMARY.match(v) and (number_col is None or c == number_col or not row[number_col]):
            return True
    return False


def _data_rows(sheet: Sheet, header: int) -> list[int]:
    rows: list[int] = []
    blanks = 0
    for r in range(header + 1, len(sheet.rows)):
        row = sheet.rows[r]
        if _blank(row):
            blanks += 1
            if blanks >= _BLANK_RUN:
                break
            continue
        if _header_like(row) and rows:
            break  # another table starts
        blanks = 0
        rows.append(r)
    return rows


def _column_strings(sheet: Sheet, rows: list[int], col: int) -> list[str]:
    out: list[str] = []
    for r in rows:
        row = sheet.rows[r]
        if col < len(row) and isinstance(row[col], str):
            out.append(str(row[col]))
    return out


@dataclass(slots=True)
class _Parsed:
    value: Any
    confidence: float
    notes: list[str]


def _text_value(name: str, cell: CellValue) -> _Parsed | None:
    if isinstance(cell, bool):
        return None
    if isinstance(cell, Decimal):
        text = str(int(cell)) if cell == cell.to_integral_value() else format(cell, "f")
    else:
        text = render_cell(cell)
    cleaned = clean_text_value(text)
    if cleaned is None or not re.search(r"\w", cleaned):
        return None
    if name == "invoice_number" and not re.search(r"\d", cleaned):
        return None
    return _Parsed(cleaned, TYPED, [])


def _money_value(name: str, cell: CellValue, hint: Literal[",", "."] | None, dollar: str) -> _Parsed | None:
    if isinstance(cell, bool):
        return None
    value: Decimal | None = None
    confidence = TYPED
    if isinstance(cell, Decimal):
        value = cell
    elif isinstance(cell, int):
        value = Decimal(cell)
    elif isinstance(cell, float):
        value = Decimal(str(cell))
    elif isinstance(cell, str):
        found = find_money_values(cell, hint, dollar)
        value, confidence = (found[0].value, PARSED) if len(found) == 1 else (None, PARSED)
    if value is None:
        return None
    return _Parsed(abs(value) if name == "discount" else value, confidence, [])


def _date_value(cell: CellValue, order: DateOrderArg, prior: str | None) -> _Parsed | None:
    if isinstance(cell, datetime):
        return _Parsed(cell.date(), TYPED, [])
    if isinstance(cell, date):
        return _Parsed(cell, TYPED, [])
    if not isinstance(cell, str):
        return None
    found = find_dates(cell, order)
    if not found:
        return None
    parsed = found[0].parsed
    if not parsed.ambiguous:
        return _Parsed(parsed.value, PARSED, [])
    if prior == "DMY":
        parsed = find_dates(cell[found[0].start : found[0].end], "DMY")[0].parsed
    return _Parsed(parsed.value, min(PARSED, AMBIGUOUS_CAP), [MetaNote.AMBIGUOUS_DATE_ORDER.value])


def _currency_value(cell: CellValue, dollar: str) -> _Parsed | None:
    if not isinstance(cell, str) or not cell.strip():
        return None
    code = cell.strip().upper()
    if code in ISO_CURRENCIES:
        return _Parsed(code, TYPED, [])
    guess = detect_currency(cell, default_dollar=dollar)
    return _guess(guess)


def _guess(guess: CurrencyGuess | None) -> _Parsed | None:
    if guess is None:
        return None
    return _Parsed(guess.code, guess.confidence, [MetaNote.CURRENCY_INFERRED.value] if guess.inferred else [])


@dataclass(slots=True)
class _Ledger:
    sheet: Sheet
    header: int
    cols: dict[str, int]
    rows: list[int]
    hints: dict[int, Literal[",", "."] | None]
    orders: dict[int, DateOrderArg]
    prior: str | None
    currency: tuple[_Parsed, str] | None  # sheet-level currency and its evidence
    dollar: str


def _ledger(sheet: Sheet, header: int, cols: dict[str, int], dollar: str, date_order: DateOrderArg) -> _Ledger:
    rows = _data_rows(sheet, header)
    hints = {c: infer_decimal_separator(_column_strings(sheet, rows, c)) for n, c in cols.items() if n in MONEY_FIELDS}
    orders: dict[int, DateOrderArg] = {}
    for name, c in cols.items():
        if name in DATE_FIELDS:
            inferred = None if date_order != "auto" else infer_date_order(_column_strings(sheet, rows, c))
            orders[c] = date_order if date_order != "auto" else (inferred or "auto")
    heading = [render_cell(v) for r in range(header + 1) for v in sheet.rows[r] if isinstance(v, str)]
    sheet_guess = _guess(detect_currency("\n".join(heading), default_dollar=dollar))
    currency = None
    if sheet_guess is not None:
        currency = (sheet_guess, f"{sheet.name}!{header + 1}: " + " | ".join(h for h in heading if h.strip()))
    codes = [str(sheet.rows[r][cols["currency"]]).strip().upper() for r in rows
             if "currency" in cols and cols["currency"] < len(sheet.rows[r]) and sheet.rows[r][cols["currency"]]]
    code = sheet_guess.value if sheet_guess else (codes[0] if codes else None)
    decimal_hint = next((h for h in hints.values() if h), None)
    prior = date_order_prior([], code, decimal_hint) if date_order == "auto" else None
    return _Ledger(sheet, header, cols, rows, hints, orders, prior, currency, dollar)


def _ledger_row(ledger: _Ledger, r: int) -> SheetRecord | None:
    sheet, row = ledger.sheet, ledger.sheet.rows[r]
    values: dict[str, Any] = {}
    meta: dict[str, FieldMeta] = {}

    def put(name: str, parsed: _Parsed | None, evidence: str) -> None:
        if parsed is None:
            return
        values[name] = parsed.value
        meta[name] = FieldMeta(
            confidence=parsed.confidence, source=ExtractionSource.SPREADSHEET, evidence=evidence[:EVIDENCE_LIMIT],
            page=None, notes=parsed.notes,
        )

    for name, c in ledger.cols.items():
        cell = row[c] if c < len(row) else None
        if cell is None or (isinstance(cell, str) and not cell.strip()):
            continue
        if name in _TEXT_FIELDS:
            parsed = _text_value(name, cell)
        elif name in MONEY_FIELDS:
            parsed = _money_value(name, cell, ledger.hints.get(c), ledger.dollar)
        elif name in DATE_FIELDS:
            parsed = _date_value(cell, ledger.orders.get(c, "auto"), ledger.prior)
        elif name == "currency":
            parsed = _currency_value(cell, ledger.dollar)
        else:
            continue
        put(name, parsed, _cell_ref(sheet, r, c, cell))
    if "currency" not in values:
        if ledger.currency is not None:
            put("currency", ledger.currency[0], ledger.currency[1])
        else:
            for name in ("total", "subtotal", "tax"):
                col = ledger.cols.get(name)
                if col is None or col >= len(row) or not isinstance(row[col], str):
                    continue
                guess = _guess(detect_currency(str(row[col]), default_dollar=ledger.dollar))
                put("currency", guess, _cell_ref(sheet, r, col, row[col]))
                if "currency" in values:
                    break
    if not ({"invoice_number", "total"} & values.keys()) and len(_KEY_FIELDS & values.keys()) < 2:
        return None
    result = ExtractionResult(data=InvoiceData(**values), meta=meta, method="spreadsheet")
    result.compute_overall_confidence()
    return SheetRecord(result, sheet.name, r + 1, SourceKind.LEDGER_ROW)


def _ledger_records(
    sheet: Sheet, header: int, cols: dict[str, int], dollar: str, date_order: DateOrderArg
) -> tuple[list[SheetRecord], int]:
    """Records of the ledger's rows, and how many non-summary rows carried no invoice signal (skipped)."""
    ledger = _ledger(sheet, header, cols, dollar, date_order)
    out: list[SheetRecord] = []
    skipped = 0
    for r in ledger.rows:
        if _summary(sheet.rows[r], cols):
            continue
        record = _ledger_row(ledger, r)
        if record is None:
            skipped += 1
        else:
            out.append(record)
    return out, skipped


# ------------------------------------------------------------------------------------------ invoice layout


def _sheet_lines(sheet: Sheet) -> tuple[list[Line], list[int], list[list[int]]]:
    """One line per non-empty row (one span per cell, columns at fixed x); the sheet row of each line and the
    sheet column of each span. Column widths come from rows with several cells, so a long title in a row of
    its own does not push every column to the right."""
    widths: dict[int, int] = {}
    for row in sheet.rows:
        cells = [(c, render_cell(v).strip()) for c, v in enumerate(row) if render_cell(v).strip()]
        if len(cells) >= 2:
            for c, text in cells:
                widths[c] = max(widths.get(c, 1), len(text))
    last_col = max((len(row) for row in sheet.rows), default=0)
    x_of: list[float] = []
    x = 0.0
    for c in range(last_col):
        x_of.append(x)
        x += (widths.get(c, 1) + _COL_GAP) * _CHAR_W
    lines: list[Line] = []
    rows: list[int] = []
    cols: list[list[int]] = []
    for r, row in enumerate(sheet.rows):
        spans: list[Span] = []
        span_cols: list[int] = []
        for c, v in enumerate(row):
            text = render_cell(v).strip()
            if text:
                y0 = r * _ROW_H
                spans.append(Span(text, x_of[c], y0, x_of[c] + len(text) * _CHAR_W, y0 + 10.0))
                span_cols.append(c)
        if spans:
            lines.append(Line(spans=spans))
            rows.append(r)
            cols.append(span_cols)
    return lines, rows, cols


def _has_invoice_signals(labelled: set[str]) -> bool:
    strong = labelled & {"invoice_number", "invoice_date", "due_date", "total", "subtotal", "tax", "customer_name"}
    return bool(labelled & {"invoice_number", "total"}) and len(strong) >= 2


def _typed_captions(sheet: Sheet, missing: set[str], skip_rows: set[int]) -> dict[str, tuple[Decimal, int, int]]:
    """Money values the text parser cannot read back from a typed cell (an integer like 2000 reads as a year):
    a caption cell ("Total:") followed in its row by a typed number. Item-table rows are skipped."""
    found: dict[str, tuple[Decimal, int, int]] = {}
    for r, row in enumerate(sheet.rows):
        if r in skip_rows:
            continue
        for c, cell in enumerate(row):
            if not isinstance(cell, str):
                continue
            labels = find_field_labels(cell)
            if not labels or labels[0].start > 1 or labels[0].field not in missing or labels[0].field in found:
                continue
            if not money_only(cell[labels[0].end :]):
                continue
            nxt = next((k for k in range(c + 1, len(row)) if render_cell(row[k]).strip()), None)
            value = row[nxt] if nxt is not None else None
            if nxt is not None and isinstance(value, Decimal | int) and not isinstance(value, bool):
                amount = Decimal(value)
                found[labels[0].field] = (abs(amount) if labels[0].field == "discount" else amount, r, nxt)
    return found


def _typed_items(sheet: Sheet, hit: FieldHit, header_row: int, item_rows: list[int]) -> list[LineItem]:
    """Item fields the text parse left empty, taken from typed number cells under the header's quantity / unit
    price / amount column (an integral amount such as 2000 reads as a year in text)."""
    items: list[LineItem] = list(hit.value)
    columns: dict[str, list[int]] = {}
    for c, cell in enumerate(sheet.rows[header_row]):
        name = match_column(cell, LINE_ITEM_COLUMNS) if isinstance(cell, str) else None
        if name in ("quantity", "unit_price", "amount"):
            columns.setdefault(name, []).append(c)
    if len(item_rows) != len(items):
        return items  # wrapped descriptions: rows and items no longer pair up
    out: list[LineItem] = []
    for item, r in zip(items, item_rows, strict=True):
        data = item.model_dump()
        for name, cols in columns.items():
            cell = sheet.rows[r][cols[0]] if len(cols) == 1 and cols[0] < len(sheet.rows[r]) else None
            if data[name] is None and isinstance(cell, Decimal | int) and not isinstance(cell, bool):
                data[name] = Decimal(cell)
        out.append(LineItem(**data))
    return out


def _typed_upgrade(name: str, hit: FieldHit, cell: CellValue) -> None:
    """A value read from a single typed cell is that cell's value: typed confidence, no ambiguity note. Only when
    the hit comes from one line and the cell holds the same value - an aggregated pick (CGST + SGST summed into
    the tax) or a value read elsewhere keeps what the rules read."""
    if len(hit.lines) != 1:
        return
    if name in MONEY_FIELDS and isinstance(cell, Decimal | int) and not isinstance(cell, bool):
        value = abs(Decimal(cell)) if name == "discount" else Decimal(cell)
        if value == hit.value:
            hit.value = value
            hit.confidence = max(hit.confidence, TYPED)
    elif name in DATE_FIELDS and isinstance(cell, date):
        day = cell.date() if isinstance(cell, datetime) else cell
        if day == hit.value:
            hit.confidence = max(hit.confidence, TYPED)
            hit.notes = [n for n in hit.notes if n != MetaNote.AMBIGUOUS_DATE_ORDER.value]


def _invoice_layout_record(sheet: Sheet, dollar: str, date_order: DateOrderArg) -> SheetRecord | None:
    lines, line_rows, line_cols = _sheet_lines(sheet)
    if not lines:
        return None
    out = run_rules(lines, default_dollar=dollar, date_order=date_order, page_numbers=[1] * len(lines))
    missing = {name for name in MONEY_FIELDS if name not in out.hits}
    captions = _typed_captions(sheet, missing, {line_rows[s] for s in out.table_sources})
    labelled = {name for name, hit in out.hits.items() if hit.confidence >= 0.7} | captions.keys()
    if not _has_invoice_signals(labelled):
        return None
    meta: dict[str, FieldMeta] = {}
    values: dict[str, Any] = {}
    for name, (amount, r, c) in captions.items():
        values[name] = amount
        meta[name] = FieldMeta(
            confidence=TYPED, source=ExtractionSource.SPREADSHEET, evidence=_cell_ref(sheet, r, c, sheet.rows[r][c])
        )
    for name, hit in out.hits.items():
        if name == "line_items":
            header_row = line_rows[out.lines[hit.lines[0]].source]
            rows = sorted(line_rows[s] for s in out.table_sources)
            item_rows = [r for r in rows if r != header_row and not _header_like(sheet.rows[r])]
            hit.value = _typed_items(sheet, hit, header_row, item_rows)
        refs: list[str] = []
        for k, index in enumerate(hit.lines):
            source = out.lines[index].source
            r, span_cols = line_rows[source], line_cols[source]
            span = hit.span if k == 0 and hit.span is not None and hit.span < len(span_cols) else len(span_cols) - 1
            c = span_cols[span]
            cell = sheet.rows[r][c]
            if k == 0 and name != "line_items":
                _typed_upgrade(name, hit, cell)
            if name == "line_items":
                refs.append(f"{sheet.name}!{r + 1}: {out.lines[index].text}")
            else:
                refs.append(_cell_ref(sheet, r, c, cell))
        values[name] = hit.value
        meta[name] = FieldMeta(
            confidence=round(min(max(hit.confidence, 0.0), 1.0), 4), source=ExtractionSource.SPREADSHEET,
            evidence=" | ".join(refs)[:EVIDENCE_LIMIT], page=None, notes=list(dict.fromkeys(hit.notes)),
        )
    result = ExtractionResult(data=InvoiceData(**values), meta=meta, method="spreadsheet", warnings=out.warnings)
    result.compute_overall_confidence()
    return SheetRecord(result, sheet.name, None, SourceKind.INVOICE_DOCUMENT)


# ---------------------------------------------------------------------------------------------- entry point


def records_from_sheets(
    sheets: Sequence[Sheet],
    *,
    default_dollar: str = "USD",
    date_order: DateOrderArg = "auto",
    diagnostics: list[str] | None = None,
) -> list[SheetRecord]:
    """Invoice records of a workbook (or CSV): ledger rows and invoice-layout sheets, in sheet order.

    ``diagnostics``, when given, receives count-only codes (never document text): ``"ledger_rows_skipped:<sheet
    index>:<n>"`` for ledger rows dropped for carrying no invoice signal.
    """
    records: list[SheetRecord] = []
    for sheet in sheets:
        header = _ledger_header(sheet)
        ledger, skipped = _ledger_records(sheet, *header, default_dollar, date_order) if header else ([], 0)
        if len(ledger) <= 1:
            layout = _invoice_layout_record(sheet, default_dollar, date_order)
            if layout is not None and (layout.result.data.line_items or not ledger):
                records.append(layout)
                continue
        records.extend(ledger)
        if skipped and diagnostics is not None:
            diagnostics.append(f"ledger_rows_skipped:{sheet.index}:{skipped}")
    return records
