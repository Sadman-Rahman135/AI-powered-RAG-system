"""Line-item tables: header detection, column geometry, row parsing, wrapped descriptions, tables continuing on
the next page, and a conservative fallback for header-less receipts.

A header line has >= 2 ``LINE_ITEM_COLUMNS`` (one of them description or amount); its cells give the columns'
x-ranges. Row cells are assigned to columns by overlap (nearest centre otherwise). Rows whose cells do not line up
(plain text without column gaps) are read with a row pattern instead: description, then the numbers mapped to the
numeric columns in order. A row needs an amount, or a quantity and a unit price. Values are never computed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from app.domain.schemas import LineItem
from app.extraction.labels import LINE_ITEM_COLUMNS, match_column, normalize_header
from app.extraction.normalize import find_dates, parse_number
from app.extraction.rules_parts import values, vocab
from app.extraction.rules_parts.kv import labels_of
from app.extraction.rules_parts.layout import Chunk, TextLine, build_text_lines, overlap
from app.extraction.rules_parts.values import DocContext, build_context
from app.ingestion.types import Line

_NUMERIC = ("quantity", "unit_price", "amount")
_TOTAL_FIELDS = frozenset({"subtotal", "total"})
_MONEY_LABELS = frozenset({"subtotal", "total", "tax", "discount", "shipping"})
_MAX_RUN = 3  # consecutive non-item lines that end a table
_HEADERLESS_CONFIDENCE = 0.6
_NO_CHECK_CONFIDENCE = 0.75
_QTY_PREFIX = re.compile(r"^\s*(\d{1,4})\s*[xX×@]\s+(?=\S)")
_QTY_SUFFIX = re.compile(r"\s+[xX×]\s?(\d{1,4})\s*$")
_PLAIN = DocContext(None, "auto", None, None, "USD")


@dataclass(slots=True)
class Column:
    field: str | None  # None: a column the item schema does not use ("#", "HSN", "Tax %")
    x0: float
    x1: float
    header: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass(slots=True)
class LineItemTable:
    items: list[LineItem]
    confidence: float
    header_line: int | None  # TextLine index of the (first) header; None for a header-less receipt
    lines: set[int] = field(default_factory=set)  # TextLine indices that belong to the table


@dataclass(slots=True)
class _Row:
    description: str | None
    quantity: Decimal | None
    unit_price: Decimal | None
    amount: Decimal | None

    @property
    def valid(self) -> bool:
        has_values = self.amount is not None or (self.quantity is not None and self.unit_price is not None)
        return has_values and self.description is not None

    @property
    def checkable(self) -> bool:
        return self.quantity is not None and self.unit_price is not None and self.amount is not None

    @property
    def consistent(self) -> bool:
        if self.quantity is None or self.unit_price is None or self.amount is None:
            return False
        tolerance = max(Decimal("0.02"), abs(self.quantity) * Decimal("0.005"))
        return abs(self.quantity * self.unit_price - self.amount) <= tolerance


def _clean_description(text: str | None) -> str | None:
    if text is None:
        return None
    s = re.sub(r"\s+", " ", text).strip(" |-–:")
    return s if re.search(r"[^\W\d_]", s) else None


# --------------------------------------------------------------------------------------------- header


def _exact_alias(text: str) -> str | None:
    h = normalize_header(text)
    for name, aliases in LINE_ITEM_COLUMNS.items():
        if h in aliases:
            return name
    return None


def _split_words(line: TextLine, chunk: Chunk) -> list[Column]:
    """Greedy longest-alias split of a multi-word header cell ("Qty Unit Price" -> qty | unit price)."""
    words = [(m.group(0), chunk.start + m.start(), chunk.start + m.end()) for m in re.finditer(r"\S+", chunk.text)]
    out: list[Column] = []
    i = 0
    while i < len(words):
        for n in range(min(4, len(words) - i), 0, -1):
            phrase = " ".join(w for w, _, _ in words[i : i + n])
            name = _exact_alias(phrase)
            if name is not None or n == 1:
                x0 = line.x_at(words[i][1])
                x1 = line.x_at(words[i + n - 1][2])
                out.append(Column(name, x0, x1, phrase))
                i += n
                break
    return out


def header_columns(line: TextLine) -> list[Column] | None:
    """Columns of a line-item table header, or ``None`` when ``line`` is not one."""
    if find_dates(line.text):
        return None
    if any(values.has_fraction(m) or m.currency for m in values.money_values(line.text, _PLAIN)):
        return None
    columns: list[Column] = []
    for chunk in line.chunks:
        if " " in chunk.text and _exact_alias(chunk.text) is None:
            parts = _split_words(line, chunk)
            if len({c.field for c in parts if c.field}) >= 2:
                columns.extend(parts)
                continue
        columns.append(Column(match_column(chunk.text, LINE_ITEM_COLUMNS), chunk.x0, chunk.x1, chunk.text))
    known = [c for c in columns if c.field]
    names = {c.field for c in known}
    if len(columns) < 2 or len(names) < 2 or not names & {"description", "amount"}:
        return None
    if sum(len(c.header.split()) for c in known) / len(line.text.split()) < 0.5:
        return None  # a sentence that happens to mention "item" and "price"
    return _dedupe(columns)


def _dedupe(columns: list[Column]) -> list[Column]:
    """One description column (the most specific header), one quantity and one unit-price column (the rightmost);
    several amount columns are kept and chosen by arithmetic in ``_amount_column``."""
    descriptions = [c for c in columns if c.field == "description"]
    if len(descriptions) > 1:
        keep = next(
            (c for c in descriptions if re.search(r"descr|particular|detail", c.header, re.IGNORECASE)), descriptions[0]
        )
        for c in descriptions:
            if c is not keep:
                c.field = None
    for name in ("quantity", "unit_price"):
        for c in [c for c in columns if c.field == name][:-1]:
            c.field = None
    return columns


# ------------------------------------------------------------------------------------------------- rows


def _column_for(cols: list[Column], x0: float, x1: float) -> int:
    best, best_ov = -1, 0.0
    for i, c in enumerate(cols):
        ov = overlap(c.x0, c.x1, x0, x1)
        if ov > best_ov:
            best, best_ov = i, ov
    if best >= 0:
        return best
    centre = (x0 + x1) / 2
    return min(range(len(cols)), key=lambda i: abs(cols[i].cx - centre))


def _cells(line: TextLine, cols: list[Column]) -> dict[int, list[str]]:
    """Text of the row per column index; a cell spanning several columns is split into its words."""
    cells: dict[int, list[str]] = {}
    tol = line.char_width
    for chunk in line.chunks:
        touched = sum(1 for c in cols if overlap(c.x0, c.x1, chunk.x0, chunk.x1) >= tol)
        if touched >= 2 and chunk.last > chunk.first:
            for s in range(chunk.first, chunk.last + 1):
                span = line.spans[s]
                cells.setdefault(_column_for(cols, span.x0, span.x1), []).append(span.text)
        else:
            cells.setdefault(_column_for(cols, chunk.x0, chunk.x1), []).append(chunk.text)
    return cells


def _money(text: str, ctx: DocContext, last: bool) -> Decimal | None:
    found = values.money_values(text, ctx)
    if not found:
        return None
    return (found[-1] if last else found[0]).value


def _index(cols: list[Column], name: str) -> int | None:
    return next((i for i, c in enumerate(cols) if c.field == name), None)


def _geometric_row(line: TextLine, cols: list[Column], amount_col: int | None, ctx: DocContext) -> _Row:
    text_of = {i: " ".join(parts) for i, parts in _cells(line, cols).items()}
    desc_idx, qty_idx, price_idx = _index(cols, "description"), _index(cols, "quantity"), _index(cols, "unit_price")
    if desc_idx is not None:
        description = text_of.get(desc_idx)
    else:
        loose = [t for i, t in sorted(text_of.items()) if cols[i].field is None and re.search(r"[A-Za-z]{2}", t)]
        description = " ".join(loose) or None
    qty_text = text_of.get(qty_idx) if qty_idx is not None else None
    price_text = text_of.get(price_idx) if price_idx is not None else None
    amount_text = text_of.get(amount_col) if amount_col is not None else None
    return _Row(
        description=_clean_description(description),
        quantity=parse_number(qty_text, ctx.decimal_hint) if qty_text else None,
        unit_price=_money(price_text, ctx, last=False) if price_text else None,
        amount=_money(amount_text, ctx, last=True) if amount_text else None,
    )


def _pattern_row(line: TextLine, cols: list[Column], amount_col: int | None, ctx: DocContext) -> _Row | None:
    """Row read as "description  n1 n2 ... nk": trailing numbers mapped to the numeric columns in order."""
    text = line.text
    trailing: list[tuple[int, Decimal]] = []
    end = len(text.rstrip())
    for m in reversed(values.money_values(text, ctx)):
        if text[m.end : end].strip(" $€£₹¥"):
            break
        start = m.start
        trailing.insert(0, (start, m.value))
        end = start
    if not trailing:
        return None
    desc_idx = _index(cols, "description")
    after = -1 if desc_idx is None else desc_idx
    numeric = [
        i for i, c in enumerate(cols)
        if i > after and (c.field is None or c.field in _NUMERIC) and (c.field != "amount" or i == amount_col)
    ]
    if not numeric:
        return None
    pairs = list(zip(numeric[-len(trailing) :], trailing[-len(numeric) :], strict=False))
    by_field = {cols[i].field: value for i, (_, value) in pairs if cols[i].field}
    description = _clean_description(text[: pairs[0][1][0]] if pairs else text)
    return _Row(description, by_field.get("quantity"), by_field.get("unit_price"), by_field.get("amount"))


def _row(line: TextLine, cols: list[Column], amount_col: int | None, ctx: DocContext) -> _Row:
    """Geometric reading, unless the row pattern explains the numbers better (qty x price = amount)."""
    geo = _geometric_row(line, cols, amount_col, ctx)
    if geo.valid and (geo.consistent or not geo.checkable):
        if not geo.checkable:
            pattern = _pattern_row(line, cols, amount_col, ctx)
            if pattern is not None and pattern.consistent:
                return pattern
        return geo
    pattern = _pattern_row(line, cols, amount_col, ctx)
    if pattern is not None and pattern.valid and (pattern.consistent or not geo.valid):
        return pattern
    return geo


def _numeric_cells(line: TextLine, cols: list[Column]) -> int:
    return sum(
        1
        for i, parts in _cells(line, cols).items()
        if cols[i].field in _NUMERIC and any(re.search(r"\d", p) for p in parts)
    )


def _stops_table(line: TextLine, cols: list[Column]) -> bool:
    """A totals label ends the table - unless the line is a full item row ("Delivery  1  5.00  5.00")."""
    labels = labels_of(line)
    names = {lab.field for lab in labels}
    if not names & _MONEY_LABELS:
        return False
    numeric = _numeric_cells(line, cols)
    if names & _TOTAL_FIELDS:
        label_text = " ".join(lab.text.lower() for lab in labels)
        words = [w for w in re.findall(r"[A-Za-z]{2,}", line.text) if w.lower() not in label_text]
        return not (numeric >= 2 and words)
    return numeric < 2


def _wrapped(line: TextLine, prev: TextLine, cols: list[Column]) -> bool:
    """Text-only line right under an item, inside the description column: the description continues."""
    desc = next((c for c in cols if c.field == "description"), None)
    if desc is None or line.y0 - prev.y1 > 1.8 * line.height or labels_of(line):
        return False
    if _numeric_cells(line, cols) or not re.search(r"[A-Za-z]{2}", line.text):
        return False
    right = min((c.x0 for c in cols if c.x0 > desc.x1), default=float("inf"))
    return line.x0 >= desc.x0 - 2 * line.char_width and line.x1 <= right + line.char_width


# ------------------------------------------------------------------------------------------------ table


def _continuation(lines: list[TextLine], start: int, marker: bool) -> tuple[int, list[Column] | None] | None:
    """Where a table resumes on the page starting at ``start``: at a repeated header, or (after a "continued"
    marker) at the top of the page; ``None`` when the new page does not continue the table."""
    page = lines[start].page
    for j in range(start, min(len(lines), start + 15)):
        line = lines[j]
        if line.page != page:
            break
        cols = header_columns(line)
        if cols is not None:
            return j, cols
        if {lab.field for lab in labels_of(line)} & _TOTAL_FIELDS:
            break
        marker = marker or bool(vocab.CONTINUED_RE.search(line.text))
    return (start, None) if marker else None


def _amount_column(lines: list[TextLine], header: int, cols: list[Column], ctx: DocContext) -> int | None:
    """With several amount columns, the one where qty x price = amount holds most often (else the rightmost)."""
    candidates = [i for i, c in enumerate(cols) if c.field == "amount"]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    sample = [ln for ln in lines[header + 1 : header + 9] if ln.page == lines[header].page]
    scores = {i: sum(1 for ln in sample if _geometric_row(ln, cols, i, ctx).consistent) for i in candidates}
    best = max(scores.values())
    return max(i for i in candidates if scores[i] == best)


def _next_page(lines: list[TextLine], i: int) -> int | None:
    page = lines[i].page
    return next((j for j in range(i, min(len(lines), i + 8)) if lines[j].page != page), None)


def _parse_table(
    lines: list[TextLine], header: int, cols: list[Column], ctx: DocContext
) -> tuple[list[_Row], set[int]]:
    rows: list[_Row] = []
    used = {header}
    amount_col = _amount_column(lines, header, cols, ctx)
    prev: TextLine | None = None
    run, marker = 0, False
    i = header + 1
    while i < len(lines):
        line = lines[i]
        new_page = line.page != lines[i - 1].page
        if new_page or run >= _MAX_RUN:
            start = i if new_page else _next_page(lines, i)
            resume = _continuation(lines, start, marker) if start is not None else None
            if resume is None:
                break
            i, new_cols = resume
            if new_cols is not None:
                cols = new_cols
                used.add(i)
                amount_col = _amount_column(lines, i, cols, ctx)
                i += 1
            run, prev, marker = 0, None, False
            continue
        if _stops_table(line, cols):
            break
        if vocab.CONTINUED_RE.search(line.text) or vocab.PAGE_OF_RE.search(line.text):
            marker = marker or bool(vocab.CONTINUED_RE.search(line.text))
            run += 1  # "Continued on next page", "Page 1 of 2": page furniture, never an item
            i += 1
            continue
        row = _row(line, cols, amount_col, ctx)
        if row.valid:
            rows.append(row)
            used.add(i)
            prev, run = line, 0
        elif prev is not None and _wrapped(line, prev, cols):
            rows[-1].description = f"{rows[-1].description} {_clean_description(line.text)}"
            used.add(i)
            prev = line
        else:
            run += 1
        i += 1
    return rows, used


def _confidence(rows: list[_Row]) -> float:
    """Share of rows where qty x unit price = amount (rows with all three), else 0.75."""
    checkable = [r for r in rows if r.checkable]
    if not checkable:
        return _NO_CHECK_CONFIDENCE
    return sum(1 for r in checkable if r.consistent) / len(checkable)


def _items(rows: list[_Row]) -> list[LineItem]:
    return [
        LineItem(description=r.description, quantity=r.quantity, unit_price=r.unit_price, amount=r.amount)
        for r in rows
    ]


# -------------------------------------------------------------------------------------- header-less receipt


def _receipt_row(line: TextLine, ctx: DocContext) -> _Row | None:
    """"Latte (large)   5.25", "2 x Croissant   7.00": text, then one or two amounts with cents at the line end."""
    text = line.text
    if labels_of(line) or vocab.PAYMENT_WORDS.search(text) or find_dates(text) or vocab.KEY_RE.match(text):
        return None
    found = [m for m in values.money_values(text, ctx) if values.has_fraction(m) or m.currency]
    if not found or text[found[-1].end :].strip():
        return None
    amounts = found[-2:]
    head = text[: amounts[0].start]
    quantity: Decimal | None = None
    prefix, suffix = _QTY_PREFIX.match(head), _QTY_SUFFIX.search(head)
    if prefix:
        quantity, head = Decimal(prefix.group(1)), head[prefix.end() :]
    elif suffix:
        quantity, head = Decimal(suffix.group(1)), head[: suffix.start()]
    description = _clean_description(head)
    if description is None or (re.search(r"\d", description) and len(re.findall(r"[A-Za-z]", description)) < 3):
        return None
    unit_price = amounts[0].value if len(amounts) == 2 else None
    return _Row(description, quantity, unit_price, amounts[-1].value)


def _headerless(lines: list[TextLine], ctx: DocContext) -> LineItemTable | None:
    """Item lines directly above the first totals-block line (subtotal, tax, total...) of a receipt without a
    table header."""
    anchor = None
    for line in lines:
        labels = [lab for lab in labels_of(line) if lab.field in _MONEY_LABELS]
        if labels and values.labelled_amount(line.text[labels[0].end :], ctx) is not None:
            anchor = line.index
            break
    if anchor is None:
        return None
    rows: list[tuple[int, _Row]] = []
    j = anchor - 1
    while j >= 0 and lines[j].page == lines[anchor].page:
        row = _receipt_row(lines[j], ctx)
        if row is None:
            break
        rows.insert(0, (j, row))
        j -= 1
    if not rows:
        return None
    parsed = [r for _, r in rows]
    return LineItemTable(_items(parsed), min(_confidence(parsed), _HEADERLESS_CONFIDENCE), None, {j for j, _ in rows})


# ---------------------------------------------------------------------------------------------- entry points


def find_line_items(lines: list[TextLine], ctx: DocContext) -> LineItemTable | None:
    """The document's line-item table (the first header that yields rows), else a header-less receipt list."""
    for line in lines:
        cols = header_columns(line)
        if cols is None:
            continue
        rows, used = _parse_table(lines, line.index, cols, ctx)
        if rows:
            return LineItemTable(_items(rows), _confidence(rows), line.index, used)
    return _headerless(lines, ctx)


def extract_line_items(lines: Sequence[Line]) -> tuple[list[LineItem], float]:
    """Line items of a document (or segment) and the table's confidence; ``([], 0.0)`` when there is no table."""
    text_lines = build_text_lines(lines)
    if not text_lines:
        return [], 0.0
    ctx = build_context([ln.text for ln in text_lines], "USD", "auto")
    table = find_line_items(text_lines, ctx)
    if table is None:
        return [], 0.0
    return table.items, table.confidence
