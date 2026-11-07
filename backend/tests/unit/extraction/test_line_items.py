"""Line-item table detection and parsing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.extraction.line_items import extract_line_items
from app.extraction.rules import lines_from_text
from tests.unit.extraction.rules_layouts import US, as_lines, item, pages_as_lines, span_line

MODES = ["text", "pdf", "ocr"]


def _dump(items: list) -> list[dict]:
    return [li.model_dump() for li in items]


@pytest.mark.parametrize("mode", MODES)
def test_line_items_basic_table(mode: str) -> None:
    items, confidence = extract_line_items(as_lines(US, mode))
    assert _dump(items) == [
        item("Widget A", "10", "25.00", "250.00"),
        item("Widget B (blue)", "5", "40.00", "200.00"),
        item("Installation service", "1", "150.00", "150.00"),
    ]
    assert confidence == pytest.approx(1.0)


@pytest.mark.parametrize("mode", MODES)
def test_line_items_wrapped_description(mode: str) -> None:
    text = """\
Description                         Qty    Unit Price      Amount
Annual maintenance contract           1      1,200.00    1,200.00
covering servers and network
equipment (12 months)
Spare parts kit                       2         75.00      150.00
Subtotal                                                 1,350.00
"""
    items, _ = extract_line_items(as_lines(text, mode))
    assert _dump(items) == [
        item(
            "Annual maintenance contract covering servers and network equipment (12 months)", "1", "1200.00", "1200.00"
        ),
        item("Spare parts kit", "2", "75.00", "150.00"),
    ]


def test_line_items_qty_x_price_check() -> None:
    text = """\
Description            Qty     Price     Amount
Widget                   2     10.00      20.00
Gadget                   3      5.00      15.00
Gizmo                    4      2.50      12.00
"""
    items, confidence = extract_line_items(lines_from_text(text))
    assert len(items) == 3
    assert items[2].amount == Decimal("12.00")  # printed value kept, even though 4 x 2.50 = 10.00
    assert confidence == pytest.approx(2 / 3)

    amounts_only = (
        "Description                   Amount\n"
        "Consulting                  500.00\n"
        "Travel                       80.00\n"
    )
    items, confidence = extract_line_items(lines_from_text(amounts_only))
    assert [i.amount for i in items] == [Decimal("500.00"), Decimal("80.00")]
    assert [i.quantity for i in items] == [None, None]
    assert confidence == pytest.approx(0.75)


def test_line_items_stop_at_totals() -> None:
    text = """\
Description            Qty     Price     Amount
Widget                   2     10.00      20.00
Delivery                 1      5.00       5.00
                             Subtotal     25.00
                             Tax 10%       2.50
                             Total        27.50

Payment schedule              Due       Amount
First instalment              1         13.75
"""
    items, _ = extract_line_items(lines_from_text(text))
    assert _dump(items) == [item("Widget", "2", "10.00", "20.00"), item("Delivery", "1", "5.00", "5.00")]


def test_line_items_none_when_no_table() -> None:
    text = "ACME Corp\nInvoice No: 1\nDate: 2024-01-01\nThank you for your business\nTotal: 110.00\n"
    assert extract_line_items(lines_from_text(text)) == ([], 0.0)
    assert extract_line_items([]) == ([], 0.0)


def test_line_items_header_words_in_a_sentence_are_not_a_table() -> None:
    text = "Please describe each item and the agreed price when you reply\nTotal: 110.00\n"
    assert extract_line_items(lines_from_text(text)) == ([], 0.0)


def test_line_items_right_aligned_geometry() -> None:
    """PDF geometry where numbers are right-aligned and do not start under their header text."""
    lines = [
        span_line(100, ("Item", 40), ("Quantity", 300), ("Unit Cost", 380), ("Line Total", 480)),
        span_line(116, ("Copper cable 2.5mm", 40), ("120", 322), ("1.10", 404), ("132.00", 512)),
        span_line(132, ("Junction box", 40), ("8", 333), ("4.50", 404), ("36.00", 517)),
        span_line(170, ("Subtotal", 380), ("168.00", 512)),
    ]
    items, confidence = extract_line_items(lines)
    assert _dump(items) == [
        item("Copper cable 2.5mm", "120", "1.10", "132.00"),
        item("Junction box", "8", "4.50", "36.00"),
    ]
    assert confidence == pytest.approx(1.0)


@pytest.mark.parametrize("mode", MODES)
def test_line_items_table_continues_on_next_page(mode: str) -> None:
    page1 = """\
Description               Qty     Price      Amount
Item one                    1     10.00       10.00
Item two                    2     10.00       20.00
Continued on next page                  Page 1 of 2
"""
    page2 = """\
Invoice No: 77 (continued)              Page 2 of 2
Description               Qty     Price      Amount
Item three                  3     10.00       30.00
Total                                         60.00
"""
    lines, _ = pages_as_lines([page1, page2], mode)
    items, _ = extract_line_items(lines)
    assert [i.description for i in items] == ["Item one", "Item two", "Item three"]


def test_line_items_rows_without_column_gaps_use_row_pattern() -> None:
    text = "Description Qty Price Amount\nWidget blue 2 10.00 20.00\nGadget 1 5.50 5.50\nTotal 25.50\n"
    items, confidence = extract_line_items(lines_from_text(text))
    assert _dump(items) == [item("Widget blue", "2", "10.00", "20.00"), item("Gadget", "1", "5.50", "5.50")]
    assert confidence == pytest.approx(1.0)


def test_line_items_headerless_receipt() -> None:
    text = """\
CORNER DELI
Date: 2024-03-05
Turkey sandwich              8.50
Iced tea                     2.75
3 x Cookie                   4.50
SUBTOTAL                    15.75
TOTAL                       15.75
CASH                        20.00
"""
    items, confidence = extract_line_items(lines_from_text(text))
    assert _dump(items) == [
        item("Turkey sandwich", None, None, "8.50"),
        item("Iced tea", None, None, "2.75"),
        item("Cookie", "3", None, "4.50"),
    ]
    assert 0.0 < confidence <= 0.75


def test_line_items_european_decimal_comma() -> None:
    text = """\
Bezeichnung / Description      Menge / Qty     Preis / Price     Betrag / Amount
Schrauben M6                           100            0,12 €            12,00 €
Dübel 8mm                               50            0,30 €            15,00 €
"""
    items, confidence = extract_line_items(lines_from_text(text))
    assert _dump(items) == [
        item("Schrauben M6", "100", "0.12", "12.00"),
        item("Dübel 8mm", "50", "0.30", "15.00"),
    ]
    assert confidence == pytest.approx(1.0)
