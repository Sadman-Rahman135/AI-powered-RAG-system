"""Splitting long multi-invoice PDFs into one segment per invoice."""

from __future__ import annotations

import time

from app.extraction.rules import lines_from_text
from app.extraction.segment import Segment, segment_pages
from app.ingestion.types import Page


def _page(number: int, text: str) -> Page:
    return Page(number=number, lines=lines_from_text(text))


def _invoice_header(number: str, customer: str = "Globex Inc") -> str:
    return (
        "ACME Corp\n"
        "INVOICE\n"
        f"Invoice No: {number}            Date: 2024-03-05\n"
        "Bill To:\n"
        f"{customer}\n"
        "Description              Qty     Price      Amount\n"
    )


_ITEMS = "Widget                     2     10.00       20.00\nGadget                     1      5.00        5.00\n"
_TOTALS = (
    "Subtotal                                    25.00\n"
    "Tax                                          2.50\n"
    "Total                                       27.50\n"
)


def _spans(segments: list[Segment]) -> list[tuple[int, int, str | None]]:
    return [(s.page_start, s.page_end, s.invoice_number_hint) for s in segments]


def test_segment_three_invoices_mixed_lengths() -> None:
    pages = [
        _page(1, _invoice_header("INV-001") + _ITEMS + _TOTALS),
        _page(2, _invoice_header("INV-002", "Initech") + _ITEMS + _ITEMS),
        _page(3, _ITEMS + _TOTALS),  # no signal: continues INV-002
        _page(4, _invoice_header("INV-003", "Hooli") + _ITEMS + _TOTALS),
    ]
    segments = segment_pages(pages)
    assert _spans(segments) == [(1, 1, "INV-001"), (2, 3, "INV-002"), (4, 4, "INV-003")]
    assert [p.number for p in segments[1].pages] == [2, 3]


def test_segment_repeated_number_on_every_page_stays_together() -> None:
    pages = [
        _page(1, _invoice_header("INV-010") + _ITEMS),
        _page(2, "Invoice No: INV-010 (continued)\n" + _ITEMS),
        _page(3, "Invoice No: INV-010\n" + _ITEMS + _TOTALS),
        _page(4, _invoice_header("INV-011") + _ITEMS + _TOTALS),
    ]
    assert _spans(segment_pages(pages)) == [(1, 3, "INV-010"), (4, 4, "INV-011")]


def test_segment_page_k_of_n() -> None:
    pages = [
        _page(1, "Northwind Ltd\nINVOICE\nPage 1 of 2\n" + _ITEMS),
        _page(2, "Page 2 of 2\n" + _ITEMS + _TOTALS),
        _page(3, "Northwind Ltd\nINVOICE\nPage 1 of 3\n" + _ITEMS),
        _page(4, "Page 2 of 3\n" + _ITEMS),
        _page(5, "Page 3 of 3\n" + _TOTALS),
        _page(6, "Northwind Ltd\nINVOICE\nPage 1 of 1\n" + _ITEMS + _TOTALS),
    ]
    assert _spans(segment_pages(pages)) == [(1, 2, None), (3, 5, None), (6, 6, None)]


def test_segment_page_marker_beats_a_misread_number() -> None:
    pages = [
        _page(1, _invoice_header("INV-0042") + "Page 1 of 2\n" + _ITEMS),
        _page(2, "Invoice No: INV-OO42                Page 2 of 2\n" + _ITEMS + _TOTALS),  # OCR: O for 0
        _page(3, _invoice_header("INV-0043") + "Page 1 of 1\n" + _ITEMS + _TOTALS),
    ]
    assert _spans(segment_pages(pages)) == [(1, 2, "INV-0042"), (3, 3, "INV-0043")]


def test_segment_single_invoice_no_numbers() -> None:
    pages = [
        _page(1, "ACME Corp\nINVOICE\nBill To:\nGlobex Inc\n" + _ITEMS),
        _page(2, _ITEMS + _ITEMS),
        _page(3, _ITEMS + _TOTALS + "Thank you for your business\n"),
    ]
    segments = segment_pages(pages)
    assert _spans(segments) == [(1, 3, None)]
    assert len(segments[0].pages) == 3


def test_segment_new_header_after_totals_starts_new_invoice() -> None:
    pages = [
        _page(1, "ACME Corp\nINVOICE\nBill To:\nGlobex Inc\n" + _ITEMS + _TOTALS),
        _page(2, "ACME Corp\nINVOICE\nBill To:\nInitech\n" + _ITEMS + _TOTALS),
        _page(3, ""),  # blank page: no signal, stays with the current invoice
    ]
    assert _spans(segment_pages(pages)) == [(1, 1, None), (2, 3, None)]


def test_segment_empty_and_single_page() -> None:
    assert segment_pages([]) == []
    only = segment_pages([_page(1, "")])
    assert _spans(only) == [(1, 1, None)]


def test_segment_1000_pages_fast() -> None:
    pages: list[Page] = []
    expected: list[tuple[int, int, str | None]] = []
    number = 1
    invoice = 0
    while number <= 1000:
        invoice += 1
        length = min(1 + invoice % 4, 1001 - number)  # invoices of 1..4 pages
        inv_no = f"INV-{invoice:05d}"
        for k in range(length):
            text = _invoice_header(inv_no) + _ITEMS * 8 if k == 0 else _ITEMS * 12
            if k == length - 1:
                text += _TOTALS
            pages.append(_page(number + k, text))
        expected.append((number, number + length - 1, inv_no))
        number += length

    start = time.perf_counter()
    segments = segment_pages(pages)
    elapsed = time.perf_counter() - start

    assert _spans(segments) == expected
    assert elapsed < 1.0, f"segmentation took {elapsed:.3f}s for 1000 pages"
