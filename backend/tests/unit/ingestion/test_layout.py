"""Grouping word spans (digital PDFs) and line boxes (OCR) into visual lines."""

from __future__ import annotations

import random
import time

import pytest

from app.ingestion import layout
from app.ingestion.layout import build_lines, limit_spans
from app.ingestion.types import Span


def _texts(spans: list[Span]) -> list[str]:
    return [line.text for line in build_lines(spans)]


def test_build_lines_groups_and_orders() -> None:
    # PyMuPDF-style word spans (points) with a slightly larger amount font on the subtotal row.
    words = [
        Span("Invoice", 72.0, 87.1, 110.0, 103.6),
        Span("No:", 113.4, 87.1, 132.0, 103.6),
        Span("INV-2024-0042", 135.4, 87.1, 216.7, 103.6),
        Span("Subtotal:", 72.0, 117.1, 118.1, 133.6),
        Span("20.00", 450.0, 115.0, 480.0, 134.2),
    ]
    # RapidOCR-style line boxes (pixels): label and amount of one row are separate boxes far apart.
    boxes = [
        Span("Invoice No: INV-2024-0042", 192.5, 237.0, 676.2, 290.0, conf=0.99),
        Span("Date: 2024-03-05", 193.7, 326.6, 466.1, 370.2, conf=0.97),
        Span("Subtotal:", 192.5, 519.9, 339.3, 567.1, conf=0.99),
        Span("20.00", 1240.9, 518.7, 1340.7, 568.2, conf=0.99),
        Span("Total: USD 22.00", 192.5, 602.4, 460.2, 648.4, conf=0.99),
    ]
    for spans, expected in (
        (words, ["Invoice No: INV-2024-0042", "Subtotal:  20.00"]),
        (boxes, ["Invoice No: INV-2024-0042", "Date: 2024-03-05", "Subtotal:  20.00", "Total: USD 22.00"]),
    ):
        shuffled = spans[:]
        random.Random(3).shuffle(shuffled)
        lines = build_lines(shuffled)
        assert [line.text for line in lines] == expected
        for line in lines:
            assert [s.x0 for s in line.spans] == sorted(s.x0 for s in line.spans)
        assert [line.y0 for line in lines] == sorted(line.y0 for line in lines)


def test_vertical_overlap_threshold_is_half_the_smaller_height() -> None:
    # 4 of 10 units overlap (< 50 %) -> separate lines; 6 of 10 (>= 50 %) -> one line.
    apart = [Span("A", 0, 100, 10, 110), Span("B", 50, 106, 60, 116)]
    together = [Span("C", 0, 200, 10, 210), Span("D", 50, 204, 60, 214)]
    assert _texts(apart) == ["A", "B"]
    assert _texts(together) == ["C  D"]


def test_tall_span_does_not_chain_neighbouring_lines() -> None:
    spans = [
        Span("Line one", 0, 100, 80, 110),
        Span("Line two", 0, 113, 80, 123),
        Span("BIG", 300, 99, 360, 124),  # e.g. a large total amount spanning both rows
    ]
    lines = build_lines(spans)
    assert len(lines) == 2
    assert {line.spans[0].text for line in lines} == {"Line one", "Line two"}


def test_blank_and_duplicate_spans_are_dropped() -> None:
    spans = [
        Span("Total", 72.0, 100.0, 100.0, 110.0),
        Span("Total", 72.4, 100.2, 100.4, 110.2),  # "fake bold": the same word drawn twice, offset slightly
        Span("  ", 120.0, 100.0, 130.0, 110.0),
        Span("110.00", 200.0, 100.0, 236.0, 110.0),
    ]
    assert _texts(spans) == ["Total  110.00"]
    assert build_lines([]) == []


def test_one_tall_span_and_many_tiny_lines_stay_fast_and_separate() -> None:
    # Hostile layout: a span as tall as the page next to 16 000 one-unit lines must neither keep every line
    # "active" (quadratic) nor swallow the tiny lines into its own line.
    tiny = [Span(f"w{i}", 0.0, 2.0 * i, 5.0, 2.0 * i + 1.0) for i in range(16_000)]
    spans = [*tiny, Span("TALL", 100.0, 0.0, 120.0, 32_000.0)]
    random.Random(5).shuffle(spans)
    start = time.perf_counter()
    lines = build_lines(spans)
    assert time.perf_counter() - start < 2.0
    assert len(lines) == 16_001
    assert sum(len(line.spans) for line in lines) == 16_001


def test_limit_spans_caps_a_page(monkeypatch: pytest.MonkeyPatch) -> None:
    spans = [Span(f"w{i}", 0.0, float(i), 5.0, i + 1.0) for i in range(30)]
    assert limit_spans(spans) == (spans, False)
    monkeypatch.setattr(layout, "MAX_SPANS_PER_PAGE", 10)
    kept, truncated = limit_spans(spans)
    assert kept == spans[:10] and truncated
