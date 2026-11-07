"""Group text spans into visual lines.

The same routine serves word spans from digital PDFs and line-level boxes from OCR: a span joins the line
whose vertical band it overlaps by at least half of the smaller of the two heights. A line's band is the mean
top and mean bottom of its spans, so one tall span (a large total amount, a logo letter) cannot chain two
neighbouring rows together the way a union of boxes would; and a span more than four times taller (or
shorter) than a line's band never joins it, so a page-high stray box cannot swallow the rows it crosses.

Spans are swept top to bottom; a line stops accepting spans once the sweep has passed its band by a few
typical (median) line heights, which keeps the set of open lines small and the sweep close to linear even on
hostile pages. Callers cap a page at :data:`MAX_SPANS_PER_PAGE` spans with :func:`limit_spans`.
"""

from __future__ import annotations

import statistics

from app.ingestion.types import Line, Span

MAX_SPANS_PER_PAGE = 20_000
_MIN_OVERLAP = 0.5  # of the smaller height
_MAX_HEIGHT_RATIO = 4.0  # between a span and the band of a line it joins
_REACH_HEIGHTS = 8.0  # how many median heights above the sweep a line stays open
_DUPLICATE_IOU = 0.6  # same text drawn twice at almost the same place ("fake bold")
_MIN_HEIGHT = 1e-6


def limit_spans(spans: list[Span]) -> tuple[list[Span], bool]:
    """The first :data:`MAX_SPANS_PER_PAGE` spans of a page, and whether any had to be dropped."""
    if len(spans) <= MAX_SPANS_PER_PAGE:
        return spans, False
    return spans[:MAX_SPANS_PER_PAGE], True


class _LineBuilder:
    __slots__ = ("count", "spans", "sum_y0", "sum_y1")

    def __init__(self, span: Span) -> None:
        self.spans = [span]
        self.sum_y0 = span.y0
        self.sum_y1 = span.y1
        self.count = 1

    @property
    def y0(self) -> float:
        return self.sum_y0 / self.count

    @property
    def y1(self) -> float:
        return self.sum_y1 / self.count

    @property
    def cy(self) -> float:
        return (self.sum_y0 + self.sum_y1) / (2 * self.count)

    def add(self, span: Span) -> None:
        self.spans.append(span)
        self.sum_y0 += span.y0
        self.sum_y1 += span.y1
        self.count += 1


def build_lines(spans: list[Span]) -> list[Line]:
    """Group spans into lines, ordered top to bottom with spans left to right.

    Blank spans are dropped, and so is a span that repeats the text of an overlapping span on the same line.
    """
    items = [s for s in spans if s.text.strip()]
    if not items:
        return []
    items.sort(key=lambda s: (s.cy, s.x0))
    heights = [max(s.height, _MIN_HEIGHT) for s in items]
    # Spans arrive in order of their centre, so a span of height h reaches at most h / 2 above the sweep.
    # Using the median height (not one outlier's) bounds how long lines stay open.
    reach = min(max(heights), _REACH_HEIGHTS * statistics.median(heights)) / 2

    active: list[_LineBuilder] = []
    finished: list[_LineBuilder] = []
    for span in items:
        cutoff = span.cy - reach
        if any(line.y1 < cutoff for line in active):
            finished.extend(line for line in active if line.y1 < cutoff)
            active = [line for line in active if line.y1 >= cutoff]
        target = _best_line(active, span)
        if target is None:
            active.append(_LineBuilder(span))
        else:
            target.add(span)

    builders = finished + active
    builders.sort(key=lambda line: (line.cy, min(s.x0 for s in line.spans)))
    return [Line(spans=_ordered_unique(line.spans)) for line in builders]


def _best_line(candidates: list[_LineBuilder], span: Span) -> _LineBuilder | None:
    best: _LineBuilder | None = None
    best_key = (0.0, 0.0)
    span_height = max(span.height, _MIN_HEIGHT)
    for line in candidates:
        overlap = min(line.y1, span.y1) - max(line.y0, span.y0)
        if overlap <= 0:
            continue
        line_height = max(line.y1 - line.y0, _MIN_HEIGHT)
        if max(line_height, span_height) > _MAX_HEIGHT_RATIO * min(line_height, span_height):
            continue
        ratio = overlap / min(line_height, span_height)
        if ratio < _MIN_OVERLAP:
            continue
        key = (ratio, -abs(line.cy - span.cy))
        if best is None or key > best_key:
            best, best_key = line, key
    return best


def _ordered_unique(spans: list[Span]) -> list[Span]:
    ordered = sorted(spans, key=lambda s: (s.x0, s.y0))
    kept: list[Span] = []
    for span in ordered:
        if any(k.text == span.text and _iou(k, span) >= _DUPLICATE_IOU for k in kept[-3:]):
            continue
        kept.append(span)
    return kept


def _iou(a: Span, b: Span) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0
