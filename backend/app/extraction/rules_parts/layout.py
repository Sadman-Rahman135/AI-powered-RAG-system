"""Line geometry shared by every rules component.

A ``TextLine`` wraps an ingestion ``Line`` with its rendered text, the character offset of every span inside that
text and its ``chunks``: runs of spans separated by less than a column gap. ``Line.text`` renders a column gap
(at least two average character widths) as a double space, so a chunk is exactly a double-space separated part
of the text - one "cell" of the visual layout. The same code therefore serves digital PDF words, OCR text boxes
and plain text with pseudo-geometry.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median

from app.extraction.labels import LabelMatch
from app.ingestion.types import Line, Span


@dataclass(slots=True)
class Chunk:
    text: str
    start: int  # character offsets in the line text
    end: int
    x0: float
    x1: float
    first: int  # index of the first / last span of the chunk
    last: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass(slots=True)
class TextLine:
    index: int  # position in reading order (page, top to bottom)
    source: int  # index of the line in the caller's sequence
    text: str
    spans: list[Span]
    starts: list[int]  # character offset of each span in ``text``
    chunks: list[Chunk]
    page: int
    y0: float
    y1: float
    conf: float | None  # mean OCR confidence of the spans (None for digital text)
    labels_cache: list[LabelMatch] | None = field(default=None, repr=False)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def x0(self) -> float:
        return self.spans[0].x0

    @property
    def x1(self) -> float:
        return max(s.x1 for s in self.spans)

    @property
    def char_width(self) -> float:
        widths = [s.char_width for s in self.spans]
        return sum(widths) / len(widths)

    def span_index_at(self, pos: int) -> int:
        """Index of the span covering character ``pos`` (the span to its left when ``pos`` is in a gap)."""
        return max(bisect_right(self.starts, pos) - 1, 0)

    def x_at(self, pos: int) -> float:
        """Horizontal position of character ``pos``, interpolated inside its span."""
        i = self.span_index_at(pos)
        span, start = self.spans[i], self.starts[i]
        n = max(len(span.text), 1)
        offset = min(max(pos - start, 0), n)
        return span.x0 + span.width * offset / n

    def x_range(self, start: int, end: int) -> tuple[float, float]:
        return self.x_at(start), self.x_at(max(end, start + 1))

    def chunk_at(self, pos: int) -> Chunk:
        for chunk in self.chunks:
            if pos < chunk.end:
                return chunk
        return self.chunks[-1]


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def best_chunk(line: TextLine, x0: float, x1: float, tol: float = 0.0) -> Chunk | None:
    """The chunk of ``line`` with the largest horizontal overlap with ``[x0, x1]`` (widened by ``tol``)."""
    best: Chunk | None = None
    best_ov = 0.0
    for chunk in line.chunks:
        ov = overlap(chunk.x0, chunk.x1, x0 - tol, x1 + tol)
        if ov > best_ov:
            best, best_ov = chunk, ov
    return best


def _offsets(text: str, spans: Sequence[Span]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for span in spans:
        found = text.find(span.text, pos)
        found = pos if found < 0 else found
        starts.append(found)
        pos = found + len(span.text)
    return starts


def _chunks(text: str, spans: Sequence[Span], starts: Sequence[int]) -> list[Chunk]:
    chunks: list[Chunk] = []
    first = 0
    for i in range(1, len(spans) + 1):
        boundary = i == len(spans) or starts[i] - (starts[i - 1] + len(spans[i - 1].text)) >= 2
        if boundary:
            start, end = starts[first], starts[i - 1] + len(spans[i - 1].text)
            x1 = max(s.x1 for s in spans[first:i])
            chunks.append(Chunk(text[start:end], start, end, spans[first].x0, x1, first, i - 1))
            first = i
    return chunks


def _derived_pages(lines: Sequence[Line]) -> list[int]:
    """Page number per line from vertical resets (lines are top-to-bottom within a page)."""
    heights = [ln.y1 - ln.y0 for ln in lines if ln.spans]
    step = 5 * (median(heights) if heights else 10.0)
    pages: list[int] = []
    page, prev_y = 1, None
    for line in lines:
        y = line.y0 if line.spans else prev_y
        if prev_y is not None and y is not None and y < prev_y - max(step, 50.0):
            page += 1
        pages.append(page)
        prev_y = y if y is not None else prev_y
    return pages


_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})


def _clean_span(span: Span) -> Span:
    """Span text stripped, typographic ligatures expanded ("muﬃn" -> "muffin"; PDF text keeps them)."""
    text = span.text.strip().translate(_LIGATURES)
    return span if text == span.text else Span(text, span.x0, span.y0, span.x1, span.y1, span.conf)


def build_text_lines(lines: Sequence[Line], page_numbers: Sequence[int] | None = None) -> list[TextLine]:
    """``TextLine`` records in reading order (page, then top to bottom); lines without text are dropped."""
    pages = list(page_numbers) if page_numbers is not None and len(page_numbers) == len(lines) else None
    if pages is None:
        pages = _derived_pages(lines)
    records: list[TextLine] = []
    for source, line in enumerate(lines):
        spans = sorted((s for s in line.spans if s.text and s.text.strip()), key=lambda s: s.x0)
        if not spans:
            continue
        spans = [_clean_span(s) for s in spans]
        text = Line(spans=spans).text
        starts = _offsets(text, spans)
        confs = [s.conf for s in spans if s.conf is not None]
        records.append(
            TextLine(
                index=0,
                source=source,
                text=text,
                spans=spans,
                starts=starts,
                chunks=_chunks(text, spans, starts),
                page=pages[source],
                y0=min(s.y0 for s in spans),
                y1=max(s.y1 for s in spans),
                conf=sum(confs) / len(confs) if confs else None,
            )
        )
    records.sort(key=lambda r: (r.page, r.y0, r.x0))
    for i, record in enumerate(records):
        record.index = i
    return records


def median_height(lines: Sequence[TextLine]) -> float:
    return median([ln.height for ln in lines]) if lines else 10.0
