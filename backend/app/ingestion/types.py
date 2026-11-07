"""Types shared by every ingestion path.

A single layout model serves digital PDF text and OCR output alike: ``Span`` (a word or text box with its
bounding box, in page units, origin top-left) -> ``Line`` (spans on one visual line, left to right) ->
``Page``. Spreadsheets produce ``Sheet`` grids instead of pages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal


class FileKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"


@dataclass(slots=True)
class Span:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float | None = None  # OCR confidence in [0, 1]; None for digital text

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def char_width(self) -> float:
        n = len(self.text.strip()) or 1
        return max(self.width / n, 0.1)


@dataclass(slots=True)
class Line:
    spans: list[Span] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Spans joined left to right; a horizontal gap of at least two average character widths becomes two
        spaces so that column structure ("Total:      110.00") survives as text."""
        parts: list[str] = []
        prev: Span | None = None
        for span in self.spans:
            if prev is not None:
                gap = span.x0 - prev.x1
                char_w = (prev.char_width + span.char_width) / 2
                parts.append("  " if gap >= 2 * char_w else " ")
            parts.append(span.text)
            prev = span
        return "".join(parts)

    @property
    def x0(self) -> float:
        return min((s.x0 for s in self.spans), default=0.0)

    @property
    def x1(self) -> float:
        return max((s.x1 for s in self.spans), default=0.0)

    @property
    def y0(self) -> float:
        return min((s.y0 for s in self.spans), default=0.0)

    @property
    def y1(self) -> float:
        return max((s.y1 for s in self.spans), default=0.0)


@dataclass(slots=True)
class Page:
    number: int  # 1-based
    lines: list[Line] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    method: Literal["text", "ocr", "none"] = "text"
    ocr_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


CellValue = str | int | Decimal | date | datetime | bool | None


@dataclass(slots=True)
class Sheet:
    name: str
    index: int  # 0-based position in the workbook (CSV = 0)
    rows: list[list[CellValue]] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Tab-separated rendering used for search indexing and LLM input."""
        out: list[str] = []
        for row in self.rows:
            out.append("\t".join("" if v is None else _cell_str(v) for v in row).rstrip())
        return "\n".join(out)


def _cell_str(value: CellValue) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


@dataclass(slots=True)
class ParsedDocument:
    kind: FileKind
    pages: list[Page] = field(default_factory=list)
    sheets: list[Sheet] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages) if self.pages else len(self.sheets)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.method == "ocr")

    @property
    def text(self) -> str:
        if self.pages:
            return "\n\n".join(p.text for p in self.pages)
        return "\n\n".join(s.text for s in self.sheets)


# (current, total, stage) — e.g. (12, 1000, "ocr")
ProgressCallback = Callable[[int, int, str], None]


class IngestionError(Exception):
    """A document could not be ingested. ``permanent`` errors are not retried."""

    code: str = "ingestion_error"
    permanent: bool = True

    def __init__(self, message: str, *, code: str | None = None, permanent: bool | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if permanent is not None:
            self.permanent = permanent


class UnsupportedFileError(IngestionError):
    code = "unsupported_file_type"


class CorruptFileError(IngestionError):
    code = "corrupt_file"


class EncryptedFileError(IngestionError):
    code = "encrypted_file"


class FileTooLargeError(IngestionError):
    code = "file_too_large"


class EmptyFileError(IngestionError):
    code = "empty_file"
