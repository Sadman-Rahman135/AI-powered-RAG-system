"""Spreadsheet parsing: XLSX (openpyxl, read-only, defusedxml), XLS (xlrd) and CSV/TSV text.

Every worksheet becomes a :class:`Sheet`; the result has no pages. Trailing empty rows and columns are trimmed
and rows are padded to one width; leading empty rows and columns are kept, so ``rows[i][j]`` is the cell at
sheet row ``i + 1``, column ``j + 1`` and row numbers reported downstream match what the user sees in Excel.

Cell types: workbooks keep numbers (integers as ``int``, other floats as ``Decimal(str(value))`` - in .xls every
number is a float, so integral ones become ``int`` like openpyxl reads them from .xlsx), dates (a midnight
datetime is a ``date``), datetimes and booleans; time-only cells become ISO time strings in both formats; text
is stripped and an empty cell is ``None``. CSV cells stay text - typed conversion (locale-aware money and dates)
belongs to the extractor. CSV encoding: BOM, then strict UTF-8, then charset-normalizer with a preference for
Windows-1252 among near-equal candidates (it ranks short Western European text as Baltic or Central European
otherwise); delimiter: ``csv.Sniffer`` on the first 8 KiB among ``, ; TAB |``.

Limits: ``MAX_SHEET_ROWS`` / ``MAX_SHEET_COLS`` of data per sheet, and for XLSX a zip-bomb guard (total
uncompressed size and per-member compression ratio) before openpyxl touches the file. A worksheet's declared
dimension is ignored (it can claim a million rows).
"""

from __future__ import annotations

import codecs
import csv
import functools
import io
import math
import warnings
import zipfile
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import openpyxl
import xlrd
from charset_normalizer import from_bytes

from app.config import Settings, get_settings
from app.ingestion.guard import typed_errors
from app.ingestion.types import (
    CellValue,
    CorruptFileError,
    EncryptedFileError,
    FileKind,
    FileTooLargeError,
    IngestionError,
    ParsedDocument,
    Sheet,
    UnsupportedFileError,
)

CSV_SHEET_NAME = "CSV"

_DELIMITERS = ",;\t|"
_SNIFF_CHARS = 8 * 1024  # csv.Sniffer's quote regex is quadratic on hostile samples (64 KiB: ~30 s)
_ENCODING_SAMPLE_BYTES = 1024 * 1024
_WESTERN_TOLERANCE = 0.05  # how much more "chaos" cp1252 may show than charset-normalizer's best guess
_RATIO_EXEMPT_BYTES = 1024 * 1024  # a member smaller than this cannot be a bomb, whatever its ratio
_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),  # before UTF-16: the UTF-32 LE BOM starts with the UTF-16 LE BOM
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

Reader = Callable[[Path, Settings], tuple[list[Sheet], dict[str, Any], list[str]]]


def parse_spreadsheet(path: Path, kind: FileKind) -> ParsedDocument:
    """Parse every sheet of an XLSX/XLS workbook, or a CSV file as one sheet.

    Only the content matters, never the file name. Every failure is a typed
    :class:`~app.ingestion.types.IngestionError` (see :mod:`app.ingestion.guard`).
    """
    readers: dict[FileKind, Reader] = {FileKind.XLSX: _read_xlsx, FileKind.XLS: _read_xls, FileKind.CSV: _read_csv}
    reader = readers.get(kind)
    if reader is None:
        raise UnsupportedFileError(f"{kind} is not a spreadsheet kind")
    with typed_errors("spreadsheet"):
        sheets, metadata, notes = reader(path, get_settings())
    return ParsedDocument(kind=kind, sheets=sheets, warnings=notes, metadata=metadata)


class _Grid:
    """One sheet's rows: trims trailing empty rows and columns and enforces the size limits."""

    def __init__(self, name: str, max_rows: int, max_cols: int) -> None:
        self.name = name
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.rows: list[list[CellValue]] = []
        self.pending_empty = 0  # empty rows seen since the last row with data
        self.width = 0

    def add(self, raw: Sequence[Any], convert: Callable[[Any], CellValue]) -> None:
        if len(raw) > self.max_cols:
            overflow = raw[self.max_cols :]
            # tuple/list .count is C-level: cheap even for rows padded to Excel's 16 384 columns
            if overflow.count(None) != len(overflow) and any(convert(v) is not None for v in overflow):
                raise FileTooLargeError(f"sheet {self.name!r} has data beyond column {self.max_cols}")
            raw = raw[: self.max_cols]
        values = [convert(v) for v in raw]
        last = len(values)
        while last and values[last - 1] is None:
            last -= 1
        if not last:
            self.pending_empty += 1
            return
        row_number = len(self.rows) + self.pending_empty + 1
        if row_number > self.max_rows:
            raise FileTooLargeError(f"sheet {self.name!r} has more than {self.max_rows} rows")
        self.rows.extend([] for _ in range(self.pending_empty))
        self.pending_empty = 0
        del values[last:]
        self.rows.append(values)
        self.width = max(self.width, last)

    def sheet(self, index: int) -> Sheet:
        for row in self.rows:
            row.extend([None] * (self.width - len(row)))
        return Sheet(name=self.name, index=index, rows=self.rows)


# --- XLSX ----------------------------------------------------------------------------------------------------


def _read_xlsx(path: Path, settings: Settings) -> tuple[list[Sheet], dict[str, Any], list[str]]:
    _check_zip(path, settings)
    # openpyxl rejects a *path* whose name does not end in .xlsx/.xlsm/...; stored uploads have UUID names, so
    # it gets a file object. In read-only mode it reads lazily, so the file stays open until wb.close().
    with path.open("rb") as handle, warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
        try:
            wb = openpyxl.load_workbook(handle, read_only=True, data_only=True, keep_links=False)
        except Exception as exc:
            raise CorruptFileError("the workbook cannot be opened: it is damaged or not an XLSX file") from exc
        try:
            names = list(wb.sheetnames)
            sheets: list[Sheet] = []
            for ws in wb.worksheets:
                ws.reset_dimensions()  # never trust the declared used range
                grid = _Grid(ws.title, settings.max_sheet_rows, settings.max_sheet_cols)
                for row in ws.iter_rows(values_only=True):
                    grid.add(row, _excel_value)
                sheets.append(grid.sheet(names.index(ws.title)))
        except IngestionError:
            raise
        except Exception as exc:
            raise CorruptFileError("the workbook is damaged") from exc
        finally:
            wb.close()
    return sheets, {"sheets": [s.name for s in sheets]}, []


def _check_zip(path: Path, settings: Settings) -> None:
    """Reject zip bombs from the central directory alone, before any member is inflated."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (FileNotFoundError, PermissionError):
        raise
    except Exception as exc:  # BadZipFile, NotImplementedError ("zip file version 7.9"), OSError, ValueError ...
        raise CorruptFileError("the workbook is not a valid ZIP archive (damaged or truncated)") from exc
    limit = settings.max_xlsx_uncompressed_mb * 1024 * 1024
    total = 0
    for member in members:
        total += member.file_size
        if total > limit:
            raise FileTooLargeError(f"the workbook expands to more than {settings.max_xlsx_uncompressed_mb} MB")
        if member.file_size >= _RATIO_EXEMPT_BYTES and (
            member.file_size > settings.max_xlsx_compression_ratio * max(member.compress_size, 1)
        ):
            raise FileTooLargeError(
                f"a workbook member is compressed more than {settings.max_xlsx_compression_ratio}:1 (zip bomb)"
            )


def _excel_value(value: Any) -> CellValue:
    if value is None or isinstance(value, bool | int | Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value)) if math.isfinite(value) else None
    if isinstance(value, datetime):
        return _date_or_datetime(value)
    if isinstance(value, date):
        return value
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    return _text(str(value))


# --- XLS -----------------------------------------------------------------------------------------------------


def _read_xls(path: Path, settings: Settings) -> tuple[list[Sheet], dict[str, Any], list[str]]:
    sink = io.StringIO()  # xlrd writes its diagnostics to a log file (stdout by default)
    try:
        book = xlrd.open_workbook(str(path), on_demand=True, logfile=sink)
    except xlrd.XLRDError as exc:
        message = str(exc).lower()
        if "encrypted" in message:
            raise EncryptedFileError("the workbook is password-protected") from exc
        if "can't find workbook" in message:
            if "EncryptedPackage" in _ole2_stream_names(path):
                raise EncryptedFileError("the workbook is password-protected") from exc
            raise UnsupportedFileError("the OLE2 file holds no Excel workbook") from exc
        raise CorruptFileError("the workbook cannot be read: it is damaged or not an XLS file") from exc
    except Exception as exc:
        raise CorruptFileError("the workbook cannot be read: it is damaged or not an XLS file") from exc
    convert = functools.partial(_xls_value, datemode=book.datemode)
    try:
        sheets: list[Sheet] = []
        for index in range(book.nsheets):
            sheet = book.sheet_by_index(index)
            grid = _Grid(sheet.name, settings.max_sheet_rows, settings.max_sheet_cols)
            for r in range(sheet.nrows):
                grid.add(sheet.row(r), convert)
            sheets.append(grid.sheet(index))
            book.unload_sheet(index)
    except IngestionError:
        raise
    except Exception as exc:
        raise CorruptFileError("the workbook is damaged") from exc
    finally:
        book.release_resources()
    return sheets, {"sheets": [s.name for s in sheets]}, []


def _ole2_stream_names(path: Path) -> set[str]:
    try:
        return {node.name for node in xlrd.compdoc.CompDoc(path.read_bytes(), logfile=io.StringIO()).dirlist}
    except Exception:
        return set()


def _xls_value(cell: Any, datemode: int) -> CellValue:
    ctype, value = cell.ctype, cell.value
    if ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
        return None
    if ctype == xlrd.XL_CELL_NUMBER:
        return _xls_number(value)
    if ctype == xlrd.XL_CELL_DATE:
        try:
            year, month, day, hour, minute, second = xlrd.xldate.xldate_as_tuple(value, datemode)
            if (year, month, day) == (0, 0, 0):  # a time-only cell: the same ISO string openpyxl gives for .xlsx
                return time(hour, minute, second).isoformat()
            return _date_or_datetime(xlrd.xldate.xldate_as_datetime(value, datemode))
        except (xlrd.xldate.XLDateError, ValueError, OverflowError):
            return _xls_number(value)
    if ctype == xlrd.XL_CELL_BOOLEAN:
        return bool(value)
    if ctype == xlrd.XL_CELL_ERROR:
        return xlrd.error_text_from_code.get(value, "#ERROR")
    return _text(str(value))


def _xls_number(value: float) -> CellValue:
    if not math.isfinite(value):
        return None
    if value.is_integer() and abs(value) < 2**53:
        return int(value)
    return Decimal(str(value))


# --- CSV -----------------------------------------------------------------------------------------------------


def _read_csv(path: Path, settings: Settings) -> tuple[list[Sheet], dict[str, Any], list[str]]:
    text, encoding, lossy = _decode(path.read_bytes())
    delimiter = _sniff_delimiter(text)
    grid = _Grid(CSV_SHEET_NAME, settings.max_sheet_rows, settings.max_sheet_cols)
    try:
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter):
            grid.add(row, _text)
    except csv.Error as exc:
        raise CorruptFileError("the CSV file is malformed") from exc
    return [grid.sheet(0)], {"encoding": encoding, "delimiter": delimiter}, ["csv_decoding_errors"] if lossy else []


def _decode(data: bytes) -> tuple[str, str, bool]:
    """Text, the codec name used, and whether undecodable bytes had to be replaced."""
    for bom, codec in _BOMS:
        if data.startswith(bom):
            return _decode_with(data, codec)
    try:
        return data.decode("utf-8"), "utf-8", False
    except UnicodeDecodeError:
        return _decode_with(data, _legacy_encoding(data[:_ENCODING_SAMPLE_BYTES]))


def _decode_with(data: bytes, codec: str) -> tuple[str, str, bool]:
    name = codecs.lookup(codec).name
    try:
        return data.decode(codec), name, False
    except UnicodeDecodeError:
        return data.decode(codec, errors="replace"), name, True


def _legacy_encoding(sample: bytes) -> str:
    best = from_bytes(sample).best()
    western = from_bytes(sample, cp_isolation=["cp1252"]).best()
    if western is not None and (best is None or western.chaos <= best.chaos + _WESTERN_TOLERANCE):
        return "cp1252"
    return best.encoding if best is not None else "cp1252"


def _sniff_delimiter(text: str) -> str:
    sample = text[:_SNIFF_CHARS]
    if len(text) > _SNIFF_CHARS and "\n" in sample:
        sample = sample[: sample.rindex("\n") + 1]  # whole lines only
    try:
        return csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        return _most_consistent_delimiter(sample)


def _most_consistent_delimiter(sample: str) -> str:
    """The candidate that splits most of the first lines into the same number (> 1) of fields."""
    lines = [line for line in sample.splitlines() if line.strip()][:50]
    best, best_score = ",", 0
    for delimiter in _DELIMITERS:
        counts = [len(fields) for fields in csv.reader(lines, delimiter=delimiter)]
        multi = [c for c in counts if c > 1]
        if not multi:
            continue
        modal = max(set(multi), key=multi.count)
        score = multi.count(modal)
        if score > best_score:
            best, best_score = delimiter, score
    return best


# --- shared --------------------------------------------------------------------------------------------------


def _text(value: str) -> str | None:
    value = value.strip()
    return value or None


def _date_or_datetime(value: datetime) -> date | datetime:
    if value.tzinfo is None and value.time() == time.min:
        return value.date()
    return value
