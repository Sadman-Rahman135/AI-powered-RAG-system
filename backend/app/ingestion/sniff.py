"""File-kind detection from content ("magic bytes").

The extension of the uploaded filename is only a hint: content decides, and a disagreement is reported by
:func:`extension_mismatch` so callers can record a warning. Only the kinds the system can parse are accepted;
everything else raises :class:`UnsupportedFileError`.
"""

from __future__ import annotations

import codecs
import os
import re
import unicodedata

from app.ingestion.types import EmptyFileError, FileKind, UnsupportedFileError

HEAD_BYTES = 64 * 1024
"""How many leading bytes of a file callers should pass to :func:`detect_file_kind` (ZIP member names of an
XLSX workbook appear in its first local headers, usually within the first few kilobytes)."""

MIME_TYPES: dict[FileKind, str] = {
    FileKind.PDF: "application/pdf",
    FileKind.IMAGE: "image/*",  # the image family; detect_mime_type() gives the exact subtype
    FileKind.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    FileKind.XLS: "application/vnd.ms-excel",
    FileKind.CSV: "text/csv",
}

EXTENSIONS: dict[FileKind, frozenset[str]] = {
    FileKind.PDF: frozenset({".pdf"}),
    FileKind.IMAGE: frozenset({".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".tif", ".tiff"}),
    FileKind.XLSX: frozenset({".xlsx", ".xlsm"}),
    FileKind.XLS: frozenset({".xls"}),
    FileKind.CSV: frozenset({".csv", ".tsv", ".txt"}),
}

_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"
_TIFF = (b"II*\x00", b"MM\x00*")
_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_LOCAL_HEADER = b"PK\x03\x04"
_PDF_HEADER = b"%PDF-"
_PDF_HEADER_WINDOW = 1024  # readers accept the header anywhere in the first KiB

_DELIMITERS = ",;\t|"
_MARKUP = re.compile(r"<[!?/a-zA-Z]")
_NON_TEXT_PREFIXES = ("{\\rtf", "%!", "#!")
_MAX_CONTROL_RATIO = 0.01


def detect_file_kind(head: bytes, filename: str) -> FileKind:
    """Return the kind of a file from its first bytes (pass at least :data:`HEAD_BYTES`).

    Raises :class:`EmptyFileError` for an empty file and :class:`UnsupportedFileError` for anything that is not
    a PDF, PNG/JPEG/TIFF image, XLSX/XLS workbook or delimited text file.
    """
    if not head:
        raise EmptyFileError("the file is empty")
    if _PDF_HEADER in head[:_PDF_HEADER_WINDOW]:
        return FileKind.PDF
    if head.startswith(_PNG) or head.startswith(_JPEG) or head[:4] in _TIFF:
        return FileKind.IMAGE
    if head.startswith(_ZIP_LOCAL_HEADER):
        return _zip_kind(head)
    if head.startswith(_OLE2):
        return FileKind.XLS  # OLE2 containers that hold no workbook are rejected by the XLS parser
    if _is_delimited_text(head, filename):
        return FileKind.CSV
    raise UnsupportedFileError("unsupported file type: expected PDF, PNG, JPEG, TIFF, XLSX, XLS or CSV")


def detect_mime_type(head: bytes, kind: FileKind) -> str:
    """The MIME type of a file already classified as ``kind`` (exact subtype for images)."""
    if kind is FileKind.IMAGE:
        if head.startswith(_PNG):
            return "image/png"
        if head.startswith(_JPEG):
            return "image/jpeg"
        if head[:4] in _TIFF:
            return "image/tiff"
    return MIME_TYPES[kind]


def extension_mismatch(kind: FileKind, filename: str) -> bool:
    """True when the filename has an extension that does not belong to ``kind`` (no extension = no mismatch)."""
    suffix = os.path.splitext(filename.strip())[1].lower()
    return bool(suffix) and suffix not in EXTENSIONS[kind]


def _zip_kind(head: bytes) -> FileKind:
    names = [name.replace("\\", "/").lower() for name in _zip_member_names(head)]
    if "xl/workbook.bin" in names:
        raise UnsupportedFileError("binary Excel workbooks (.xlsb) are not supported")
    if any(name.startswith("xl/") for name in names):
        return FileKind.XLSX
    raise UnsupportedFileError("ZIP archive that is not an Excel workbook")


def _zip_member_names(head: bytes) -> list[str]:
    """Member names from the ZIP local file headers present in ``head``.

    Scanning for every header signature (rather than walking sizes) also works for members written with a
    trailing data descriptor, whose local header carries no compressed size.
    """
    names: list[str] = []
    start = 0
    while (pos := head.find(_ZIP_LOCAL_HEADER, start)) != -1 and pos + 30 <= len(head):
        flags = int.from_bytes(head[pos + 6 : pos + 8], "little")
        name_len = int.from_bytes(head[pos + 26 : pos + 28], "little")
        raw = head[pos + 30 : pos + 30 + name_len]
        if name_len and len(raw) == name_len:
            names.append(raw.decode("utf-8" if flags & 0x800 else "cp437", errors="replace"))
        start = pos + 4
    return names


def _is_delimited_text(head: bytes, filename: str) -> bool:
    text = _decode_text(head)
    if text is None:
        return False
    stripped = text.lstrip("\ufeff \t\r\n")
    if not stripped or _MARKUP.match(stripped) or stripped.startswith(_NON_TEXT_PREFIXES):
        return False
    if os.path.splitext(filename.strip())[1].lower() in EXTENSIONS[FileKind.CSV]:
        return True
    return _has_consistent_delimiter(stripped)


def _decode_text(head: bytes) -> str | None:
    """Decode the head of a file as text, or return None when it looks binary."""
    if head.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        text = head.decode("utf-32", errors="ignore")
    elif head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = head.decode("utf-16", errors="ignore")
    elif b"\x00" in head:
        return None
    else:
        try:
            text = head.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason == "unexpected end of data":  # a multi-byte character cut off by the head boundary
                text = head[: exc.start].decode("utf-8", errors="replace")
            else:
                text = head.decode("cp1252", errors="replace")
    suspicious = sum(
        1 for ch in text if ch == "\ufffd" or (unicodedata.category(ch) == "Cc" and ch not in "\t\n\r\f\v")
    )
    if suspicious > _MAX_CONTROL_RATIO * max(len(text), 1):
        return None
    return text


def _has_consistent_delimiter(text: str) -> bool:
    """At least two lines that split into the same number (> 1) of fields on one candidate delimiter."""
    lines = [line for line in text.splitlines() if line.strip()][:20]
    if len(lines) < 2:
        return False
    for delimiter in _DELIMITERS:
        counts = [line.count(delimiter) for line in lines]
        if counts[0] > 0 and sum(1 for c in counts if c == counts[0]) >= max(2, int(0.8 * len(counts))):
            return True
    return False
