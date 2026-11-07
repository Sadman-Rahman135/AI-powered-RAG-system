"""Entry point of the ingestion layer: dispatch a stored file to the parser for its (sniffed) kind."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.images import parse_image
from app.ingestion.pdf import parse_pdf
from app.ingestion.spreadsheets import parse_spreadsheet
from app.ingestion.types import EmptyFileError, FileKind, ParsedDocument, ProgressCallback, UnsupportedFileError


def parse_document(path: Path, kind: FileKind, on_progress: ProgressCallback | None = None) -> ParsedDocument:
    """Parse ``path`` as ``kind`` (a :class:`FileKind` or its stored string value).

    ``on_progress(done, total, stage)`` is called per page for PDFs and images, and once for spreadsheets.
    Raises an :class:`~app.ingestion.types.IngestionError` subclass when the file cannot be ingested.
    """
    try:
        kind = FileKind(kind)
    except ValueError as exc:
        raise UnsupportedFileError(f"unsupported file kind {kind!r}") from exc
    if path.stat().st_size == 0:
        raise EmptyFileError("the file is empty")
    if kind is FileKind.PDF:
        return parse_pdf(path, on_progress)
    if kind is FileKind.IMAGE:
        return parse_image(path, on_progress)
    doc = parse_spreadsheet(path, kind)
    if on_progress is not None:
        on_progress(len(doc.sheets), len(doc.sheets), "parsing")
    return doc
