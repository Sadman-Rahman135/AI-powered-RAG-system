"""PDF parsing with PyMuPDF: digital text with geometry, and per-page OCR for scanned pages.

For every page the words of the text layer (in display orientation, ligatures expanded, unmapped glyphs as
U+FFFD) become spans and then lines. A page with fewer than ``OCR_MIN_TEXT_CHARS`` alphanumeric characters is
rendered at ``OCR_DPI`` and OCR'd - unless a tiny render shows it is blank - and the OCR geometry is converted
from pixels back to PDF points so digital and scanned pages share one coordinate system. With
``OCR_WORKERS > 0`` the OCR of those pages runs in the process pool (each job opens the PDF by path);
otherwise pages are OCR'd inline, one rendered image at a time.
An OCR *infrastructure* failure (engine unavailable, worker pool broken) raises the non-permanent
:class:`~app.ingestion.ocr.OcrUnavailableError` so the job is retried; it never becomes a permanent
per-document error or a partial document.

Failures are isolated per page: a page that cannot be read becomes an empty ``method="none"`` page with a
``page_<n>_extraction_failed`` warning and the document continues. A document that cannot be opened, has no
pages, or of which no page could be read raises :class:`CorruptFileError`; a password-protected one raises
:class:`EncryptedFileError`; more than ``MAX_PDF_PAGES`` pages raise :class:`FileTooLargeError`. When MuPDF had
to repair a damaged file, pages that come out empty are counted as failed (truncated files lose their content
without raising).

PyMuPDF runs MuPDF in single-threaded mode, so every MuPDF call of this process is serialised by one lock
(documents parsed from several threads stay safe); OCR inference runs outside the lock.
"""
from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image

from app.config import Settings, get_settings
from app.ingestion.guard import guard_progress, typed_errors
from app.ingestion.layout import build_lines, limit_spans
from app.ingestion.ocr import (
    OcrEngine,
    OcrUnavailableError,
    get_ocr_engine,
    ocr_image,
    pooled_page,
    run_ocr_jobs,
)
from app.ingestion.types import (
    CorruptFileError,
    EmptyFileError,
    EncryptedFileError,
    FileKind,
    FileTooLargeError,
    Page,
    ParsedDocument,
    ProgressCallback,
    Span,
)

logger = logging.getLogger(__name__)

# MuPDF reports structural problems of damaged files on stderr; the parser records them as warnings instead.
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

_MUPDF = threading.RLock()
_WORD_FLAGS = pymupdf.TEXTFLAGS_WORDS & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_CID_FOR_UNKNOWN_UNICODE
_BLANK_CHECK_DPI = 24
_BLANK_TOLERANCE = 8  # grey levels between the darkest and lightest pixel of a blank page
_MAX_RENDER_PIXELS = 40_000_000  # oversized pages are rendered at a lower DPI


def parse_pdf(path: Path, on_progress: ProgressCallback | None = None) -> ParsedDocument:
    """Parse a PDF into pages of lines; ``on_progress(done, total, stage)`` is called once per page.

    Every failure is a typed :class:`~app.ingestion.types.IngestionError` (see :mod:`app.ingestion.guard`).
    """
    with typed_errors("PDF"):
        return _parse_pdf(path, guard_progress(on_progress))


def _parse_pdf(path: Path, on_progress: ProgressCallback | None) -> ParsedDocument:
    settings = get_settings()
    with _MUPDF:
        doc = _open(path)
    try:
        with _MUPDF:
            try:
                total = doc.page_count
                encrypted = bool((doc.metadata or {}).get("encryption"))
            except Exception as exc:  # e.g. RuntimeError "Invalid number of pages" from a truncated page tree
                raise CorruptFileError("the PDF page tree is damaged") from exc
        if total == 0:
            raise CorruptFileError("the PDF has no readable pages")
        if total > settings.max_pdf_pages:
            raise FileTooLargeError(f"the PDF has {total} pages; the limit is {settings.max_pdf_pages}")
        pages: list[Page] = []
        pooled: list[int] = []
        engine: OcrEngine | None = None
        done = 0
        for index in range(total):
            with _MUPDF:
                page, needs_ocr = _read_text_page(doc, index, settings.ocr_min_text_chars)
            if needs_ocr and settings.ocr_workers > 0:
                pages.append(page)
                pooled.append(index)
                continue
            if needs_ocr:
                engine = engine or get_ocr_engine()
                page = _ocr_or_keep(doc, index, page, settings.ocr_dpi, engine)
            pages.append(page)
            done += 1
            _progress(on_progress, done, total, "ocr" if needs_ocr else "parsing")

        for index, page in _ocr_in_pool(path, pooled, pages, settings):
            pages[index] = page
            done += 1
            _progress(on_progress, done, total, "ocr")
        with _MUPDF:
            repaired = bool(doc.is_repaired)
    finally:
        with _MUPDF:
            doc.close()
    return _assemble(pages, repaired=repaired, encrypted=encrypted, settings=settings)


def _open(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except pymupdf.EmptyFileError as exc:
        raise EmptyFileError("the PDF is empty") from exc
    except Exception as exc:
        raise CorruptFileError("the PDF cannot be opened: it is damaged or not a PDF") from exc
    if doc.needs_pass and not doc.authenticate(""):
        doc.close()
        raise EncryptedFileError("the PDF is password-protected")
    return doc


def _read_text_page(doc: pymupdf.Document, index: int, min_chars: int) -> tuple[Page, bool]:
    """The text-layer page and whether it still needs OCR (call with the MuPDF lock held)."""
    number = index + 1
    try:
        page = doc.load_page(index)
        rect = page.rect
        spans = _page_spans(page)
    except Exception as exc:
        logger.warning("pdf page %d: text layer unreadable (%s)", number, type(exc).__name__)
        return Page(number=number, method="none", warnings=[f"page_{number}_text_extraction_failed"]), True
    spans, truncated = limit_spans(spans)
    result = Page(
        number=number,
        lines=build_lines(spans),
        width=rect.width,
        height=rect.height,
        method="text",
        warnings=[f"page_{number}_spans_truncated"] if truncated else [],
    )
    if _has_enough_text(spans, min_chars) or (not spans and _renders_blank(page)):
        return result, False
    return result, True


def _page_spans(page: pymupdf.Page) -> list[Span]:
    """Words of the text layer as spans in display coordinates (points, origin top-left)."""
    words = page.get_text("words", flags=_WORD_FLAGS)
    if not page.rotation:
        return [Span(w[4], w[0], w[1], w[2], w[3]) for w in words]
    matrix = page.rotation_matrix  # text coordinates are unrotated; the page is displayed rotated
    spans: list[Span] = []
    for x0, y0, x1, y1, text, *_ in words:
        box = pymupdf.Rect(x0, y0, x1, y1) * matrix
        spans.append(Span(text, box.x0, box.y0, box.x1, box.y1))
    return spans


def _has_enough_text(spans: list[Span], min_chars: int) -> bool:
    count = 0
    for span in spans:
        count += sum(1 for ch in span.text if ch.isalnum())
        if count >= min_chars:
            return True
    return False


def _renders_blank(page: pymupdf.Page) -> bool:
    try:
        pix = page.get_pixmap(dpi=_BLANK_CHECK_DPI, colorspace=pymupdf.csGRAY, alpha=False)
    except Exception:
        return False
    samples = np.frombuffer(pix.samples, dtype=np.uint8)
    return samples.size == 0 or int(samples.max()) - int(samples.min()) <= _BLANK_TOLERANCE


def _render(doc: pymupdf.Document, index: int, dpi: int) -> tuple[Image.Image, float, float, float]:
    """Render page ``index`` as RGB; returns the image, the page size in points and the zoom used."""
    with _MUPDF:
        page = doc.load_page(index)
        rect = page.rect
        zoom = dpi / 72
        pixels = rect.width * rect.height * zoom * zoom
        if pixels > _MAX_RENDER_PIXELS:
            zoom *= math.sqrt(_MAX_RENDER_PIXELS / pixels)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        del pix, page
    return img, rect.width, rect.height, zoom


def _ocr_page(doc: pymupdf.Document, index: int, dpi: int, engine: OcrEngine) -> Page:
    """Render one page (one image in memory at a time) and OCR it; geometry in PDF points."""
    img, width, height, zoom = _render(doc, index, dpi)
    result = ocr_image(img, index + 1, engine=engine, units_per_pixel=1 / zoom)
    result.width, result.height = width, height
    return result


def _ocr_or_keep(doc: pymupdf.Document, index: int, text_page: Page, dpi: int, engine: OcrEngine) -> Page:
    try:
        ocr_page = _ocr_page(doc, index, dpi, engine)
    except OcrUnavailableError:
        raise
    except Exception as exc:
        logger.warning("pdf page %d: OCR failed (%s)", index + 1, type(exc).__name__)
        return _ocr_failed(text_page)
    return _merge(text_page, ocr_page)


def _merge(text_page: Page, ocr_page: Page) -> Page:
    """Prefer the OCR result (it also sees the rendered text layer) unless it found nothing."""
    if not ocr_page.lines:
        if text_page.lines:
            return text_page
        if text_page.method == "none":  # the text layer raised and OCR recovered nothing either
            return _failed_page(text_page)
    ocr_page.warnings = [*text_page.warnings, *ocr_page.warnings]
    return ocr_page


def _ocr_failed(text_page: Page) -> Page:
    if text_page.lines:  # keep the sparse digital text
        text_page.warnings.append(f"page_{text_page.number}_ocr_failed")
        return text_page
    return _failed_page(text_page)


def _failed_page(page: Page) -> Page:
    return Page(
        number=page.number,
        width=page.width,
        height=page.height,
        method="none",
        warnings=[f"page_{page.number}_extraction_failed"],
    )


def _ocr_in_pool(path: Path, indexes: list[int], pages: list[Page], settings: Settings) -> Iterator[tuple[int, Page]]:
    if not indexes:
        return
    jobs = {index: (str(path), index, settings.ocr_dpi, settings.ocr_engine) for index in indexes}
    results = run_ocr_jobs(_ocr_page_job, jobs, settings.ocr_workers, timeout_s=settings.ocr_page_timeout_s)
    for index, result in results:
        page, _error = pooled_page(result, settings.ocr_engine)  # infrastructure failures raise (retryable)
        text_page = pages[index]
        yield index, _ocr_failed(text_page) if page is None else _merge(text_page, page)


def _ocr_page_job(path: str, index: int, dpi: int, engine_name: str) -> tuple[Page | None, str | None]:
    """Process-pool job: open the PDF by path, render page ``index`` and OCR it (engine cached per worker)."""
    try:
        engine = get_ocr_engine(engine_name)
    except OcrUnavailableError:
        return None, OcrUnavailableError.code
    try:
        with _MUPDF:
            doc = pymupdf.open(path, filetype="pdf")
            if doc.needs_pass:
                doc.authenticate("")
        try:
            return _ocr_page(doc, index, dpi, engine), None
        finally:
            with _MUPDF:
                doc.close()
    except Exception as exc:
        logger.warning("pdf page %d: OCR failed in worker (%s)", index + 1, type(exc).__name__)
        return None, "ocr_failed"


def _assemble(pages: list[Page], *, repaired: bool, encrypted: bool, settings: Settings) -> ParsedDocument:
    if repaired:
        pages = [_failed_page(p) if not p.lines and p.method != "none" else p for p in pages]
    failed = sum(1 for p in pages if p.method == "none")
    if failed == len(pages):
        raise CorruptFileError("no page of the PDF could be read")
    ocr_pages = sum(1 for p in pages if p.method == "ocr")
    metadata: dict[str, Any] = {
        "pdf_pages": len(pages),
        "ocr_pages": ocr_pages,
        "failed_pages": failed,
        "repaired": repaired,
        "encrypted": encrypted,
    }
    if ocr_pages:
        metadata["ocr_engine"] = settings.ocr_engine
        metadata["ocr_dpi"] = settings.ocr_dpi
    warnings = (["pdf_repaired"] if repaired else []) + [w for p in pages for w in p.warnings]
    return ParsedDocument(kind=FileKind.PDF, pages=pages, warnings=warnings, metadata=metadata)


def _progress(callback: ProgressCallback | None, done: int, total: int, stage: str) -> None:
    if callback is not None:
        callback(done, total, stage)
