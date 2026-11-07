"""Image parsing: PNG, JPEG and TIFF (every frame of a multi-frame TIFF is a page) through OCR.

Pillow decodes only these formats (no other plugin ever sees hostile input). ``MAX_IMAGE_PIXELS`` guards against
decompression bombs: Pillow's bomb error *and* its bomb warning become :class:`FileTooLargeError`, and every
frame's size is checked against the limit. The file is ``verify()``-ed and reopened before decoding; damaged
files raise :class:`CorruptFileError` (a single damaged frame of a multi-frame TIFF only empties that page).
EXIF orientation is applied, so phone photos are read the way they are displayed. Page geometry is in pixels.
An OCR *infrastructure* failure (engine unavailable, worker pool broken) raises the non-permanent
:class:`~app.ingestion.ocr.OcrUnavailableError` so the job is retried - never a permanent ``corrupt_file``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import (
    Image,
    ImageOps,
    JpegImagePlugin,  # noqa: F401 - importing a plugin registers its decoder for Image.open(formats=...)
    PngImagePlugin,  # noqa: F401
    TiffImagePlugin,  # noqa: F401
)

from app.config import Settings, get_settings
from app.ingestion.guard import guard_progress, typed_errors
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
    FileKind,
    FileTooLargeError,
    IngestionError,
    Page,
    ParsedDocument,
    ProgressCallback,
)

logger = logging.getLogger(__name__)

# Camera multi-picture JPEGs open through the JPEG decoder (format "MPO"); only their first picture is read.
_FORMATS = ("PNG", "JPEG", "TIFF")


def parse_image(path: Path, on_progress: ProgressCallback | None = None) -> ParsedDocument:
    """OCR an image file into one page per frame; ``on_progress(done, total, "ocr")`` is called per frame.

    Every failure is a typed :class:`~app.ingestion.types.IngestionError` (see :mod:`app.ingestion.guard`).
    """
    with typed_errors("image"):
        return _parse_image(path, guard_progress(on_progress))


def _parse_image(path: Path, on_progress: ProgressCallback | None) -> ParsedDocument:
    settings = get_settings()
    image_format, frames = _inspect(path, settings.max_image_pixels, settings.max_pdf_pages)
    if settings.ocr_workers > 0:
        pages = _ocr_in_pool(path, frames, settings, on_progress)
    else:
        pages = _ocr_inline(path, frames, settings, on_progress)
    if all(p.method == "none" for p in pages):
        raise CorruptFileError("no frame of the image could be read")
    return ParsedDocument(
        kind=FileKind.IMAGE,
        pages=pages,
        warnings=[w for p in pages for w in p.warnings],
        metadata={"image_format": image_format, "frames": frames, "ocr_engine": settings.ocr_engine},
    )


@contextmanager
def _decoding(max_pixels: int) -> Iterator[None]:
    """Map Pillow's many decoding failures to ingestion errors; bomb warnings become errors."""
    Image.MAX_IMAGE_PIXELS = max_pixels
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            yield
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise FileTooLargeError(f"the image has more than {max_pixels} pixels") from exc
        except IngestionError:
            raise
        except Exception as exc:  # Pillow raises OSError, SyntaxError, ValueError, EOFError ... on damaged input
            raise CorruptFileError("the image cannot be decoded: it is damaged or truncated") from exc


def _inspect(path: Path, max_pixels: int, max_frames: int) -> tuple[str, int]:
    """Validate the file (format, integrity, size of every frame) and return its format and frame count."""
    with _decoding(max_pixels):
        with Image.open(path, formats=_FORMATS) as img:
            img.verify()
        with Image.open(path, formats=_FORMATS) as img:
            image_format = str(img.format)
            frames = int(getattr(img, "n_frames", 1)) if image_format == "TIFF" else 1
            if frames > max_frames:
                raise FileTooLargeError(f"the image has {frames} frames; the limit is {max_frames}")
            for index in range(frames):
                if index:
                    img.seek(index)
                width, height = img.size
                if width * height > max_pixels:
                    raise FileTooLargeError(f"frame {index + 1} has more than {max_pixels} pixels")
                if width == 0 or height == 0:
                    raise CorruptFileError(f"frame {index + 1} has no pixels")
    return ("JPEG" if image_format == "MPO" else image_format), frames


def _frame(img: Image.Image, index: int) -> Image.Image:
    """Decode frame ``index`` as an independent image in display orientation."""
    if index:
        img.seek(index)
    img.load()
    try:
        return ImageOps.exif_transpose(img)
    except Exception:  # a damaged EXIF block must not cost the page: read it as stored
        return img.copy()


def _ocr_inline(path: Path, frames: int, settings: Settings, on_progress: ProgressCallback | None) -> list[Page]:
    engine = get_ocr_engine()
    pages: list[Page] = []
    with _decoding(settings.max_image_pixels):
        img = Image.open(path, formats=_FORMATS)
    try:
        for index in range(frames):
            number = index + 1
            try:
                with _decoding(settings.max_image_pixels):
                    frame = _frame(img, index)
            except CorruptFileError:
                logger.warning("image frame %d cannot be decoded", number)
                pages.append(_failed_page(number))
            else:
                pages.append(_ocr_frame(frame, number, engine))
                del frame
            if on_progress is not None:
                on_progress(number, frames, "ocr")
    finally:
        img.close()
    return pages


def _ocr_frame(frame: Image.Image, number: int, engine: OcrEngine) -> Page:
    try:
        return ocr_image(frame, number, engine=engine)
    except OcrUnavailableError:
        raise
    except Exception as exc:
        logger.warning("image frame %d: OCR failed (%s)", number, type(exc).__name__)
        page = _failed_page(number)
        page.width, page.height = float(frame.width), float(frame.height)
        return page


def _ocr_in_pool(path: Path, frames: int, settings: Settings, on_progress: ProgressCallback | None) -> list[Page]:
    jobs: dict[int, tuple[Any, ...]] = {
        index: (str(path), index, settings.ocr_engine, settings.max_image_pixels) for index in range(frames)
    }
    pages: list[Page | None] = [None] * frames
    for done, (index, result) in enumerate(
        run_ocr_jobs(_ocr_frame_job, jobs, settings.ocr_workers, timeout_s=settings.ocr_page_timeout_s), start=1
    ):
        page, _error = pooled_page(result, settings.ocr_engine)  # infrastructure failures raise (retryable)
        pages[index] = page or _failed_page(index + 1)
        if on_progress is not None:
            on_progress(done, frames, "ocr")
    return [p or _failed_page(i + 1) for i, p in enumerate(pages)]


def _ocr_frame_job(path: str, index: int, engine_name: str, max_pixels: int) -> tuple[Page | None, str | None]:
    """Process-pool job: open the image by path, decode frame ``index`` and OCR it (engine cached per worker)."""
    try:
        engine = get_ocr_engine(engine_name)
    except OcrUnavailableError:
        return None, OcrUnavailableError.code
    try:
        with _decoding(max_pixels), Image.open(path, formats=_FORMATS) as img:
            frame = _frame(img, index)
        return ocr_image(frame, index + 1, engine=engine), None
    except Exception as exc:
        logger.warning("image frame %d: OCR failed in worker (%s)", index + 1, type(exc).__name__)
        return None, "ocr_failed"


def _failed_page(number: int) -> Page:
    return Page(number=number, method="none", warnings=[f"page_{number}_extraction_failed"])
