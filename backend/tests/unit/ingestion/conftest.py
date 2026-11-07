"""Fixture builders for the ingestion tests.

Every binary fixture (PDF, PNG/JPEG/TIFF, XLSX/XLS, CSV) is generated at test time under ``tmp_path`` with
PyMuPDF, Pillow, openpyxl or xlwt, so no binary files live in git.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageFilter

from app.config import reset_settings_cache
from app.ingestion.types import Line, Page, Span

INVOICE_NUMBER = "INV-2024-0042"
A4 = (595.0, 842.0)

# (x, baseline y, text, font size) in PDF points on an A4 page.
TextItem = tuple[float, float, str, float]


def invoice_items(number: str = INVOICE_NUMBER) -> list[TextItem]:
    """A small invoice: header, a two-row item table and a right-aligned totals block."""
    return [
        (72, 80, "ACME Supplies Ltd.", 16),
        (72, 104, "12 Harbour Road, Springfield", 10),
        (72, 160, f"Invoice No: {number}", 12),
        (72, 182, "Invoice Date: 2024-03-05", 12),
        (72, 204, "Bill To: Globex Corporation", 12),
        (72, 260, "Description", 11),
        (300, 260, "Qty", 11),
        (380, 260, "Price", 11),
        (470, 260, "Amount", 11),
        (72, 284, "Widget A", 11),
        (300, 284, "2", 11),
        (380, 284, "5.00", 11),
        (470, 284, "10.00", 11),
        (72, 306, "Widget B", 11),
        (300, 306, "1", 11),
        (380, 306, "10.00", 11),
        (470, 306, "10.00", 11),
        (300, 350, "Subtotal:", 11),
        (470, 350, "20.00", 11),
        (300, 372, "Tax (10%):", 11),
        (470, 372, "2.00", 11),
        (300, 400, "Total: USD 22.00", 12),
    ]


def draw_items(page: pymupdf.Page, items: Sequence[TextItem]) -> None:
    for x, y, text, size in items:
        page.insert_text((x, y), text, fontsize=size, fontname="helv")


def render_items(items: Sequence[TextItem], dpi: int = 200, size: tuple[float, float] = A4) -> Image.Image:
    """Render text items onto a white page and return it as an RGB image (like a clean scan)."""
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    draw_items(page, items)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def degrade(
    img: Image.Image, *, angle: float = 0.0, noise: float = 0.0, blur: float = 0.0, seed: int = 7
) -> Image.Image:
    """Simulate a poor scan: rotation (counter-clockwise degrees), Gaussian blur and Gaussian pixel noise."""
    out = img.convert("RGB")
    if angle:
        out = out.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        rng = np.random.default_rng(seed)
        arr = np.asarray(out).astype(np.float32)
        arr += rng.normal(0.0, noise, size=arr.shape[:2])[..., None]
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def add_text_page(doc: pymupdf.Document, items: Sequence[TextItem]) -> None:
    draw_items(doc.new_page(width=A4[0], height=A4[1]), items)


def add_scanned_page(doc: pymupdf.Document, img: Image.Image) -> None:
    """An image-only page (no text layer), as a scanner would produce."""
    page = doc.new_page(width=A4[0], height=A4[1])
    page.insert_image(page.rect, stream=png_bytes(img))


def make_pdf(path: Path, pages: Sequence[Sequence[TextItem] | Image.Image]) -> Path:
    """Write a PDF: a list of text items becomes a digital page, a PIL image becomes a scanned page."""
    doc = pymupdf.open()
    for spec in pages:
        if isinstance(spec, Image.Image):
            add_scanned_page(doc, spec)
        else:
            add_text_page(doc, spec)
    doc.save(path)
    doc.close()
    return path


def set_env(monkeypatch: pytest.MonkeyPatch, **values: object) -> None:
    """Override settings through the environment and drop the cached Settings."""
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))
    reset_settings_cache()


@pytest.fixture
def invoice_image() -> Image.Image:
    return render_items(invoice_items())


PoolOutcome = Callable[[int], object]


def fake_ocr_pool(outcome: PoolOutcome) -> Callable[..., Iterator[tuple[int, object]]]:
    """A stand-in for ``ocr.run_ocr_jobs`` that yields ``outcome(key)`` per job without starting any process."""

    def run(_fn: object, jobs: Mapping[int, object], _workers: int, **_options: object) -> Iterator[tuple[int, object]]:
        for key in jobs:
            yield key, outcome(key)

    return run


def broken_pool(_key: int) -> object:
    return BrokenProcessPool("the spawned workers could not import app")  # what a dead pool yields per job


def engine_unavailable(_key: int) -> object:
    return None, "ocr_unavailable"


def undecodable(_key: int) -> object:
    return None, "ocr_failed"  # a genuine per-page failure reported by the worker


def first_read_then_broken(key: int) -> object:
    if key == 0:
        page = Page(number=1, lines=[Line(spans=[Span("INV-POOL-1", 10, 10, 80, 20, conf=0.95)])], method="ocr")
        return page, None
    return broken_pool(key)


INFRASTRUCTURE_FAILURES = {"broken-pool": broken_pool, "engine-unavailable": engine_unavailable}


class FakeWorker:
    """Stands in for a pool process: records being terminated."""

    def __init__(self) -> None:
        self.terminated = False

    def is_alive(self) -> bool:
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True


class FakePool:
    """Stands in for the OCR ProcessPoolExecutor without starting processes.

    Job ``n`` finishes ``delays[n]`` seconds after it is submitted, running ``fn`` on a timer thread; a delay of
    ``None`` means the job never finishes (an OCR child stuck in native code).
    """

    def __init__(self, delays: Sequence[float | None]) -> None:
        self._delays = list(delays)
        self._processes = {1: FakeWorker(), 2: FakeWorker()}  # what ocr reads to terminate a pool's workers
        self.shut_down = False

    def submit(self, fn: Callable[..., object], *args: object) -> Future[object]:
        future: Future[object] = Future()
        delay = self._delays.pop(0) if self._delays else None
        if delay is not None:
            threading.Timer(delay, lambda: future.set_result(fn(*args))).start()
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shut_down = True

    @property
    def workers(self) -> list[FakeWorker]:
        return list(self._processes.values())
