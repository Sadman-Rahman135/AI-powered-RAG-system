"""OCR engines, image preprocessing and OCR'd pages."""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.ingestion import ocr
from app.ingestion.ocr import (
    OcrUnavailableError,
    RapidOcrEngine,
    get_ocr_engine,
    ocr_image,
    run_ocr_jobs,
    shutdown_ocr_pool,
    threads_per_worker,
)
from app.ingestion.preprocess import estimate_skew, prepare_for_ocr
from app.ingestion.types import Span

from .conftest import INVOICE_NUMBER, FakePool, degrade, invoice_items, render_items

HAS_TESSERACT = shutil.which("tesseract") is not None


def test_ocr_reads_rendered_invoice_text(invoice_image: Image.Image) -> None:
    page = ocr_image(invoice_image, page_number=1)
    assert page.method == "ocr"
    assert page.number == 1
    assert INVOICE_NUMBER in page.text
    assert "22.00" in page.text
    assert page.ocr_confidence is not None and 0.8 < page.ocr_confidence <= 1.0
    assert (page.width, page.height) == invoice_image.size
    # A label and its far-right amount are separate OCR boxes but one line, joined with a column gap.
    subtotal = next(line for line in page.lines if "Subtotal" in line.text)
    assert subtotal.text.endswith("20.00") and "  " in subtotal.text
    # A table row made of four separate boxes reads left to right.
    assert any(line.text.split() == ["Widget", "A", "2", "5.00", "10.00"] for line in page.lines)
    # Geometry is in pixels of the input image (72 pt at 200 DPI = 200 px).
    number_line = next(line for line in page.lines if INVOICE_NUMBER in line.text)
    assert 180 < number_line.x0 < 220


def test_ocr_handles_rotated_noisy_image(invoice_image: Image.Image) -> None:
    poor = degrade(invoice_image, angle=2.0, noise=20.0, blur=1.0)
    page = ocr_image(poor, page_number=3)
    assert INVOICE_NUMBER in page.text
    assert "22.00" in page.text
    assert page.number == 3


def test_ocr_reads_faded_low_contrast_scan(invoice_image: Image.Image) -> None:
    faded = Image.fromarray((150 + np.asarray(invoice_image, dtype=np.float32) * (80 / 255)).astype(np.uint8))
    assert INVOICE_NUMBER in ocr_image(faded, page_number=1).text


class _RecordingEngine:
    """Returns one fixed box in the coordinates of whatever (preprocessed) array it is given."""

    name = "recording"

    def __init__(self, upscale_small_images: bool) -> None:
        self.upscale_small_images = upscale_small_images
        self.shape: tuple[int, ...] | None = None

    def recognize(self, image: np.ndarray) -> list[Span]:
        self.shape = image.shape
        return [Span("Total", 100.0, 200.0, 300.0, 250.0, conf=0.9)]


def test_ocr_image_maps_coordinates_back_to_input_pixels() -> None:
    small = render_items(invoice_items(), dpi=60)  # 496 x 702 px: upscaled to a 2000 px long side
    engine = _RecordingEngine(upscale_small_images=True)
    page = ocr_image(small, page_number=1, engine=engine, units_per_pixel=0.5)
    assert engine.shape is not None and max(engine.shape[:2]) >= 1900
    span = page.lines[0].spans[0]
    assert span.x0 == pytest.approx(100 * small.width / engine.shape[1] * 0.5)
    assert span.y1 == pytest.approx(250 * small.height / engine.shape[0] * 0.5)
    assert (page.width, page.height) == (small.width * 0.5, small.height * 0.5)
    assert page.ocr_confidence == 0.9


def test_rapidocr_reads_small_images_at_native_size() -> None:
    small = render_items(invoice_items(), dpi=60)
    engine = _RecordingEngine(upscale_small_images=get_ocr_engine("rapidocr").upscale_small_images)
    ocr_image(small, page_number=1, engine=engine)
    assert engine.shape == (small.height, small.width, 3)
    assert INVOICE_NUMBER in ocr_image(small, page_number=1).text


def test_blank_image_gives_an_empty_ocr_page() -> None:
    page = ocr_image(Image.new("RGB", (1200, 1600), "white"), page_number=2)
    assert page.method == "ocr"
    assert page.lines == []
    assert page.ocr_confidence is None


def test_prepare_for_ocr_upscales_small_images() -> None:
    small = render_items(invoice_items(), dpi=60)  # 496 x 702 px
    arr = prepare_for_ocr(small)
    assert arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3
    assert 1900 <= max(arr.shape[:2]) <= 2100
    assert abs(arr.shape[1] / arr.shape[0] - small.width / small.height) < 0.01


def test_prepare_for_ocr_leaves_clean_scans_untouched(invoice_image: Image.Image) -> None:
    arr = prepare_for_ocr(invoice_image)
    assert arr.shape == (invoice_image.height, invoice_image.width, 3)
    assert np.array_equal(arr, np.asarray(invoice_image))


def test_prepare_for_ocr_normalizes_modes() -> None:
    transparent = Image.new("RGBA", (1700, 900), (0, 0, 0, 0))
    ImageDraw.Draw(transparent).rectangle((100, 100, 400, 200), fill=(0, 0, 0, 255))
    arr = prepare_for_ocr(transparent)
    assert arr.shape[2] == 3
    assert tuple(arr[10, 10]) == (255, 255, 255)  # transparency becomes white paper, not black
    assert tuple(arr[150, 200]) == (0, 0, 0)
    samples = np.full((1700, 900), 60000, dtype=np.uint16)
    samples[400:600, 100:500] = 1000  # dark ink in 16-bit terms
    gray16 = prepare_for_ocr(Image.fromarray(samples))
    # 16-bit samples are scaled to 8 bits, not clipped (clipping would turn the ink white).
    assert gray16[500, 300].max() < 50
    assert gray16[50, 50].min() > 200


def test_estimate_skew_measures_rotation(invoice_image: Image.Image) -> None:
    def skew(img: Image.Image) -> float:
        return estimate_skew(np.asarray(img.convert("L")))

    assert abs(skew(invoice_image)) < 0.3
    assert abs(skew(degrade(invoice_image, angle=3.0)) - 3.0) < 0.4
    assert abs(skew(degrade(invoice_image, angle=-1.5)) + 1.5) < 0.4
    assert estimate_skew(np.full((800, 600), 255, dtype=np.uint8)) == 0.0


def test_prepare_for_ocr_deskews(invoice_image: Image.Image) -> None:
    straightened = prepare_for_ocr(degrade(invoice_image, angle=4.0))
    assert abs(estimate_skew(np.asarray(Image.fromarray(straightened).convert("L")))) < 0.4


def test_get_ocr_engine_is_cached_per_process() -> None:
    engine = get_ocr_engine()
    assert engine is get_ocr_engine("rapidocr")
    assert engine.name == "rapidocr"
    with pytest.raises(ValueError, match="unknown OCR engine"):
        get_ocr_engine("does-not-exist")


def test_rapidocr_engine_returns_line_spans(invoice_image: Image.Image) -> None:
    spans = get_ocr_engine("rapidocr").recognize(np.asarray(invoice_image))
    assert any(span.text == f"Invoice No: {INVOICE_NUMBER}" for span in spans)
    for span in spans:
        assert span.conf is not None and 0.0 < span.conf <= 1.0
        assert span.x0 < span.x1 and span.y0 < span.y1
    assert get_ocr_engine("rapidocr").recognize(np.full((600, 800, 3), 255, dtype=np.uint8)) == []


def test_rapidocr_engine_applies_its_thread_budget() -> None:
    engine = RapidOcrEngine(threads=2)
    engine.recognize(np.full((64, 256, 3), 255, dtype=np.uint8))  # RapidOCR creates its sessions lazily
    detector = engine._ocr.text_det
    assert detector is not None
    options = detector.session.session.get_session_options()  # the detector's ONNX Runtime session
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (2, 1)
    in_process = get_ocr_engine("rapidocr")
    assert isinstance(in_process, RapidOcrEngine)
    assert in_process.threads is None  # outside a pool worker: ONNX Runtime's own choice (measured fastest)


def test_threads_per_worker_splits_the_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "_cpu_budget", lambda: 12)
    assert [threads_per_worker(w) for w in (1, 2, 3, 4, 6, 12, 24)] == [12, 6, 4, 3, 2, 1, 1]


def _report_thread_budget() -> tuple[object, ...]:
    """Pool job: the CPU thread budget this worker process runs with."""
    engine = get_ocr_engine("rapidocr")
    assert isinstance(engine, RapidOcrEngine)
    return ocr._process_threads, cv2.getNumThreads(), os.environ.get("OMP_NUM_THREADS"), engine.threads


@pytest.mark.slow
def test_pool_workers_get_their_share_of_the_cpus() -> None:
    try:
        results = dict(run_ocr_jobs(_report_thread_budget, {0: ()}, workers=2))
    finally:
        shutdown_ocr_pool()
    share = threads_per_worker(2)
    assert results[0] == (share, share, str(share), share)


def _echo(value: object) -> object:
    return value


def test_hung_ocr_job_times_out_and_terminates_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = FakePool([0.05, None])  # the second page never finishes
    monkeypatch.setattr(ocr, "_pool", pool)
    monkeypatch.setattr(ocr, "_get_pool", lambda _workers: pool)
    received = []
    start = time.perf_counter()
    with pytest.raises(OcrUnavailableError) as err:
        for key, result in run_ocr_jobs(_echo, {0: ("page 1",), 1: ("page 2",)}, workers=2, timeout_s=0.3):
            received.append((key, result))
    assert time.perf_counter() - start < 3
    assert err.value.code == "ocr_unavailable" and err.value.permanent is False  # the job is re-queued
    assert received == [(0, "page 1")]
    assert all(worker.terminated for worker in pool.workers) and pool.shut_down
    assert ocr._pool is None  # the next document gets a fresh pool


def test_slow_but_steady_pool_is_not_timed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    # The timeout is per page (time without any page finishing), not for the whole document.
    pool = FakePool([0.1, 0.2, 0.3, 0.4, 0.5])
    monkeypatch.setattr(ocr, "_get_pool", lambda _workers: pool)
    jobs = {n: (n,) for n in range(5)}
    assert dict(run_ocr_jobs(_echo, jobs, workers=2, timeout_s=0.3)) == {n: n for n in range(5)}
    assert not any(worker.terminated for worker in pool.workers)


def _crash_until_marker(marker: str) -> str:
    """Pool job: kills its worker process the first time, succeeds once the marker file exists."""
    path = Path(marker)
    if not path.exists():
        path.write_text("crashed once", encoding="utf-8")
        os._exit(1)
    return "ok"


def _always_crash() -> str:
    os._exit(1)


@pytest.mark.slow
def test_ocr_pool_survives_a_dead_worker(tmp_path: Path) -> None:
    marker = str(tmp_path / "marker")
    try:
        assert dict(run_ocr_jobs(_crash_until_marker, {"job": (marker,)}, workers=1)) == {"job": "ok"}
        failed = dict(run_ocr_jobs(_always_crash, {"bad": ()}, workers=1))
        assert isinstance(failed["bad"], BrokenProcessPool)  # reported, never silently dropped
        assert dict(run_ocr_jobs(_crash_until_marker, {"again": (marker,)}, workers=2)) == {"again": "ok"}
    finally:
        shutdown_ocr_pool()


@pytest.mark.skipif(not HAS_TESSERACT, reason="the tesseract binary is not installed")
def test_tesseract_engine_returns_word_spans(invoice_image: Image.Image) -> None:
    spans = get_ocr_engine("tesseract").recognize(np.asarray(invoice_image))
    assert any(span.text == INVOICE_NUMBER for span in spans)
    assert all(" " not in span.text for span in spans)
    assert all(span.conf is not None and 0.0 <= span.conf <= 1.0 for span in spans)


@pytest.mark.skipif(HAS_TESSERACT, reason="the tesseract binary is installed")
def test_tesseract_engine_unavailable_is_reported() -> None:
    with pytest.raises(OcrUnavailableError) as err:
        get_ocr_engine("tesseract")
    assert err.value.code == "ocr_unavailable"
    assert not err.value.permanent
