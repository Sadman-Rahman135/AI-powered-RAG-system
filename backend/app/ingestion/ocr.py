"""OCR engines and the OCR process pool.

Engines turn an RGB image array into :class:`Span` boxes: RapidOCR (PP-OCR models on ONNX Runtime, the
default) yields one span per detected text line; Tesseract (optional, needs the ``tesseract`` binary) yields
one span per word. One engine instance is created per process and cached, because initialisation takes
seconds. :func:`ocr_image` turns a page image into a :class:`Page` (geometry scaled back to the input image,
or to any other unit via ``units_per_pixel``).

With ``OCR_WORKERS > 0`` the PDF and image parsers run OCR in a process pool (``spawn`` start method, one
engine per worker process): :func:`run_ocr_jobs` submits module-level job functions that open the file by path
themselves, so no document or image is ever pickled. Each worker gets an even share of the CPUs for ONNX
Runtime, OpenCV, OpenMP and BLAS (:func:`threads_per_worker`); without that limit every worker takes all
cores and the pool is slower than a single process. A page that does not finish within ``OCR_PAGE_TIMEOUT_S``
(a child stuck in native inference or rendering) terminates the pool and raises the retryable
:class:`OcrUnavailableError`. In-process OCR (``OCR_WORKERS=0``) has no per-page timeout - a native call cannot
be interrupted from its own thread - so the pipeline's attempt-level deadline is the backstop there.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from collections.abc import Callable, Hashable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor
from concurrent.futures import wait as wait_for_any
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.ingestion.layout import build_lines, limit_spans
from app.ingestion.preprocess import prepare_for_ocr
from app.ingestion.types import IngestionError, Page, Span

logger = logging.getLogger(__name__)

# CPU threads this process may use for OCR: set in OCR pool workers by _init_pool_worker (their share of the
# machine), None in any other process.
_process_threads: int | None = None


class OcrUnavailableError(IngestionError):
    """The configured OCR engine cannot run here (missing binary or models); not the document's fault."""

    code = "ocr_unavailable"
    permanent = False


class OcrEngine(Protocol):
    name: str
    upscale_small_images: bool  # whether preprocessing should enlarge small images for this engine

    def recognize(self, image: np.ndarray) -> list[Span]:
        """Recognise text in an 8-bit RGB (or grey) image; span coordinates are pixels of ``image``."""
        ...


class RapidOcrEngine:
    """RapidOCR: line-level boxes, each converted to one span (axis-aligned hull of the polygon).

    Small images are not upscaled for RapidOCR: its detector rescales input itself, and on 50-72 DPI renders
    an upscaled copy was read measurably worse than the native image.
    """

    name = "rapidocr"
    upscale_small_images = False

    def __init__(self, threads: int | None = None) -> None:
        """``threads`` = ONNX Runtime intra-op threads; default: this process's share in an OCR pool worker,
        else ONNX Runtime's own choice, which already uses every physical core (measured fastest for a single
        process: forcing all 12 logical CPUs of a 6-core machine was 1.7x slower)."""
        self.threads = threads or _process_threads
        params: dict[str, Any] = {"Global.log_level": "error"}
        if self.threads:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = self.threads
            params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
        try:
            from rapidocr import RapidOCR

            self._ocr = RapidOCR(params=params)
        except Exception as exc:
            raise OcrUnavailableError("RapidOCR could not be initialised") from exc
        self._lock = threading.Lock()  # one inference at a time per engine; ONNX Runtime parallelises inside

    def recognize(self, image: np.ndarray) -> list[Span]:
        # RapidOCR treats ndarray input as BGR (OpenCV order).
        bgr = np.ascontiguousarray(_as_rgb(image)[:, :, ::-1])
        with self._lock:
            out = self._ocr(bgr)
        boxes, texts, scores = getattr(out, "boxes", None), getattr(out, "txts", None), getattr(out, "scores", None)
        if boxes is None or texts is None or scores is None:
            return []
        spans: list[Span] = []
        for box, text, score in zip(boxes, texts, scores, strict=False):
            text = text.strip()
            if not text:
                continue
            points = np.asarray(box, dtype=np.float64).reshape(-1, 2)
            (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
            spans.append(Span(text, float(x0), float(y0), float(x1), float(y1), conf=_unit(score)))
        return spans


class TesseractEngine:
    """Tesseract via pytesseract: word-level spans, confidence scaled from 0-100 to 0-1."""

    name = "tesseract"
    upscale_small_images = True  # Tesseract wants text about 30 px high

    def __init__(self, lang: str = "eng") -> None:
        import pytesseract

        try:
            pytesseract.get_tesseract_version()
        except (pytesseract.TesseractNotFoundError, OSError) as exc:
            raise OcrUnavailableError("the tesseract binary is not installed") from exc
        self._tesseract = pytesseract
        self._lang = lang

    def recognize(self, image: np.ndarray) -> list[Span]:
        data = self._tesseract.image_to_data(
            _as_rgb(image), lang=self._lang, config="--psm 3", output_type=self._tesseract.Output.DICT
        )
        spans: list[Span] = []
        for text, conf, left, top, width, height in zip(
            data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"], strict=False
        ):
            text = str(text).strip()
            confidence = float(conf)
            if not text or confidence < 0:
                continue
            x0, y0 = float(left), float(top)
            spans.append(Span(text, x0, y0, x0 + float(width), y0 + float(height), conf=_unit(confidence / 100)))
        return spans


_ENGINE_FACTORIES: dict[str, Callable[[], OcrEngine]] = {"rapidocr": RapidOcrEngine, "tesseract": TesseractEngine}
_engines: dict[str, OcrEngine] = {}
_engines_lock = threading.Lock()


def get_ocr_engine(name: str | None = None) -> OcrEngine:
    """The process-wide engine called ``name`` (default: the ``OCR_ENGINE`` setting), created on first use."""
    key = (name or get_settings().ocr_engine).strip().lower()
    factory = _ENGINE_FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"unknown OCR engine {key!r}; expected one of {sorted(_ENGINE_FACTORIES)}")
    with _engines_lock:
        engine = _engines.get(key)
        if engine is None:
            engine = factory()
            _engines[key] = engine
    return engine


def ocr_image(
    img: Image.Image, page_number: int, *, engine: OcrEngine | None = None, units_per_pixel: float = 1.0
) -> Page:
    """OCR one page image into a :class:`Page` (``method="ocr"``, ``ocr_confidence`` = mean span confidence).

    Span coordinates are in pixels of ``img`` times ``units_per_pixel`` (pass ``72 / dpi`` for PDF points),
    whatever scaling preprocessing applied internally.
    """
    engine = engine or get_ocr_engine()
    prepared = prepare_for_ocr(img, upscale=engine.upscale_small_images)
    sx = img.width / prepared.shape[1] * units_per_pixel
    sy = img.height / prepared.shape[0] * units_per_pixel
    spans = [Span(s.text, s.x0 * sx, s.y0 * sy, s.x1 * sx, s.y1 * sy, conf=s.conf) for s in engine.recognize(prepared)]
    del prepared
    spans, truncated = limit_spans(spans)
    confidences = [s.conf for s in spans if s.conf is not None]
    return Page(
        number=page_number,
        lines=build_lines(spans),
        width=img.width * units_per_pixel,
        height=img.height * units_per_pixel,
        method="ocr",
        ocr_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
        warnings=[f"page_{page_number}_spans_truncated"] if truncated else [],
    )


# --- process pool --------------------------------------------------------------------------------------------

_pool: ProcessPoolExecutor | None = None
_pool_workers = 0
_pool_lock = threading.Lock()


_THREAD_ENV = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _cpu_budget() -> int:
    """Logical CPUs this process may run on (the affinity mask where the OS has one, e.g. a Docker cpuset)."""
    affinity = getattr(os, "sched_getaffinity", None)  # Linux only; Windows and macOS have none in os
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


def threads_per_worker(workers: int) -> int:
    """CPU threads given to each of ``workers`` OCR pool processes: an even share of the logical CPUs.

    Without it every worker's ONNX Runtime takes all cores: on 6 cores / 12 threads, 4 workers then needed
    2.77 s per page against 1.52 s with 3 threads each.
    """
    return max(1, _cpu_budget() // max(workers, 1))


def _init_pool_worker(threads: int) -> None:
    """Runs first in every OCR pool process: confine ONNX Runtime, OpenCV, OpenMP and BLAS to ``threads``."""
    global _process_threads
    _process_threads = threads
    for variable in _THREAD_ENV:  # OMP_THREAD_LIMIT also bounds the tesseract processes started from here
        os.environ[variable] = str(threads)
    cv2.setNumThreads(threads)


def _get_pool(workers: int) -> ProcessPoolExecutor:
    global _pool, _pool_workers
    with _pool_lock:
        if _pool is not None and _pool_workers != workers:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
        if _pool is None:
            _pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_pool_worker,
                initargs=(threads_per_worker(workers),),
            )
            _pool_workers = workers
        return _pool


def _discard_pool(pool: ProcessPoolExecutor) -> None:
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    pool.shutdown(wait=False, cancel_futures=True)


def _terminate_pool(pool: ProcessPoolExecutor) -> None:
    """Discard ``pool`` and kill its worker processes: ``shutdown`` alone never stops a worker stuck in native code."""
    processes = list((getattr(pool, "_processes", None) or {}).values())
    _discard_pool(pool)
    for process in processes:
        if process.is_alive():
            process.terminate()


def shutdown_ocr_pool(wait: bool = True) -> None:
    """Stop the OCR worker processes of this process (they are started again on demand)."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=True)


def run_ocr_jobs[K: Hashable](
    fn: Callable[..., Any], jobs: Mapping[K, tuple[Any, ...]], workers: int, *, timeout_s: float | None = None
) -> Iterator[tuple[K, Any]]:
    """Run ``fn(*args)`` for every job in the OCR pool and yield ``(key, result)`` as jobs finish.

    ``fn`` must be a module-level function taking only picklable arguments. A job that raised yields its
    exception as the result. If a worker process dies (the pool breaks), the pool is replaced once and the
    unfinished jobs are resubmitted; jobs that still cannot run yield a :class:`BrokenProcessPool`.

    ``timeout_s`` (``OCR_PAGE_TIMEOUT_S``) bounds the time *without any job finishing* - a page stuck in native
    OCR or rendering - not the whole document, so a long healthy scan is never cut off. When it passes, the pool
    is discarded, its worker processes are terminated and the non-permanent :class:`OcrUnavailableError` is
    raised so the document is retried. ``None`` or 0 waits forever.
    """
    remaining = dict(jobs)
    for _attempt in range(2):
        if not remaining:
            return
        pool = _get_pool(workers)
        futures: dict[Future[Any], K] = {}
        try:
            for key, args in remaining.items():  # submit raises at once if the pool broke while idle
                futures[pool.submit(fn, *args)] = key
            pending = set(futures)
            while pending:
                done, pending = wait_for_any(pending, timeout=timeout_s or None, return_when=FIRST_COMPLETED)
                if not done:
                    logger.warning("no OCR job finished within %.0f s; terminating the OCR pool", timeout_s)
                    _terminate_pool(pool)
                    raise OcrUnavailableError(
                        f"an OCR page did not finish within {timeout_s:g} s; the document can be retried"
                    )
                for future in done:
                    key = futures[future]
                    try:
                        result = future.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:
                        result = exc
                    del remaining[key]
                    yield key, result
            return
        except BrokenProcessPool:
            logger.warning("OCR worker pool broke; %d job(s) will be retried once", len(remaining))
            _discard_pool(pool)
        finally:
            for future in futures:
                future.cancel()
    for key in list(remaining):
        yield key, BrokenProcessPool("an OCR worker process died")


def pooled_page(result: Any, engine_name: str) -> tuple[Page | None, str | None]:
    """A pooled job's ``(page, error)``; raise a *non-permanent* :class:`OcrUnavailableError` for infrastructure.

    A job that raised (above all :class:`BrokenProcessPool`: workers that died or could not even import the app)
    or a worker without a usable engine says nothing about the document, so it must lead to a retry rather than a
    permanent per-document failure - and never to a partial document with pages silently missing.
    """
    if isinstance(result, BaseException):
        logger.warning("OCR worker pool failed (%s); the document can be retried", type(result).__name__)
        raise OcrUnavailableError("the OCR worker processes failed; the document can be retried") from result
    page, error = result
    if error == OcrUnavailableError.code:
        raise OcrUnavailableError(f"OCR engine {engine_name!r} is unavailable in the worker processes")
    return page, error


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return np.ascontiguousarray(image[:, :, :3])
    return image


def _unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
