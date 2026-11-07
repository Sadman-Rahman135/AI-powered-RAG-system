"""Image parsing: PNG/JPEG/TIFF through OCR, multi-frame TIFF, EXIF orientation, damaged and oversized images."""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from app.ingestion import images, ocr
from app.ingestion.images import parse_image
from app.ingestion.ocr import OcrUnavailableError, shutdown_ocr_pool
from app.ingestion.types import CorruptFileError, FileKind, FileTooLargeError

from .conftest import (
    INFRASTRUCTURE_FAILURES,
    INVOICE_NUMBER,
    FakePool,
    PoolOutcome,
    fake_ocr_pool,
    first_read_then_broken,
    invoice_items,
    png_bytes,
    render_items,
    set_env,
    undecodable,
)


def _png_claiming(width: int, height: int) -> bytes:
    """A 1x1 PNG whose IHDR header claims ``width`` x ``height`` pixels (a decompression bomb's calling card)."""
    data = bytearray(png_bytes(Image.new("L", (1, 1), 255)))
    data[16:24] = struct.pack(">II", width, height)
    data[29:33] = struct.pack(">I", zlib.crc32(bytes(data[12:29])))
    return bytes(data)


def test_png_jpg_ocr(tmp_path: Path, invoice_image: Image.Image) -> None:
    png, jpg = tmp_path / "scan.png", tmp_path / "photo.jpg"
    invoice_image.save(png)
    invoice_image.save(jpg, quality=90)
    for path, fmt in ((png, "PNG"), (jpg, "JPEG")):
        doc = parse_image(path)
        assert doc.kind is FileKind.IMAGE
        assert len(doc.pages) == 1 and doc.sheets == []
        page = doc.pages[0]
        assert page.method == "ocr" and page.number == 1
        assert INVOICE_NUMBER in page.text and "22.00" in page.text
        assert (page.width, page.height) == invoice_image.size  # pixel units
        assert doc.metadata["image_format"] == fmt and doc.metadata["frames"] == 1
        assert doc.warnings == []


def test_multiframe_tiff_pages(tmp_path: Path) -> None:
    frames = [render_items(invoice_items(f"INV-TIF-{n}"), dpi=150) for n in (1, 2, 3)]
    path = tmp_path / "fax.tif"
    frames[0].save(path, save_all=True, append_images=frames[1:], compression="tiff_deflate")
    calls: list[tuple[int, int, str]] = []
    doc = parse_image(path, on_progress=lambda current, total, stage: calls.append((current, total, stage)))
    assert [p.number for p in doc.pages] == [1, 2, 3]
    for number, page in enumerate(doc.pages, start=1):
        assert f"INV-TIF-{number}" in page.text
    assert calls == [(1, 3, "ocr"), (2, 3, "ocr"), (3, 3, "ocr")]
    assert doc.metadata["frames"] == 3 and doc.metadata["image_format"] == "TIFF"


def test_one_unreadable_frame_does_not_fail_tiff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [render_items(invoice_items(f"INV-F{n}"), dpi=150) for n in (1, 2, 3)]
    path = tmp_path / "scan.tif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    real_frame = images._frame

    def flaky_frame(img: Image.Image, index: int) -> Image.Image:
        if index == 1:
            raise OSError("broken strip")
        return real_frame(img, index)

    monkeypatch.setattr(images, "_frame", flaky_frame)
    doc = parse_image(path)
    assert [p.method for p in doc.pages] == ["ocr", "none", "ocr"]
    assert "INV-F1" in doc.pages[0].text and "INV-F3" in doc.pages[2].text
    assert doc.warnings == ["page_2_extraction_failed"]


def test_truncated_png_raises_corrupt(tmp_path: Path, invoice_image: Image.Image) -> None:
    data = png_bytes(invoice_image)
    path = tmp_path / "truncated.png"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(CorruptFileError) as err:
        parse_image(path)
    assert err.value.code == "corrupt_file"


def test_truncated_jpeg_raises_corrupt(tmp_path: Path, invoice_image: Image.Image) -> None:
    buf = io.BytesIO()
    invoice_image.save(buf, format="JPEG", quality=90)
    path = tmp_path / "truncated.jpg"
    path.write_bytes(buf.getvalue()[: len(buf.getvalue()) // 3])
    with pytest.raises(CorruptFileError):
        parse_image(path)


def test_decompression_bomb_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bomb = tmp_path / "bomb.png"
    bomb.write_bytes(_png_claiming(40_000, 40_000))  # 1.6 gigapixels in a 70-byte file
    with pytest.raises(FileTooLargeError) as err:
        parse_image(bomb)
    assert err.value.code == "file_too_large"
    # The limit is MAX_IMAGE_PIXELS, enforced even where Pillow itself would only warn (between 1x and 2x).
    set_env(monkeypatch, MAX_IMAGE_PIXELS=1_000_000)
    big = tmp_path / "big.png"
    Image.new("RGB", (1200, 1000), "white").save(big)
    with pytest.raises(FileTooLargeError):
        parse_image(big)


def test_exif_rotated_jpeg(tmp_path: Path, invoice_image: Image.Image) -> None:
    stored = invoice_image.rotate(90, expand=True)  # the camera stored the pixels on their side
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display
    path = tmp_path / "phone.jpg"
    stored.save(path, format="JPEG", quality=92, exif=exif)
    page = parse_image(path).pages[0]
    assert INVOICE_NUMBER in page.text
    assert (page.width, page.height) == invoice_image.size  # displayed orientation


def test_transparent_png_is_read_on_white(tmp_path: Path, invoice_image: Image.Image) -> None:
    ink = ImageOps.invert(invoice_image.convert("L"))
    transparent = Image.new("RGBA", invoice_image.size, (0, 0, 0, 0))
    transparent.putalpha(ink)  # black text, fully transparent paper
    path = tmp_path / "transparent.png"
    transparent.save(path)
    assert INVOICE_NUMBER in parse_image(path).text


def test_unexpected_parser_errors_become_corrupt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoice_image: Image.Image
) -> None:
    path = tmp_path / "scan.png"
    invoice_image.save(path)

    def explode(*_args: object, **_kwargs: object) -> object:
        raise ValueError("unexpected decoder state")

    monkeypatch.setattr(images, "_inspect", explode)
    with pytest.raises(CorruptFileError):
        parse_image(path)


def test_progress_callback_errors_are_not_mistaken_for_corruption(tmp_path: Path, invoice_image: Image.Image) -> None:
    path = tmp_path / "scan.png"
    invoice_image.save(path)

    def failing_progress(_current: int, _total: int, _stage: str) -> None:
        raise ConnectionError("progress store is down")

    with pytest.raises(ConnectionError):
        parse_image(path, on_progress=failing_progress)


@pytest.mark.parametrize("failure", list(INFRASTRUCTURE_FAILURES.values()), ids=list(INFRASTRUCTURE_FAILURES))
def test_ocr_infrastructure_failure_is_retryable_not_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoice_image: Image.Image, failure: PoolOutcome
) -> None:
    # A healthy photo must never be failed permanently because the OCR workers are down.
    set_env(monkeypatch, OCR_WORKERS=2)
    monkeypatch.setattr(images, "run_ocr_jobs", fake_ocr_pool(failure))
    path = tmp_path / "photo.png"
    invoice_image.save(path)
    with pytest.raises(OcrUnavailableError) as err:
        parse_image(path)
    assert err.value.code == "ocr_unavailable" and err.value.permanent is False


def test_frames_lost_to_the_pool_are_not_returned_as_a_partial_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_env(monkeypatch, OCR_WORKERS=2)
    monkeypatch.setattr(images, "run_ocr_jobs", fake_ocr_pool(first_read_then_broken))
    frames = [render_items(invoice_items(f"INV-POOL-{n}"), dpi=100) for n in (1, 2)]
    path = tmp_path / "two.tif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    with pytest.raises(OcrUnavailableError) as err:
        parse_image(path)
    assert err.value.permanent is False


def test_pooled_frames_that_cannot_be_read_are_still_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoice_image: Image.Image
) -> None:
    set_env(monkeypatch, OCR_WORKERS=2)
    monkeypatch.setattr(images, "run_ocr_jobs", fake_ocr_pool(undecodable))
    path = tmp_path / "photo.png"
    invoice_image.save(path)
    with pytest.raises(CorruptFileError) as err:
        parse_image(path)
    assert err.value.permanent is True


def test_hung_ocr_page_raises_retryable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoice_image: Image.Image
) -> None:
    set_env(monkeypatch, OCR_WORKERS=2, OCR_PAGE_TIMEOUT_S=0.3)
    pool = FakePool([None])  # the OCR child never returns
    monkeypatch.setattr(ocr, "_get_pool", lambda _workers: pool)
    path = tmp_path / "photo.png"
    invoice_image.save(path)
    with pytest.raises(OcrUnavailableError) as err:
        parse_image(path)
    assert err.value.permanent is False
    assert all(worker.terminated for worker in pool.workers)


@pytest.mark.slow
def test_image_ocr_in_process_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, OCR_WORKERS=2)
    frames = [render_items(invoice_items(f"INV-PT-{n}"), dpi=150) for n in (1, 2)]
    path = tmp_path / "pool.tif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    try:
        doc = parse_image(path)
    finally:
        shutdown_ocr_pool()
    assert [p.number for p in doc.pages] == [1, 2]
    assert "INV-PT-1" in doc.pages[0].text and "INV-PT-2" in doc.pages[1].text
