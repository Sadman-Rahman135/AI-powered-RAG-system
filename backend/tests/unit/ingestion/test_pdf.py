"""PDF parsing: digital text with geometry, per-page OCR fallback, failure isolation, limits and scale."""

from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from app.ingestion import layout, ocr, pdf
from app.ingestion.ocr import OcrUnavailableError, shutdown_ocr_pool
from app.ingestion.pdf import parse_pdf
from app.ingestion.types import (
    CorruptFileError,
    EncryptedFileError,
    FileKind,
    FileTooLargeError,
    IngestionError,
    Page,
    ParsedDocument,
)

from .conftest import (
    A4,
    INFRASTRUCTURE_FAILURES,
    INVOICE_NUMBER,
    FakePool,
    PoolOutcome,
    add_text_page,
    broken_pool,
    fake_ocr_pool,
    invoice_items,
    make_pdf,
    render_items,
    set_env,
    undecodable,
)


def _no_ocr(*_args: object, **_kwargs: object) -> Page:
    raise AssertionError("this page must not be OCR'd")


def _broken_ocr(*_args: object, **_kwargs: object) -> Page:
    raise RuntimeError("cannot render page")


def test_digital_pdf_text_and_geometry(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "digital.pdf", [invoice_items(), invoice_items("INV-2024-0043")])
    doc = parse_pdf(path)
    assert doc.kind is FileKind.PDF
    assert [p.number for p in doc.pages] == [1, 2]
    assert all(p.method == "text" and p.ocr_confidence is None for p in doc.pages)
    first = doc.pages[0]
    assert (first.width, first.height) == A4
    lines = first.text.splitlines()
    assert lines[0] == "ACME Supplies Ltd."
    assert f"Invoice No: {INVOICE_NUMBER}" in lines
    assert "Subtotal:  20.00" in lines  # the column gap survives as two spaces
    assert "Widget A  2  5.00  10.00" in lines
    number = next(s for line in first.lines for s in line.spans if s.text == INVOICE_NUMBER)
    # insert_text((72, 160), "Invoice No: ...", fontsize=12): the number starts after "Invoice No: ".
    assert 130 < number.x0 < 140 and 145 < number.y0 < 150 and number.y1 < 165
    assert "INV-2024-0043" in doc.pages[1].text
    assert doc.metadata["pdf_pages"] == 2 and doc.metadata["ocr_pages"] == 0
    assert doc.warnings == []


def test_scanned_pdf_page_uses_ocr(tmp_path: Path, invoice_image: Image.Image) -> None:
    doc = parse_pdf(make_pdf(tmp_path / "scan.pdf", [invoice_image]))
    page = doc.pages[0]
    assert page.method == "ocr"
    assert INVOICE_NUMBER in page.text and "22.00" in page.text
    assert page.ocr_confidence is not None and page.ocr_confidence > 0.8
    # OCR geometry is converted from pixels back to PDF points, comparable with digital pages.
    assert (page.width, page.height) == A4
    number_line = next(line for line in page.lines if INVOICE_NUMBER in line.text)
    assert 65 < number_line.x0 < 80 and 140 < number_line.y0 < 160
    assert all(0 <= line.x0 <= line.x1 <= A4[0] + 1 for line in page.lines)
    assert doc.metadata["ocr_pages"] == 1 and doc.ocr_page_count == 1


def test_mixed_pdf_per_page_method(tmp_path: Path) -> None:
    scanned = render_items(invoice_items("INV-SCAN-2"))
    path = make_pdf(tmp_path / "mixed.pdf", [invoice_items("INV-TEXT-1"), scanned, invoice_items("INV-TEXT-3")])
    doc = parse_pdf(path)
    assert [p.method for p in doc.pages] == ["text", "ocr", "text"]
    assert "INV-TEXT-1" in doc.pages[0].text
    assert "INV-SCAN-2" in doc.pages[1].text
    assert "INV-TEXT-3" in doc.pages[2].text
    assert doc.ocr_page_count == 1 and doc.metadata["ocr_pages"] == 1


def test_encrypted_pdf_raises(tmp_path: Path) -> None:
    plain = make_pdf(tmp_path / "plain.pdf", [invoice_items()])
    src = pymupdf.open(plain)
    src.save(tmp_path / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    src.save(tmp_path / "owner_only.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="", owner_pw="owner")
    src.close()
    with pytest.raises(EncryptedFileError) as err:
        parse_pdf(tmp_path / "locked.pdf")
    assert err.value.code == "encrypted_file"
    # Only an owner password (empty user password): opens without asking anyone.
    doc = parse_pdf(tmp_path / "owner_only.pdf")
    assert INVOICE_NUMBER in doc.text
    assert doc.metadata["encrypted"] is True


@pytest.mark.parametrize(
    ("save_options", "fraction"),
    [({}, 0.05), ({"garbage": 4, "deflate": True, "use_objstms": 1}, 0.5)],
    ids=["repairable-xref-no-content", "object-streams-unopenable"],
)
def test_truncated_pdf_raises_corrupt(tmp_path: Path, save_options: dict[str, object], fraction: float) -> None:
    doc = pymupdf.open()
    for i in range(3):
        add_text_page(doc, invoice_items(f"INV-{i}"))
    full = tmp_path / "full.pdf"
    doc.save(full, **save_options)
    doc.close()
    data = full.read_bytes()
    truncated = tmp_path / "truncated.pdf"
    truncated.write_bytes(data[: int(len(data) * fraction)])
    with pytest.raises(CorruptFileError) as err:
        parse_pdf(truncated)
    assert err.value.code == "corrupt_file"


def test_partially_truncated_pdf_keeps_recovered_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = make_pdf(tmp_path / "full.pdf", [invoice_items(f"INV-{i}") for i in range(3)])
    data = full.read_bytes()
    # Cut the file right after the third page object: the page survives, its content streams do not.
    third_page = [m.start() for m in re.finditer(rb"/Type\s*/Page\b(?!s)", data)][2]
    truncated = tmp_path / "truncated.pdf"
    truncated.write_bytes(data[: data.index(b"endobj", third_page) + len(b"endobj")])
    monkeypatch.setattr(pdf, "_ocr_page", _no_ocr)  # lost pages render blank: no OCR needed to notice
    doc = parse_pdf(truncated)
    assert len(doc.pages) == 3
    assert "INV-0" in doc.pages[0].text and doc.pages[0].method == "text"
    assert doc.pages[2].method == "none" and doc.pages[2].lines == []
    assert "pdf_repaired" in doc.warnings
    assert "page_3_extraction_failed" in doc.warnings
    assert doc.metadata["repaired"] is True


def _render_only(doc: pymupdf.Document, index: int, _dpi: int, _engine: object) -> Page:
    """Stands in for OCR: still renders the damaged page through MuPDF, but skips recognition (fast)."""
    pdf._render(doc, index, 36)
    return Page(number=index + 1, method="ocr")


@pytest.mark.parametrize(
    "save_options",
    [{}, {"garbage": 4, "deflate": True, "use_objstms": 1}],
    ids=["plain", "object-streams"],
)
def test_truncation_sweep_never_raises_untyped_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_options: dict[str, object]
) -> None:
    monkeypatch.setattr(pdf, "_ocr_page", _render_only)
    doc = pymupdf.open()
    for i in range(3):  # three pages: cutting a plain save at bytes 177-192 broke page_count (RuntimeError)
        add_text_page(doc, invoice_items(f"INV-{i}"))
    full = tmp_path / "full.pdf"
    doc.save(full, **save_options)
    doc.close()
    data = full.read_bytes()
    cuts = sorted({*range(1, 400), *range(400, len(data), max(1, len(data) // 150))})
    outcomes: dict[str, int] = {}
    for cut in cuts:
        truncated = tmp_path / "cut.pdf"
        truncated.write_bytes(data[:cut])
        try:
            result = parse_pdf(truncated)
        except IngestionError as exc:
            outcomes[exc.code] = outcomes.get(exc.code, 0) + 1
        else:
            assert isinstance(result, ParsedDocument)
            outcomes["parsed"] = outcomes.get("parsed", 0) + 1
    assert set(outcomes) <= {"parsed", "corrupt_file", "empty_file"}
    assert outcomes.get("corrupt_file", 0) > 0


def test_progress_callback_errors_are_not_mistaken_for_corruption(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "ok.pdf", [invoice_items()])

    def failing_progress(_current: int, _total: int, _stage: str) -> None:
        raise ConnectionError("progress store is down")

    with pytest.raises(ConnectionError):
        parse_pdf(path, on_progress=failing_progress)


def test_unexpected_parser_errors_become_corrupt_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "ok.pdf", [invoice_items()])

    def explode(*_args: object, **_kwargs: object) -> ParsedDocument:
        raise ValueError("unexpected MuPDF state")

    monkeypatch.setattr(pdf, "_assemble", explode)
    with pytest.raises(CorruptFileError):
        parse_pdf(path)


def test_span_cap_per_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(layout, "MAX_SPANS_PER_PAGE", 20)
    doc = parse_pdf(make_pdf(tmp_path / "dense.pdf", [invoice_items()]))
    assert sum(len(line.spans) for line in doc.pages[0].lines) == 20
    assert "page_1_spans_truncated" in doc.pages[0].warnings
    assert "page_1_spans_truncated" in doc.warnings


def test_random_bytes_pdf_raises_corrupt(tmp_path: Path) -> None:
    rng = random.Random(1234)
    path = tmp_path / "random.pdf"
    path.write_bytes(b"%PDF-1.7\n" + bytes(rng.getrandbits(8) for _ in range(20_000)))
    with pytest.raises(CorruptFileError):
        parse_pdf(path)


def test_page_limit_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, MAX_PDF_PAGES=3)
    with pytest.raises(FileTooLargeError) as err:
        parse_pdf(make_pdf(tmp_path / "four.pdf", [invoice_items()] * 4))
    assert err.value.code == "file_too_large"
    assert parse_pdf(make_pdf(tmp_path / "three.pdf", [invoice_items()] * 3)).page_count == 3


def test_one_corrupt_page_does_not_fail_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "three.pdf", [invoice_items(f"INV-{n}") for n in (1, 2, 3)])
    real_spans = pdf._page_spans

    def flaky_spans(page: pymupdf.Page) -> list:
        if page.number == 1:
            raise RuntimeError("broken content stream")
        return real_spans(page)

    monkeypatch.setattr(pdf, "_page_spans", flaky_spans)
    monkeypatch.setattr(pdf, "_ocr_page", _broken_ocr)
    doc = parse_pdf(path)
    assert [p.number for p in doc.pages] == [1, 2, 3]
    assert "INV-1" in doc.pages[0].text and "INV-3" in doc.pages[2].text
    broken = doc.pages[1]
    assert broken.method == "none" and broken.lines == []
    assert "page_2_extraction_failed" in broken.warnings
    assert "page_2_extraction_failed" in doc.warnings
    assert doc.metadata["failed_pages"] == 1


def test_all_pages_failing_raises_corrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "two.pdf", [invoice_items(), invoice_items()])
    monkeypatch.setattr(pdf, "_page_spans", _broken_ocr)
    monkeypatch.setattr(pdf, "_ocr_page", _broken_ocr)
    with pytest.raises(CorruptFileError):
        parse_pdf(path)


def test_text_layer_failure_falls_back_to_ocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_pdf(tmp_path / "two.pdf", [invoice_items("INV-A1"), invoice_items("INV-B2")])
    real_spans = pdf._page_spans

    def flaky_spans(page: pymupdf.Page) -> list:
        if page.number == 1:
            raise RuntimeError("bad font")
        return real_spans(page)

    monkeypatch.setattr(pdf, "_page_spans", flaky_spans)
    doc = parse_pdf(path)
    assert doc.pages[1].method == "ocr"
    assert "INV-B2" in doc.pages[1].text
    assert "page_2_text_extraction_failed" in doc.pages[1].warnings


def test_blank_page_is_not_ocrd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf, "_ocr_page", _no_ocr)
    doc = parse_pdf(make_pdf(tmp_path / "blank.pdf", [invoice_items(), []]))
    assert doc.pages[1].method == "text" and doc.pages[1].lines == []
    assert doc.warnings == []


def test_rotated_page_reads_in_display_orientation(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=A4[0], height=A4[1])
    page.insert_text((100, 700), f"Invoice No: {INVOICE_NUMBER}", fontsize=12, rotate=90)
    page.insert_text((130, 700), "Total: 99.00", fontsize=12, rotate=90)
    page.set_rotation(90)  # a landscape page stored portrait, as scanners and print drivers do
    doc.save(tmp_path / "rotated.pdf")
    doc.close()
    parsed = parse_pdf(tmp_path / "rotated.pdf").pages[0]
    assert parsed.text.splitlines() == [f"Invoice No: {INVOICE_NUMBER}", "Total: 99.00"]
    assert (parsed.width, parsed.height) == (A4[1], A4[0])


def test_concurrent_parses_from_threads_are_safe(tmp_path: Path) -> None:
    # The local queue backend parses documents from a thread pool; MuPDF itself runs single-threaded.
    paths = [make_pdf(tmp_path / f"doc{n}.pdf", [invoice_items(f"INV-T{n}-{p}") for p in range(15)]) for n in range(6)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        docs = list(executor.map(parse_pdf, paths))
    for n, doc in enumerate(docs):
        assert [f"INV-T{n}-{p}" in page.text for p, page in enumerate(doc.pages)] == [True] * 15


def test_progress_callback_called_per_page(tmp_path: Path) -> None:
    scanned = render_items(invoice_items("INV-SCAN"))
    path = make_pdf(tmp_path / "mixed.pdf", [invoice_items(), scanned, invoice_items()])
    calls: list[tuple[int, int, str]] = []
    parse_pdf(path, on_progress=lambda current, total, stage: calls.append((current, total, stage)))
    assert calls == [(1, 3, "parsing"), (2, 3, "ocr"), (3, 3, "parsing")]


@pytest.mark.parametrize("failure", list(INFRASTRUCTURE_FAILURES.values()), ids=list(INFRASTRUCTURE_FAILURES))
def test_ocr_infrastructure_failure_is_retryable_not_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoice_image: Image.Image, failure: PoolOutcome
) -> None:
    set_env(monkeypatch, OCR_WORKERS=2)
    monkeypatch.setattr(pdf, "run_ocr_jobs", fake_ocr_pool(failure))
    with pytest.raises(OcrUnavailableError) as err:
        parse_pdf(make_pdf(tmp_path / "scan.pdf", [invoice_image]))
    assert err.value.code == "ocr_unavailable" and err.value.permanent is False


def test_scanned_pages_lost_to_the_pool_are_not_returned_as_a_partial_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoice_image: Image.Image
) -> None:
    # The digital page reads fine, but returning it alone would silently drop the scanned page.
    set_env(monkeypatch, OCR_WORKERS=2)
    monkeypatch.setattr(pdf, "run_ocr_jobs", fake_ocr_pool(broken_pool))
    with pytest.raises(OcrUnavailableError) as err:
        parse_pdf(make_pdf(tmp_path / "mixed.pdf", [invoice_items(), invoice_image]))
    assert err.value.permanent is False


def test_pooled_page_that_cannot_be_read_is_still_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoice_image: Image.Image
) -> None:
    set_env(monkeypatch, OCR_WORKERS=2)
    monkeypatch.setattr(pdf, "run_ocr_jobs", fake_ocr_pool(undecodable))
    with pytest.raises(CorruptFileError) as err:
        parse_pdf(make_pdf(tmp_path / "scan.pdf", [invoice_image]))
    assert err.value.permanent is True


def test_hung_ocr_page_raises_retryable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoice_image: Image.Image
) -> None:
    set_env(monkeypatch, OCR_WORKERS=2, OCR_PAGE_TIMEOUT_S=0.3)
    pool = FakePool([None])  # the OCR child never returns
    monkeypatch.setattr(ocr, "_get_pool", lambda _workers: pool)
    with pytest.raises(OcrUnavailableError) as err:
        parse_pdf(make_pdf(tmp_path / "scan.pdf", [invoice_image]))
    assert err.value.permanent is False
    assert all(worker.terminated for worker in pool.workers)


@pytest.mark.slow
def test_large_pdf_1000_pages_parses_under_budget(tmp_path: Path) -> None:
    doc = pymupdf.open()
    for i in range(1000):
        add_text_page(doc, invoice_items(f"INV-{i:05d}"))
    path = tmp_path / "large.pdf"
    doc.save(path)
    doc.close()
    calls: list[int] = []
    start = time.perf_counter()
    parsed = parse_pdf(path, on_progress=lambda current, _total, _stage: calls.append(current))
    elapsed = time.perf_counter() - start
    assert elapsed < 20, f"1000 digital pages took {elapsed:.1f}s"
    assert parsed.page_count == 1000 and parsed.ocr_page_count == 0
    assert calls == list(range(1, 1001))
    assert "INV-00999" in parsed.pages[-1].text


@pytest.mark.slow
def test_ocr_process_pool_parses_scanned_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, OCR_WORKERS=2)
    scans = [render_items(invoice_items(f"INV-POOL-{n}")) for n in (1, 3)]
    path = make_pdf(tmp_path / "pool.pdf", [scans[0], invoice_items("INV-POOL-2"), scans[1]])
    calls: list[tuple[int, int, str]] = []
    try:
        doc = parse_pdf(path, on_progress=lambda current, total, stage: calls.append((current, total, stage)))
    finally:
        shutdown_ocr_pool()
    assert [p.method for p in doc.pages] == ["ocr", "text", "ocr"]
    for number, page in enumerate(doc.pages, start=1):
        assert f"INV-POOL-{number}" in page.text
    assert doc.pages[0].width == A4[0] and all(line.x1 <= A4[0] + 1 for line in doc.pages[0].lines)
    assert [c[0] for c in calls] == [1, 2, 3] and all(c[1] == 3 for c in calls)
    assert sorted(c[2] for c in calls) == ["ocr", "ocr", "parsing"]
