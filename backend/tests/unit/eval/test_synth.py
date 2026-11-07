"""Tests for the synthetic, labeled evaluation dataset generator (``eval.synth``).

Plan-level tests (``make_plan``) check the dataset *design* — composition, error rates, groups, uniqueness — without
rendering anything. Build-level tests render a small, deterministic subset (``--limit``) and check the files against
their labels: files exist, labels validate as ``InvoiceData``, arithmetic is consistent except where an error is
declared, corrupt files expect failure, the build is deterministic, and digital PDFs really print what the labels say
(and do not print what the labels say is absent).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from PIL import Image

from app.domain.schemas import INVOICE_FIELDS, InvoiceData, IssueCode, LineItem, SourceKind
from eval.labels import (
    ERROR_CODES,
    LOCATION_KEYS,
    LabeledFile,
    LabeledRecord,
    issue_codes_for,
    load_labels,
    write_labels,
)
from eval.synth.build import build_dataset, main, make_plan, render_plan
from eval.synth.data import NumberRegistry, ascii_slug, make_invoice
from eval.synth.degrade import QUALITIES, degrade_image
from eval.synth.render import render_pdf, restore_ligature_text
from eval.synth.templates import TEMPLATES, fmt_date, fmt_number, fmt_qty, number_in_words_inr

TOLERANCE = Decimal("0.02")
SMALL_LIMIT = 32
ARITHMETIC_CODES = {"TOTAL_MISMATCH", "LINE_ITEMS_SUM_MISMATCH", "LINE_ITEM_MATH"}
DEV_TEMPLATES = {"T1", "T2", "T3"}
HELDOUT_TEMPLATES = {"T4", "T5", "T6"}
ALL_TEMPLATES = sorted(DEV_TEMPLATES | HELDOUT_TEMPLATES)
# MuPDF draws a fresh random file id when it encrypts, so these two cannot be byte-identical between runs
# (their labels and names are).
NOT_BYTE_REPRODUCIBLE = {"corrupt/encrypted.pdf", "corrupt/owner_locked.pdf"}
CURRENCY_TOKEN = re.compile(r"[$€£₹]|\b(USD|EUR|GBP|INR|CAD|AUD|Rupees)\b|\bRs\.")


# --------------------------------------------------------------------------------------------------------------------
# helpers (deliberately independent of the generator's own arithmetic)
# --------------------------------------------------------------------------------------------------------------------
def _dec(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _arithmetic_codes(record: dict[str, Any]) -> set[str]:
    """Which arithmetic problems a record exhibits, computed from first principles."""
    codes: set[str] = set()
    subtotal, total = _dec(record["subtotal"]), _dec(record["total"])
    discount, shipping, tax = (_dec(record[k]) or Decimal(0) for k in ("discount", "shipping", "tax"))
    if subtotal is not None and total is not None and abs(subtotal - discount + shipping + tax - total) > TOLERANCE:
        codes.add("TOTAL_MISMATCH")
    items = record["line_items"]
    amounts = [_dec(i["amount"]) for i in items]
    known = [a for a in amounts if a is not None]
    if (
        subtotal is not None
        and items
        and len(known) == len(items)
        and abs(sum(known, Decimal(0)) - subtotal) > TOLERANCE
    ):
        codes.add("LINE_ITEMS_SUM_MISMATCH")
    for item in items:
        qty, price, amount = _dec(item["quantity"]), _dec(item["unit_price"]), _dec(item["amount"])
        if qty is not None and price is not None and amount is not None and abs(qty * price - amount) > TOLERANCE:
            codes.add("LINE_ITEM_MATH")
    return codes


def _record_dict(record: LabeledRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _invoice_part(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in LOCATION_KEYS}


def _amount_pattern(value: Decimal) -> re.Pattern[str]:
    """The amount as printed in any supported locale: at most one separator between digits, `.`/`,` before the two
    decimals (``1,234.56``, ``1.234,56``, ``1 234,56``, ``1,23,456.00``)."""
    whole, _, cents = f"{value:.2f}".partition(".")
    return re.compile(r"\D?".join(re.escape(d) for d in whole) + r"[.,]" + cents)


def _page_text(pdf_path: Path, start: int, end: int) -> str:
    with pymupdf.open(pdf_path) as doc:
        return "\n".join(doc[i - 1].get_text() for i in range(start, end + 1))


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _success_digital_pdfs(labels: list[LabeledFile]) -> list[LabeledFile]:
    return [f for f in labels if f.kind == "pdf" and f.quality == "clean" and f.expect == "success"]


# --------------------------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plan_labels() -> list[LabeledFile]:
    return [p.labeled() for p in make_plan(42)]


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[LabeledFile]]:
    out = tmp_path_factory.mktemp("synth_small")
    build_dataset(out, seed=7, limit=SMALL_LIMIT)
    return out, load_labels(out / "labels.json")


# --------------------------------------------------------------------------------------------------------------------
# Small dataset build
# --------------------------------------------------------------------------------------------------------------------
def test_build_small_dataset(small_dataset: tuple[Path, list[LabeledFile]]) -> None:
    out, labels = small_dataset
    raw = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["seed"] == 7
    assert 20 <= len(labels) <= SMALL_LIMIT

    names = [f.file for f in labels]
    assert len(names) == len(set(names)), "file names must be unique"
    for f in labels:
        path = out / f.file
        assert path.is_file(), f"{f.file} missing"
        if f.expected_error_code != "empty_file":
            assert path.stat().st_size > 0, f"{f.file} is unexpectedly empty"
        assert f.split in {"dev", "heldout", "na"}
        assert f.quality in QUALITIES

    for raw_file in raw["files"]:
        for record in raw_file["records"]:
            assert set(record) == set(INVOICE_FIELDS) | set(LOCATION_KEYS)
            InvoiceData.model_validate(_invoice_part(record))  # labels validate against the shared contract

    # arithmetic is consistent unless the file declares the corresponding problem
    for f in labels:
        declared = {c for c in f.expected_issue_codes if c in ARITHMETIC_CODES}
        found: set[str] = set()
        for rec in f.records:
            found |= _arithmetic_codes(_record_dict(rec))
        assert found == declared, f"{f.file}: arithmetic {found} vs declared {declared}"

    # corrupt / invalid files expect failure with a known code; everything else expects success
    corrupt = [f for f in labels if f.kind == "corrupt"]
    assert len(corrupt) >= 8, "the cheap corrupt files are front-loaded into every subset"
    for f in labels:
        if f.expect == "failure":
            assert f.kind == "corrupt"
            assert f.expected_error_code in ERROR_CODES
            assert f.records == []
            assert f.template is None
            assert f.split == "na"
        else:
            assert f.expected_error_code is None
            assert f.records, f"{f.file}: a file expected to succeed must carry labels"

    # the declared page counts are the real ones
    for f in labels:
        if f.expect == "success" and f.kind == "pdf":
            with pymupdf.open(out / f.file) as doc:
                assert doc.page_count == f.pages, f.file
        if f.kind == "tiff":
            with Image.open(out / f.file) as img:
                assert getattr(img, "n_frames", 1) == f.pages, f.file


def test_build_is_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    build_dataset(first, seed=11, limit=16)
    build_dataset(second, seed=11, limit=16)
    assert (first / "labels.json").read_bytes() == (second / "labels.json").read_bytes()
    files_a = sorted(p.relative_to(first).as_posix() for p in first.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(second).as_posix() for p in second.rglob("*") if p.is_file())
    assert files_a == files_b
    for rel in files_a:
        if rel in NOT_BYTE_REPRODUCIBLE:
            continue
        digest_a = hashlib.sha256((first / rel).read_bytes()).hexdigest()
        digest_b = hashlib.sha256((second / rel).read_bytes()).hexdigest()
        assert digest_a == digest_b, f"{rel} differs between runs with the same seed"


def test_different_seeds_give_different_data() -> None:
    a = [p.labeled().model_dump(mode="json") for p in make_plan(1)]
    b = [p.labeled().model_dump(mode="json") for p in make_plan(2)]
    assert [x["file"] for x in a] == [x["file"] for x in b], "the file list is part of the design, not of the seed"
    assert json.dumps(a) != json.dumps(b)
    assert json.dumps(a) == json.dumps([p.labeled().model_dump(mode="json") for p in make_plan(1)])


def test_digital_pdfs_print_what_the_labels_say(small_dataset: tuple[Path, list[LabeledFile]]) -> None:
    out, labels = small_dataset
    digital = _success_digital_pdfs(labels)
    assert {f.template for f in digital} >= set(ALL_TEMPLATES), "every template has a digital PDF in every subset"
    checked = 0
    for f in digital:
        for rec in f.records:
            assert rec.page_start is not None and rec.page_end is not None
            text = _page_text(out / f.file, rec.page_start, rec.page_end)
            flat = _norm_space(text)
            if rec.invoice_number is not None:
                assert rec.invoice_number in flat, f"{f.file}: invoice number {rec.invoice_number!r} not printed"
            if rec.total is not None:
                assert _amount_pattern(rec.total).search(text), f"{f.file}: total {rec.total} not printed"
            if rec.supplier_name is not None:
                assert _norm_space(rec.supplier_name) in flat, f"{f.file}: supplier not printed"
            if rec.customer_name is not None:
                assert _norm_space(rec.customer_name) in flat, f"{f.file}: customer not printed"
            for field_name in ("invoice_date", "due_date"):
                value = getattr(rec, field_name)
                if value is not None:
                    tpl = TEMPLATES[f.template or ""]
                    shown = {fmt_date(value, style) for style, _ in tpl.date_styles}
                    assert any(s in flat for s in shown), f"{f.file}: {field_name} {value} not printed"
            for item in rec.line_items:
                assert _norm_space(item.description or "") in flat, f"{f.file}: item text missing"
                if item.amount is not None:
                    assert _amount_pattern(item.amount).search(text), f"{f.file}: line amount {item.amount} missing"
            checked += 1
    assert checked >= 6


OMITTABLE = (
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "customer_name",
    "total",
    "subtotal",
    "tax",
    "currency",
    "line_items",
)
RECEIPT_ALWAYS_HAS = {"invoice_date", "supplier_name", "total", "subtotal", "line_items"}


def _printed_figures(spec: Any) -> set[Decimal]:
    """Every amount the document prints (so a coincidence between two figures is not mistaken for a leak)."""
    record = spec.record()
    figures = {i.amount for i in record.line_items if i.amount is not None}
    figures |= {i.unit_price for i in record.line_items if i.unit_price is not None}
    figures |= {
        v for v in (record.subtotal, record.discount, record.shipping, record.tax, record.total) if v is not None
    }
    return figures | {t.amount for t in spec.taxes}


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_omitted_fields_are_really_not_printed(template: str) -> None:
    """A label may only be null if the document does not print the value anywhere (header, footer, notes, words)."""
    tpl = TEMPLATES[template]
    for index, field_name in enumerate(OMITTABLE):
        if template == "T5" and field_name in RECEIPT_ALWAYS_HAS:
            continue
        seed = f"{template}-{field_name}"
        kwargs: dict[str, Any] = {"n_items": 4, "discount_shipping": template != "T5"}
        shown = make_invoice(random.Random(seed), template, seed, NumberRegistry(), **kwargs)
        if field_name == "customer_name" and shown.customer is None:
            continue  # most receipts have no customer: nothing to hide
        hidden = make_invoice(
            random.Random(seed), template, seed, NumberRegistry(), omit=frozenset({field_name}), **kwargs
        )
        hidden.style = shown.style = tpl.new_style(random.Random(f"style-{index}"))
        with pymupdf.open(stream=render_pdf([hidden]).data, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        flat = _norm_space(text)
        label = hidden.record()
        elsewhere = _printed_figures(hidden)

        def amount_leaked(value: Decimal | None, text: str = text, elsewhere: set[Decimal] = elsewhere) -> bool:
            return value is not None and value not in elsewhere and bool(_amount_pattern(value).search(text))

        if field_name == "invoice_number":
            assert label.invoice_number is None and shown.invoice_number
            assert shown.invoice_number not in flat
        elif field_name == "invoice_date":
            assert label.invoice_date is None and shown.invoice_date
            assert fmt_date(shown.invoice_date, shown.style["date"]) not in flat
        elif field_name == "supplier_name":
            assert label.supplier_name is None and shown.supplier.name
            assert shown.supplier.name not in flat
            assert ascii_slug(shown.supplier.name)[:12] not in flat.lower(), "the name leaks through an e-mail or URL"
        elif field_name == "customer_name":
            assert label.customer_name is None and shown.customer and shown.customer.name
            assert shown.customer.name not in flat
        elif field_name == "total":
            assert label.total is None
            assert not amount_leaked(shown.total)
        elif field_name == "subtotal":
            assert label.subtotal is None
            assert not amount_leaked(shown.subtotal)
        elif field_name == "tax":
            assert label.tax is None and label.tax_rate is None and shown.taxes
            assert not any(amount_leaked(line.amount) for line in shown.taxes)
        elif field_name == "currency":
            assert label.currency is None
            assert not CURRENCY_TOKEN.search(flat), CURRENCY_TOKEN.search(flat)
        else:  # line_items
            assert label.line_items == []
            assert all(_norm_space(item.description) not in flat for item in shown.items)


def test_images_and_spreadsheets_are_readable_and_match_labels(small_dataset: tuple[Path, list[LabeledFile]]) -> None:
    import openpyxl
    import xlrd
    from charset_normalizer import from_bytes

    out, labels = small_dataset
    seen: Counter[str] = Counter()
    for f in labels:
        path = out / f.file
        if f.kind in {"png", "jpg", "tiff"}:
            with Image.open(path) as img:
                assert getattr(img, "n_frames", 1) == f.pages
                img.load()
                assert img.width > 100 and img.height > 100
            seen[f.kind] += 1
        elif f.kind == "xlsx":
            wb = openpyxl.load_workbook(path)
            for ws in wb:
                for row in ws.iter_rows():
                    for cell in row:
                        assert cell.data_type != "f", f"{f.file}: formula in {ws.title}!{cell.coordinate}"
            blob = "\n".join(str(c.value) for ws in wb for row in ws.iter_rows() for c in row if c.value is not None)
            assert len(wb.sheetnames) >= 1
            for rec in f.records:
                if rec.row_number is not None and rec.sheet_name is not None:
                    cells = [str(c.value) for c in wb[rec.sheet_name][rec.row_number]]
                    assert rec.invoice_number is not None and rec.invoice_number in cells, (
                        f"{f.file}: row {rec.row_number} is not {rec.invoice_number}"
                    )
                elif rec.invoice_number is not None:
                    assert rec.invoice_number in blob
                    assert rec.sheet_name in wb.sheetnames
            seen["xlsx"] += 1
        elif f.kind == "xls":
            book = xlrd.open_workbook(path)
            for rec in f.records:
                sheet = book.sheet_by_name(rec.sheet_name) if rec.sheet_name else book.sheet_by_index(0)
                if rec.row_number is not None:
                    cells = [str(sheet.cell_value(rec.row_number - 1, c)) for c in range(sheet.ncols)]
                    assert rec.invoice_number is not None and rec.invoice_number in cells, (
                        f"{f.file}: row {rec.row_number} is not {rec.invoice_number}"
                    )
                elif rec.invoice_number is not None:
                    blob = "\n".join(
                        str(sheet.cell_value(r, c)) for r in range(sheet.nrows) for c in range(sheet.ncols)
                    )
                    assert rec.invoice_number in blob
            seen["xls"] += 1
        elif f.kind == "csv":
            text = str(from_bytes(path.read_bytes()).best())
            lines = text.splitlines()
            for rec in f.records:
                if rec.row_number is not None:
                    assert rec.invoice_number is not None
                    assert rec.invoice_number in lines[rec.row_number - 1], f"{f.file}: line {rec.row_number}"
                elif rec.invoice_number is not None:
                    assert rec.invoice_number in text
            rows = list(csv.reader(io.StringIO(text), delimiter=max(",;\t|", key=text[:2000].count)))
            assert len(rows) >= len(f.records)
            seen["csv"] += 1
    assert {"png", "jpg", "tiff", "xlsx", "xls", "csv"} <= set(seen), f"small subset lacks kinds: {dict(seen)}"


def test_corrupt_files_really_are_broken(small_dataset: tuple[Path, list[LabeledFile]]) -> None:
    out, labels = small_dataset
    failing = [f for f in labels if f.expect == "failure"]
    assert {f.expected_error_code for f in failing} == set(ERROR_CODES)
    for f in failing:
        path = out / f.file
        data = path.read_bytes()
        code = f.expected_error_code
        if code == "empty_file":
            assert data == b""
        elif code == "encrypted_file":
            with pymupdf.open(path) as doc:
                assert doc.needs_pass
        elif code == "unsupported_file_type":
            assert not data.startswith((b"%PDF", b"\x89PNG", b"PK\x03\x04", b"\xff\xd8", b"II*\x00", b"MM\x00*"))
            assert data.startswith(b"MZ") or b"<html" in data[:200].lower()
        elif path.suffix == ".png":
            with pytest.raises(OSError), Image.open(path) as img:
                img.load()
        elif path.suffix == ".xlsx":
            import zipfile

            with pytest.raises(zipfile.BadZipFile):
                zipfile.ZipFile(path)
        elif path.name == "truncated.pdf":
            assert data.startswith(b"%PDF")
            with pymupdf.open(path) as doc:  # PyMuPDF repairs what it can: a cut-off file loses its pages' content
                assert doc.page_count == 0 or not _pdf_readable(doc)
        else:  # random_bytes.pdf
            assert not data.startswith(b"%PDF")
    renamed = [f for f in labels if f.kind == "png" and f.file.endswith(".pdf")]
    assert len(renamed) == 1
    assert renamed[0].expect == "success"
    with Image.open(out / renamed[0].file) as img:
        assert img.format == "PNG"
    locked = [f for f in labels if f.file.endswith("owner_locked.pdf")]
    assert len(locked) == 1
    with pymupdf.open(out / locked[0].file) as doc:
        assert doc.page_count == 1 and not doc.needs_pass


def _pdf_readable(doc: pymupdf.Document) -> bool:
    try:
        return any(page.get_text().strip() for page in doc)
    except (RuntimeError, ValueError):
        return False


def test_multi_invoice_pdf_page_ranges_partition_the_file(tmp_path: Path) -> None:
    plan = next(p for p in make_plan(42) if p.path.startswith("multi/") and p.fmt == "pdf_digital")
    render_plan(plan, tmp_path, 42)
    labeled = plan.labeled()
    assert 10 <= len(labeled.records) <= 25
    expected_start = 1
    with pymupdf.open(tmp_path / plan.path) as doc:
        assert doc.page_count == labeled.pages
    for rec in labeled.records:
        assert rec.page_start == expected_start and rec.page_end is not None
        assert 1 <= rec.page_end - expected_start + 1 <= 3, "1-3 pages per invoice"
        text = _page_text(tmp_path / plan.path, expected_start, rec.page_end)
        assert rec.invoice_number is not None and rec.invoice_number in _norm_space(text)
        assert rec.total is not None and _amount_pattern(rec.total).search(text)
        expected_start = rec.page_end + 1
    assert expected_start - 1 == labeled.pages


def test_long_invoice_paginates_with_continuation_and_totals_on_the_last_page() -> None:
    spec = make_invoice(
        random.Random(3),
        "T1",
        "long",
        NumberRegistry(),
        n_items=25,
        industry="consulting",
        long_text=True,
        max_desc=400,
        discount_shipping=False,
    )
    spec.style = TEMPLATES["T1"].new_style(random.Random(4))
    rendered = render_pdf([spec])
    assert rendered.pages >= 2 and rendered.ranges == [(1, rendered.pages)]
    with pymupdf.open(stream=rendered.data, filetype="pdf") as doc:
        pages = [p.get_text() for p in doc]
    assert "Continued on next page" in pages[0]
    assert "Continued on next page" not in pages[-1]
    for number, text in enumerate(pages, start=1):
        assert re.search(rf"Page\s*{number}\s*(of|/)\s*{len(pages)}", text), f"page {number} has no 'Page k of n'"
        assert spec.style["v"]["desc"] in text, "the table header is repeated on every page"
    assert spec.total is not None
    assert _amount_pattern(spec.total).search(pages[-1])
    assert not any(_amount_pattern(spec.total).search(p) for p in pages[:-1])


def test_ligature_glyphs_do_not_leak_into_the_text_layer() -> None:
    spec = make_invoice(random.Random(8), "T3", "lig", NumberRegistry(), n_items=2)
    spec.items[0].description = "Office file X-ray flow affluent staff"
    spec.style = TEMPLATES["T3"].new_style(random.Random(9))
    rendered = render_pdf([spec])
    with pymupdf.open(stream=rendered.data, filetype="pdf") as doc:
        text = doc[0].get_text()
    assert "Office file X-ray flow affluent staff" in _norm_space(text)
    assert not any(0xFB00 <= ord(c) <= 0xFB06 for c in text) and "\xad" not in text and "‐" not in text
    with pymupdf.open() as doc:  # and the repair itself reports what it fixed
        page = doc.new_page()
        page.insert_htmlbox(
            pymupdf.Rect(20, 20, 300, 80), "<p>efficient office</p>", css="body {font-family: Helvetica}"
        )
        assert restore_ligature_text(doc) >= 1


def test_formulas_are_never_written(tmp_path: Path) -> None:
    import openpyxl

    from eval.synth import spreadsheets

    spec = make_invoice(random.Random(5), "T2", "wb", NumberRegistry(), n_items=3)
    spec.style = TEMPLATES["T2"].new_style(random.Random(6))
    data = spreadsheets.invoice_xlsx_bytes([spec], ["Invoice"], typed=True)
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    ws = wb["Invoice"]
    values = [c.value for row in ws.iter_rows() for c in row if c.value is not None]
    assert any(isinstance(v, (int, float)) for v in values)
    assert not any(
        c.data_type == "f" for row in openpyxl.load_workbook(io.BytesIO(data))["Invoice"].iter_rows() for c in row
    )
    spec.items[0].description = "=SUM(A1:A2)"
    with pytest.raises(ValueError, match="formula"):
        spreadsheets.invoice_xlsx_bytes([spec], ["Invoice"], typed=True)


# --------------------------------------------------------------------------------------------------------------------
# plan-level design checks (the whole default dataset, nothing rendered)
# --------------------------------------------------------------------------------------------------------------------
def test_plan_size_and_split(plan_labels: list[LabeledFile]) -> None:
    assert 150 <= len(plan_labels) <= 190
    names = [f.file for f in plan_labels]
    assert len(names) == len(set(names))
    assert not any(n.startswith("scale/") for n in names)
    for f in plan_labels:
        if f.template in DEV_TEMPLATES:
            assert f.split == "dev"
        elif f.template in HELDOUT_TEMPLATES:
            assert f.split == "heldout"
        else:
            assert f.split == "na"
    # corrupt files are cheap, so they are front-loaded: any --limit subset exercises failure handling
    assert any(f.kind == "corrupt" for f in plan_labels[:20])


def test_every_template_appears_in_every_applicable_format(plan_labels: list[LabeledFile]) -> None:
    for tpl in ALL_TEMPLATES:
        mine = [f for f in plan_labels if f.template == tpl and f.expect == "success"]
        kinds = Counter(f.kind for f in mine)
        digital = [f for f in mine if f.kind == "pdf" and f.quality == "clean"]
        scanned = [f for f in mine if f.kind == "pdf" and f.quality != "clean"]
        assert len(digital) >= 5, f"{tpl}: digital PDFs"
        assert len(scanned) >= 2, f"{tpl}: scanned PDFs"
        for kind in ("png", "jpg", "tiff", "xlsx", "xls", "csv"):
            assert kinds[kind] >= 1, f"{tpl}: no {kind}"
        qualities = {f.quality for f in mine}
        assert {"clean", "scan_light", "scan_heavy"} <= qualities, f"{tpl}: {qualities}"
    assert {f.quality for f in plan_labels} == set(QUALITIES)  # clean, scan_light, scan_heavy, photo, lowres
    assert len([f for f in plan_labels if f.kind == "tiff" and f.pages >= 2]) >= 2, "multi-frame TIFFs"
    multi_invoice = [f for f in plan_labels if f.kind == "pdf" and len(f.records) >= 10]
    assert len(multi_invoice) == 4
    assert all(10 <= len(f.records) <= 25 for f in multi_invoice)
    assert {f.quality for f in multi_invoice} == {"clean", "scan_light"}, "digital, mixed and scanned batches"


def test_plan_labels_are_internally_consistent(plan_labels: list[LabeledFile]) -> None:
    for f in plan_labels:
        assert len(f.record_issues) == len(f.records)
        union: set[str] = set()
        for rec, codes in zip(f.records, f.record_issues, strict=True):
            InvoiceData.model_validate(_invoice_part(_record_dict(rec)))
            declared = {c for c in codes if c in ARITHMETIC_CODES}
            assert _arithmetic_codes(_record_dict(rec)) == declared, f"{f.file}: record {rec.invoice_number}"
            union |= set(codes)
        assert union == set(f.expected_issue_codes), f.file
        if any(r.invoice_number is None for r in f.records):
            assert "MISSING_FIELD" in f.expected_issue_codes, (
                f"{f.file}: an unnumbered document must flag MISSING_FIELD"
            )


def test_error_rates_match_the_brief(plan_labels: list[LabeledFile]) -> None:
    docs = [
        (f, r, c)
        for f in plan_labels
        if f.expect == "success"
        for r, c in zip(f.records, f.record_issues, strict=True)
        if r.source_kind is SourceKind.INVOICE_DOCUMENT
    ]
    n = len(docs)
    assert n >= 150
    rate = {code: sum(code in c for _, _, c in docs) / n for code in ARITHMETIC_CODES}
    assert 0.06 <= rate["TOTAL_MISMATCH"] <= 0.11, rate
    assert 0.035 <= rate["LINE_ITEMS_SUM_MISMATCH"] <= 0.07, rate
    assert 0.02 <= rate["LINE_ITEM_MATH"] <= 0.045, rate
    required = ("invoice_number", "invoice_date", "supplier_name", "total")
    missing_required = [d for d in docs if any(getattr(d[1], k) is None for k in required)]
    assert 8 <= len(missing_required) <= 30
    assert all("MISSING_FIELD" in c for _, _, c in missing_required)
    invoices = [d for d in docs if d[1].document_type.value != "receipt"]  # a retail receipt has no customer anyway
    no_customer = sum(1 for _, r, _ in invoices if r.customer_name is None)
    assert 0.06 <= no_customer / len(invoices) <= 0.16, no_customer / len(invoices)
    unnumbered_receipts = [d for d in docs if d[1].document_type.value == "receipt" and d[1].invoice_number is None]
    assert len(unnumbered_receipts) >= 2


def test_content_variety(plan_labels: list[LabeledFile]) -> None:
    records = [r for f in plan_labels if f.expect == "success" for r in f.records]
    currencies = Counter(r.currency for r in records)
    assert {"USD", "EUR", "GBP", "INR", "CAD", "AUD"} <= set(currencies)
    assert None in currencies, "a few documents print no currency at all"
    sizes = [len(r.line_items) for r in records if r.line_items]
    assert min(sizes) == 1 and 20 <= max(sizes) <= 25
    docs = [r for r in records if r.source_kind is SourceKind.INVOICE_DOCUMENT and r.document_type.value != "receipt"]
    with_due = sum(1 for r in docs if r.due_date is not None)
    assert with_due / len(docs) >= 0.7
    assert any(r.discount is not None for r in docs) and any(r.shipping is not None for r in docs)
    assert any(r.document_type.value == "credit_note" for r in docs)
    assert any(r.tax_rate is not None for r in docs) and any(r.tax_rate is None and r.tax is not None for r in docs)
    names = [r.supplier_name for r in records if r.supplier_name]
    assert any("&" in n for n in names)
    assert any(re.search(r"[À-ÿ]", n) for n in names), "accented names"
    assert any(len(n) >= 40 for n in names), "long names"
    for r in records:
        if r.invoice_date is not None:
            assert date(2023, 1, 1) <= r.invoice_date <= date(2025, 12, 31)
            if r.due_date is not None:
                assert r.due_date >= r.invoice_date
        for amount in (r.subtotal, r.tax, r.total):
            assert amount is None or amount == amount.quantize(Decimal("0.01"))


def test_invoice_numbers_are_unique_except_for_intended_repeats(plan_labels: list[LabeledFile]) -> None:
    seen: dict[str, str] = {}
    repeats = 0
    for f in plan_labels:
        for r in f.records:
            if r.invoice_number is None:
                continue
            key = re.sub(r"[^A-Z0-9]", "", r.invoice_number.upper())
            if key in seen:
                repeats += 1
                assert f.group is not None, f"{f.file}: {r.invoice_number} repeated outside a duplicate/xv group"
            seen[key] = f.file
    assert repeats, "duplicate pairs and the cross-verification set repeat numbers by design"


def test_duplicate_pairs(plan_labels: list[LabeledFile]) -> None:
    groups: dict[str, list[LabeledFile]] = {}
    for f in plan_labels:
        if f.group and f.group.startswith("dup-"):
            groups.setdefault(f.group, []).append(f)
    assert sorted(groups) == [f"dup-{i}" for i in range(1, 6)]
    for name, members in groups.items():
        assert len(members) == 2, name
        original, copy = members  # listed (and to be ingested) in this order
        assert original.kind == "pdf" and original.quality == "clean"
        assert copy.kind in {"jpg", "png", "pdf"} and copy.quality != "clean"
        assert _invoice_part(_record_dict(original.records[0])) == _invoice_part(_record_dict(copy.records[0]))
        assert "DUPLICATE_INVOICE" in copy.expected_issue_codes
        assert "DUPLICATE_INVOICE" not in original.expected_issue_codes
    assert sum(1 for m in groups.values() if m[1].kind == "jpg") >= 2


def test_cross_verification_set(plan_labels: list[LabeledFile]) -> None:
    xv = [f for f in plan_labels if f.group == "xv"]
    ledgers = [f for f in xv if f.kind == "xlsx"]
    pdfs = [f for f in xv if f.kind == "pdf"]
    assert len(ledgers) == 1
    ledger = ledgers[0]
    assert len({r.sheet_name for r in ledger.records}) == 2, "the ledger has two sheets of rows"
    assert 18 <= len(ledger.records) <= 22
    assert len(pdfs) == len(ledger.records)
    by_number = {f.records[0].invoice_number: f for f in pdfs}
    flagged_pdf = [f for f in pdfs if "CROSS_VERIFICATION_MISMATCH" in f.expected_issue_codes]
    flagged_rows = [i for i, codes in enumerate(ledger.record_issues) if "CROSS_VERIFICATION_MISMATCH" in codes]
    assert len(flagged_pdf) == 3 and len(flagged_rows) == 3
    assert "CROSS_VERIFICATION_MISMATCH" in ledger.expected_issue_codes
    mismatching: set[str | None] = set()
    for i, row in enumerate(ledger.records):
        assert row.source_kind is SourceKind.LEDGER_ROW and row.row_number is not None
        pdf = by_number[row.invoice_number].records[0]
        same = all(getattr(row, k) == getattr(pdf, k) for k in ("invoice_date", "total", "currency", "supplier_name"))
        if i in flagged_rows:
            assert not same, "a flagged row must really differ from its PDF (different total or date)"
            mismatching.add(row.invoice_number)
        else:
            assert same, f"row {row.invoice_number} should match its PDF"
    assert mismatching == {f.records[0].invoice_number for f in flagged_pdf}
    for f in pdfs:
        assert f.template in ALL_TEMPLATES and not (ARITHMETIC_CODES & set(f.expected_issue_codes))
    other = {
        (r.sheet_name, r.row_number) for f in plan_labels if f.group != "xv" and f.kind == "xlsx" for r in f.records
    }
    assert len(other) > 0


def test_corrupt_set_matches_the_brief(plan_labels: list[LabeledFile]) -> None:
    failing = [f for f in plan_labels if f.expect == "failure"]
    assert Counter(f.expected_error_code for f in failing) == Counter(
        {"empty_file": 1, "corrupt_file": 4, "encrypted_file": 1, "unsupported_file_type": 2}
    )
    assert Counter(Path(f.file).suffix for f in failing) == Counter({".pdf": 5, ".xlsx": 2, ".png": 1})
    lookalikes = [f for f in plan_labels if f.file.startswith("corrupt/") and f.expect == "success"]
    assert sorted(Path(f.file).name for f in lookalikes) == ["owner_locked.pdf", "scan_saved_as.pdf"]
    sniffed = [f for f in lookalikes if f.kind == "png"]
    assert len(sniffed) == 1 and sniffed[0].file.endswith(".pdf") and sniffed[0].records


def test_scale_plan_adds_the_throughput_files() -> None:
    labels = [p.labeled() for p in make_plan(42, scale=True)]
    scale = {f.file: f for f in labels if f.file.startswith("scale/")}
    assert set(scale) == {"scale/large_1000_pages.pdf", "scale/scanned_100_pages.pdf"}
    # the plan holds more documents than fit: the renderer keeps those that make the page count come out exact
    assert len(scale["scale/large_1000_pages.pdf"].records) >= 300
    assert scale["scale/large_1000_pages.pdf"].quality == "clean"
    assert scale["scale/scanned_100_pages.pdf"].quality == "scan_light"
    assert len(scale["scale/scanned_100_pages.pdf"].records) >= 30
    base = [f.file for f in labels if not f.file.startswith("scale/")]
    assert base == [p.labeled().file for p in make_plan(42)]


def test_limit_keeps_groups_whole() -> None:
    plan_all = [p.labeled() for p in make_plan(42)]
    for limit in (9, 40, 77):
        plan = [p.labeled() for p in make_plan(42, limit=limit)]
        assert len(plan) <= limit
        assert [f.file for f in plan] == [f.file for f in plan_all if f.file in {x.file for x in plan}]
        for group in {f.group for f in plan if f.group}:
            assert sum(1 for f in plan if f.group == group) == sum(1 for f in plan_all if f.group == group)


# --------------------------------------------------------------------------------------------------------------------
# helpers: formatting, label derivation, label files
# --------------------------------------------------------------------------------------------------------------------
def test_number_and_date_formats() -> None:
    assert fmt_number(Decimal("1234.5")) == "1,234.50"
    assert fmt_number(Decimal("1234.56"), group=".", dec=",") == "1.234,56"
    assert fmt_number(Decimal("1234.56"), group=" ", dec=",") == "1 234,56"
    assert fmt_number(Decimal("123456.78"), indian=True) == "1,23,456.78"
    assert fmt_number(Decimal("12345678.9"), indian=True) == "1,23,45,678.90"
    assert fmt_number(Decimal("999.99"), indian=True) == "999.99"
    assert fmt_qty(Decimal("7.50")) == "7.5"
    assert fmt_qty(Decimal("12"), places=2) == "12.00"
    assert fmt_qty(Decimal("7.5"), dec=",") == "7,5"
    day = date(2024, 3, 5)
    assert fmt_date(day, "us") == "03/05/2024"
    assert fmt_date(day, "eu_dot") == "05.03.2024"
    assert fmt_date(day, "uk_long") == "5 March 2024"
    assert fmt_date(day, "in_mon") == "05-Mar-2024"
    assert fmt_date(day, "iso") == "2024-03-05"
    assert number_in_words_inr(Decimal("255719.58")) == (
        "Rupees Two Lakh Fifty-Five Thousand Seven Hundred Nineteen and Fifty-Eight Paise Only"
    )


def test_issue_codes_are_derived_from_the_labels() -> None:
    item = LineItem(description="x", quantity=Decimal(2), unit_price=Decimal("50.00"), amount=Decimal("100.00"))
    base: dict[str, Any] = {
        "invoice_number": "A1",
        "invoice_date": date(2024, 3, 5),
        "supplier_name": "S",
        "customer_name": "C",
        "subtotal": Decimal("100.00"),
        "tax": Decimal("10.00"),
        "total": Decimal("110.00"),
        "currency": "USD",
        "line_items": [item],
    }

    def codes(**changes: Any) -> list[str]:
        record = LabeledRecord.model_validate({**base, **changes})
        return [c.value for c in issue_codes_for(record, record.source_kind)]

    assert codes() == []
    assert codes(total=Decimal("110.02")) == []  # within the 0.02 tolerance
    assert codes(total=Decimal("110.03")) == ["TOTAL_MISMATCH"]
    assert (
        codes(discount=Decimal("10.00"), shipping=Decimal("5.00"), tax=Decimal("9.00"), total=Decimal("104.00")) == []
    )
    assert codes(subtotal=Decimal("120.00"), total=Decimal("130.00")) == ["LINE_ITEMS_SUM_MISMATCH"]
    wrong = LineItem(description="x", quantity=Decimal(2), unit_price=Decimal("50.00"), amount=Decimal("90.00"))
    assert codes(line_items=[wrong], subtotal=Decimal("90.00"), total=Decimal("100.00")) == ["LINE_ITEM_MATH"]
    assert codes(customer_name=None) == ["MISSING_FIELD"]
    assert codes(line_items=[]) == ["MISSING_FIELD"]
    ledger = LabeledRecord.model_validate({**base, "line_items": [], "source_kind": SourceKind.LEDGER_ROW})
    assert issue_codes_for(ledger, SourceKind.LEDGER_ROW) == []  # ledger rows have no line items by nature


def test_labels_round_trip_and_reject_other_versions(tmp_path: Path) -> None:
    record = LabeledRecord(invoice_number="A1", total=Decimal("12.50"), currency="EUR", page_start=1, page_end=2)
    entry = LabeledFile(
        file="pdf/a.pdf",
        kind="pdf",
        template="T2",
        split="dev",
        quality="clean",
        expect="success",
        records=[record],
        expected_issue_codes=[IssueCode.MISSING_FIELD],
        pages=2,
        record_issues=[[IssueCode.MISSING_FIELD]],
    )
    path = tmp_path / "labels.json"
    write_labels(path, 3, [entry])
    assert load_labels(path) == [entry]
    assert json.loads(path.read_text(encoding="utf-8"))["files"][0]["records"][0]["total"] == "12.50"
    path.write_text(json.dumps({"version": 2, "seed": 3, "files": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_labels(path)


# --------------------------------------------------------------------------------------------------------------------
# degradations
# --------------------------------------------------------------------------------------------------------------------
def _sample_page() -> Image.Image:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_htmlbox(pymupdf.Rect(40, 40, 555, 400), "<h2>Invoice INV-2024-0042</h2><p>Total: 1,234.56</p>" * 6)
    pix = page.get_pixmap(dpi=100, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def test_degradations_are_deterministic_and_ordered_by_severity() -> None:
    base = _sample_page()
    gray = np.asarray(base.convert("L"), dtype=np.float32)
    distance: dict[str, float] = {}
    for quality in ("scan_light", "scan_heavy", "photo"):
        a, jpeg_q_a = degrade_image(base, quality, np.random.default_rng(5))
        b, jpeg_q_b = degrade_image(base, quality, np.random.default_rng(5))
        assert np.array_equal(np.asarray(a), np.asarray(b)) and jpeg_q_a == jpeg_q_b
        resized = np.asarray(a.convert("L").resize(base.size), dtype=np.float32)
        distance[quality] = float(np.abs(resized - gray).mean())
    assert distance["scan_light"] < distance["scan_heavy"]
    _, light_q = degrade_image(base, "scan_light", np.random.default_rng(1))
    _, heavy_q = degrade_image(base, "scan_heavy", np.random.default_rng(1))
    assert light_q == 70 and heavy_q == 35
    assert degrade_image(base, "clean", np.random.default_rng(1))[1] is None
    photo, _ = degrade_image(base, "photo", np.random.default_rng(2))
    arr = np.asarray(photo.convert("L"), dtype=np.float32)
    left, right = arr[:, : arr.shape[1] // 3].mean(), arr[:, -arr.shape[1] // 3 :].mean()
    assert abs(left - right) > 2.0, "a photo has uneven lighting"
    with pytest.raises(ValueError, match="unknown quality"):
        degrade_image(base, "sepia", np.random.default_rng(1))


def test_main_prints_one_line_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--out", str(tmp_path / "ds"), "--seed", "3", "--limit", "6"])
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert re.search(r"\b6 files\b", out[0]) and "pages" in out[0] and "pdf=" in out[0]
    assert (tmp_path / "ds" / "labels.json").is_file()
