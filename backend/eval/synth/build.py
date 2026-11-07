"""Build the labeled evaluation dataset: composition plan -> files -> ``labels.json``.

``python -m eval.synth.build --out eval/dataset --seed 42 [--scale] [--limit N]``

Two phases keep the labels honest. ``make_plan`` is pure data (fast, no rendering): it lays out every file, creates the
content of every document, schedules the intentional anomalies (arithmetic errors, missing fields, credit notes), the
duplicate pairs and the cross-verification set. ``build_dataset`` then renders each planned file; the labels are
derived from the very same specs, plus the page ranges the renderer produced. Every file is generated from its own
``random.Random`` keyed by ``seed`` and the file's name, so a file does not depend on which other files are built
(``--limit`` yields an identical prefix) and the same seed reproduces the dataset byte for byte.

Composition (default run, 164 files):

====================  =====  ==========================================================================
digital PDFs             42  7 per template; 20 of them (T1-T4, T6) are the cross-verification set
scanned PDFs             24  4 per template: scan_light x2, scan_heavy, lowres
PNG / JPG / TIFF         36  per template 2 PNG (clean, lowres), 3 JPG (light, heavy, photo), 1 TIFF
invoice sheets           27  XLSX typed + text, XLS, CSV per template, three multi-invoice workbooks
ledgers                  11  one XLSX cross-verification ledger, 4 XLSX/XLS, 6 CSV (delimiters, encodings)
multi-invoice PDFs        4  10-25 invoices of 1-3 pages each (digital, mixed digital+scan, scanned)
duplicate pairs          10  5 pairs: digital PDF + JPG / scanned PDF, group ``dup-N``
corrupt / look-alikes    10  8 failing files + a PNG named .pdf + an owner-locked PDF (both must succeed)
====================  =====  ==========================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.domain.schemas import DocumentType, IssueCode
from eval.labels import LabeledFile, LabeledRecord, issue_codes_for, write_labels
from eval.synth import corrupt, spreadsheets
from eval.synth.data import InvoiceSpec, NumberRegistry, make_invoice, q2
from eval.synth.degrade import DPI, degrade_image, encode_jpeg
from eval.synth.render import (
    RenderedPdf,
    ScanPage,
    encrypt_pdf,
    image_pdf,
    mixed_pdf,
    page_images,
    png_bytes,
    render_pdf,
    tiff_bytes,
)
from eval.synth.spreadsheets import LedgerOptions, sheet_title
from eval.synth.templates import TEMPLATES

TEMPLATE_IDS = ("T1", "T2", "T3", "T4", "T5", "T6")
SPLIT = {"T1": "dev", "T2": "dev", "T3": "dev", "T4": "heldout", "T5": "heldout", "T6": "heldout"}
# Item counts that always fit one page (short descriptions) — measured per template, with a safety margin.
SINGLE_PAGE_ITEMS = {"T1": 12, "T2": 12, "T3": 9, "T4": 12, "T5": 12, "T6": 8}
MULTI_PAGE_ITEMS = 22  # with long descriptions every template needs at least two pages
SCAN_QUALITIES = ("scan_light", "scan_light", "scan_heavy", "lowres")
ERROR_COUNTS = {"TOTAL_MISMATCH": 16, "LINE_ITEMS_SUM_MISMATCH": 10, "LINE_ITEM_MATH": 6}
OUTPUT_DIRS = ("pdf", "images", "sheets", "ledgers", "multi", "groups", "corrupt", "scale")
SCALE_PAGES = 1000
SCANNED_SCALE_PAGES = 100


# ====================================================================================================================
# plan model
# ====================================================================================================================
@dataclass
class Slot:
    """One document to create. ``pool`` says which intentional anomalies it may receive."""

    key: str
    template: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    pool: str = "all"  # all | errors | none  (errors: no omitted fields — e.g. documents inside a multi-invoice PDF)
    error: str | None = None
    omit: frozenset[str] = frozenset()
    doc_type: DocumentType | None = None
    spec: InvoiceSpec | None = None


@dataclass
class Ledger:
    options: LedgerOptions
    sheets: list[tuple[str, list[LabeledRecord]]]
    notes_sheet: bool = False


@dataclass
class FilePlan:
    path: str
    kind: str  # label kind
    fmt: str  # renderer
    template: str | None
    quality: str = "clean"
    slots: list[Slot] = field(default_factory=list)
    expect: str = "success"
    error_code: str | None = None
    group: str | None = None
    extra_codes: dict[int, set[IssueCode]] = field(default_factory=dict)  # record index -> extra expected codes
    options: dict[str, Any] = field(default_factory=dict)
    ledger: Ledger | None = None
    min_pages: int = 1
    pages: int | None = None  # set by the renderer
    ranges: list[tuple[int, int]] | None = None  # set by the renderer

    @property
    def specs(self) -> list[InvoiceSpec]:
        return [s.spec for s in self.slots if s.spec is not None]

    def sheet_names(self) -> list[str | None]:
        if self.fmt in {"xlsx_invoice", "xls_invoice"}:
            titles: list[str] = []
            for i, spec in enumerate(self.specs):
                titles.append(sheet_title(spec.invoice_number, i, titles))
            return list(titles)
        return [None] * len(self.specs)

    def records(self) -> list[LabeledRecord]:
        if self.ledger is not None:
            return [r for _, rows in self.ledger.sheets for r in rows]
        sheets = self.sheet_names()
        out: list[LabeledRecord] = []
        for i, spec in enumerate(self.specs):
            update: dict[str, Any] = {}
            if self.fmt.endswith("_invoice"):
                update = {"sheet_name": sheets[i]}
            elif self.fmt in {"png", "jpg"}:
                update = {"page_start": 1, "page_end": 1}
            elif self.ranges is not None:
                update = {"page_start": self.ranges[i][0], "page_end": self.ranges[i][1]}
            out.append(spec.record().model_copy(update=update))
        return out

    def labeled(self) -> LabeledFile:
        """The ``labels.json`` entry (page ranges are only filled in once the file has been rendered)."""
        if self.expect == "failure":
            return LabeledFile(
                file=self.path,
                kind=self.kind,
                template=None,
                split="na",
                quality="clean",
                expect="failure",
                expected_error_code=self.error_code,
                group=self.group,
            )
        records = self.records()
        per_record: list[list[IssueCode]] = []
        for i, rec in enumerate(records):
            codes = set(issue_codes_for(rec, rec.source_kind)) | self.extra_codes.get(i, set())
            per_record.append(sorted(codes, key=lambda c: c.value))
        union = sorted({c for codes in per_record for c in codes}, key=lambda c: c.value)
        if self.ledger is not None:
            pages = len(self.ledger.sheets) + (1 if self.ledger.notes_sheet else 0)
            template, split = "ledger", "na"
        else:
            pages = self.pages if self.pages is not None else self._planned_pages()
            template = self.template
            split = SPLIT.get(template or "", "na")
        return LabeledFile(
            file=self.path,
            kind=self.kind,
            template=template,
            split=split,  # type: ignore[arg-type]
            quality=self.quality,  # type: ignore[arg-type]
            expect="success",
            records=records,
            expected_issue_codes=union,
            group=self.group,
            pages=pages,
            record_issues=per_record,
        )

    def _planned_pages(self) -> int:
        if self.fmt in {"csv_invoice"}:
            return 1
        if self.fmt in {"xlsx_invoice", "xls_invoice"}:
            return len(self.specs)
        return self.min_pages


# ====================================================================================================================
# planner
# ====================================================================================================================
class Planner:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.registry = NumberRegistry()

    def rng(self, *parts: object) -> random.Random:
        return random.Random("|".join(str(p) for p in (self.seed, *parts)))

    def instantiate(self, slot: Slot) -> InvoiceSpec:
        spec = make_invoice(
            self.rng("spec", slot.key),
            slot.template,
            slot.key,
            self.registry,
            error=slot.error,
            omit=slot.omit,
            doc_type=slot.doc_type,
            **slot.kwargs,
        )
        spec.style = TEMPLATES[slot.template].new_style(self.rng("style", slot.key))
        slot.spec = spec
        return spec


def _single(planner: Planner, key: str, tpl: str, *, pool: str = "all", **extra: Any) -> Slot:
    """A slot for a document that must fit on one page."""
    n = planner.rng("n", key).randint(1, SINGLE_PAGE_ITEMS[tpl])
    return Slot(key, tpl, {"n_items": n, "max_desc": 70, **extra}, pool)


def _long(tpl: str, key: str, n: int = MULTI_PAGE_ITEMS, *, pool: str = "all") -> Slot:
    """A slot for a document with long service descriptions: at least two pages, at most three."""
    return Slot(
        key,
        tpl,
        {"n_items": n, "industry": "consulting" if tpl != "T5" else None, "long_text": True, "max_desc": 400},
        pool,
    )


def _file(path: str, kind: str, fmt: str, tpl: str | None, slots: list[Slot], **kw: Any) -> FilePlan:
    return FilePlan(path, kind, fmt, tpl, slots=slots, **kw)


Bundle = list[FilePlan]


def compose(planner: Planner) -> dict[str, list[Bundle]]:
    """Every file of the default dataset, grouped by category (a bundle is files that must stay together)."""
    cats: dict[str, list[Bundle]] = {
        k: []
        for k in ("digital", "scanned", "png", "jpg", "tiff", "xlsx", "xls", "csv", "multi", "dup", "corrupt", "ledger")
    }
    for tpl in TEMPLATE_IDS:
        low = tpl.lower()
        for k in range(7):
            xv = tpl != "T5" and 2 <= k <= 5  # the cross-verification PDFs: clean, no discount or shipping
            slot = Slot(f"dig-{tpl}-{k}", tpl, {"discount_shipping": False} if xv else {}, "none" if xv else "all")
            cats["digital"].append(
                [
                    _file(
                        f"pdf/{low}_digital_{k + 1:02d}.pdf",
                        "pdf",
                        "pdf_digital",
                        tpl,
                        [slot],
                        group="xv" if xv else None,
                    )
                ]
            )
        counters: Counter[str] = Counter()
        for k, quality in enumerate(SCAN_QUALITIES):
            counters[quality] += 1
            n = planner.rng("scan-n", tpl, k).randint(1, 12)
            slot = Slot(f"scn-{tpl}-{k}", tpl, {"n_items": n})
            cats["scanned"].append(
                [
                    _file(
                        f"pdf/{low}_{quality}_{counters[quality]:02d}.pdf",
                        "pdf",
                        "pdf_scanned",
                        tpl,
                        [slot],
                        quality=quality,
                    )
                ]
            )
        cats["png"].append(
            [_file(f"images/{low}_clean_01.png", "png", "png", tpl, [_single(planner, f"png-{tpl}-0", tpl)])]
        )
        cats["png"].append(
            [
                _file(
                    f"images/{low}_lowres_01.png",
                    "png",
                    "png",
                    tpl,
                    [_single(planner, f"png-{tpl}-1", tpl)],
                    quality="lowres",
                )
            ]
        )
        for k, quality in enumerate(("scan_light", "scan_heavy", "photo")):
            exif = {"T1": 6, "T4": 8}.get(tpl) if quality == "photo" else None
            cats["jpg"].append(
                [
                    _file(
                        f"images/{low}_{quality}_01.jpg",
                        "jpg",
                        "jpg",
                        tpl,
                        [_single(planner, f"jpg-{tpl}-{k}", tpl)],
                        quality=quality,
                        options={"exif": exif} if exif else {},
                    )
                ]
            )
        if tpl in {"T1", "T4", "T6"}:
            tiff_slots, min_pages = [_long(tpl, f"tif-{tpl}-0")], 2
        elif tpl == "T2":
            tiff_slots, min_pages = [_long(tpl, "tif-T2-0"), _single(planner, "tif-T2-1", tpl)], 3
        else:
            tiff_slots, min_pages = [_single(planner, f"tif-{tpl}-0", tpl)], 1
        gray = tpl in {"T3", "T5"}  # grey scans stay lossless; colour scans are JPEG-in-TIFF
        cats["tiff"].append(
            [
                _file(
                    f"images/{low}_scan_01.tiff",
                    "tiff",
                    "tiff",
                    tpl,
                    tiff_slots,
                    quality="scan_light",
                    min_pages=min_pages,
                    options={"gray": gray, "compression": "tiff_lzw" if gray else "jpeg"},
                )
            ]
        )
        cats["xlsx"].append(
            [
                _file(
                    f"sheets/{low}_invoice_typed.xlsx",
                    "xlsx",
                    "xlsx_invoice",
                    tpl,
                    [Slot(f"xlt-{tpl}", tpl)],
                    options={"typed": True},
                )
            ]
        )
        cats["xlsx"].append(
            [
                _file(
                    f"sheets/{low}_invoice_text.xlsx",
                    "xlsx",
                    "xlsx_invoice",
                    tpl,
                    [Slot(f"xlx-{tpl}", tpl)],
                    options={"typed": False},
                )
            ]
        )
        cats["xls"].append([_file(f"sheets/{low}_invoice.xls", "xls", "xls_invoice", tpl, [Slot(f"xls-{tpl}", tpl)])])
        delimiter, encoding = {
            "T1": (",", "utf-8"),
            "T2": (";", "cp1252"),
            "T3": (",", "utf-8-sig"),
            "T4": ("\t", "utf-8"),
            "T5": (",", "utf-8"),
            "T6": (";", "utf-8"),
        }[tpl]
        cats["csv"].append(
            [
                _file(
                    f"sheets/{low}_invoice.csv",
                    "csv",
                    "csv_invoice",
                    tpl,
                    [Slot(f"csv-{tpl}", tpl)],
                    options={"delimiter": delimiter, "encoding": encoding},
                )
            ]
        )
    for tpl, count, typed in (("T1", 3, True), ("T3", 2, False), ("T6", 3, True)):
        slots = [Slot(f"wbk-{tpl}-{i}", tpl) for i in range(count)]
        cats["xlsx"].append(
            [
                _file(
                    f"sheets/{tpl.lower()}_workbook_{count}_invoices.xlsx",
                    "xlsx",
                    "xlsx_invoice",
                    tpl,
                    slots,
                    options={"typed": typed},
                )
            ]
        )
    cats["multi"] = _multi_invoice_files(planner)
    cats["dup"] = _duplicate_pairs(planner)
    cats["corrupt"] = _corrupt_files(planner)
    return cats


def _multi_invoice_files(planner: Planner) -> list[Bundle]:
    """Four PDFs holding 10-25 invoices of 1-3 pages each (batch exports from one billing system)."""
    plans = (
        ("T1", 12, "pdf_digital", "clean"),
        ("T3", 18, "pdf_digital", "clean"),
        ("T4", 10, "pdf_mixed", "scan_light"),
        ("T6", 14, "pdf_scanned", "scan_light"),
    )
    out: list[Bundle] = []
    for index, (tpl, count, fmt, quality) in enumerate(plans, start=1):
        rng = planner.rng("multi", tpl)
        slots: list[Slot] = []
        for i in range(count):
            key = f"mul-{tpl}-{i}"
            if rng.random() < 0.15:
                slots.append(_long(tpl, key, rng.randint(21, 25), pool="errors"))
            else:
                slots.append(
                    Slot(key, tpl, {"n_items": rng.choice((1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 25))}, "errors")
                )
        options = {"scan_invoices": {i for i in range(count) if i % 3 == 1}} if fmt == "pdf_mixed" else {}
        out.append(
            [
                _file(
                    f"multi/batch_{index:02d}_{tpl.lower()}.pdf",
                    "pdf",
                    fmt,
                    tpl,
                    slots,
                    quality=quality,
                    options=options,
                )
            ]
        )
    return out


def _duplicate_pairs(planner: Planner) -> list[Bundle]:
    """The same invoice as a digital PDF (first, the original) and as a scan or photo (second, the duplicate)."""
    plan = (
        ("T1", "jpg", "scan_light"),
        ("T2", "jpg", "photo"),
        ("T3", "pdf_scanned", "scan_heavy"),
        ("T4", "jpg", "scan_light"),
        ("T6", "jpg", "photo"),
    )
    out: list[Bundle] = []
    for n, (tpl, copy_fmt, quality) in enumerate(plan, start=1):
        slot = _single(planner, f"dup-{n}", tpl, pool="none")
        low = tpl.lower()
        original = _file(f"groups/dup_{n}_a_digital.pdf", "pdf", "pdf_digital", tpl, [slot], group=f"dup-{n}")
        ext = "jpg" if copy_fmt == "jpg" else "pdf"
        copy = _file(
            f"groups/dup_{n}_b_{low}_{quality}.{ext}",
            "jpg" if copy_fmt == "jpg" else "pdf",
            copy_fmt,
            tpl,
            [slot],
            quality=quality,
            group=f"dup-{n}",
            extra_codes={0: {IssueCode.DUPLICATE_INVOICE}},
        )
        out.append([original, copy])
    return out


def _corrupt_files(planner: Planner) -> list[Bundle]:
    out: list[Bundle] = []
    for case in corrupt.CASES:
        slots: list[Slot] = []
        tpl: str | None = None
        if case.expect == "success":
            tpl = "T6" if case.kind == "png" else "T1"
            slots = [_single(planner, f"cor-{Path(case.path).stem}", tpl, pool="none")]
        out.append(
            [
                FilePlan(
                    case.path,
                    case.kind,
                    f"corrupt:{Path(case.path).stem}",
                    tpl,
                    slots=slots,
                    expect=case.expect,
                    error_code=case.error_code,
                )
            ]
        )
    return out


# --------------------------------------------------------------------------------------------------------------------
# anomalies
# --------------------------------------------------------------------------------------------------------------------
def assign_anomalies(planner: Planner, slots: list[Slot]) -> None:
    """Schedule the intentional problems over the eligible slots (exact counts, reproducible per seed)."""
    rng = planner.rng("anomalies")
    pool = [s for s in slots if s.pool != "none"]
    rng.shuffle(pool)
    taken: set[str] = set()

    def pick(count: int, ok: Any = lambda s: True) -> list[Slot]:
        chosen = [s for s in pool if s.key not in taken and ok(s)][:count]
        taken.update(s.key for s in chosen)
        return chosen

    not_receipt = lambda s: s.template != "T5"  # noqa: E731
    every = lambda s: s.pool == "all"  # noqa: E731
    for field_name, count in (
        ("invoice_number", 3),
        ("invoice_date", 3),
        ("supplier_name", 3),
        ("total", 3),
        ("subtotal", 3),
        ("line_items", 3),
    ):
        for slot in pick(count, lambda s: not_receipt(s) and every(s)):
            slot.omit = frozenset({field_name})
    receipts = [s for s in pool if s.template == "T5" and every(s)]
    for slot in pick(max(2, round(0.3 * len(receipts))), lambda s: s.template == "T5" and every(s)):
        slot.omit = frozenset({"invoice_number"})
    for field_name, count in (("tax", 4), ("currency", 5)):
        for slot in pick(count, every):
            slot.omit = frozenset({field_name})
    for code, count in ERROR_COUNTS.items():
        for slot in pick(count):
            slot.error = code
    for tpl in ("T2", "T3"):
        for slot in pick(2, lambda s, tpl=tpl: s.template == tpl and every(s)):
            slot.doc_type = DocumentType.CREDIT_NOTE
    for slot in pick(19, not_receipt):
        slot.omit = slot.omit | {"customer_name"}


# ====================================================================================================================
# ledgers and the cross-verification set
# ====================================================================================================================
def _ledger_rows(
    planner: Planner, key: str, tpl: str, count: int, layout: str, *, currency: str | None
) -> list[LabeledRecord]:
    kwargs: dict[str, Any] = {"force_currency": currency} if currency else {}
    if layout != "D":
        kwargs["discount_shipping"] = False
    rows: list[LabeledRecord] = []
    for i in range(count):
        slot = Slot(f"{key}-{i}", tpl, kwargs, "none")
        rows.append(planner.instantiate(slot).record())
    rows.sort(key=lambda r: (r.invoice_date or r.due_date, r.invoice_number or ""))
    return rows


def _located(rows: Sequence[LabeledRecord], layout: str, sheet: str | None, first_row: int) -> list[LabeledRecord]:
    return [
        spreadsheets.ledger_record(r, layout, sheet_name=sheet, row_number=first_row + i) for i, r in enumerate(rows)
    ]


def _ledger_files(planner: Planner) -> list[Bundle]:
    """Ledger workbooks and CSVs with their own (non-PDF) invoices."""
    out: list[Bundle] = []

    def add(
        path: str, kind: str, fmt: str, sheets: list[tuple[str, list[LabeledRecord]]], options: LedgerOptions, **kw: Any
    ) -> None:
        out.append(
            [FilePlan(path, kind, fmt, None, ledger=Ledger(options, sheets, kw.pop("notes_sheet", False)), **kw)]
        )

    # US quarterly workbook: layout D, three quarters + a notes sheet, a total row under each quarter
    opts = LedgerOptions("D", True, 4, "Sales ledger 2024 - US customers", total_row=True)
    sheets = []
    for q, count in (("Q1", 8), ("Q2", 9), ("Q3", 7)):
        rows = _ledger_rows(planner, f"led-us-{q}", "T1", count, "D", currency="USD")
        sheets.append((q, _located(rows, "D", q, opts.header_row + 1)))
    add("ledgers/ledger_us_quarterly.xlsx", "xlsx", "ledger_xlsx", sheets, opts, notes_sheet=True)
    # EU text export: layout B, amounts as "1.234,56", dates as "05.03.2024"
    opts = LedgerOptions("B", False, 1, money_fmt="eu", date_fmt="eu_dot")
    rows = _ledger_rows(planner, "led-eu-text", "T2", 18, "B", currency="EUR")
    add("ledgers/ledger_eu_text.xlsx", "xlsx", "ledger_xlsx", [("Sales EU", _located(rows, "B", "Sales EU", 2))], opts)
    # UK half-year workbook as legacy .xls
    opts = LedgerOptions("A", True, 3, "UK sales ledger", total_row=True)
    sheets = []
    for name, count in (("Jan-Jun", 10), ("Jul-Dec", 9)):
        rows = _ledger_rows(planner, f"led-uk-{name}", "T3", count, "A", currency="GBP")
        sheets.append((name, _located(rows, "A", name, opts.header_row + 1)))
    add("ledgers/ledger_uk_halves.xls", "xls", "ledger_xls", sheets, opts)
    # AU minimal ledger (no subtotal / tax columns) as .xls
    opts = LedgerOptions("C", True, 3, "Debtors - Australia")
    rows = _ledger_rows(planner, "led-au", "T6", 14, "C", currency="AUD")
    add("ledgers/ledger_au_minimal.xls", "xls", "ledger_xls", [("Debtors", _located(rows, "C", "Debtors", 4))], opts)
    # CSV ledgers: delimiters, encodings, decimal commas, quoting, a title above the header, a total row
    csv_cases = (
        (
            "ledger_de_semicolon_cp1252.csv",
            "T2",
            "EUR",
            16,
            LedgerOptions("B", False, 1, money_fmt="plain_comma", date_fmt="eu_dot", delimiter=";", encoding="cp1252"),
        ),
        ("ledger_us_comma_utf8.csv", "T1", "USD", 14, LedgerOptions("A", False, 1, money_fmt="plain", date_fmt="iso")),
        (
            "ledger_us_tab_bom.csv",
            "T1",
            "USD",
            12,
            LedgerOptions(
                "D", False, 1, money_fmt="sym_us", date_fmt="us", delimiter="\t", encoding="utf-8-sig", newline="\r\n"
            ),
        ),
        (
            "ledger_uk_pipe_utf8.csv",
            "T3",
            "GBP",
            13,
            LedgerOptions("A", False, 1, money_fmt="plain", date_fmt="uk_long", delimiter="|"),
        ),
        (
            "ledger_eu_semicolon_utf8_title.csv",
            "T2",
            "EUR",
            15,
            LedgerOptions(
                "A",
                False,
                3,
                "Ledger export",
                total_row=True,
                money_fmt="eu",
                date_fmt="eu_slash",
                delimiter=";",
                quote_all=True,
            ),
        ),
        (
            "ledger_uk_comma_cp1252.csv",
            "T3",
            "GBP",
            11,
            LedgerOptions("B", False, 1, money_fmt="sym_gbp", date_fmt="eu_slash", encoding="cp1252", newline="\r\n"),
        ),
    )
    for name, tpl, cur, count, options in csv_cases:
        rows = _ledger_rows(planner, f"led-{Path(name).stem}", tpl, count, options.layout, currency=cur)
        located = _located(rows, options.layout, None, options.header_row + 1)
        add(f"ledgers/{name}", "csv", "ledger_csv", [("", located)], options)
    return out


def _shift_dates(rec: LabeledRecord, days: int) -> LabeledRecord:
    """The record with its invoice and due date moved by ``days`` (later if that stays in range, else earlier)."""
    assert rec.invoice_date is not None
    latest = date(2025, 12, 31)
    shift = timedelta(days=days if (rec.due_date or rec.invoice_date) + timedelta(days=days) <= latest else -days)
    return rec.model_copy(
        update={"invoice_date": rec.invoice_date + shift, "due_date": rec.due_date + shift if rec.due_date else None}
    )


def _cross_verification(planner: Planner, pdfs: list[FilePlan]) -> FilePlan:
    """A two-sheet XLSX ledger describing the 20 ``xv`` PDFs, three rows deliberately disagreeing with their PDF."""
    rng = planner.rng("xv")
    records = [p.specs[0].record() for p in pdfs]
    mismatches = dict(zip(rng.sample(range(len(records)), 3), rng.sample(("total", "total", "date"), 3), strict=True))
    rows: list[LabeledRecord] = []
    flagged: set[str] = set()
    for i, rec in enumerate(records):
        row = rec
        kind = mismatches.get(i)
        if kind == "total":
            subtotal = q2((rec.subtotal or Decimal(0)) + Decimal(rng.choice((15, 40, 75, 120, 250))) + Decimal("0.35"))
            row = rec.model_copy(update={"subtotal": subtotal, "total": q2(subtotal + (rec.tax or Decimal(0)))})
        elif kind == "date" and rec.invoice_date is not None:
            row = _shift_dates(rec, rng.randint(3, 40))
        if kind:
            flagged.add(rec.invoice_number or "")
        rows.append(row)
    domestic = [r for r in rows if r.currency in {"USD", "CAD"}]
    international = [r for r in rows if r.currency not in {"USD", "CAD"}]
    opts = LedgerOptions("A", True, 3, "Receivables reconciliation")
    sheets = []
    for name, group in (("Domestic", domestic), ("International", international)):
        group.sort(key=lambda r: (r.invoice_date, r.invoice_number))
        sheets.append((name, _located(group, "A", name, opts.header_row + 1)))
    ledger = FilePlan("ledgers/xv_ledger.xlsx", "xlsx", "ledger_xlsx", None, ledger=Ledger(opts, sheets), group="xv")
    flat = ledger.records()
    for i, row in enumerate(flat):
        if row.invoice_number in flagged:
            ledger.extra_codes[i] = {IssueCode.CROSS_VERIFICATION_MISMATCH}
    for pdf in pdfs:
        if pdf.specs[0].invoice_number in flagged:
            pdf.extra_codes[0] = {IssueCode.CROSS_VERIFICATION_MISMATCH}
    return ledger


# ====================================================================================================================
# plan assembly
# ====================================================================================================================
# Cheap, diverse files first so that any --limit prefix exercises every kind (category, index within category).
FRONT: tuple[tuple[str, int], ...] = (
    *(("digital", 7 * i) for i in range(6)),  # the first digital PDF of every template
    *(("corrupt", i) for i in range(len(corrupt.CASES))),
    ("scanned", 0),  # T1 scan_light PDF
    ("png", 2),  # T2 clean PNG
    ("jpg", 8),  # T3 photo JPG
    ("tiff", 3),  # T4 multi-frame TIFF
    ("xlsx", 8),  # T5 typed XLSX
    ("xls", 5),  # T6 XLS
    ("csv", 0),  # T1 CSV
    ("dup", 0),  # a duplicate pair
    ("ledger", 4),  # CSV ledger, semicolons, cp1252
    ("ledger", 1),  # text XLSX ledger
)


def _round_robin(cats: dict[str, list[Bundle]], used: set[int]) -> Iterator[Bundle]:
    queues = [[b for b in bundles if id(b) not in used] for bundles in cats.values()]
    while any(queues):
        for q in queues:
            if q:
                yield q.pop(0)


def _order(cats: dict[str, list[Bundle]]) -> list[Bundle]:
    """Cheap, diverse files first (so any ``--limit`` prefix exercises every kind), then round-robin over the rest."""
    first = [cats[name][index] for name, index in FRONT]
    used = {id(b) for b in first}
    return [*first, *_round_robin(cats, used)]


def make_plan(seed: int, *, scale: bool = False, limit: int | None = None) -> list[FilePlan]:
    """The full ordered list of files to build (nothing is rendered). Files are listed in ingestion order."""
    planner = Planner(seed)
    cats = compose(planner)
    ledgers = _ledger_files(planner)
    all_slots = [s for bundles in cats.values() for bundle in bundles for fp in bundle for s in fp.slots]
    unique: dict[str, Slot] = {}
    for slot in all_slots:
        unique.setdefault(slot.key, slot)
    assign_anomalies(planner, list(unique.values()))
    for slot in unique.values():
        if slot.spec is None:
            planner.instantiate(slot)
    xv_pdfs = [fp for bundle in cats["digital"] for fp in bundle if fp.group == "xv"]
    cats["ledger"] = [*ledgers, [_cross_verification(planner, xv_pdfs)]]
    ordered = [fp for bundle in _order(cats) for fp in bundle]
    if limit is not None:
        ordered = _apply_limit(ordered, limit)
    if scale:
        ordered += _scale_plans(planner)
    return ordered


def _apply_limit(plan: list[FilePlan], limit: int) -> list[FilePlan]:
    """The first ``limit`` files, minus any group that the cut would have split."""
    kept = plan[:limit]
    full = Counter(p.group for p in plan if p.group)
    have = Counter(p.group for p in kept if p.group)
    return [p for p in kept if p.group is None or have[p.group] == full[p.group]]


def _scale_plans(planner: Planner) -> list[FilePlan]:
    """The throughput files: a ~1000-page digital PDF and a 100-page scanned PDF (documents are chosen at render time
    so that the page counts are exact)."""
    rng = planner.rng("scale")
    cycle = ("T1", "T2", "T3", "T4", "T5", "T6")
    mix = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 25)  # about a third of the invoices run to a second page
    large = [Slot(f"big-{i}", cycle[i % 6], {"n_items": rng.choice(mix)}, "none") for i in range(1300)]
    small = [Slot(f"scn-100-{i}", cycle[i % 6], {"n_items": rng.choice(mix[:-1])}, "none") for i in range(160)]
    plans = [
        FilePlan(
            "scale/large_1000_pages.pdf",
            "pdf",
            "pdf_digital",
            "mixed",
            slots=large,
            min_pages=SCALE_PAGES,
            options={"max_pages": SCALE_PAGES},
        ),
        FilePlan(
            "scale/scanned_100_pages.pdf",
            "pdf",
            "pdf_scanned",
            "mixed",
            slots=small,
            quality="scan_light",
            min_pages=SCANNED_SCALE_PAGES,
            options={"max_pages": SCANNED_SCALE_PAGES},
        ),
    ]
    for plan in plans:
        for slot in plan.slots:
            planner.instantiate(slot)
    return plans


# ====================================================================================================================
# rendering
# ====================================================================================================================
def _np_rng(seed: int, *parts: object) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in (seed, *parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _page_dpi(plan: FilePlan, page_templates: Sequence[str], index: int) -> int:
    tpl = TEMPLATES[page_templates[index]]
    return round(DPI[plan.quality] * tpl.dpi_factor)


def _page_templates(rendered: RenderedPdf, specs: Sequence[InvoiceSpec]) -> list[str]:
    out: list[str] = []
    for (start, end), spec_index in zip(rendered.ranges, rendered.included, strict=True):
        out += [specs[spec_index].template] * (end - start + 1)
    return out


def _degraded_pages(plan: FilePlan, rendered: RenderedPdf, seed: int) -> Iterator[tuple[Image.Image, int | None, int]]:
    """(pixels, jpeg quality or None, dpi) for every page of a rendered PDF."""
    templates = _page_templates(rendered, plan.specs)
    for index in range(len(templates)):
        dpi = _page_dpi(plan, templates, index)
        image = next(page_images(rendered.data, dpi, [index]))
        out, jpeg_q = degrade_image(image, plan.quality, _np_rng(seed, plan.path, index), dpi=dpi)
        yield out, jpeg_q, dpi


def _jpeg_roundtrip(image: Image.Image, quality: int, dpi: int) -> Image.Image:
    return Image.open(io.BytesIO(encode_jpeg(image, quality, dpi=dpi))).convert("RGB")


def _orient(image: Image.Image, exif: int | None) -> tuple[Image.Image, Any]:
    """Store the pixels rotated and say so in EXIF, like a phone camera does (viewers rotate them back)."""
    if not exif:
        return image, None
    stored = image.transpose(Image.Transpose.ROTATE_90 if exif == 6 else Image.Transpose.ROTATE_270)
    tags = Image.Exif()
    tags[0x0112] = exif
    return stored, tags


def _write_image(plan: FilePlan, path: Path, rendered: RenderedPdf, seed: int) -> int:
    pages = list(_degraded_pages(plan, rendered, seed))
    if plan.fmt == "tiff":
        codec, gray, dpi = plan.options.get("compression", "tiff_lzw"), bool(plan.options.get("gray")), pages[0][2]
        if codec == "jpeg":
            frames = [img for img, _, _ in pages]
            path.write_bytes(tiff_bytes(frames, dpi=dpi, gray=gray, compression="jpeg", quality=pages[0][1] or 75))
        else:  # lossless container: bake the JPEG artefacts of the scan into the pixels first
            frames = [_jpeg_roundtrip(img, q, d) if q else img for img, q, d in pages]
            path.write_bytes(tiff_bytes(frames, dpi=dpi, gray=gray))
        return len(frames)
    if len(pages) != 1:
        raise ValueError(f"{plan.path}: a single-image file rendered {len(pages)} pages")
    image, jpeg_q, dpi = pages[0]
    if plan.fmt == "png":
        path.write_bytes(png_bytes(image, dpi=dpi))
    else:
        stored, exif = _orient(image, plan.options.get("exif"))
        buffer = io.BytesIO()
        kwargs: dict[str, Any] = {"exif": exif} if exif is not None else {}
        stored.save(buffer, format="JPEG", quality=jpeg_q or 90, dpi=(dpi, dpi), **kwargs)
        path.write_bytes(buffer.getvalue())
    return 1


def _scan_pages(plan: FilePlan, rendered: RenderedPdf, seed: int, only: set[int] | None = None) -> dict[int, ScanPage]:
    out: dict[int, ScanPage] = {}
    for index, (image, jpeg_q, dpi) in enumerate(_degraded_pages(plan, rendered, seed)):
        if only is not None and index not in only:
            continue
        data = encode_jpeg(image, jpeg_q or 85, dpi=dpi)
        out[index] = ScanPage(data, image.width * 72 / dpi, image.height * 72 / dpi)
    return out


def _render_pdf_family(plan: FilePlan, path: Path, seed: int) -> None:
    rendered = render_pdf(plan.specs, max_pages=plan.options.get("max_pages"))
    if len(rendered.included) != len(plan.slots):  # an exact page budget keeps only the documents that fit
        plan.slots = [plan.slots[i] for i in rendered.included]
        rendered = RenderedPdf(rendered.data, rendered.ranges, list(range(len(plan.slots))))
    if plan.fmt == "pdf_digital":
        path.write_bytes(rendered.data)
    elif plan.fmt == "pdf_scanned":
        scans = _scan_pages(plan, rendered, seed)
        path.write_bytes(image_pdf([scans[i] for i in sorted(scans)]))
    else:  # pdf_mixed: whole invoices replaced by scans
        only: set[int] = set()
        for i in plan.options["scan_invoices"]:
            start, end = rendered.ranges[i]
            only.update(range(start - 1, end))
        path.write_bytes(mixed_pdf(rendered.data, _scan_pages(plan, rendered, seed, only)))
    budget = plan.options.get("max_pages")
    if budget is not None and rendered.pages != budget:
        raise ValueError(f"{plan.path}: planned documents give {rendered.pages} pages, not {budget}")
    plan.pages = rendered.pages
    plan.ranges = rendered.ranges


def _render_sheet(plan: FilePlan, path: Path) -> None:
    specs = plan.specs
    if plan.fmt == "xlsx_invoice":
        spreadsheets.write_invoice_xlsx(path, specs, [n or "" for n in plan.sheet_names()], typed=plan.options["typed"])
        plan.pages = len(specs)
    elif plan.fmt == "xls_invoice":
        spreadsheets.write_invoice_xls(path, specs, [n or "" for n in plan.sheet_names()])
        plan.pages = len(specs)
    else:
        spreadsheets.write_invoice_csv(
            path, specs[0], delimiter=plan.options["delimiter"], encoding=plan.options["encoding"]
        )
        plan.pages = 1


def _render_ledger(plan: FilePlan, path: Path) -> None:
    ledger = plan.ledger
    assert ledger is not None
    sheets = [(name, rows) for name, rows in ledger.sheets]
    if plan.fmt == "ledger_xlsx":
        extra = [("Notes", spreadsheets.notes_sheet())] if ledger.notes_sheet else []
        spreadsheets.write_ledger_xlsx(path, sheets, ledger.options, extra_sheets=extra)
    elif plan.fmt == "ledger_xls":
        spreadsheets.write_ledger_xls(path, sheets, ledger.options)
    else:
        spreadsheets.write_ledger_csv(path, sheets[0][1], ledger.options)
    plan.pages = len(sheets) + (1 if ledger.notes_sheet else 0)


def _render_corrupt(plan: FilePlan, path: Path, seed: int) -> None:
    stem = path.stem
    rng = random.Random(f"{seed}|corrupt|{path.name}")
    if plan.expect == "success":
        rendered = render_pdf(plan.specs)
        if stem == "scan_saved_as":
            path.write_bytes(png_bytes(next(page_images(rendered.data, 200)), dpi=200))
            plan.pages, plan.ranges = 1, [(1, 1)]
        else:
            path.write_bytes(encrypt_pdf(rendered.data, user_pw="", owner_pw=f"owner-{rng.randint(1000, 9999)}"))
            plan.pages, plan.ranges = rendered.pages, rendered.ranges
        return
    base = _corrupt_base(seed)
    makers = {
        ("truncated", ".pdf"): lambda: corrupt.truncate(base["pdf"], 0.55),
        ("truncated", ".xlsx"): lambda: corrupt.truncate(base["xlsx"], 0.6),
        ("truncated", ".png"): lambda: corrupt.truncate(base["png"], 0.45),
        ("random_bytes", ".pdf"): lambda: corrupt.random_noise(rng),
        ("empty", ".pdf"): lambda: b"",
        ("encrypted", ".pdf"): lambda: encrypt_pdf(
            base["pdf"], user_pw=f"s3cret-{rng.randint(100, 999)}", owner_pw="owner"
        ),
        ("fake_invoice", ".pdf"): lambda: corrupt.fake_executable(rng),
        ("session_expired", ".xlsx"): lambda: corrupt.html_error_page(rng),
    }
    path.write_bytes(makers[(stem, path.suffix)]())
    plan.pages = 0


_BASE_CACHE: dict[int, dict[str, bytes]] = {}


def _corrupt_base(seed: int) -> dict[str, bytes]:
    """A small valid PDF, XLSX and PNG to damage."""
    if seed not in _BASE_CACHE:
        spec = Planner(seed).instantiate(Slot("corrupt-base", "T1", {"n_items": 5, "max_desc": 60}, "none"))
        rendered = render_pdf([spec])
        _BASE_CACHE[seed] = {
            "pdf": rendered.data,
            "xlsx": spreadsheets.invoice_xlsx_bytes([spec], ["Invoice"], typed=True),
            "png": png_bytes(next(page_images(rendered.data, 120)), dpi=120),
        }
    return _BASE_CACHE[seed]


def render_plan(plan: FilePlan, root: Path, seed: int) -> None:
    """Write one planned file under ``root`` and record the page information the labels need."""
    path = root / plan.path
    path.parent.mkdir(parents=True, exist_ok=True)
    if plan.fmt.startswith("corrupt:"):
        _render_corrupt(plan, path, seed)
    elif plan.fmt in {"pdf_digital", "pdf_scanned", "pdf_mixed"}:
        _render_pdf_family(plan, path, seed)
    elif plan.fmt in {"png", "jpg", "tiff"}:
        rendered = render_pdf(plan.specs)
        frames = _write_image(plan, path, rendered, seed)
        if plan.fmt == "tiff" and frames < plan.min_pages:
            raise ValueError(f"{plan.path}: expected at least {plan.min_pages} frames, got {frames}")
        plan.pages, plan.ranges = frames, rendered.ranges
    elif plan.fmt.endswith("_invoice"):
        _render_sheet(plan, path)
    else:
        _render_ledger(plan, path)


# ====================================================================================================================
# driver and CLI
# ====================================================================================================================
@dataclass(frozen=True)
class BuildSummary:
    files: int
    by_kind: dict[str, int]
    pages: int
    seconds: float
    labels_path: Path

    def line(self) -> str:
        kinds = " ".join(f"{k}={v}" for k, v in sorted(self.by_kind.items()))
        return f"built {self.files} files ({kinds}), {self.pages} pages in {self.seconds:.1f}s -> {self.labels_path}"


def build_dataset(
    out_dir: Path | str, *, seed: int = 42, scale: bool = False, limit: int | None = None
) -> BuildSummary:
    """Render the dataset into ``out_dir`` and write ``labels.json``."""
    started = time.perf_counter()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_DIRS:  # only what this tool writes; never anything else
        shutil.rmtree(root / name, ignore_errors=True)
    (root / "labels.json").unlink(missing_ok=True)
    plan = make_plan(seed, scale=scale, limit=limit)
    for file_plan in plan:
        render_plan(file_plan, root, seed)
    files = [p.labeled() for p in plan]
    labels_path = root / "labels.json"
    write_labels(labels_path, seed, files)
    return BuildSummary(
        len(files),
        dict(Counter(f.kind for f in files)),
        sum(f.pages for f in files),
        time.perf_counter() - started,
        labels_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.synth.build", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="output directory (e.g. eval/dataset)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", action="store_true", help="also write the ~1000-page and 100-page throughput PDFs")
    parser.add_argument("--limit", type=int, default=None, help="build only the first N files of the plan")
    args = parser.parse_args(argv)
    summary = build_dataset(args.out, seed=args.seed, scale=args.scale, limit=args.limit)
    sys.stdout.write(summary.line() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
