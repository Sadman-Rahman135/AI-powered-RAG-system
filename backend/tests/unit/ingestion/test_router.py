"""The router dispatches a stored file to the parser for its kind."""

from __future__ import annotations

import shutil
from pathlib import Path

import openpyxl
import pytest
import xlwt
from PIL import Image

from app.ingestion.router import parse_document
from app.ingestion.types import EmptyFileError, FileKind, ProgressCallback, UnsupportedFileError

from .conftest import INVOICE_NUMBER, invoice_items, make_pdf


def _files(tmp_path: Path, invoice_image: Image.Image) -> dict[FileKind, Path]:
    pdf = make_pdf(tmp_path / "invoice.pdf", [invoice_items()])
    png = tmp_path / "invoice.png"
    invoice_image.save(png)
    xlsx = tmp_path / "ledger.xlsx"
    wb = openpyxl.Workbook()
    wb.active.append(["Invoice No", "Total"])
    wb.active.append([INVOICE_NUMBER, 22.0])
    wb.save(xlsx)
    xls = tmp_path / "ledger.xls"
    book = xlwt.Workbook()
    sheet = book.add_sheet("Ledger")
    sheet.write(0, 0, "Invoice No")
    sheet.write(1, 0, INVOICE_NUMBER)
    book.save(str(xls))
    csv = tmp_path / "ledger.csv"
    csv.write_text(f"Invoice No,Total\n{INVOICE_NUMBER},22.00\n", encoding="utf-8")
    return {FileKind.PDF: pdf, FileKind.IMAGE: png, FileKind.XLSX: xlsx, FileKind.XLS: xls, FileKind.CSV: csv}


def _recorder(calls: list[tuple[int, int, str]]) -> ProgressCallback:
    return lambda current, total, stage: calls.append((current, total, stage))


def test_parse_document_dispatches_by_kind(tmp_path: Path, invoice_image: Image.Image) -> None:
    for kind, path in _files(tmp_path, invoice_image).items():
        calls: list[tuple[int, int, str]] = []
        doc = parse_document(path, kind, on_progress=_recorder(calls))
        assert doc.kind is kind
        assert INVOICE_NUMBER in doc.text, kind
        assert bool(doc.pages) == (kind in (FileKind.PDF, FileKind.IMAGE))
        assert bool(doc.sheets) == (kind in (FileKind.XLSX, FileKind.XLS, FileKind.CSV))
        assert calls and calls[-1][0] == calls[-1][1] == doc.page_count, kind


def test_parse_document_does_not_depend_on_the_file_name(tmp_path: Path, invoice_image: Image.Image) -> None:
    for kind, path in _files(tmp_path, invoice_image).items():
        stored = Path(shutil.copy(path, tmp_path / f"stored-{kind.value}"))  # storage keys carry no extension
        assert INVOICE_NUMBER in parse_document(stored, kind).text, kind


def test_parse_document_accepts_kind_values_and_rejects_unknown(tmp_path: Path) -> None:
    csv = tmp_path / "ledger.csv"
    csv.write_text("Invoice No,Total\nINV-1,10.00\n", encoding="utf-8")
    assert parse_document(csv, "csv").kind is FileKind.CSV  # type: ignore[arg-type]  # as stored in the DB
    with pytest.raises(UnsupportedFileError):
        parse_document(csv, "docx")  # type: ignore[arg-type]


def test_parse_document_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(EmptyFileError):
        parse_document(empty, FileKind.PDF)
