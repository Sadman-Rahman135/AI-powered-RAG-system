"""File-kind detection: magic bytes decide, the filename extension is only a hint."""

from __future__ import annotations

import io
import zipfile

import openpyxl
import pytest
import xlwt
from PIL import Image

from app.ingestion.sniff import (
    HEAD_BYTES,
    MIME_TYPES,
    detect_file_kind,
    detect_mime_type,
    extension_mismatch,
)
from app.ingestion.types import EmptyFileError, FileKind, UnsupportedFileError


def _image_bytes(fmt: str, **save_args: object) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format=fmt, **save_args)
    return buf.getvalue()


def _xlsx_bytes() -> bytes:
    wb = openpyxl.Workbook()
    wb.active.append(["Invoice No", "Total"])
    wb.active.append(["INV-1", 10.5])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _xls_bytes() -> bytes:
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Ledger")
    ws.write(0, 0, "Invoice No")
    ws.write(1, 0, "INV-1")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _zip_bytes(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, "<xml/>" * 50)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("head", "filename", "expected"),
    [
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n", "invoice.pdf", FileKind.PDF),
        (_image_bytes("PNG"), "scan.png", FileKind.IMAGE),
        (_image_bytes("JPEG"), "photo.jpg", FileKind.IMAGE),
        (_image_bytes("TIFF"), "fax.tif", FileKind.IMAGE),  # Pillow writes little-endian "II*\0"
        (b"MM\x00*\x00\x00\x00\x08" + b"\x00" * 32, "fax.tiff", FileKind.IMAGE),
        (_xlsx_bytes(), "book.xlsx", FileKind.XLSX),
        (_xls_bytes(), "book.xls", FileKind.XLS),
        (b"Invoice No,Date,Total\nINV-1,2024-03-05,110.00\n", "ledger.csv", FileKind.CSV),
        ("Rechnung;Betrag\nRE-1;1.234,56 €\n".encode("cp1252"), "export.csv", FileKind.CSV),
        ("Invoice\tTotal\nINV-1\t10.00\n".encode("utf-16"), "unicode.txt", FileKind.CSV),
    ],
    ids=["pdf", "png", "jpeg", "tiff-le", "tiff-be", "xlsx", "xls", "csv", "csv-cp1252", "tsv-utf16"],
)
def test_detects_each_kind_by_magic(head: bytes, filename: str, expected: FileKind) -> None:
    assert detect_file_kind(head[:HEAD_BYTES], filename) is expected
    assert not extension_mismatch(expected, filename)


def test_extension_is_only_a_hint() -> None:
    png = _image_bytes("PNG")
    assert detect_file_kind(png, "invoice.pdf") is FileKind.IMAGE
    assert extension_mismatch(FileKind.IMAGE, "invoice.pdf")
    assert not extension_mismatch(FileKind.IMAGE, "SCAN.JPEG")
    assert not extension_mismatch(FileKind.PDF, "no_extension")
    assert detect_file_kind(b"%PDF-1.4\n", "report.xlsx") is FileKind.PDF
    assert extension_mismatch(FileKind.PDF, "report.xlsx")


def test_rejects_exe_zip_html_empty() -> None:
    exe = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00" + b"\x00" * 64 + b"This program cannot be run"
    with pytest.raises(UnsupportedFileError) as err:
        detect_file_kind(exe, "invoice.pdf")
    assert err.value.code == "unsupported_file_type"
    with pytest.raises(UnsupportedFileError):
        detect_file_kind(_zip_bytes(["readme.txt", "data/payload.bin"]), "archive.xlsx")
    with pytest.raises(UnsupportedFileError):
        detect_file_kind(_zip_bytes(["[Content_Types].xml", "word/document.xml"]), "letter.xlsx")
    html = b"<!DOCTYPE html>\n<html><head><title>Invoice</title></head><body>a,b,c</body></html>"
    with pytest.raises(UnsupportedFileError):
        detect_file_kind(html, "invoice.csv")
    with pytest.raises(UnsupportedFileError):
        detect_file_kind(b"GIF89a\x01\x00\x01\x00\x80\x00\x00", "anim.gif")
    with pytest.raises(EmptyFileError) as empty:
        detect_file_kind(b"", "empty.pdf")
    assert empty.value.code == "empty_file"


def test_plain_text_needs_delimiters_or_a_csv_extension() -> None:
    assert detect_file_kind(b"Invoice\nINV-1\nINV-2\n", "single_column.csv") is FileKind.CSV
    with pytest.raises(UnsupportedFileError):
        detect_file_kind(b"Just a note without any separators\n", "notes.md")


def test_mime_types_cover_every_kind() -> None:
    assert set(MIME_TYPES) == set(FileKind)
    assert MIME_TYPES[FileKind.PDF] == "application/pdf"
    assert detect_mime_type(_image_bytes("PNG"), FileKind.IMAGE) == "image/png"
    assert detect_mime_type(_image_bytes("JPEG"), FileKind.IMAGE) == "image/jpeg"
    assert detect_mime_type(_image_bytes("TIFF"), FileKind.IMAGE) == "image/tiff"
    assert detect_mime_type(b"a,b\n", FileKind.CSV) == MIME_TYPES[FileKind.CSV]
