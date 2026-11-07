"""Spreadsheet parsing: every sheet, trimmed grids, typed cells, CSV encodings/delimiters, hostile workbooks."""

from __future__ import annotations

import random
import re
import shutil
import time
import zipfile
from datetime import date, datetime
from datetime import time as dtime
from decimal import Decimal
from pathlib import Path

import openpyxl
import pytest
import xlwt
from openpyxl.styles import PatternFill

from app.ingestion import spreadsheets
from app.ingestion.spreadsheets import parse_spreadsheet
from app.ingestion.types import CorruptFileError, FileKind, FileTooLargeError

from .conftest import set_env

Rows = list[list[object]]


def _xlsx(path: Path, sheets: dict[str, Rows]) -> Path:
    wb = openpyxl.Workbook()
    for position, (name, rows) in enumerate(sheets.items()):
        ws = wb.active if position == 0 else wb.create_sheet()
        ws.title = name
        for row in rows:
            ws.append(row)
    wb.save(path)
    return path


def _rewrite_member(path: Path, member: str, transform: object) -> None:
    with zipfile.ZipFile(path) as src:
        contents = {info.filename: src.read(info.filename) for info in src.infolist()}
    contents[member] = transform(contents[member])  # type: ignore[operator]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in contents.items():
            dst.writestr(name, data)


def test_xlsx_all_sheets_typed_cells(tmp_path: Path) -> None:
    path = _xlsx(
        tmp_path / "book.xlsx",
        {
            "Invoices": [
                ["Invoice No", "Date", "Qty", "Total", "Paid", "Issued at"],
                ["INV-1", date(2024, 3, 5), 3, 110.5, True, datetime(2024, 3, 5, 14, 30)],
                ["INV-2", date(2024, 3, 6), 1, 20.1, False, None],
            ],
            "Notes": [["Supplier", "ACME Corp."]],
        },
    )
    doc = parse_spreadsheet(path, FileKind.XLSX)
    assert doc.kind is FileKind.XLSX and doc.pages == []
    assert [(s.name, s.index) for s in doc.sheets] == [("Invoices", 0), ("Notes", 1)]
    first = doc.sheets[0].rows[1]
    assert first == ["INV-1", date(2024, 3, 5), 3, Decimal("110.5"), True, datetime(2024, 3, 5, 14, 30)]
    assert [type(v) for v in first] == [str, date, int, Decimal, bool, datetime]
    assert doc.sheets[0].rows[2][3] == Decimal("20.1")  # Decimal(str(float)), not the binary expansion
    assert doc.sheets[0].rows[2][5] is None
    assert doc.sheets[1].rows == [["Supplier", "ACME Corp."]]
    assert doc.metadata["sheets"] == ["Invoices", "Notes"]


def test_xlsx_trims_empty_rows_cols(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["B2"], ws["C2"] = "Invoice No", "Total"
    ws["B3"], ws["C3"] = "INV-1", 12.5
    ws["B5"] = "   "  # whitespace only: empty
    ws["H40"].fill = PatternFill("solid", fgColor="FFFF00")  # formatted but empty, far away
    path = tmp_path / "sparse.xlsx"
    wb.save(path)
    sheet = parse_spreadsheet(path, FileKind.XLSX).sheets[0]
    # Trailing empty rows and columns are trimmed; leading ones are kept so that rows[i][j] is sheet cell
    # (row i + 1, column j + 1) - row numbers reported downstream match what the user sees in Excel.
    assert sheet.rows == [[None, None, None], [None, "Invoice No", "Total"], [None, "INV-1", Decimal("12.5")]]


def test_xlsx_lying_dimension_is_not_trusted(tmp_path: Path) -> None:
    path = _xlsx(tmp_path / "book.xlsx", {"S": [["a", "b"], ["c", "d"]]})
    _rewrite_member(
        path,
        "xl/worksheets/sheet1.xml",
        lambda xml: re.sub(rb'<dimension ref="[^"]+"\s*/>', b'<dimension ref="A1:XFD1048576"/>', xml),
    )
    start = time.perf_counter()
    sheet = parse_spreadsheet(path, FileKind.XLSX).sheets[0]
    assert time.perf_counter() - start < 5
    assert sheet.rows == [["a", "b"], ["c", "d"]]


def test_xls_read(tmp_path: Path) -> None:
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Ledger")
    for col, header in enumerate(["Invoice No", "Date", "Qty", "Total", "Paid"]):
        ws.write(0, col, header)
    ws.write(1, 0, "INV-1")
    ws.write(1, 1, date(2024, 3, 5), xlwt.easyxf(num_format_str="YYYY-MM-DD"))
    ws.write(1, 2, 3)
    ws.write(1, 3, 110.5)
    ws.write(1, 4, True)
    wb.add_sheet("Empty")
    wb.add_sheet("Other").write(0, 0, "x")
    path = tmp_path / "ledger.xls"
    wb.save(str(path))
    doc = parse_spreadsheet(path, FileKind.XLS)
    assert [(s.name, s.index) for s in doc.sheets] == [("Ledger", 0), ("Empty", 1), ("Other", 2)]
    row = doc.sheets[0].rows[1]
    assert row == ["INV-1", date(2024, 3, 5), 3, Decimal("110.5"), True]
    # .xls stores every number as a float; integral ones come back as int, as openpyxl reads them from .xlsx.
    assert [type(v) for v in row] == [str, date, int, Decimal, bool]
    assert doc.sheets[1].rows == []
    assert doc.metadata["sheets"] == ["Ledger", "Empty", "Other"]


def test_csv_semicolon_cp1252(tmp_path: Path) -> None:
    text = (
        "Rechnungsnr;Datum;Lieferant;Betrag\n"
        "RE-1001;05.03.2024;Müller GmbH;1.234,56 €\n"
        "RE-1002;06.03.2024;Société Générale;99,00 €\n"
    )
    path = tmp_path / "export.csv"
    path.write_bytes(text.encode("cp1252"))
    doc = parse_spreadsheet(path, FileKind.CSV)
    assert doc.kind is FileKind.CSV and len(doc.sheets) == 1 and doc.sheets[0].index == 0
    rows = doc.sheets[0].rows
    assert rows[1] == ["RE-1001", "05.03.2024", "Müller GmbH", "1.234,56 €"]  # cells stay text
    assert rows[2][2] == "Société Générale"
    assert doc.metadata["encoding"] == "cp1252" and doc.metadata["delimiter"] == ";"


def test_csv_utf8_bom_tab(tmp_path: Path) -> None:
    text = (
        "\ufeffInvoice No\tSupplier\tTotal\tNote\n"
        "0042\tACME, Inc.\t1,234.50\t\n"
        'INV-2\t"Globex\tGroup"\t99.00\tquoted tab\n'
    )
    path = tmp_path / "export.tsv"
    path.write_bytes(text.encode("utf-8"))
    doc = parse_spreadsheet(path, FileKind.CSV)
    rows = doc.sheets[0].rows
    assert rows[0] == ["Invoice No", "Supplier", "Total", "Note"]  # the BOM is not part of the first header
    assert rows[1] == ["0042", "ACME, Inc.", "1,234.50", None]  # leading zeros kept, empty cell -> None
    assert rows[2][1] == "Globex\tGroup"
    assert doc.metadata["encoding"] == "utf-8-sig" and doc.metadata["delimiter"] == "\t"


def test_csv_other_encodings_and_single_column(tmp_path: Path) -> None:
    polish = tmp_path / "pl.csv"
    polish.write_bytes("Faktura;Kwota\nŁódź Spółka;1.234,56 zł\nZażółć gęślą jaźń;12,00\n".encode("cp1250"))
    doc = parse_spreadsheet(polish, FileKind.CSV)
    assert doc.sheets[0].rows[1][0] == "Łódź Spółka"
    utf16 = tmp_path / "unicode.txt"
    utf16.write_bytes("Invoice\tTotal\nINV-1\t10.00\n".encode("utf-16"))
    assert parse_spreadsheet(utf16, FileKind.CSV).sheets[0].rows == [["Invoice", "Total"], ["INV-1", "10.00"]]
    single = tmp_path / "single.csv"
    single.write_bytes(b"Invoice\r\nINV-1\r\n\r\n")
    assert parse_spreadsheet(single, FileKind.CSV).sheets[0].rows == [["Invoice"], ["INV-1"]]


def test_corrupt_xlsx_raises(tmp_path: Path) -> None:
    data = _xlsx(tmp_path / "good.xlsx", {"S": [["a", 1]] * 50}).read_bytes()
    truncated = tmp_path / "truncated.xlsx"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(CorruptFileError) as err:
        parse_spreadsheet(truncated, FileKind.XLSX)
    assert err.value.code == "corrupt_file"
    not_a_workbook = tmp_path / "other.xlsx"
    with zipfile.ZipFile(not_a_workbook, "w") as zf:
        zf.writestr("xl/readme.txt", "hello")
    with pytest.raises(CorruptFileError):
        parse_spreadsheet(not_a_workbook, FileKind.XLSX)
    garbage_xls = tmp_path / "garbage.xls"
    garbage_xls.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + random.Random(3).randbytes(4096))
    with pytest.raises(CorruptFileError):
        parse_spreadsheet(garbage_xls, FileKind.XLS)


def test_xlsx_zip_bomb_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bomb = _xlsx(tmp_path / "bomb.xlsx", {"S": [["a", 1]]})
    with zipfile.ZipFile(bomb, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/media/padding.bin", b"\0" * (8 * 1024 * 1024))  # inflates about 1000:1
    with pytest.raises(FileTooLargeError) as err:
        parse_spreadsheet(bomb, FileKind.XLSX)
    assert err.value.code == "file_too_large"
    set_env(monkeypatch, MAX_XLSX_UNCOMPRESSED_MB=1)
    big = _xlsx(tmp_path / "big.xlsx", {"S": [["a", 1]]})
    with zipfile.ZipFile(big, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/media/photo.bin", random.Random(5).randbytes(2 * 1024 * 1024))  # incompressible
    with pytest.raises(FileTooLargeError):
        parse_spreadsheet(big, FileKind.XLSX)


def test_row_limit_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, MAX_SHEET_ROWS=10, MAX_SHEET_COLS=5)
    rows: Rows = [[f"INV-{i}", i] for i in range(11)]
    with pytest.raises(FileTooLargeError):
        parse_spreadsheet(_xlsx(tmp_path / "eleven.xlsx", {"S": rows}), FileKind.XLSX)
    csv_path = tmp_path / "eleven.csv"
    csv_path.write_text("".join(f"INV-{i},{i}\n" for i in range(11)), encoding="utf-8")
    with pytest.raises(FileTooLargeError):
        parse_spreadsheet(csv_path, FileKind.CSV)
    # Formatting below the data is not data: ten rows plus styled empty rows are within the limit.
    wb = openpyxl.Workbook()
    for row in rows[:10]:
        wb.active.append(row)
    wb.active["A30"].fill = PatternFill("solid", fgColor="FFFF00")
    ok = tmp_path / "ten.xlsx"
    wb.save(ok)
    assert len(parse_spreadsheet(ok, FileKind.XLSX).sheets[0].rows) == 10
    wide = _xlsx(tmp_path / "wide.xlsx", {"S": [["a", None, None, None, None, None, "far right"]]})
    with pytest.raises(FileTooLargeError):
        parse_spreadsheet(wide, FileKind.XLSX)


def test_sheet_text_rendering(tmp_path: Path) -> None:
    path = _xlsx(
        tmp_path / "book.xlsx",
        {"Ledger": [["Invoice No", "Date", "Total"], ["INV-1", date(2024, 3, 5), 12.5], ["INV-2", None, 7]]},
    )
    doc = parse_spreadsheet(path, FileKind.XLSX)
    assert doc.sheets[0].text == "Invoice No\tDate\tTotal\nINV-1\t2024-03-05\t12.5\nINV-2\t\t7"
    assert doc.text == doc.sheets[0].text
    assert doc.page_count == 1


def test_workbooks_parse_whatever_the_file_name(tmp_path: Path) -> None:
    # Stored uploads have UUID names; only the content decides how a workbook is read.
    xlsx = _xlsx(tmp_path / "book.xlsx", {"S": [["Invoice No", "Total"], ["INV-1", 12.5]]})
    for name in ("noext", "data.csv", "0f3c9a.bin"):
        copy = shutil.copy(xlsx, tmp_path / name)
        assert parse_spreadsheet(Path(copy), FileKind.XLSX).sheets[0].rows[1] == ["INV-1", Decimal("12.5")]
    book = xlwt.Workbook()
    book.add_sheet("Ledger").write(0, 0, "INV-1")
    xls = tmp_path / "ledger.xls"
    book.save(str(xls))
    copy = shutil.copy(xls, tmp_path / "ledger.bin")
    assert parse_spreadsheet(Path(copy), FileKind.XLS).sheets[0].rows == [["INV-1"]]


def test_bad_zip_version_raises_corrupt(tmp_path: Path) -> None:
    data = bytearray(_xlsx(tmp_path / "good.xlsx", {"S": [["a", 1]]}).read_bytes())
    central = data.index(b"PK\x01\x02")
    data[central + 6 : central + 8] = (79).to_bytes(2, "little")  # "version needed to extract" = 7.9
    path = tmp_path / "future.xlsx"
    path.write_bytes(bytes(data))
    with pytest.raises(CorruptFileError):
        parse_spreadsheet(path, FileKind.XLSX)


def test_csv_sniffing_hostile_sample_is_bounded(tmp_path: Path) -> None:
    # csv.Sniffer's quote regex is quadratic: 64 KiB of ',"a' takes about 30 s to sniff.
    path = tmp_path / "hostile.csv"
    path.write_text((',"a' * 100 + "\n") * 220, encoding="utf-8")  # 220 rows of 101 fields, 65 KiB
    start = time.perf_counter()
    doc = parse_spreadsheet(path, FileKind.CSV)
    assert time.perf_counter() - start < 3
    assert len(doc.sheets[0].rows) == 220 and doc.metadata["delimiter"] == ","


def test_time_only_cells_match_between_xls_and_xlsx(tmp_path: Path) -> None:
    xlsx = tmp_path / "times.xlsx"
    wb = openpyxl.Workbook()
    wb.active.append(["Start", dtime(14, 30)])
    wb.save(xlsx)
    book = xlwt.Workbook()
    sheet = book.add_sheet("Times")
    sheet.write(0, 0, "Start")
    sheet.write(0, 1, dtime(14, 30), xlwt.easyxf(num_format_str="HH:MM:SS"))
    xls = tmp_path / "times.xls"
    book.save(str(xls))
    from_xlsx = parse_spreadsheet(xlsx, FileKind.XLSX).sheets[0].rows[0]
    from_xls = parse_spreadsheet(xls, FileKind.XLS).sheets[0].rows[0]
    assert from_xlsx == from_xls == ["Start", "14:30:00"]


def test_unexpected_parser_errors_become_corrupt_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ledger.csv"
    path.write_text("Invoice,Total\nINV-1,10\n", encoding="utf-8")

    def explode(_text: str) -> str:
        raise LookupError("unexpected codec state")

    monkeypatch.setattr(spreadsheets, "_sniff_delimiter", explode)
    with pytest.raises(CorruptFileError):
        parse_spreadsheet(path, FileKind.CSV)
