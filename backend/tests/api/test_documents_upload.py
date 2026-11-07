"""POST /api/v1/documents: multi-file upload and its hardening (type sniffing, names, sizes, counts)."""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.config import Settings
from app.db.models import Document
from tests.api.conftest import ClientFactory, RecordingQueue
from tests.api.factories import EXE_BYTES, csv_bytes, pdf_bytes, png_bytes, xlsx_bytes

UPLOAD = "/api/v1/documents"
ITEM_KEYS = {"filename", "status", "document_id", "job_id", "duplicate_of_id", "error"}
STORAGE_KEY = re.compile(r"^\d{4}/\d{2}/[0-9a-f]{32}\.[a-z]{3,4}$")


def _stored_files(settings: Settings) -> list[Path]:
    root = Path(settings.storage_dir)
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def _document_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Document)) or 0


def test_upload_accepts_every_supported_kind_in_request_order(client: TestClient, queue: RecordingQueue) -> None:
    files = [
        ("files", ("a.pdf", pdf_bytes("Invoice A"), "application/pdf")),
        ("files", ("b.png", png_bytes(), "image/png")),
        ("files", ("c.xlsx", xlsx_bytes(), "application/octet-stream")),
        ("files", ("d.csv", csv_bytes(), "text/csv")),
    ]
    response = client.post(UPLOAD, files=files)

    assert response.status_code == 202
    items = response.json()["items"]
    assert [item["filename"] for item in items] == ["a.pdf", "b.png", "c.xlsx", "d.csv"]
    assert all(set(item) == ITEM_KEYS for item in items)
    assert all(item["status"] == "queued" and item["error"] is None for item in items)
    assert all(item["duplicate_of_id"] is None for item in items)
    assert queue.job_ids == [UUID(item["job_id"]) for item in items]

    kinds = [client.get(f"{UPLOAD}/{item['document_id']}").json() for item in items]
    assert [(d["file_kind"], d["content_type"], d["status"]) for d in kinds] == [
        ("pdf", "application/pdf", "queued"),
        ("image", "image/png", "queued"),
        ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "queued"),
        ("csv", "text/csv", "queued"),
    ]


def test_upload_same_bytes_twice_reports_duplicate(client: TestClient) -> None:
    data = pdf_bytes("Invoice twice")
    first = client.post(UPLOAD, files=[("files", ("one.pdf", data, "application/pdf"))]).json()["items"][0]
    second = client.post(UPLOAD, files=[("files", ("two.pdf", data, "application/pdf"))]).json()["items"][0]

    assert first["duplicate_of_id"] is None
    assert second["status"] == "queued"
    assert second["duplicate_of_id"] == first["document_id"]
    assert second["document_id"] != first["document_id"]


def test_upload_rejects_path_traversal_name(client: TestClient, db_session: Session, settings: Settings) -> None:
    data = pdf_bytes("Traversal")
    files = [
        ("files", ("../../etc/passwd.pdf", data, "application/pdf")),
        ("files", ("..\\..\\Windows\\System32\\evil.pdf", pdf_bytes("Other"), "application/pdf")),
        ("files", ("/abs/path/../..", pdf_bytes("Dots"), "application/pdf")),
    ]
    items = client.post(UPLOAD, files=files).json()["items"]

    names = [item["filename"] for item in items]
    assert names[:2] == ["passwd.pdf", "evil.pdf"]
    for name in names:
        assert "/" not in name and "\\" not in name and name not in ("", ".", "..")
    rows = db_session.scalars(select(Document)).all()
    assert {row.filename for row in rows} == set(names)
    assert all(STORAGE_KEY.match(row.storage_key) for row in rows)
    stored = _stored_files(settings)
    assert len(stored) == 3
    root = Path(settings.storage_dir).resolve()
    assert all(path.resolve().is_relative_to(root) for path in stored)
    assert not any(path.name in ("passwd.pdf", "evil.pdf") for path in stored)


def test_upload_rejects_renamed_exe(
    client: TestClient, queue: RecordingQueue, db_session: Session, settings: Settings
) -> None:
    files = [
        ("files", ("invoice.pdf", EXE_BYTES, "application/pdf")),
        ("files", ("ok.pdf", pdf_bytes("Fine"), "application/pdf")),
    ]
    response = client.post(UPLOAD, files=files)

    assert response.status_code == 202
    rejected, accepted = response.json()["items"]
    assert rejected["status"] == "rejected"
    assert rejected["error"]["code"] == "unsupported_file_type"
    assert rejected["error"]["message"]
    assert rejected["document_id"] is None and rejected["job_id"] is None
    assert accepted["status"] == "queued"
    assert _document_count(db_session) == 1
    assert len(queue.job_ids) == 1
    assert len(_stored_files(settings)) == 1


def test_upload_rejects_empty_file(client: TestClient, db_session: Session) -> None:
    item = client.post(UPLOAD, files=[("files", ("empty.csv", b"", "text/csv"))]).json()["items"][0]

    assert item["status"] == "rejected"
    assert item["error"]["code"] == "empty_file"
    assert _document_count(db_session) == 0


def test_upload_rejects_oversize(make_client: ClientFactory, db_session: Session, settings: Settings) -> None:
    client = make_client(env={"MAX_UPLOAD_MB": "1"})
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 512)
    files = [
        ("files", ("big.pdf", big, "application/pdf")),
        ("files", ("small.pdf", pdf_bytes("Small"), "application/pdf")),
    ]
    response = client.post(UPLOAD, files=files)

    assert response.status_code == 202
    big_item, small_item = response.json()["items"]
    assert big_item["status"] == "rejected"
    assert big_item["error"]["code"] == "file_too_large"
    assert "1 MB" in big_item["error"]["message"]
    assert small_item["status"] == "queued"
    assert _document_count(db_session) == 1
    assert len(_stored_files(settings)) == 1


def test_upload_request_over_total_limit_is_413_before_reading(make_client: ClientFactory, db_session: Session) -> None:
    client = make_client(env={"MAX_UPLOAD_MB": "1", "MAX_FILES_PER_REQUEST": "1"})
    response = client.post(
        UPLOAD, files=[("files", ("huge.pdf", b"%PDF-1.4\n" + b"0" * (3 * 1024 * 1024), "application/pdf"))]
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    assert _document_count(db_session) == 0


def test_upload_chunked_body_over_total_limit_is_413(make_client: ClientFactory, db_session: Session) -> None:
    client = make_client(env={"MAX_UPLOAD_MB": "1", "MAX_FILES_PER_REQUEST": "1"})
    boundary = "testboundary123"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="x.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n%PDF-1.4\n"
    ).encode()

    def body() -> Iterator[bytes]:  # a generator body is sent without Content-Length (chunked)
        yield head
        for _ in range(3):
            yield b"0" * (1024 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    response = client.post(
        UPLOAD, content=body(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    assert _document_count(db_session) == 0


@pytest.fixture
def spooled(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Bytes written to each upload's temporary spool file, by filename (the most the file ever held)."""
    written: dict[str, int] = {}
    real_write = UploadFile.write

    async def counting_write(self: UploadFile, data: bytes) -> None:
        written[self.filename or ""] = written.get(self.filename or "", 0) + len(data)
        await real_write(self, data)

    monkeypatch.setattr(UploadFile, "write", counting_write)
    return written


def _stream_post(app: Any, path: str, content_type: str, chunks: Iterator[bytes]) -> tuple[int, Any, int]:
    """POST ``chunks`` straight into the ASGI ``app`` as separate ``http.request`` messages (a chunked body without a
    Content-Length, as a proxy streams it; the TestClient would read a generator body whole first). Returns the
    status, the JSON body and how many body bytes the app pulled."""
    pulled = 0
    status = 0
    body = bytearray()

    async def receive() -> dict[str, Any]:
        nonlocal pulled
        chunk = next(chunks, None)
        if chunk is None:
            return {"type": "http.request", "body": b"", "more_body": False}
        pulled += len(chunk)
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": [(b"content-type", content_type.encode()), (b"host", b"testserver")],
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }  # fmt: skip
    asyncio.run(app(scope, receive, send))
    return status, json.loads(body), pulled


def test_oversized_file_is_cut_off_while_it_streams(
    make_client: ClientFactory, db_session: Session, spooled: dict[str, int], queue: RecordingQueue
) -> None:
    client = make_client(env={"MAX_UPLOAD_MB": "1", "MAX_FILES_PER_REQUEST": "5"})
    boundary = "streamboundary42"

    def part(name: str, payload: Iterator[bytes]) -> Iterator[bytes]:
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
        yield from payload
        yield b"\r\n"

    def huge() -> Iterator[bytes]:  # 3 MiB in 64 KiB chunks
        yield b"%PDF-1.4\n"
        for _ in range(48):
            yield b"0" * (64 * 1024)

    def body() -> Iterator[bytes]:
        yield from part("huge.pdf", huge())
        yield from part("small.pdf", iter([pdf_bytes("Small")]))
        yield f"--{boundary}--\r\n".encode()

    status, payload, pulled = _stream_post(client.app, UPLOAD, f"multipart/form-data; boundary={boundary}", body())

    assert status == 202
    huge_item, small_item = payload["items"]
    assert (huge_item["status"], huge_item["error"]["code"]) == ("rejected", "file_too_large")
    assert "1 MB" in huge_item["error"]["message"]
    assert small_item["status"] == "queued"
    assert pulled > 3 * 1024 * 1024  # the whole part arrived ...
    assert 0 < spooled["huge.pdf"] <= 1024 * 1024  # ... but no more than MAX_UPLOAD_MB of it reached the disk
    assert _document_count(db_session) == 1 and len(queue.job_ids) == 1


def test_file_of_exactly_the_limit_is_accepted(make_client: ClientFactory, spooled: dict[str, int]) -> None:
    client = make_client(env={"MAX_UPLOAD_MB": "1"})
    exact = b"%PDF-1.4\n" + b"0" * (1024 * 1024 - 9)

    item = client.post(UPLOAD, files=[("files", ("exact.pdf", exact, "application/pdf"))]).json()["items"][0]

    assert item["status"] == "queued"
    assert spooled["exact.pdf"] == 1024 * 1024


def test_upload_request_cap_allows_five_full_size_files(make_client: ClientFactory, settings: Settings) -> None:
    from app.security.uploads import MULTIPART_OVERHEAD_PER_FILE, max_upload_request_bytes

    client = make_client(env={"MAX_UPLOAD_MB": "1", "MAX_FILES_PER_REQUEST": "50"})
    cap = 5 * (1024 * 1024 + MULTIPART_OVERHEAD_PER_FILE)
    assert client.get("/api/v1/config").json()["max_request_mb"] == cap // (1024 * 1024)

    response = client.post(  # declared over the cap: refused before a byte of the body is read
        UPLOAD,
        content=b"x" * (cap + 1),
        headers={"Content-Type": "multipart/form-data; boundary=b"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    assert max_upload_request_bytes(settings) == 5 * (200 * 1024 * 1024 + MULTIPART_OVERHEAD_PER_FILE)  # defaults
    small = make_client(env={"MAX_UPLOAD_MB": "1", "MAX_FILES_PER_REQUEST": "2"})
    assert small.get("/api/v1/config").json()["max_request_mb"] == 2


def test_upload_too_many_files(make_client: ClientFactory, queue: RecordingQueue, db_session: Session) -> None:
    client = make_client(env={"MAX_FILES_PER_REQUEST": "2"})
    for count in (3, 6):
        files = [("files", (f"f{i}.csv", csv_bytes() + str(i).encode(), "text/csv")) for i in range(count)]
        response = client.post(UPLOAD, files=files)
        assert response.status_code == 422, count
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert "2" in error["message"]
    assert _document_count(db_session) == 0
    assert queue.job_ids == []


def test_upload_without_files_is_422(client: TestClient) -> None:
    for response in (
        client.post(UPLOAD, files={"note": (None, "no files")}),  # multipart with a text part only
        client.post(UPLOAD, files=[("file", ("a.csv", csv_bytes(), "text/csv"))]),
    ):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


def test_upload_requires_multipart_body(client: TestClient) -> None:
    response = client.post(UPLOAD, json={"files": []})

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"


def test_upload_unicode_filename(client: TestClient, db_session: Session) -> None:
    decomposed = unicodedata.normalize("NFD", "Facture été №42 請求書.pdf")
    item = client.post(UPLOAD, files=[("files", (decomposed, pdf_bytes("Unicode"), "application/pdf"))]).json()[
        "items"
    ][0]

    expected = unicodedata.normalize("NFC", decomposed)
    assert item["status"] == "queued"
    assert item["filename"] == expected
    assert db_session.get(Document, UUID(item["document_id"])).filename == expected

    download = client.get(f"{UPLOAD}/{item['document_id']}/file")
    disposition = download.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert "filename*=UTF-8''Facture%20%C3%A9t%C3%A9%20%E2%84%9642%20%E8%AB%8B%E6%B1%82%E6%9B%B8.pdf" in disposition
    ascii_part = disposition.split("filename*=")[0]
    assert ascii_part.isascii()


def test_upload_extension_mismatch_is_recorded(client: TestClient, db_session: Session) -> None:
    item = client.post(UPLOAD, files=[("files", ("scan.pdf", png_bytes(), "application/pdf"))]).json()["items"][0]

    assert item["status"] == "queued"
    document = db_session.get(Document, UUID(item["document_id"]))
    assert document.file_kind == "image" and document.content_type == "image/png"  # the content decides
    assert document.storage_key.endswith(".png")
    assert document.warnings == ["extension_mismatch"]
