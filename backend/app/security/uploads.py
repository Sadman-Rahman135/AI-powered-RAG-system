"""Upload hardening: request size limits, multipart parsing limits, content sniffing and safe download headers.

- ``RequestSizeLimitMiddleware`` caps every request body before the app reads it: the upload route at
  ``max_upload_request_bytes`` (``min(MAX_FILES_PER_REQUEST, 5) x (MAX_UPLOAD_MB + multipart framing)``, about
  1 GB with the defaults), every other route at 1 MiB. A declared ``Content-Length`` over the cap is refused with
  413 before a byte is read; bodies without one (chunked, as the web proxy streams uploads) are counted while they
  arrive and cut off at the cap.
- ``read_upload_form`` parses multipart with bounded file / field counts and part sizes, and enforces
  ``MAX_UPLOAD_MB`` per file *while the body streams in*: past the limit the rest of a file part is counted but not
  written to the temporary spool file, so an oversized file costs at most ``MAX_UPLOAD_MB`` of temporary disk (and
  that is released as soon as parsing ends); the file is then rejected as ``file_too_large`` while the other files of
  the request are processed.
- ``inspect_upload`` decides a file's kind from its bytes (the extension is only a hint; intake records a mismatch
  as a document warning), enforces the per-file size limit, and rejects empty and unsupported files with the
  contract's codes.

Temporary disk per upload request is therefore bounded by the request cap; per file by ``MAX_UPLOAD_MB``.
- ``content_disposition`` builds an RFC 6266 / RFC 5987 ``attachment`` header from a display name.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, BinaryIO
from urllib.parse import quote

from fastapi import Request
from starlette.datastructures import FormData, Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import PayloadTooLargeError, UnsupportedMediaTypeError, ValidationFailedError, error_response
from app.config import Settings, get_settings
from app.ingestion.sniff import HEAD_BYTES, detect_file_kind, detect_mime_type
from app.ingestion.types import FileKind, IngestionError
from app.storage import suffix_for

UPLOAD_PATH = "/api/v1/documents"
UPLOAD_FIELD = "files"
MAX_JSON_BODY_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_PER_FILE = 16 * 1024  # part headers and boundaries, generously
MAX_FORM_FIELDS = 16  # non-file parts tolerated in an upload
MAX_FORM_FIELD_BYTES = 64 * 1024
MAX_FULL_SIZE_FILES_PER_REQUEST = 5


def max_upload_request_bytes(settings: Settings) -> int:
    """The cap of one upload request body: room for ``min(MAX_FILES_PER_REQUEST, 5)`` files of the full
    ``MAX_UPLOAD_MB`` (1000 MB + framing with the defaults); more, smaller files fit as long as the body does.

    Why not ``MAX_FILES_PER_REQUEST x MAX_UPLOAD_MB`` (10 GB)? The per-file check while streaming
    (``read_upload_form``) stops one oversized file from filling the disk, but 50 files that are each just under the
    limit - or 50 files of junk that are rejected only after parsing - would still be spooled to temporary disk, so
    the whole request needs its own bound. A client with more data sends more requests (the UI can batch by bytes
    using ``GET /config`` ``max_request_mb``).
    """
    files = min(settings.max_files_per_request, MAX_FULL_SIZE_FILES_PER_REQUEST)
    return files * (settings.max_upload_bytes + MULTIPART_OVERHEAD_PER_FILE)


def _body_limit(scope: Scope) -> int:
    if scope.get("method") == "POST" and scope.get("path", "").rstrip("/") == UPLOAD_PATH:
        return max_upload_request_bytes(get_settings())
    return MAX_JSON_BODY_BYTES


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class RequestSizeLimitMiddleware:
    """Refuse or cut off request bodies over the route's cap (see the module docstring)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = _body_limit(scope)
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            response = error_response(413, PayloadTooLargeError.code, _too_large_message(limit))
            await response(scope, receive, send)
            return
        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise PayloadTooLargeError(_too_large_message(limit))
            return message

        await self.app(scope, counted_receive, send)


def _too_large_message(limit: int) -> str:
    return f"The request body is larger than the {_megabytes(limit)} limit"


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.0f} MB" if size >= 1024 * 1024 else f"{size // 1024} KB"


class _BoundedMultiPartParser(MultiPartParser):
    """Starlette's multipart parser with a per-file byte budget enforced while the body streams in.

    Once a file part has passed ``max_file_bytes``, the rest of it is counted but not handed on for spooling; when
    parsing ends the part's spool file is emptied and its ``size`` is set to the true byte count, so
    ``inspect_upload`` rejects it as ``file_too_large``. Relies on Starlette's parser callbacks
    (``on_part_begin`` / ``on_part_data`` / ``on_part_end``, ``_current_part``), pinned by the upload tests.
    """

    def __init__(self, headers: Headers, stream: AsyncGenerator[bytes], *, max_file_bytes: int, **limits: Any) -> None:
        super().__init__(headers, stream, **limits)
        self._max_file_bytes = max_file_bytes
        self._part_bytes = 0
        self._oversized: list[tuple[UploadFile, int]] = []

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._part_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self._part_bytes += end - start
            if self._part_bytes > self._max_file_bytes:
                return  # over the per-file limit: counted, never written to disk
        super().on_part_data(data, start, end)

    def on_part_end(self) -> None:
        upload = self._current_part.file
        if upload is not None and self._part_bytes > self._max_file_bytes:
            self._oversized.append((upload, self._part_bytes))
        super().on_part_end()

    async def parse(self) -> FormData:
        form = await super().parse()
        for upload, size in self._oversized:
            upload.file.seek(0)
            upload.file.truncate()  # release the spooled prefix now, not when the request ends
            upload.size = size
        return form


async def read_upload_form(request: Request, settings: Settings) -> FormData:
    """The multipart form, with at most ``MAX_FILES_PER_REQUEST + 1`` files parsed (one more than allowed, so the
    caller can report the count), small non-file parts, and no file spooled past ``MAX_UPLOAD_MB`` (see
    ``_BoundedMultiPartParser``). Not multipart -> 415; malformed multipart -> 422. The caller closes the form."""
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise UnsupportedMediaTypeError(
            f"Send the files as multipart/form-data in the '{UPLOAD_FIELD}' field",
        )
    try:
        async with aclosing(request.stream()) as stream:
            parser = _BoundedMultiPartParser(
                request.headers,
                stream,
                max_file_bytes=settings.max_upload_bytes,
                max_files=settings.max_files_per_request + 1,
                max_fields=MAX_FORM_FIELDS,
                max_part_size=MAX_FORM_FIELD_BYTES,
            )
            return await parser.parse()
    except MultiPartException as exc:
        if "too many files" in exc.message.lower():
            raise too_many_files(settings) from None
        raise ValidationFailedError("The multipart body could not be read", details={"reason": exc.message}) from None


def too_many_files(settings: Settings, received: int | None = None) -> ValidationFailedError:
    limit = settings.max_files_per_request
    details: dict[str, int] = {"max_files": limit}
    if received is not None:
        details["received"] = received
    return ValidationFailedError(f"Too many files: at most {limit} per request", details=details)


class UploadRejectedError(Exception):
    """A file that is not accepted; ``code`` is the contract's per-item error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class UploadInspection:
    kind: FileKind
    content_type: str  # exact MIME type from the content (image/png, not image/*)

    @property
    def storage_suffix(self) -> str:
        """The stored file's suffix, from the detected type (never from the user's filename)."""
        return suffix_for(self.kind, self.content_type)


def inspect_upload(file: BinaryIO, filename: str, *, size: int | None, max_bytes: int) -> UploadInspection:
    """Check one uploaded file (left positioned at its start) or raise ``UploadRejectedError``."""
    if size is not None and size > max_bytes:
        raise UploadRejectedError("file_too_large", size_limit_message(max_bytes))
    file.seek(0)
    head = file.read(HEAD_BYTES)
    file.seek(0)
    try:
        kind = detect_file_kind(head, filename)
    except IngestionError as exc:
        message = "The file is empty" if exc.code == "empty_file" else exc.message[:1].upper() + exc.message[1:]
        raise UploadRejectedError(exc.code, message) from None
    return UploadInspection(kind, detect_mime_type(head, kind))


def size_limit_message(max_bytes: int) -> str:
    return f"The file is larger than the {_megabytes(max_bytes)} limit"


_HEADER_UNSAFE = re.compile(r'[\x00-\x1f\x7f"\\]')


def content_disposition(filename: str) -> str:
    """``attachment; filename="<ASCII fallback>"; filename*=UTF-8''<percent-encoded name>``."""
    name = _HEADER_UNSAFE.sub("_", os.path.basename(filename.replace("\\", "/"))) or "document"
    fallback = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").strip() or "document"
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(name, safe='')}"
