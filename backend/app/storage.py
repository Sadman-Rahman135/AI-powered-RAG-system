"""Local file storage for uploaded originals, behind a small interface (an object store can replace it later).

Keys look like ``"2026/11/<uuid4 hex><suffix>"``: they are generated here and never derived from the user's
filename, so an upload cannot choose (or traverse to) its own path. ``save`` streams in 1 MiB chunks, hashing as
it goes, writes to a temporary ``.part`` file and renames it into place only when complete, so a reader never
sees a half-written file and an oversized upload leaves nothing behind.
"""

from __future__ import annotations

import hashlib
import io
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from app.config import get_settings
from app.ingestion.types import FileKind, FileTooLargeError

CHUNK_BYTES = 1024 * 1024

SUFFIX_BY_KIND: dict[FileKind, str] = {
    FileKind.PDF: ".pdf",
    FileKind.XLSX: ".xlsx",
    FileKind.XLS: ".xls",
    FileKind.CSV: ".csv",
}
"""Storage suffix per file kind; images take theirs from the detected MIME type (``IMAGE_SUFFIX_BY_MIME``)."""

IMAGE_SUFFIX_BY_MIME: dict[str, str] = {"image/png": ".png", "image/jpeg": ".jpg", "image/tiff": ".tif"}
_GENERIC_IMAGE_SUFFIX = ".img"
_SUFFIX_RE = re.compile(r"^(?:\.[a-z0-9]{1,8})?$")
_KEY_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")

__all__ = [
    "CHUNK_BYTES",
    "IMAGE_SUFFIX_BY_MIME",
    "SUFFIX_BY_KIND",
    "Storage",
    "StoredFile",
    "get_storage",
    "suffix_for",
]


@dataclass(frozen=True, slots=True)
class StoredFile:
    key: str  # "{yyyy}/{mm}/{uuid4hex}{suffix}" - never derived from the user's filename
    size: int
    sha256: str


def suffix_for(kind: FileKind, content_type: str | None = None) -> str:
    """The storage suffix for a sniffed file: ``SUFFIX_BY_KIND``, or for images the one of its MIME type."""
    if kind is FileKind.IMAGE:
        return IMAGE_SUFFIX_BY_MIME.get((content_type or "").lower(), _GENERIC_IMAGE_SUFFIX)
    return SUFFIX_BY_KIND[kind]


class Storage:
    """Files under one root directory (default ``STORAGE_DIR``, created if missing)."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root if root is not None else get_settings().storage_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, stream: BinaryIO, *, suffix: str, max_bytes: int | None = None) -> StoredFile:
        """Copy ``stream`` into a new file and return its key, size and SHA-256.

        More than ``max_bytes`` bytes delete the partial file and raise ``FileTooLargeError``; any other failure
        also leaves no file behind.
        """
        if not _SUFFIX_RE.match(suffix):
            raise ValueError(f"invalid storage suffix {suffix!r}")
        now = datetime.now(UTC)
        key = f"{now:%Y}/{now:%m}/{uuid.uuid4().hex}{suffix}"
        final = self.path(key)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(final.name + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with partial.open("xb") as out:
                while chunk := stream.read(CHUNK_BYTES):
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise FileTooLargeError(f"the file is larger than {max_bytes} bytes")
                    digest.update(chunk)
                    out.write(chunk)
            partial.replace(final)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return StoredFile(key=key, size=size, sha256=digest.hexdigest())

    def save_bytes(self, data: bytes, *, suffix: str) -> StoredFile:
        return self.save(io.BytesIO(data), suffix=suffix)

    def path(self, key: str) -> Path:
        """The file of ``key``. Absolute keys, ``..``/``.`` parts, backslashes, drive letters, NUL and anything but
        plain ``[A-Za-z0-9._-]`` path parts raise ``ValueError``; the result is always inside the root."""
        if not key or key.startswith("/"):
            raise ValueError("invalid storage key")
        parts = key.split("/")
        if any(part in ("", ".", "..") or not _KEY_PART_RE.match(part) for part in parts):
            raise ValueError("invalid storage key")
        candidate = self.root.joinpath(*parts).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("invalid storage key")
        return candidate

    def open(self, key: str) -> BinaryIO:
        return self.path(key).open("rb")

    def delete(self, key: str) -> None:
        """Remove the file of ``key``; a missing file is fine."""
        self.path(key).unlink(missing_ok=True)


@lru_cache(maxsize=8)
def _storage_at(root: str) -> Storage:
    return Storage(Path(root))


def get_storage() -> Storage:
    """The storage under the configured ``STORAGE_DIR`` (one cached instance per directory)."""
    return _storage_at(str(get_settings().storage_dir))
