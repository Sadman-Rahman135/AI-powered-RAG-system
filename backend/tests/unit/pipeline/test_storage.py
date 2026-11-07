"""File storage."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pytest

from app.ingestion.types import FileKind, FileTooLargeError
from app.storage import CHUNK_BYTES, SUFFIX_BY_KIND, Storage, get_storage, suffix_for


def test_save_streams_hashes_and_places_the_file_under_a_generated_key(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "files")
    data = b"%PDF-1.7 " + bytes(range(256)) * (3 * CHUNK_BYTES // 256 + 7)  # several chunks
    stored = storage.save(io.BytesIO(data), suffix=".pdf", max_bytes=len(data))
    assert re.fullmatch(r"\d{4}/\d{2}/[0-9a-f]{32}\.pdf", stored.key)
    assert stored.size == len(data) and stored.sha256 == hashlib.sha256(data).hexdigest()
    path = storage.path(stored.key)
    assert path.is_relative_to((tmp_path / "files").resolve()) and path.read_bytes() == data
    with storage.open(stored.key) as handle:
        assert handle.read(9) == b"%PDF-1.7 "
    assert storage.save_bytes(data, suffix=".pdf").key != stored.key  # never derived from the content or a name
    assert not list((tmp_path / "files").rglob("*.part"))


def test_oversize_upload_leaves_nothing_behind(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "files")
    with pytest.raises(FileTooLargeError):
        storage.save(io.BytesIO(b"x" * (CHUNK_BYTES + 10)), suffix=".csv", max_bytes=CHUNK_BYTES)
    assert [p for p in (tmp_path / "files").rglob("*") if p.is_file()] == []


@pytest.mark.parametrize(
    "key",
    ["../outside.pdf", "2026/../../outside.pdf", "/etc/passwd", "a\\b.pdf", "a\x00b.pdf", "C:/x.pdf", "C:x.pdf", "",
     "2026//x.pdf", "./x.pdf"],
)
def test_path_rejects_keys_that_could_escape_the_root(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError):
        Storage(tmp_path).path(key)


def test_delete_is_idempotent_and_bad_suffixes_are_refused(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    stored = storage.save_bytes(b"a,b\n1,2\n", suffix=".csv")
    storage.delete(stored.key)
    storage.delete(stored.key)  # missing file is fine
    assert not storage.path(stored.key).exists()
    for suffix in ("pdf", ".PDF/../x", ".toolongsuffix"):
        with pytest.raises(ValueError):
            storage.save_bytes(b"x", suffix=suffix)


def test_suffixes_per_kind_and_cached_storage(tmp_path: Path) -> None:
    assert SUFFIX_BY_KIND == {FileKind.PDF: ".pdf", FileKind.XLSX: ".xlsx", FileKind.XLS: ".xls", FileKind.CSV: ".csv"}
    assert suffix_for(FileKind.IMAGE, "image/png") == ".png"
    assert suffix_for(FileKind.IMAGE, "image/jpeg") == ".jpg"
    assert suffix_for(FileKind.IMAGE, "image/tiff") == ".tif"
    assert suffix_for(FileKind.PDF, "application/pdf") == ".pdf"
    storage = get_storage()
    assert get_storage() is storage and storage.root.is_dir()  # STORAGE_DIR from the test settings
