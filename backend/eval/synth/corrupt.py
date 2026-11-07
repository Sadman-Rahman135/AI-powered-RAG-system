"""Corrupt and invalid files: each must be rejected with a stated reason — except the two look-alikes that must work.

The failing cases and the error code a correct system reports (magic bytes win over the extension):

=====================  ============================================  ======================
``truncated.pdf``      a real PDF cut short (header intact)          ``corrupt_file``
``random_bytes.pdf``   40 KB of noise named ``.pdf``                 ``corrupt_file``
``empty.pdf``          zero bytes                                    ``empty_file``
``encrypted.pdf``      RC4-128, non-empty user password              ``encrypted_file``
``truncated.xlsx``     a real workbook cut short (no central dir)    ``corrupt_file``
``truncated.png``      a real PNG cut inside its image data          ``corrupt_file``
``fake_invoice.pdf``   a Windows executable (``MZ`` header)          ``unsupported_file_type``
``session_expired.xlsx``  an HTML error page saved as a workbook     ``unsupported_file_type``
=====================  ============================================  ======================

Look-alikes that must *succeed*: ``scan_saved_as.pdf`` (a PNG: the type is sniffed from the bytes) and
``owner_locked.pdf`` (restricted by an owner password but openable with the empty user password).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CorruptCase:
    path: str  # relative to the dataset root
    expect: str  # success | failure
    error_code: str | None
    kind: str  # label kind: "corrupt" for failures, the true type for the look-alikes


CASES: tuple[CorruptCase, ...] = (
    CorruptCase("corrupt/truncated.pdf", "failure", "corrupt_file", "corrupt"),
    CorruptCase("corrupt/random_bytes.pdf", "failure", "corrupt_file", "corrupt"),
    CorruptCase("corrupt/empty.pdf", "failure", "empty_file", "corrupt"),
    CorruptCase("corrupt/encrypted.pdf", "failure", "encrypted_file", "corrupt"),
    CorruptCase("corrupt/truncated.xlsx", "failure", "corrupt_file", "corrupt"),
    CorruptCase("corrupt/truncated.png", "failure", "corrupt_file", "corrupt"),
    CorruptCase("corrupt/fake_invoice.pdf", "failure", "unsupported_file_type", "corrupt"),
    CorruptCase("corrupt/session_expired.xlsx", "failure", "unsupported_file_type", "corrupt"),
    CorruptCase("corrupt/scan_saved_as.pdf", "success", None, "png"),
    CorruptCase("corrupt/owner_locked.pdf", "success", None, "pdf"),
)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Session expired</title></head>
<body><h1>Your session has expired</h1>
<p>Please <a href="/login">sign in</a> again to download <b>invoices_2024.xlsx</b>.</p>
<script>window.setTimeout(function () {{ location.href = '/login?r={token}'; }}, 3000);</script>
</body></html>
"""


def truncate(data: bytes, fraction: float) -> bytes:
    """The first ``fraction`` of ``data`` (a download that stopped half way)."""
    return data[: max(int(len(data) * fraction), 16)]


def random_noise(rng: random.Random, size: int = 40_000) -> bytes:
    """Noise that does not start with any magic number the sniffer knows."""
    data = bytearray(rng.randbytes(size))
    data[0] = 0x01
    return bytes(data)


def fake_executable(rng: random.Random) -> bytes:
    """A DOS/Windows executable shell: ``MZ`` header, the classic stub message, then noise."""
    stub = b"This program cannot be run in DOS mode.\r\r\n$"
    return b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00" + stub + rng.randbytes(30_000)


def html_error_page(rng: random.Random) -> bytes:
    return HTML_PAGE.format(token=rng.randbytes(8).hex()).encode("utf-8")
