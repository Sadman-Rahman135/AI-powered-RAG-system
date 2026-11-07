"""The one failure guarantee of every parser entry point: a failure is a typed :class:`IngestionError`.

On hostile input the parsing libraries raise many exception types (MuPDF ``RuntimeError``, zipfile
``NotImplementedError``, Pillow ``SyntaxError`` ...). :func:`typed_errors` maps anything unexpected to
:class:`CorruptFileError` (``MemoryError`` to :class:`FileTooLargeError`) and lets three kinds through unchanged:
ingestion errors, a stored file that is missing or unreadable (an environment problem, not the document's), and
whatever the caller's progress callback raised (wrap the callback with :func:`guard_progress`) - a progress store
that is down must not mark a good document as corrupt.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from app.ingestion.types import CorruptFileError, FileTooLargeError, IngestionError, ProgressCallback


class _CallbackError(Exception):
    """Carries an exception raised by the caller's progress callback through the parser untouched."""

    def __init__(self, original: Exception) -> None:
        super().__init__(repr(original))
        self.original = original


def guard_progress(callback: ProgressCallback | None) -> ProgressCallback | None:
    """Wrap ``callback`` so that :func:`typed_errors` re-raises its exceptions as they were."""
    if callback is None:
        return None

    def report(done: int, total: int, stage: str) -> None:
        try:
            callback(done, total, stage)
        except Exception as exc:
            raise _CallbackError(exc) from exc

    return report


@contextmanager
def typed_errors(what: str) -> Iterator[None]:
    """Map unexpected exceptions raised while parsing a ``what`` (e.g. "PDF") to typed ingestion errors."""
    try:
        yield
    except _CallbackError as exc:
        raise exc.original from None
    except (IngestionError, FileNotFoundError, PermissionError, IsADirectoryError):
        raise
    except MemoryError as exc:
        raise FileTooLargeError(f"the {what} is too large to process") from exc
    except Exception as exc:
        raise CorruptFileError(f"the {what} is damaged or malformed") from exc
