"""Task-local fixtures for the pipeline tests: schema creation, queue reset and canned extraction."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest

from app.db.session import init_db


@pytest.fixture(autouse=True)
def _schema_and_queue(request: pytest.FixtureRequest) -> Iterator[None]:
    if not request.node.get_closest_marker("integration"):  # PostgreSQL tests build a throw-away database
        init_db()
    yield
    from app.queue import reset_queue

    reset_queue()


CannedExtraction = Callable[[Sequence[Any]], list[Any]]


@pytest.fixture
def canned(monkeypatch: pytest.MonkeyPatch) -> CannedExtraction:
    """``canned(records)`` makes the processor's next extraction calls return ``records`` (the parsed document is
    still produced by the real ingestion layer). Returns the list of calls made so far."""
    from app.pipeline import processor

    calls: list[Any] = []
    current: list[Sequence[Any]] = [[]]

    def fake_extract_records(doc: Any, **kwargs: Any) -> list[Any]:
        calls.append((doc, kwargs))
        return list(current[0])

    monkeypatch.setattr(processor, "extract_records", fake_extract_records)

    def use(records: Sequence[Any]) -> list[Any]:
        current[0] = records
        return calls

    return use
