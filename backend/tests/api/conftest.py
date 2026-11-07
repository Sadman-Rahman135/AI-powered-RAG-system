"""Fixtures for the API tests.

Every test gets a fresh app from ``create_app()`` (the root conftest already isolates the database, storage, queue
backend, embedder and LLM). Most tests replace the job queue with ``RecordingQueue`` through FastAPI's dependency
overrides, so uploads stay ``queued`` and the test sees which jobs were enqueued; the full-flow test uses the real
inline queue. ``make_client(env=...)`` builds a client after applying extra environment settings.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.config import reset_settings_cache


@dataclass
class RecordingQueue:
    """A ``JobQueue`` that only records what was enqueued."""

    name: str = "recording"
    job_ids: list[UUID] = field(default_factory=list)
    healthy: bool = True

    def enqueue(self, job_id: UUID) -> None:
        self.job_ids.append(job_id)

    def ping(self) -> bool:
        return self.healthy


ClientFactory = Callable[..., TestClient]


@pytest.fixture
def queue() -> RecordingQueue:
    return RecordingQueue()


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch, queue: RecordingQueue) -> Iterator[ClientFactory]:
    from app.api.deps import get_job_queue
    from app.main import create_app

    opened: list[TestClient] = []

    def factory(
        *, env: dict[str, str] | None = None, real_queue: bool = False, raise_server_exceptions: bool = True
    ) -> TestClient:
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        reset_settings_cache()
        app = create_app()
        if not real_queue:
            app.dependency_overrides[get_job_queue] = lambda: queue
        client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
        client.__enter__()
        opened.append(client)
        return client

    yield factory
    for client in opened:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    return make_client()
