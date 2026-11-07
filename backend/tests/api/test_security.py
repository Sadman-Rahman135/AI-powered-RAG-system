"""Security layer: API keys, rate limits, security headers, CORS, request ids, the error envelope, startup checks."""

from __future__ import annotations

import logging
import uuid

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.security.ratelimit import RateLimiter
from tests.api.conftest import ClientFactory
from tests.api.factories import csv_bytes

KEY = "a-very-long-test-api-key-0123456789"
OTHER_KEY = "another-long-test-api-key-9876543210"


def test_auth_required_when_keys_set(make_client: ClientFactory) -> None:
    client = make_client(env={"API_KEYS": f"{KEY}, {OTHER_KEY}"})

    for headers in ({}, {"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {KEY}"}):
        response = client.get("/api/v1/documents", headers=headers)
        assert response.status_code == 401, headers
        assert response.json()["error"]["code"] == "unauthorized"
        assert response.headers["www-authenticate"] == "Bearer"

    assert client.get("/api/v1/documents", headers={"X-API-Key": KEY}).status_code == 200
    assert client.get("/api/v1/documents", headers={"Authorization": f"Bearer {OTHER_KEY}"}).status_code == 200
    assert client.get("/api/v1/documents", headers={"authorization": f"bearer {KEY}"}).status_code == 200
    assert client.get("/api/v1/documents", headers={"X-API-Key": "ключ".encode()}).status_code == 401  # non-ASCII

    # every route under /api/v1 is protected except the liveness probe
    assert client.get("/api/v1/health").status_code == 200
    protected = [
        ("GET", "/api/v1/ready"),
        ("GET", "/api/v1/config"),
        ("GET", "/api/v1/stats"),
        ("GET", "/api/v1/search?q=x"),
        ("GET", "/api/v1/jobs"),
        ("GET", "/api/v1/invoices"),
        ("POST", "/api/v1/jobs/reprocess-failed"),
        ("GET", "/api/v1/cross-verifications"),
        ("POST", "/api/v1/cross-verify"),
        ("DELETE", f"/api/v1/documents/{uuid.uuid4()}"),
    ]
    for method, path in protected:
        assert client.request(method, path).status_code == 401, path
    upload = client.post("/api/v1/documents", files=[("files", ("a.csv", csv_bytes(), "text/csv"))])
    assert upload.status_code == 401
    assert client.get("/api/v1/config", headers={"X-API-Key": KEY}).json()["auth_required"] is True


def test_no_auth_when_keys_unset(client: TestClient) -> None:
    assert client.get("/api/v1/documents").status_code == 200
    assert client.get("/api/v1/config").json()["auth_required"] is False


def test_api_keys_never_reach_the_logs(make_client: ClientFactory, caplog: pytest.LogCaptureFixture) -> None:
    client = make_client(env={"API_KEYS": KEY})
    with caplog.at_level(logging.DEBUG):
        client.get("/api/v1/documents", headers={"X-API-Key": KEY})
        client.get("/api/v1/documents", headers={"X-API-Key": KEY + "x"})
    assert caplog.records
    assert KEY not in caplog.text


def test_rate_limit_429(make_client: ClientFactory) -> None:
    client = make_client(env={"RATE_LIMIT_ENABLED": "true", "RATE_LIMIT_DEFAULT": "2/minute",
                              "RATE_LIMIT_SEARCH": "1/minute", "API_KEYS": f"{KEY},{OTHER_KEY}"})  # fmt: skip
    mine = {"X-API-Key": KEY}

    assert [client.get("/api/v1/jobs", headers=mine).status_code for _ in range(2)] == [200, 200]
    limited = client.get("/api/v1/jobs", headers=mine)
    assert limited.status_code == 429
    error = limited.json()["error"]
    assert error["code"] == "rate_limited"
    assert 0 < int(limited.headers["retry-after"]) <= 60
    assert error["details"]["retry_after_s"] == int(limited.headers["retry-after"])

    # the limit is per API key, and search has its own budget
    assert client.get("/api/v1/jobs", headers={"X-API-Key": OTHER_KEY}).status_code == 200
    assert client.get("/api/v1/search", params={"q": "x"}, headers=mine).status_code == 200
    assert client.get("/api/v1/search", params={"q": "x"}, headers=mine).status_code == 429
    assert client.get("/api/v1/health").status_code == 200  # the liveness probe is never limited


def test_rate_limit_counts_unauthenticated_attempts_per_ip(make_client: ClientFactory) -> None:
    client = make_client(env={"RATE_LIMIT_ENABLED": "true", "RATE_LIMIT_DEFAULT": "3/minute", "API_KEYS": KEY})

    codes = [client.get("/api/v1/jobs", headers={"X-API-Key": f"guess-{i}"}).status_code for i in range(4)]
    assert codes == [401, 401, 401, 429]
    assert client.get("/api/v1/jobs", headers={"X-API-Key": KEY}).status_code == 200


def test_rate_limit_gives_each_proxy_user_its_own_budget(make_client: ClientFactory) -> None:
    client = make_client(env={"RATE_LIMIT_ENABLED": "true", "RATE_LIMIT_DEFAULT": "2/minute", "API_KEYS": KEY})
    alice, bob = ({"X-API-Key": KEY, "X-Actor": name} for name in ("alice", "bob"))

    assert [client.get("/api/v1/jobs", headers=alice).status_code for _ in range(3)] == [200, 200, 429]
    assert client.get("/api/v1/jobs", headers=bob).status_code == 200  # another UI user behind the same key
    assert client.get("/api/v1/jobs", headers={"X-API-Key": KEY}).status_code == 200  # the key without an actor
    assert client.get("/api/v1/jobs", headers={"X-API-Key": KEY, "X-Actor": "alice\x7f"}).status_code == 429

    # without a valid key, X-Actor is not trusted: failed attempts stay counted per client IP
    codes = [client.get("/api/v1/jobs", headers={"X-API-Key": "wrong", "X-Actor": f"u{i}"}).status_code
             for i in range(3)]  # fmt: skip
    assert codes == [401, 401, 429]


def test_rate_limit_ignores_x_actor_when_auth_is_off(make_client: ClientFactory) -> None:
    client = make_client(env={"RATE_LIMIT_ENABLED": "true", "RATE_LIMIT_DEFAULT": "2/minute"})

    codes = [client.get("/api/v1/jobs", headers={"X-Actor": f"user-{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limit_upload_budget(make_client: ClientFactory) -> None:
    client = make_client(env={"RATE_LIMIT_ENABLED": "true", "RATE_LIMIT_UPLOAD": "1/minute"})
    files = [("files", ("a.csv", csv_bytes(), "text/csv"))]

    assert client.post("/api/v1/documents", files=files).status_code == 202
    assert client.post("/api/v1/documents", files=files).status_code == 429
    assert client.get("/api/v1/documents").status_code == 200


def test_rate_limit_disabled_by_default_in_tests(client: TestClient) -> None:
    assert all(client.get("/api/v1/jobs").status_code == 200 for _ in range(30))


def test_security_headers_present(client: TestClient) -> None:
    paths = ("/api/v1/health", "/api/v1/documents", f"/api/v1/documents/{uuid.uuid4()}", "/nowhere")
    for response in (client.get(path) for path in paths):
        headers = response.headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert client.get("/api/v1/documents").headers["cache-control"] == "no-store"

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "default-src 'none'" not in docs.headers.get("content-security-policy", "")
    assert docs.headers["x-frame-options"] == "DENY"
    assert client.get("/openapi.json").json()["info"]["title"]


def test_cors_allow_list(make_client: ClientFactory) -> None:
    client = make_client(env={"CORS_ORIGINS": "http://ui.example"})
    preflight = {"Access-Control-Request-Method": "PATCH", "Access-Control-Request-Headers": "x-api-key,content-type"}

    allowed = client.options("/api/v1/invoices/x", headers={"Origin": "http://ui.example", **preflight})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://ui.example"
    assert "access-control-allow-credentials" not in allowed.headers

    denied = client.options("/api/v1/invoices/x", headers={"Origin": "http://evil.example", **preflight})
    assert "access-control-allow-origin" not in denied.headers
    simple = client.get("/api/v1/documents", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in simple.headers


def test_request_id_header(client: TestClient) -> None:
    generated = client.get("/api/v1/health").headers["x-request-id"]
    assert uuid.UUID(generated)
    assert (
        client.get("/api/v1/health", headers={"X-Request-ID": "abc-123.DEF"}).headers["x-request-id"] == "abc-123.DEF"
    )
    unsafe = client.get("/api/v1/health", headers={"X-Request-ID": "bad id\twith spaces" + "x" * 300})
    assert unsafe.headers["x-request-id"] != "bad id\twith spaces"
    assert uuid.UUID(unsafe.headers["x-request-id"])


def test_errors_use_the_envelope(client: TestClient) -> None:
    missing = client.get("/api/v1/nothing-here")
    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "not_found", "message": "Not Found", "details": None}}

    wrong_method = client.put("/api/v1/health")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "method_not_allowed"

    bad_json = client.patch(
        f"/api/v1/invoices/{uuid.uuid4()}", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert bad_json.status_code == 422
    assert bad_json.json()["error"]["code"] == "validation_error"


def test_json_body_size_is_limited(client: TestClient) -> None:
    huge = b'{"comment": "' + b"x" * (2 * 1024 * 1024) + b'"}'
    response = client.post(
        f"/api/v1/invoices/{uuid.uuid4()}/approve", content=huge, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_unexpected_errors_are_500_envelopes(make_client: ClientFactory, caplog: pytest.LogCaptureFixture) -> None:
    client = make_client()
    router = APIRouter()

    @router.get("/api/v1/_boom")
    def boom() -> None:
        raise RuntimeError("secret invoice text 1234.50")

    client.app.include_router(router)
    with caplog.at_level(logging.ERROR):
        response = client.get("/api/v1/_boom", headers={"X-Request-ID": "req-500"})

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "secret" not in response.text
    assert body["error"]["details"] == {"request_id": "req-500"}
    assert response.headers["x-request-id"] == "req-500"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert any(getattr(record, "request_id", None) == "req-500" or "req-500" in record.getMessage()
               for record in caplog.records)  # fmt: skip


def test_production_refuses_to_start_with_problems(make_client: ClientFactory) -> None:
    with pytest.raises(RuntimeError, match="API_KEYS"):
        make_client(env={"APP_ENV": "production", "API_KEYS": ""})


def test_openapi_lists_the_api(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    for path in ("/api/v1/documents", "/api/v1/documents/{document_id}", "/api/v1/documents/{document_id}/file",
                 "/api/v1/invoices/{invoice_id}", "/api/v1/search", "/api/v1/cross-verify", "/api/v1/stats",
                 "/api/v1/jobs/reprocess-failed", "/api/v1/health", "/api/v1/ready", "/api/v1/config"):  # fmt: skip
        assert path in paths, path
    upload = paths["/api/v1/documents"]["post"]
    assert "multipart/form-data" in upload["requestBody"]["content"]


def test_rate_limiter_fails_open_when_storage_is_down(
    make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = make_client(env={"RATE_LIMIT_ENABLED": "true", "RATE_LIMIT_DEFAULT": "1/minute"})

    def unreachable(_self: RateLimiter) -> None:
        raise ConnectionError("redis is down")

    monkeypatch.setattr(RateLimiter, "_limiter", unreachable)
    with caplog.at_level(logging.WARNING):
        assert [client.get("/api/v1/jobs").status_code for _ in range(3)] == [200, 200, 200]
    assert "rate_limiter_unavailable" in caplog.text
