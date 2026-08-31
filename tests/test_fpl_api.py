"""Tests for the API client's cache and retry behaviour.

All offline: the HTTP session is replaced with a fake that records calls and
replays scripted responses. What is being tested is the client's own logic —
that it caches, that it honours TTLs, that it backs off and retries, and that a
stale cache is preferred to no data at all when the API is down.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.config import load_config
from src.data.fpl_api import FPLAPIError, FPLClient, _cache_key, _ttl_family


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Replays a scripted list of responses and counts requests."""

    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url, timeout=None):
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self.responses.pop(0)


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> FPLClient:
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # no real backoff waits
    config = load_config()
    return FPLClient(config=config, cache_dir=tmp_path / "api")


def test_cache_key_and_ttl_family() -> None:
    assert _cache_key("element-summary/123/") == "element__summary__123"
    assert _ttl_family("bootstrap-static/") == "bootstrap-static"
    assert _ttl_family("element-summary/5/") == "element-summary"
    assert _ttl_family("event/7/live/") == "event-live"
    assert _ttl_family("entry/1/history/") == "default"


def test_second_call_is_served_from_cache(client: FPLClient) -> None:
    client.session = FakeSession([FakeResponse(payload={"value": 1})])

    first = client.get("fixtures/")
    second = client.get("fixtures/")

    assert first == second == {"value": 1}
    assert len(client.session.calls) == 1, "the second call should not hit the network"


def test_expired_cache_is_refetched(client: FPLClient) -> None:
    client.session = FakeSession([FakeResponse(payload={"v": 1}), FakeResponse(payload={"v": 2})])

    assert client.get("fixtures/") == {"v": 1}
    assert client.get("fixtures/", ttl=0) == {"v": 2}, "ttl=0 must always refetch"


def test_force_bypasses_the_cache_but_still_writes_it(client: FPLClient) -> None:
    client.session = FakeSession([FakeResponse(payload={"v": 1}), FakeResponse(payload={"v": 2})])

    client.get("fixtures/")
    assert client.get("fixtures/", force=True) == {"v": 2}
    assert client.get("fixtures/") == {"v": 2}, "the forced response should be cached"


def test_retries_then_succeeds(client: FPLClient) -> None:
    client.session = FakeSession(
        [
            FakeResponse(status_code=503),
            FakeResponse(status_code=503),
            FakeResponse(payload={"recovered": True}),
        ]
    )

    assert client.get("bootstrap-static/") == {"recovered": True}
    assert len(client.session.calls) == 3


def test_retry_after_header_is_honoured(client: FPLClient, monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    client.session = FakeSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "7"}),
            FakeResponse(payload={"ok": True}),
        ]
    )

    client.get("fixtures/")
    assert slept == [7.0]


def test_backoff_grows_between_attempts(client: FPLClient, monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    client.session = FakeSession([FakeResponse(status_code=503) for _ in range(3)] + [FakeResponse()])

    client.get("fixtures/")
    assert len(slept) == 3
    # Jittered, so compare envelopes rather than exact values.
    assert slept[0] < slept[-1]


def test_stale_cache_is_used_when_the_api_is_down(client: FPLClient) -> None:
    client.session = FakeSession([FakeResponse(payload={"cached": True})])
    client.get("fixtures/")

    client.session = FakeSession([FakeResponse(status_code=503) for _ in range(client.max_retries)])
    # Expired for normal purposes, but better than nothing.
    assert client.get("fixtures/", ttl=0) == {"cached": True}


def test_raises_when_down_with_no_cache(client: FPLClient) -> None:
    client.session = FakeSession([FakeResponse(status_code=503) for _ in range(client.max_retries)])

    with pytest.raises(FPLAPIError):
        client.get("bootstrap-static/")


def test_corrupt_cache_entry_is_refetched(client: FPLClient) -> None:
    client.session = FakeSession([FakeResponse(payload={"v": 1})])
    client.get("fixtures/")

    client._cache_path("fixtures/").write_text("{not json", encoding="utf-8")

    client.session = FakeSession([FakeResponse(payload={"v": 2})])
    assert client.get("fixtures/") == {"v": 2}


def test_cache_writes_are_atomic(client: FPLClient) -> None:
    """No half-written .json is left behind for the next process to read."""
    client.session = FakeSession([FakeResponse(payload={"v": 1})])
    client.get("fixtures/")

    leftovers = list(client.cache_dir.glob("*.tmp"))
    assert not leftovers
    assert json.loads(client._cache_path("fixtures/").read_text(encoding="utf-8")) == {"v": 1}
