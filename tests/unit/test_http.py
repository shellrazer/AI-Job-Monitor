"""Unit tests for :class:`job_monitor.sources.http.PoliteClient`.

httpx traffic is mocked with respx. The rate limiter is driven by injected fake
``now`` / ``sleep`` callables so nothing actually sleeps, and curl_cffi is
monkeypatched (it cannot be respx-mocked).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from job_monitor.config import HttpSettings
from job_monitor.models import SourceBlocked
from job_monitor.sources.http import PoliteClient


def _settings(cache_dir: Path, **overrides) -> HttpSettings:
    base = {
        "cache_dir": str(cache_dir),
        "per_host_rate_limit_rps": 0.0,  # disable spacing unless a test wants it
        "max_retries": 4,
        "backoff_base_seconds": 0.0,  # keep tenacity waits at ~0 in tests
        "timeout_seconds": 5.0,
    }
    base.update(overrides)
    return HttpSettings(**base)


def _client(cache_dir: Path, **overrides) -> PoliteClient:
    # No-op sleep so retries / rate limiting never actually block.
    return PoliteClient(_settings(cache_dir, **overrides), sleep=lambda _s: None)


@respx.mock
def test_get_json_returns_parsed_dict(tmp_path: Path) -> None:
    route = respx.get("https://api.example.com/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [{"id": 1}], "total": 1})
    )
    client = _client(tmp_path)

    data = client.get_json("https://api.example.com/jobs")

    assert data == {"jobs": [{"id": 1}], "total": 1}
    assert route.called


@respx.mock
def test_retry_on_429_then_success(tmp_path: Path) -> None:
    route = respx.get("https://api.example.com/retry").mock(
        side_effect=[
            httpx.Response(429, text="slow down"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = _client(tmp_path)

    data = client.get_json("https://api.example.com/retry")

    assert data == {"ok": True}
    assert route.call_count == 2  # retried exactly once after the 429


@respx.mock
def test_persistent_403_raises_source_blocked(tmp_path: Path) -> None:
    route = respx.get("https://api.example.com/blocked").mock(
        return_value=httpx.Response(403, text="forbidden")
    )
    client = _client(tmp_path)

    with pytest.raises(SourceBlocked):
        client.get_text("https://api.example.com/blocked")
    assert route.called


@respx.mock
def test_cache_hits_network_once(tmp_path: Path) -> None:
    route = respx.get("https://api.example.com/cached").mock(
        return_value=httpx.Response(200, json={"v": 42})
    )
    client = _client(tmp_path)

    first = client.get_json("https://api.example.com/cached", use_cache=True)
    second = client.get_json("https://api.example.com/cached", use_cache=True)

    assert first == second == {"v": 42}
    assert route.call_count == 1  # second call served from disk cache


@respx.mock
def test_use_cache_false_always_hits_network(tmp_path: Path) -> None:
    route = respx.get("https://api.example.com/fresh").mock(
        return_value=httpx.Response(200, json={"v": 1})
    )
    client = _client(tmp_path)

    client.get_json("https://api.example.com/fresh", use_cache=False)
    client.get_json("https://api.example.com/fresh", use_cache=False)

    assert route.call_count == 2  # cache bypassed for healthchecks


@respx.mock
def test_clear_cache_forces_refetch(tmp_path: Path) -> None:
    route = respx.get("https://api.example.com/clear").mock(
        return_value=httpx.Response(200, json={"v": 1})
    )
    client = _client(tmp_path)

    client.get_json("https://api.example.com/clear")
    client.clear_cache()
    client.get_json("https://api.example.com/clear")

    assert route.call_count == 2


@respx.mock
def test_rate_limiter_spaces_same_host_requests(tmp_path: Path) -> None:
    respx.get("https://rate.example.com/a").mock(return_value=httpx.Response(200, json={}))
    respx.get("https://rate.example.com/b").mock(return_value=httpx.Response(200, json={}))

    # Fake monotonic clock that only advances when our fake sleep is called.
    fake_time = {"t": 1000.0}
    sleeps: list[float] = []

    def fake_now() -> float:
        return fake_time["t"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fake_time["t"] += seconds

    # 2 rps -> 0.5s minimum spacing between same-host requests.
    settings = _settings(tmp_path, per_host_rate_limit_rps=2.0)
    client = PoliteClient(settings, sleep=fake_sleep, now=fake_now)

    # use_cache=False so both requests really go through the rate limiter.
    client.get_json("https://rate.example.com/a", use_cache=False)
    client.get_json("https://rate.example.com/b", use_cache=False)

    # First request does not wait; second must sleep ~0.5s (the min interval).
    assert sleeps == pytest.approx([0.5])


def test_get_text_impersonate_returns_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCffiResponse:
        status_code = 200
        text = "<html>seek jobs</html>"

    calls: dict[str, object] = {}

    def fake_get(url, *, impersonate, headers, timeout):
        calls["url"] = url
        calls["impersonate"] = impersonate
        return _FakeCffiResponse()

    import curl_cffi.requests as cffi

    monkeypatch.setattr(cffi, "get", fake_get)
    client = _client(tmp_path, impersonate="chrome120")

    text = client.get_text_impersonate("https://www.seek.com.au/jobs")

    assert text == "<html>seek jobs</html>"
    assert calls["url"] == "https://www.seek.com.au/jobs"
    assert calls["impersonate"] == "chrome120"


def test_get_text_impersonate_403_raises_source_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeCffiResponse:
        status_code = 403
        text = "cloudflare blocked"

    def fake_get(url, *, impersonate, headers, timeout):
        return _FakeCffiResponse()

    import curl_cffi.requests as cffi

    monkeypatch.setattr(cffi, "get", fake_get)
    client = _client(tmp_path)

    with pytest.raises(SourceBlocked):
        client.get_text_impersonate("https://www.jora.com/jobs")
