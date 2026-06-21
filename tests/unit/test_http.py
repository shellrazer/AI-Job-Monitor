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


# --------------------------------------------------------------------------- #
# get_text_rendered (Playwright)                                              #
# --------------------------------------------------------------------------- #
def test_get_text_rendered_missing_playwright_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the lazy Playwright import fails we surface a clear install hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "playwright.sync_api" or name.startswith("playwright"):
            raise ImportError("No module named 'playwright'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    client = _client(tmp_path)

    with pytest.raises(RuntimeError, match="Playwright not installed"):
        client.get_text_rendered("https://example.com/jobs", use_cache=False)


# --------------------------------------------------------------------------- #
# Render challenge-tolerance: classify on FINAL content, not the nav status    #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status


class _FakePage:
    """Minimal Playwright page stand-in driven by a scripted final HTML."""

    def __init__(self, *, goto_status: int, content: str, selector_found: bool) -> None:
        self._goto_status = goto_status
        self._content = content
        self._selector_found = selector_found
        self.goto_calls: list[dict[str, object]] = []
        self.waited_selector: str | None = None
        self.waited_timeout: int | None = None

    def goto(self, url, *, wait_until, timeout):
        self.goto_calls.append({"url": url, "wait_until": wait_until, "timeout": timeout})
        return _FakeResponse(self._goto_status)

    def wait_for_selector(self, selector, *, timeout):
        self.waited_selector = selector
        self.waited_timeout = timeout
        if not self._selector_found:
            raise TimeoutError("selector never appeared")

    def wait_for_load_state(self, state, *, timeout):
        return None

    def wait_for_timeout(self, ms):
        return None

    def content(self) -> str:
        return self._content


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self._page = page
        self.kwargs: dict[str, object] | None = None

    def new_page(self) -> _FakePage:
        return self._page


class _FakeBrowser:
    def __init__(self, context: _FakeContext) -> None:
        self._context = context
        self.launch_args: list[str] | None = None

    def new_context(self, **kwargs):
        self._context.kwargs = kwargs
        return self._context

    def close(self) -> None:
        return None


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser

    def launch(self, *, headless, args=None):
        self._browser.launch_args = args
        return self._browser


class _FakePlaywright:
    def __init__(self, browser: _FakeBrowser) -> None:
        self.chromium = _FakeChromium(browser)

    def __enter__(self) -> _FakePlaywright:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch,
    *,
    goto_status: int,
    content: str,
    selector_found: bool,
) -> _FakePage:
    """Inject a fake ``playwright.sync_api`` module returning scripted content."""
    import sys
    import types

    page = _FakePage(goto_status=goto_status, content=content, selector_found=selector_found)
    browser = _FakeBrowser(_FakeContext(page))

    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = lambda: _FakePlaywright(browser)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)
    return page


def test_rendered_403_nav_status_does_not_raise_when_selector_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 403 navigation status must NOT raise: Cloudflare returns 403 then solves.

    With the real listings present in the final content, the page is returned.
    """
    final_html = "<html><body><a href='/job/1'>Quality Manager</a></body></html>"
    page = _install_fake_playwright(
        monkeypatch, goto_status=403, content=final_html, selector_found=True
    )
    client = _client(tmp_path)

    html = client.get_text_rendered(
        "https://example.com/jobs", wait_selector="a[href*='/job/']", use_cache=False
    )

    assert html == final_html
    # Navigation used domcontentloaded so we regain control mid-challenge.
    assert page.goto_calls[0]["wait_until"] == "domcontentloaded"
    assert page.waited_selector == "a[href*='/job/']"


def test_rendered_persistent_challenge_raises_source_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final content still a challenge page AND selector absent -> SourceBlocked."""
    challenge_html = "<html><head><title>Just a moment...</title></head><body>cf-challenge</body></html>"
    _install_fake_playwright(
        monkeypatch, goto_status=403, content=challenge_html, selector_found=False
    )
    client = _client(tmp_path)

    with pytest.raises(SourceBlocked):
        client.get_text_rendered(
            "https://example.com/jobs", wait_selector="a[href*='/job/']", use_cache=False
        )


def test_rendered_challenge_marker_benign_when_selector_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A challenge marker is benign if the real listings rendered alongside it."""
    mixed_html = (
        "<html><body><noscript>Access Denied</noscript>"
        "<a href='/job/9'>Site Quality Manager</a></body></html>"
    )
    _install_fake_playwright(
        monkeypatch, goto_status=200, content=mixed_html, selector_found=True
    )
    client = _client(tmp_path)

    html = client.get_text_rendered(
        "https://example.com/jobs", wait_selector="a[href*='/job/']", use_cache=False
    )
    assert html == mixed_html


def test_rendered_uses_realistic_browser_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chromium launches with the automation flag off and an AU browser context."""
    page = _install_fake_playwright(
        monkeypatch, goto_status=200, content="<html><body>ok</body></html>", selector_found=False
    )
    client = _client(tmp_path)

    client.get_text_rendered("https://example.com/", use_cache=False)

    # No wait_selector -> networkidle + settle path; clean content -> returned.
    assert page.goto_calls[0]["wait_until"] == "domcontentloaded"


@pytest.mark.slow
def test_get_text_rendered_against_file_url(tmp_path: Path) -> None:
    """Render a local file:// page with real Playwright (no network).

    Skips cleanly when Playwright / chromium are unavailable in the environment.
    """
    page = tmp_path / "page.html"
    marker = "RENDERED-MARKER-12345"
    page.write_text(
        f"<html><body><h1 id='title'>{marker}</h1></body></html>",
        encoding="utf-8",
    )
    file_url = page.as_uri()

    client = _client(tmp_path)

    try:
        html = client.get_text_rendered(file_url, wait_selector="#title")
    except RuntimeError as exc:  # Playwright not installed
        pytest.skip(str(exc))
    except Exception as exc:  # chromium binary missing / launch failure
        pytest.skip(f"Playwright browser unavailable: {exc}")

    assert marker in html

    # Second call is served from the on-disk cache (no second browser launch).
    cached = client.get_text_rendered(file_url, wait_selector="#title")
    assert cached == html
