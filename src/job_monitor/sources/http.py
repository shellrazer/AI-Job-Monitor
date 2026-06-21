"""Polite HTTP client: on-disk caching, per-host rate limiting, and retries.

Every adapter talks to the network through :class:`PoliteClient`. It centralises
the concerns that would otherwise be re-implemented (badly) in each adapter:

* an on-disk response cache keyed by the request, with a TTL;
* a per-host rate limiter that enforces a minimum spacing between requests so we
  stay a polite neighbour;
* tenacity-based retry on transient failures (timeouts, connection errors, and
  HTTP 429/5xx) with exponential backoff + jitter.

Two transports are exposed: plain ``httpx`` (for JSON ATS APIs and ordinary HTML)
and ``curl_cffi`` browser impersonation (for Cloudflare-fronted sites such as
SEEK / Jora). An optional third path renders JavaScript-heavy pages through a
headless Playwright browser (the ``browser`` extra). A persistent ``403`` is
surfaced as :class:`SourceBlocked`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from job_monitor.config import HttpSettings, expand_path
from job_monitor.models import SourceBlocked

# HTTP status codes that warrant a retry (transient server / throttling errors).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Fallback user agent for the Playwright transport when the configured one looks
# like a bot (a realistic browser UA helps render-only / anti-bot pages).
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# Cache key tag for the Playwright render transport.
_RENDER_TAG = "RENDER"

# Generous wait (ms) for a challenge interstitial (e.g. Cloudflare "Just a
# moment") to auto-solve and navigate to the real page before we look for the
# requested ``wait_selector``.
_CHALLENGE_WAIT_MS = 25_000

# Short settle wait (ms) after networkidle when no wait_selector is supplied,
# giving late client-side rendering a moment to finish.
_SETTLE_WAIT_MS = 3_000

# Substrings that mark the final content as still a challenge / block page.
# Lower-cased; matched case-insensitively against the rendered HTML.
_CHALLENGE_MARKERS = (
    "just a moment",
    "cf-challenge",
    "attention required",
    "access denied",
    "request unsuccessful",
)


def _looks_like_challenge(html: str) -> bool:
    """Return True when ``html`` still looks like a challenge / block page."""
    lowered = html.lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


class _RetryableStatus(Exception):
    """Internal signal that a response status should trigger a tenacity retry."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"retryable HTTP status {status_code}")
        self.status_code = status_code


def _should_retry(exc: BaseException) -> bool:
    """Return True for exceptions tenacity should retry on."""
    if isinstance(exc, _RetryableStatus):
        return True
    # httpx transport-level failures: timeouts and connection errors.
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


class _CachedResponse:
    """Minimal response-like object returned from the cache or built per request.

    Carries just enough surface (``status_code`` and ``text``) for the rest of
    the client; ``json()`` parses the body lazily.
    """

    __slots__ = ("status_code", "text")

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        return json.loads(self.text)


class PoliteClient:
    """HTTP client with caching, per-host rate limiting, and retries.

    Construct once per run and share it across adapters. The ``sleep`` / ``now``
    callables are injectable so the rate limiter can be driven by a fake clock in
    tests (no real sleeping).
    """

    def __init__(
        self,
        settings: HttpSettings,
        *,
        sleep: Callable[[float], None] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self._sleep = sleep or time.sleep
        self._now = now or time.monotonic

        self._cache_dir = expand_path(settings.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Per-host last-request timestamps (monotonic seconds) for rate limiting.
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

        rps = settings.per_host_rate_limit_rps
        self._min_interval = (1.0 / rps) if rps and rps > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def get_json(
        self,
        url: str,
        *,
        method: str = "GET",
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Fetch ``url`` and return the parsed JSON body (used by ATS adapters).

        Raises :class:`SourceBlocked` on a persistent ``403``.
        """
        body_bytes = json.dumps(json_body, sort_keys=True).encode("utf-8") if json_body is not None else None
        resp = self._request(
            method=method,
            url=url,
            json_body=json_body,
            headers=headers,
            use_cache=use_cache,
            cache_body=body_bytes,
            transport="httpx",
        )
        return resp.json()

    def get_text(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> str:
        """Fetch ``url`` and return the response text via httpx.

        Raises :class:`SourceBlocked` on a persistent ``403``.
        """
        resp = self._request(
            method="GET",
            url=url,
            json_body=None,
            headers=headers,
            use_cache=use_cache,
            cache_body=None,
            transport="httpx",
        )
        return resp.text

    def get_text_impersonate(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> str:
        """Fetch ``url`` via curl_cffi browser impersonation (Cloudflare sites).

        Raises :class:`SourceBlocked` on a persistent ``403``.
        """
        resp = self._request(
            method="GET",
            url=url,
            json_body=None,
            headers=headers,
            use_cache=use_cache,
            cache_body=b"impersonate",
            transport="cffi",
        )
        return resp.text

    def get_text_rendered(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        wait_selector: str | None = None,
        wait_until: str = "networkidle",
        use_cache: bool = True,
    ) -> str:
        """Render ``url`` in a headless browser and return the post-JS HTML.

        Uses Playwright (the optional ``browser`` extra) for pages whose listings
        only materialise after client-side JavaScript runs. Shares the same
        on-disk cache and per-host rate limiter as the other transports, keyed by
        a dedicated ``RENDER`` tag.

        ``wait_until`` is the navigation lifecycle event to await (e.g.
        ``"networkidle"``, ``"load"``, ``"domcontentloaded"``). When
        ``wait_selector`` is given the call waits for that selector but swallows a
        timeout, returning whatever rendered so far.

        The navigation status alone is NOT treated as a block: Cloudflare's "Just
        a moment" interstitial returns ``403`` first, then auto-solves and
        navigates to the real page. :class:`SourceBlocked` is raised only when the
        FINAL rendered content still looks like a challenge / block page AND lacks
        the requested ``wait_selector``.

        Raises :class:`RuntimeError` if Playwright is not installed, and
        :class:`SourceBlocked` only on a persistent challenge / block page.
        """
        cache_key = self._cache_key("GET", url, _RENDER_TAG.encode("utf-8"), _RENDER_TAG)

        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached.text

        self._respect_rate_limit(url)

        html = self._render_with_playwright(
            url,
            headers=headers,
            wait_selector=wait_selector,
            wait_until=wait_until,
        )

        if use_cache:
            self._cache_put(cache_key, _CachedResponse(200, html))
        return html

    def clear_cache(self) -> None:
        """Delete every cached response file."""
        for path in self._cache_dir.glob("*.json"):
            path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Core request flow                                                  #
    # ------------------------------------------------------------------ #
    def _request(
        self,
        *,
        method: str,
        url: str,
        json_body: Any | None,
        headers: dict[str, str] | None,
        use_cache: bool,
        cache_body: bytes | None,
        transport: str,
    ) -> _CachedResponse:
        cache_key = self._cache_key(method, url, cache_body, transport)

        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        self._respect_rate_limit(url)

        resp = self._fetch_with_retry(
            method=method,
            url=url,
            json_body=json_body,
            headers=headers,
            transport=transport,
        )

        if use_cache:
            self._cache_put(cache_key, resp)
        return resp

    def _fetch_with_retry(
        self,
        *,
        method: str,
        url: str,
        json_body: Any | None,
        headers: dict[str, str] | None,
        transport: str,
    ) -> _CachedResponse:
        """Issue the request, retrying transient failures with backoff + jitter."""
        attempts = max(1, self.settings.max_retries)
        retryer = Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=self.settings.backoff_base_seconds),
            retry=retry_if_exception(_should_retry),
            sleep=self._sleep,
            reraise=True,
            before_sleep=self._before_sleep,
        )
        try:
            return retryer(
                self._do_fetch,
                method=method,
                url=url,
                json_body=json_body,
                headers=headers,
                transport=transport,
            )
        except _RetryableStatus as exc:  # exhausted retries on a 429/5xx
            raise httpx.HTTPError(f"{exc} (after {attempts} attempts) for {url}") from exc

    def _do_fetch(
        self,
        *,
        method: str,
        url: str,
        json_body: Any | None,
        headers: dict[str, str] | None,
        transport: str,
    ) -> _CachedResponse:
        """One network attempt. Raises ``_RetryableStatus`` on a 429/5xx."""
        if transport == "cffi":
            resp = self._fetch_cffi(url, headers)
        else:
            resp = self._fetch_httpx(method, url, json_body, headers)

        status = resp.status_code
        if status == 403:
            raise SourceBlocked(f"403 Forbidden for {url}")
        if status in _RETRYABLE_STATUS:
            raise _RetryableStatus(status)
        return _CachedResponse(status, resp.text)

    def _fetch_httpx(
        self,
        method: str,
        url: str,
        json_body: Any | None,
        headers: dict[str, str] | None,
    ) -> Any:
        request_headers = self._headers(headers)
        with httpx.Client(timeout=self.settings.timeout_seconds, follow_redirects=True) as client:
            return client.request(method, url, json=json_body, headers=request_headers)

    def _fetch_cffi(self, url: str, headers: dict[str, str] | None) -> Any:
        # Imported lazily: curl_cffi pulls in a native extension we only need for
        # the Cloudflare-fronted boards.
        from curl_cffi import requests as cffi

        request_headers = self._headers(headers)
        # ``impersonate`` is a plain str in config; curl_cffi types it as a
        # Literal of fingerprint names, so narrow for the type checker.
        return cffi.get(
            url,
            impersonate=self.settings.impersonate,  # type: ignore[arg-type]
            headers=request_headers,
            timeout=self.settings.timeout_seconds,
        )

    def _render_with_playwright(
        self,
        url: str,
        *,
        headers: dict[str, str] | None,
        wait_selector: str | None,
        wait_until: str,
    ) -> str:
        """Drive a headless chromium to fetch the fully rendered HTML for ``url``.

        Tolerant of anti-bot interstitials: navigation always uses
        ``domcontentloaded`` (so we get control back as soon as the document is
        parsed, even mid-challenge), then we wait for the requested selector — or
        networkidle plus a short settle — to let a Cloudflare-style challenge
        auto-solve before we read the page. ``wait_until`` is accepted for API
        compatibility but the navigation lifecycle is fixed at
        ``domcontentloaded`` here.
        """
        # Imported lazily so the package never hard-requires Playwright; it is an
        # optional dependency declared as the ``browser`` extra.
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise RuntimeError(
                "Playwright not installed; run "
                "`uv sync --extra browser && uv run playwright install chromium`"
            ) from exc

        # Use the configured UA if it looks browser-like, else a realistic one;
        # render-only pages are usually behind some anti-bot heuristic.
        configured_ua = self.settings.user_agent
        user_agent = configured_ua if "Mozilla" in configured_ua else _BROWSER_USER_AGENT
        timeout_ms = int(self.settings.timeout_seconds * 1000)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            try:
                context = browser.new_context(
                    user_agent=user_agent,
                    viewport={"width": 1366, "height": 900},
                    locale="en-AU",
                    extra_http_headers=headers or {},
                )
                page = context.new_page()

                # Do NOT bail on the navigation response status: Cloudflare's
                # interstitial returns 403 first, then JS solves it and navigates
                # to the real page. We classify on the FINAL content instead.
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

                if wait_selector:
                    # Generous wait so a challenge has time to clear and the real
                    # listings to appear. Swallow a timeout: classify on content.
                    with contextlib.suppress(Exception):
                        page.wait_for_selector(wait_selector, timeout=_CHALLENGE_WAIT_MS)
                else:
                    # No selector to anchor on: best-effort networkidle, then a
                    # short settle for late client-side rendering.
                    with contextlib.suppress(Exception):
                        page.wait_for_load_state("networkidle", timeout=_CHALLENGE_WAIT_MS)
                    page.wait_for_timeout(_SETTLE_WAIT_MS)

                html = page.content()
            finally:
                browser.close()

        # Only treat it as blocked if the FINAL content still looks like a
        # challenge / block page AND the requested selector never appeared.
        selector_present = bool(wait_selector) and self._html_has_selector(html, wait_selector)
        if _looks_like_challenge(html) and not selector_present:
            raise SourceBlocked(f"challenge/block page persisted for {url}")
        return html

    @staticmethod
    def _html_has_selector(html: str, selector: str | None) -> bool:
        """Return True when ``selector`` matches anything in ``html``.

        Used to decide whether a challenge marker in the page is benign (the real
        listings are present too) or terminal. Defensive: a malformed selector or
        parse failure yields ``False``.
        """
        if not selector:
            return False
        try:
            from selectolax.parser import HTMLParser

            return HTMLParser(html).css_first(selector) is not None
        except Exception:
            return False

    def _headers(self, headers: dict[str, str] | None) -> dict[str, str]:
        merged = {"User-Agent": self.settings.user_agent}
        if headers:
            merged.update(headers)
        return merged

    # ------------------------------------------------------------------ #
    # Rate limiting                                                      #
    # ------------------------------------------------------------------ #
    def _respect_rate_limit(self, url: str) -> None:
        """Sleep, if needed, so same-host requests are spaced by min interval."""
        if self._min_interval <= 0:
            return
        host = urlsplit(url).hostname or ""
        with self._lock:
            last = self._last_request.get(host)
            now = self._now()
            if last is not None:
                elapsed = now - last
                wait = self._min_interval - elapsed
                if wait > 0:
                    self._sleep(wait)
                    now = self._now()
            self._last_request[host] = now

    def _before_sleep(self, retry_state: RetryCallState) -> None:
        # Hook kept for observability; intentionally a no-op for now.
        return None

    # ------------------------------------------------------------------ #
    # On-disk cache                                                      #
    # ------------------------------------------------------------------ #
    def _cache_key(self, method: str, url: str, body: bytes | None, transport: str) -> str:
        h = hashlib.sha256()
        h.update(method.upper().encode("utf-8"))
        h.update(b"\x00")
        h.update(transport.encode("utf-8"))
        h.update(b"\x00")
        h.update(url.encode("utf-8"))
        h.update(b"\x00")
        if body:
            h.update(body)
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> _CachedResponse | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        ts = payload.get("timestamp", 0.0)
        ttl = self.settings.cache_ttl_seconds
        if ttl >= 0 and (time.time() - ts) > ttl:
            return None
        return _CachedResponse(int(payload["status_code"]), payload["text"])

    def _cache_put(self, key: str, resp: _CachedResponse) -> None:
        payload = {
            "status_code": resp.status_code,
            "text": resp.text,
            "timestamp": time.time(),
        }
        tmp = self._cache_path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._cache_path(key))
