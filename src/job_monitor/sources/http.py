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
SEEK / Jora). A persistent ``403`` is surfaced as :class:`SourceBlocked`.
"""

from __future__ import annotations

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
