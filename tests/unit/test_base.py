"""Unit tests for :class:`job_monitor.sources.base.BaseAdapter`.

A tiny concrete adapter exercises the concrete ``healthcheck`` /
``search_terms_from_company`` behaviour without any network access.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from job_monitor.config import CompanyConfig, Settings
from job_monitor.models import RawJob, Source, SourceBlocked
from job_monitor.sources.base import DEFAULT_SEARCH_TERMS, BaseAdapter


class _StubHttp:
    """Stand-in for PoliteClient; records the payload ``fetch`` will parse."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload


class _DemoAdapter(BaseAdapter):
    name: ClassVar[str] = "demo"
    source: ClassVar[Source] = Source.OFFICIAL_ATS

    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        # Pull a payload off the stub http and run it through the pure parser.
        return self.parse(self.http.payload)

    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        return [
            RawJob(
                source=self.source,
                title=item["title"],
                company_name="Demo Co",
                apply_url=item["url"],
            )
            for item in payload
        ]


def _company(search: dict | None = None) -> CompanyConfig:
    return CompanyConfig(
        company_id="demo",
        name="Demo Co",
        adapter="demo",
        search=search or {},
    )


def _make_adapter(http: Any, company: CompanyConfig | None = None) -> _DemoAdapter:
    return _DemoAdapter(http=http, company=company or _company(), settings=Settings())


def test_healthcheck_ok_when_jobs_returned() -> None:
    http = _StubHttp([{"title": "Quality Manager", "url": "https://x/1"}])
    adapter = _make_adapter(http)

    health = adapter.healthcheck()

    assert health.status == "ok"
    assert health.ok is True
    assert health.job_count == 1
    assert health.name == "demo"
    assert health.source is Source.OFFICIAL_ATS
    assert health.error is None
    assert health.latency_ms >= 0.0


def test_healthcheck_empty_when_no_jobs() -> None:
    adapter = _make_adapter(_StubHttp([]))

    health = adapter.healthcheck()

    assert health.status == "empty"
    assert health.ok is False
    assert health.job_count == 0


def test_healthcheck_blocked_when_fetch_raises_source_blocked() -> None:
    class _BlockedAdapter(_DemoAdapter):
        def fetch(self, search_terms: list[str]) -> list[RawJob]:
            raise SourceBlocked("403 from cloudflare")

    adapter = _BlockedAdapter(http=_StubHttp(None), company=_company(), settings=Settings())

    health = adapter.healthcheck()

    assert health.status == "blocked"
    assert health.ok is False
    assert health.job_count == 0
    assert "cloudflare" in (health.error or "")


def test_healthcheck_error_on_other_exception() -> None:
    class _BoomAdapter(_DemoAdapter):
        def fetch(self, search_terms: list[str]) -> list[RawJob]:
            raise ValueError("schema changed")

    adapter = _BoomAdapter(http=_StubHttp(None), company=_company(), settings=Settings())

    health = adapter.healthcheck()

    assert health.status == "error"
    assert health.ok is False
    assert "schema changed" in (health.error or "")
    assert "ValueError" in (health.error or "")


@pytest.mark.parametrize(
    ("search", "expected"),
    [
        ({"search_terms": ["a", "b"]}, ["a", "b"]),
        ({"keywords": ["k1"]}, ["k1"]),
        ({"q": "single"}, ["single"]),
        ({"search_terms": ["primary"], "keywords": ["secondary"]}, ["primary"]),
    ],
)
def test_search_terms_from_company_reads_config(search: dict, expected: list[str]) -> None:
    adapter = _make_adapter(_StubHttp([]), company=_company(search))
    assert adapter.search_terms_from_company() == expected


def test_search_terms_falls_back_to_default() -> None:
    adapter = _make_adapter(_StubHttp([]), company=_company({}))
    assert adapter.search_terms_from_company() == DEFAULT_SEARCH_TERMS


def test_search_terms_handles_none_company() -> None:
    adapter = _DemoAdapter(http=_StubHttp([]), company=None, settings=Settings())
    assert adapter.search_terms_from_company() == DEFAULT_SEARCH_TERMS
