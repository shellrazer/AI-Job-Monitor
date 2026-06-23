"""Offline tests for :class:`SuccessFactorsAdapter`.

``parse`` is pure (no network), so these run against small inline fixtures plus
the committed ``graincorp_careers.html`` capture with a real
:class:`PoliteClient` that is never asked to hit the network.

The adapter is dual-mode: it auto-detects whether a payload is a SuccessFactors
RSS feed or a JS-rendered HTML landing page. Both paths are exercised here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import Source
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.successfactors import SuccessFactorsAdapter

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
CAREERS_URL = "https://jobs.graincorp.com.au/"

# A small SuccessFactors-style RSS 2.0 feed: two <item>s, a Google-jobs (g:)
# namespace, <g:location> and <guid>.
RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">
  <channel>
    <title>GrainCorp Careers</title>
    <link>https://jobs.graincorp.com.au/</link>
    <item>
      <title>Quality Manager - Sydney</title>
      <link>https://jobs.graincorp.com.au/job/Quality-Manager/123456/</link>
      <description>&lt;p&gt;Lead the &lt;b&gt;quality&lt;/b&gt; team. HACCP &amp;amp; GMP.&lt;/p&gt;</description>
      <pubDate>Wed, 18 Jun 2026 09:00:00 GMT</pubDate>
      <guid>123456</guid>
      <g:location>Sydney NSW</g:location>
      <g:job_function>Quality</g:job_function>
    </item>
    <item>
      <title>Food Safety Officer</title>
      <link>https://jobs.graincorp.com.au/job/Food-Safety-Officer/789012/</link>
      <description>Entry-level food safety role at a grain handling site.</description>
      <pubDate>Thu, 19 Jun 2026 09:00:00 GMT</pubDate>
      <guid>789012</guid>
      <g:location>Newcastle NSW</g:location>
    </item>
  </channel>
</rss>
"""

# A small HTML landing page with two individual job-posting anchors (href
# contains "/job/") plus some non-posting chrome that must be ignored.
HTML_FIXTURE = """<!DOCTYPE html>
<html><body>
  <nav>
    <a href="#">Skip to content</a>
    <a href="/go/Scientific-&amp;-Quality-Assurance-Jobs/5255310/">Browse Quality Jobs</a>
    <a href="https://www.graincorp.com.au/">Corporate site</a>
  </nav>
  <ul class="results">
    <li><a href="/job/Quality-Manager/123456/">Quality Manager - Dubbo</a></li>
    <li><a href="/job/Food-Safety-Officer/789012/">Food Safety Officer</a></li>
  </ul>
</body></html>
"""


@pytest.fixture
def company() -> CompanyConfig:
    return CompanyConfig(
        company_id="graincorp",
        name="GrainCorp",
        adapter="successfactors",
        careers_url=CAREERS_URL,
    )


@pytest.fixture
def adapter(company: CompanyConfig, tmp_path: Path) -> SuccessFactorsAdapter:
    settings = Settings()
    # Point the on-disk cache at a temp dir so tests never touch the real cache.
    http = PoliteClient(HttpSettings(cache_dir=str(tmp_path / "cache")))
    return SuccessFactorsAdapter(http=http, company=company, settings=settings)


def test_parse_rss_feed(adapter: SuccessFactorsAdapter) -> None:
    jobs = adapter.parse(RSS_FIXTURE)

    assert len(jobs) == 2

    first = jobs[0]
    assert first.source is Source.OFFICIAL_ATS
    assert first.company_name == "GrainCorp"
    assert first.company_id == "graincorp"
    assert first.title == "Quality Manager - Sydney"
    assert first.apply_url == "https://jobs.graincorp.com.au/job/Quality-Manager/123456/"
    assert first.location == "Sydney NSW"  # from <g:location>
    assert first.source_job_id == "123456"  # from <guid>
    assert first.posted_date_raw == "Wed, 18 Jun 2026 09:00:00 GMT"
    # Description HTML stripped, entities decoded, whitespace collapsed.
    assert first.description is not None
    assert "<p>" not in first.description
    assert "<b>" not in first.description
    assert "&amp;" not in first.description
    assert "HACCP & GMP" in first.description

    second = jobs[1]
    assert second.title == "Food Safety Officer"
    assert second.location == "Newcastle NSW"
    assert second.source_job_id == "789012"


def test_parse_html_anchors(adapter: SuccessFactorsAdapter) -> None:
    jobs = adapter.parse(HTML_FIXTURE, base_url=CAREERS_URL)

    assert len(jobs) == 2
    titles = {j.title for j in jobs}
    assert titles == {"Quality Manager - Dubbo", "Food Safety Officer"}

    for job in jobs:
        assert job.source is Source.OFFICIAL_ATS
        assert job.company_name == "GrainCorp"
        # Relative hrefs resolved to absolute against the careers base URL.
        assert job.apply_url.startswith("https://jobs.graincorp.com.au/job/")

    # The category-browse and marketing anchors must NOT be treated as postings.
    assert all("/go/" not in j.apply_url for j in jobs)


def test_parse_html_resolves_base_from_config(adapter: SuccessFactorsAdapter) -> None:
    # No explicit base_url -> falls back to company.careers_url.
    jobs = adapter.parse(HTML_FIXTURE)

    assert len(jobs) == 2
    assert all(j.apply_url.startswith("https://jobs.graincorp.com.au/job/") for j in jobs)


def test_parse_empty_and_garbage_returns_list(adapter: SuccessFactorsAdapter) -> None:
    assert adapter.parse("") == []
    assert adapter.parse("   ") == []
    assert adapter.parse(None) == []  # type: ignore[arg-type]
    # Malformed XML that *looks* like RSS must not raise.
    assert adapter.parse("<?xml version='1.0'?><rss><item><title>oops") == []


def test_parse_real_graincorp_html_does_not_crash(adapter: SuccessFactorsAdapter) -> None:
    """The real captured SuccessFactors landing page is JS-rendered.

    Its static HTML carries only browse-by-category (``/go/...-Jobs/``) links and
    marketing chrome, no individual job postings, so ``parse`` is expected to
    return an (often empty) list rather than crash. This test documents the
    JS-rendering limitation without failing.
    """
    html = (FIXTURES_DIR / "graincorp_careers.html").read_text(encoding="utf-8")

    jobs = adapter.parse(html, base_url=CAREERS_URL)

    assert isinstance(jobs, list)
    # Empty is expected (page needs a feed_url or Playwright); print for visibility.
    print(f"\n[successfactors] graincorp_careers.html static parse -> {len(jobs)} job(s)")
    for job in jobs:
        assert job.title
        assert job.apply_url


# --------------------------------------------------------------------------- #
# fetch() transport selection                                                 #
# --------------------------------------------------------------------------- #
class _FakeHttp:
    """Records which transport was used and returns canned payload text."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []
        # Captures the wait_selector passed to the render transport (if any).
        self.wait_selector: str | None = None

    def get_text(self, url: str, **_: Any) -> str:
        self.calls.append(("httpx", url))
        return self.payload

    def get_text_impersonate(self, url: str, **_: Any) -> str:
        self.calls.append(("cffi", url))
        return self.payload

    def get_text_rendered(self, url: str, *, wait_selector: str | None = None, **_: Any) -> str:
        self.calls.append(("render", url))
        self.wait_selector = wait_selector
        return self.payload


def _adapter_with_http(http: Any, search: dict[str, Any]) -> SuccessFactorsAdapter:
    company = CompanyConfig(
        company_id="graincorp",
        name="GrainCorp",
        adapter="successfactors",
        careers_url=CAREERS_URL,
        search=search,
    )
    return SuccessFactorsAdapter(http=http, company=company, settings=Settings())


def test_fetch_defaults_to_impersonate() -> None:
    # No render / no feed_url -> SuccessFactors defaults to impersonation
    # (sidesteps the SF/Recruiting-Marketing 403 blocks, e.g. Nestlé).
    # max_pages=1 keeps this focused on transport selection (pagination has its
    # own test); the HTML path now stamps startrow=0 onto the careers URL.
    http = _FakeHttp(HTML_FIXTURE)
    adapter = _adapter_with_http(http, {"max_pages": 1})

    jobs = adapter.fetch(["quality"])

    assert len(jobs) == 2
    assert http.calls == [("cffi", f"{CAREERS_URL}?startrow=0")]


def test_fetch_feed_url_uses_impersonate_by_default() -> None:
    feed_url = "https://jobs.graincorp.com.au/rssfeed/"
    http = _FakeHttp(RSS_FIXTURE)
    adapter = _adapter_with_http(http, {"feed_url": feed_url})

    jobs = adapter.fetch(["quality"])

    assert len(jobs) == 2
    assert http.calls == [("cffi", feed_url)]


def test_fetch_uses_rendered_when_configured() -> None:
    http = _FakeHttp(HTML_FIXTURE)
    adapter = _adapter_with_http(
        http, {"render": True, "wait_selector": "ul.results", "max_pages": 1}
    )

    jobs = adapter.fetch(["quality"])

    assert len(jobs) == 2
    assert http.calls == [("render", f"{CAREERS_URL}?startrow=0")]
    # wait_selector must be threaded through to the render transport.
    assert http.wait_selector == "ul.results"


def test_fetch_render_wins_over_impersonate() -> None:
    http = _FakeHttp(HTML_FIXTURE)
    adapter = _adapter_with_http(http, {"render": True, "impersonate": True, "max_pages": 1})

    adapter.fetch(["quality"])

    # render takes precedence even when impersonate is also set.
    assert http.calls == [("render", f"{CAREERS_URL}?startrow=0")]
    assert http.wait_selector is None


def test_fetch_returns_empty_without_careers_or_feed_url() -> None:
    http = _FakeHttp(HTML_FIXTURE)
    company = CompanyConfig(
        company_id="graincorp",
        name="GrainCorp",
        adapter="successfactors",
        search={},
    )
    adapter = SuccessFactorsAdapter(http=http, company=company, settings=Settings())

    assert adapter.fetch(["quality"]) == []
    assert http.calls == []


# --------------------------------------------------------------------------- #
# startrow pagination (HTML careers_url path)                                 #
# --------------------------------------------------------------------------- #
def _page_html(*paths: str) -> str:
    anchors = "".join(f'<li><a href="{p}">{p.strip("/").split("/")[1]}</a></li>' for p in paths)
    return f"<!DOCTYPE html><html><body><ul>{anchors}</ul></body></html>"


# startrow=0 -> jobs 1,2,3 ; startrow=25 -> jobs 3 (overlap) + 4,5 ; startrow=50 -> empty.
_PAGE0 = _page_html("/job/Quality-Manager/1/", "/job/Food-Safety/2/", "/job/QA-Lead/3/")
_PAGE1 = _page_html("/job/QA-Lead/3/", "/job/Auditor/4/", "/job/Technologist/5/")
_PAGE2 = "<!DOCTYPE html><html><body><ul></ul></body></html>"


class _PagedHttp:
    """Fake client returning distinct HTML keyed by the ``startrow`` query value."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages  # startrow value (str) -> HTML
        self.calls: list[tuple[str, str]] = []

    def _payload_for(self, url: str) -> str:
        from urllib.parse import parse_qs, urlsplit

        startrow = parse_qs(urlsplit(url).query).get("startrow", ["0"])[0]
        return self.pages.get(startrow, "")

    def get_text(self, url: str, **_: Any) -> str:
        self.calls.append(("httpx", url))
        return self._payload_for(url)

    def get_text_impersonate(self, url: str, **_: Any) -> str:
        self.calls.append(("cffi", url))
        return self._payload_for(url)

    def get_text_rendered(self, url: str, *, wait_selector: str | None = None, **_: Any) -> str:
        self.calls.append(("render", url))
        return self._payload_for(url)


def test_fetch_html_paginates_and_dedupes_across_startrow_pages() -> None:
    http = _PagedHttp({"0": _PAGE0, "25": _PAGE1, "50": _PAGE2})
    adapter = _adapter_with_http(http, {"max_pages": 3})

    jobs = adapter.fetch(["quality"])

    # 5 distinct postings; the overlapping QA-Lead (job 3) appears exactly once.
    urls = [j.apply_url for j in jobs]
    assert len(jobs) == 5
    assert len(set(urls)) == 5
    qa_lead = [u for u in urls if "/QA-Lead/3/" in u]
    assert len(qa_lead) == 1

    # Walked startrow=0 then 25, then stopped at the empty startrow=50 page.
    startrows = [c[1].split("startrow=")[1] for c in http.calls]
    assert startrows == ["0", "25", "50"]


def test_fetch_feed_url_is_single_fetch_regardless_of_max_pages() -> None:
    feed_url = "https://jobs.graincorp.com.au/rssfeed/"
    # RSS feed returns everything in one response, so a large max_pages must NOT
    # paginate; _FakeHttp returns the same RSS payload for any URL it is asked.
    http = _FakeHttp(RSS_FIXTURE)
    adapter = _adapter_with_http(http, {"feed_url": feed_url, "max_pages": 5})

    jobs = adapter.fetch(["quality"])

    assert len(jobs) == 2
    # Exactly ONE fetch despite max_pages=5.
    assert http.calls == [("cffi", feed_url)]
