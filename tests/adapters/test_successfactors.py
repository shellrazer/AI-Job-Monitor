"""Offline tests for :class:`SuccessFactorsAdapter`.

``parse`` is pure (no network), so these run against small inline fixtures plus
the committed ``graincorp_careers.html`` capture with a real
:class:`PoliteClient` that is never asked to hit the network.

The adapter is dual-mode: it auto-detects whether a payload is a SuccessFactors
RSS feed or a JS-rendered HTML landing page. Both paths are exercised here.
"""

from __future__ import annotations

from pathlib import Path

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
