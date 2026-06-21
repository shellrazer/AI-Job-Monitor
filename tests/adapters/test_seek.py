"""Offline parsing tests for the SEEK adapter against a captured fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import Source
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.seek import SeekAdapter

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def seek_html() -> str:
    return (FIXTURES_DIR / "seek_search.html").read_text(encoding="utf-8")


@pytest.fixture
def adapter(tmp_path: Path) -> SeekAdapter:
    company = CompanyConfig(
        company_id="seek1",
        name="SEEK",
        adapter="seek",
        search={"keywords": "quality manager", "where": "Sydney NSW"},
    )
    settings = Settings()
    http = PoliteClient(HttpSettings(cache_dir=str(tmp_path / "cache")))
    return SeekAdapter(http=http, company=company, settings=settings)


def test_parse_returns_reasonable_job_count(adapter: SeekAdapter, seek_html: str) -> None:
    jobs = adapter.parse(seek_html)
    assert len(jobs) >= 5


def test_every_job_has_title_and_apply_url(adapter: SeekAdapter, seek_html: str) -> None:
    jobs = adapter.parse(seek_html)
    for job in jobs:
        assert job.title.strip(), "title should be non-empty"
        assert job.apply_url, "apply_url should be present"
        assert "seek.com.au/job/" in job.apply_url
        assert job.source is Source.SEEK
        assert job.company_id is None


def test_some_jobs_have_id_and_location(adapter: SeekAdapter, seek_html: str) -> None:
    jobs = adapter.parse(seek_html)
    with_id = [j for j in jobs if j.source_job_id]
    with_location = [j for j in jobs if j.location]
    assert with_id, "at least some jobs should carry a parsed source_job_id"
    assert with_location, "at least some jobs should carry a location"
    # source_job_id is the numeric id; the apply_url should contain it.
    for job in with_id:
        assert job.source_job_id is not None
        assert job.source_job_id.isdigit()
        assert f"/job/{job.source_job_id}" in job.apply_url


def test_known_job_title_present(adapter: SeekAdapter, seek_html: str) -> None:
    jobs = adapter.parse(seek_html)
    titles = {job.title for job in jobs}
    assert "Quality Manager" in titles


def test_apply_url_strips_tracking_params(adapter: SeekAdapter, seek_html: str) -> None:
    jobs = adapter.parse(seek_html)
    for job in jobs:
        # Tracking query params (type/ref/origin) must be stripped.
        assert "?" not in job.apply_url
        assert "&" not in job.apply_url


def test_parse_accepts_bytes(adapter: SeekAdapter, seek_html: str) -> None:
    jobs_from_str = adapter.parse(seek_html)
    jobs_from_bytes = adapter.parse(seek_html.encode("utf-8"))
    assert len(jobs_from_bytes) == len(jobs_from_str)


def test_custom_base_url_is_honored(adapter: SeekAdapter, seek_html: str) -> None:
    jobs = adapter.parse(seek_html, base_url="https://www.seek.co.nz")
    assert jobs
    assert all(job.apply_url.startswith("https://www.seek.co.nz/job/") for job in jobs)


def test_html_fallback_path(adapter: SeekAdapter, seek_html: str) -> None:
    # Force the data-automation HTML path by exercising it directly.
    jobs = adapter._parse_html(seek_html, "https://www.seek.com.au")
    assert len(jobs) >= 5
    for job in jobs:
        assert job.title.strip()
        assert "seek.com.au/job/" in job.apply_url
        assert job.source is Source.SEEK


# ---------------------------------------------------------------------------
# Pagination (fetch across pages)
# ---------------------------------------------------------------------------

# A distinct, small page 2 built from the data-automation HTML fallback path.
# It overlaps with page 1 on one id (to exercise cross-page de-dup) and adds two
# genuinely new postings.
_PAGE2_NEW_ID = "92737035"  # also present in the page-1 fixture (overlap)
_SEEK_PAGE2_HTML = f"""
<html><body>
  <article data-automation="normalJob">
    <a data-automation="jobTitle" href="/job/{_PAGE2_NEW_ID}?type=standard&ref=x">
      Quality Manager
    </a>
    <span data-automation="jobCompany">Acme Foods</span>
    <span data-automation="jobLocation">Melbourne VIC</span>
  </article>
  <article data-automation="normalJob">
    <a data-automation="jobTitle" href="/job/55500001?type=standard&ref=x">
      Page Two Engineer
    </a>
    <span data-automation="jobCompany">Page Two Co</span>
    <span data-automation="jobLocation">Brisbane QLD</span>
  </article>
  <article data-automation="normalJob">
    <a data-automation="jobTitle" href="/job/55500002?type=standard&ref=x">
      Page Two Analyst
    </a>
    <span data-automation="jobCompany">Page Two Co</span>
    <span data-automation="jobLocation">Perth WA</span>
  </article>
</body></html>
"""


class _FakeSeekClient:
    """Stand-in PoliteClient: serves fixtures keyed by the requested page."""

    def __init__(self, page_html: dict[int, str]) -> None:
        self._page_html = page_html
        self.requested_pages: list[int] = []

    def get_text_impersonate(self, url: str, **_: object) -> str:
        import re

        match = re.search(r"[?&]page=(\d+)", url)
        page = int(match.group(1)) if match else 1
        self.requested_pages.append(page)
        return self._page_html.get(page, "")


def _adapter_with_client(client: _FakeSeekClient, max_pages: int = 3) -> SeekAdapter:
    company = CompanyConfig(
        company_id="seek1",
        name="SEEK",
        adapter="seek",
        search={"keywords": "quality manager", "where": "Sydney NSW", "max_pages": max_pages},
    )
    return SeekAdapter(http=client, company=company, settings=Settings())  # type: ignore[arg-type]


def test_fetch_paginates_dedupes_and_stops_on_empty(seek_html: str) -> None:
    client = _FakeSeekClient(
        {
            1: seek_html,
            2: _SEEK_PAGE2_HTML,
            3: "",  # empty page => stop
        }
    )
    adapter = _adapter_with_client(client, max_pages=3)

    jobs = adapter.fetch(["quality manager"])

    # Page 3 (empty) halts the walk, so only pages 1 and 2 are fetched.
    assert client.requested_pages == [1, 2, 3]

    apply_urls = [job.apply_url for job in jobs]
    # De-duplicated across pages by apply_url.
    assert len(apply_urls) == len(set(apply_urls))

    # Page-1 fixture jobs are present.
    page1_only = adapter.parse(seek_html)
    assert len(jobs) >= len(page1_only)

    # The two genuinely-new page-2 postings made it in.
    assert "https://www.seek.com.au/job/55500001" in apply_urls
    assert "https://www.seek.com.au/job/55500002" in apply_urls
    # The overlapping id appears exactly once (cross-page de-dup).
    overlap_url = f"https://www.seek.com.au/job/{_PAGE2_NEW_ID}"
    assert apply_urls.count(overlap_url) == 1


def test_fetch_stops_when_page_adds_no_new_urls(seek_html: str) -> None:
    # Page 2 is identical to page 1 => contributes no new apply_urls => stop
    # before requesting page 3.
    client = _FakeSeekClient({1: seek_html, 2: seek_html, 3: _SEEK_PAGE2_HTML})
    adapter = _adapter_with_client(client, max_pages=3)

    jobs = adapter.fetch(["quality manager"])

    assert client.requested_pages == [1, 2]
    # fetch() de-dupes by apply_url across pages; an identical page 2 adds
    # nothing, so the result matches the de-duped page-1 set.
    page1_unique = {job.apply_url for job in adapter.parse(seek_html)}
    assert len(jobs) == len(page1_unique)
