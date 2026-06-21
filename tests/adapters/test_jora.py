"""Offline (fixture-driven) tests for the Jora source adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import RawJob, Source
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.jora import JoraAdapter

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def jora_html() -> str:
    return (FIXTURES_DIR / "jora_search.html").read_text(encoding="utf-8")


@pytest.fixture
def adapter() -> JoraAdapter:
    company = CompanyConfig(
        company_id="jora1",
        name="Jora",
        adapter="jora",
        search={"q": "quality manager", "l": "Sydney NSW"},
    )
    http = PoliteClient(HttpSettings())
    return JoraAdapter(http=http, company=company, settings=Settings())


def test_parse_returns_unique_jobs(adapter: JoraAdapter, jora_html: str) -> None:
    jobs = adapter.parse(jora_html)

    assert len(jobs) >= 5

    for job in jobs:
        assert isinstance(job, RawJob)
        assert job.source is Source.JORA
        assert job.title  # non-empty title
        assert job.apply_url  # non-empty apply_url
        assert "au.jora.com/job/" in job.apply_url
        # Query string must have been stripped.
        assert "?" not in job.apply_url


def test_apply_urls_are_deduplicated(adapter: JoraAdapter, jora_html: str) -> None:
    jobs = adapter.parse(jora_html)
    apply_urls = [job.apply_url for job in jobs]
    # Each card carries two anchors to the same posting; after stripping the
    # query string they must collapse to a single, unique apply_url.
    assert len(apply_urls) == len(set(apply_urls))


def test_spot_check_known_title(adapter: JoraAdapter, jora_html: str) -> None:
    jobs = adapter.parse(jora_html)
    titles = {job.title for job in jobs}
    assert "BreastScreen Quality Manager" in titles


def test_parse_extracts_source_job_id_hash(adapter: JoraAdapter, jora_html: str) -> None:
    jobs = adapter.parse(jora_html)
    target = next(j for j in jobs if j.title == "BreastScreen Quality Manager")
    assert target.source_job_id == "60d85bfc97dec3ff19e39604b641ef3a"
    assert target.apply_url.endswith(target.source_job_id)


def test_parse_populates_metadata(adapter: JoraAdapter, jora_html: str) -> None:
    jobs = adapter.parse(jora_html)
    target = next(j for j in jobs if j.title == "BreastScreen Quality Manager")
    assert target.company_name  # company present for this card
    assert target.location  # location present
    assert target.company_id == "jora1"


def test_parse_empty_html_is_safe(adapter: JoraAdapter) -> None:
    assert adapter.parse("") == []
    assert adapter.parse("<html><body>no jobs here</body></html>") == []


# ---------------------------------------------------------------------------
# Pagination (fetch across pages)
# ---------------------------------------------------------------------------

# A distinct, small page 2: one card overlaps page 1 (the BreastScreen posting,
# to exercise cross-page de-dup) plus two genuinely-new postings.
_OVERLAP_ID = "60d85bfc97dec3ff19e39604b641ef3a"  # present in the page-1 fixture
_JORA_PAGE2_HTML = f"""
<html><body>
  <div class="job-card result organic-job">
    <a class="job-link" href="/job/Quality-Manager-{_OVERLAP_ID}?fsv=true">
      BreastScreen Quality Manager
    </a>
    <span class="job-company">Health NSW</span>
    <span class="job-location">Sydney NSW</span>
  </div>
  <div class="job-card result organic-job">
    <a class="job-link" href="/job/Page-Two-Engineer-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?fsv=true">
      Page Two Engineer
    </a>
    <span class="job-company">Page Two Co</span>
    <span class="job-location">Brisbane QLD</span>
  </div>
  <div class="job-card result organic-job">
    <a class="job-link" href="/job/Page-Two-Analyst-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb?fsv=true">
      Page Two Analyst
    </a>
    <span class="job-company">Page Two Co</span>
    <span class="job-location">Perth WA</span>
  </div>
</body></html>
"""


class _FakeJoraClient:
    """Stand-in PoliteClient: serves fixtures keyed by the requested page (&p=)."""

    def __init__(self, page_html: dict[int, str]) -> None:
        self._page_html = page_html
        self.requested_pages: list[int] = []

    def get_text_impersonate(self, url: str, **_: object) -> str:
        import re

        match = re.search(r"[?&]p=(\d+)", url)
        page = int(match.group(1)) if match else 1
        self.requested_pages.append(page)
        return self._page_html.get(page, "")


def _adapter_with_client(client: _FakeJoraClient, max_pages: int = 3) -> JoraAdapter:
    company = CompanyConfig(
        company_id="jora1",
        name="Jora",
        adapter="jora",
        search={"q": "quality manager", "l": "Sydney NSW", "max_pages": max_pages},
    )
    return JoraAdapter(http=client, company=company, settings=Settings())  # type: ignore[arg-type]


def test_fetch_paginates_dedupes_and_stops_on_empty(jora_html: str) -> None:
    client = _FakeJoraClient(
        {
            1: jora_html,
            2: _JORA_PAGE2_HTML,
            3: "",  # empty page => stop
        }
    )
    adapter = _adapter_with_client(client, max_pages=3)

    jobs = adapter.fetch(["quality manager"])

    # The empty page 3 halts the walk after it is requested.
    assert client.requested_pages == [1, 2, 3]

    apply_urls = [job.apply_url for job in jobs]
    # De-duplicated across pages by apply_url.
    assert len(apply_urls) == len(set(apply_urls))

    page1_only = adapter.parse(jora_html)
    assert len(jobs) >= len(page1_only)

    # The two genuinely-new page-2 postings made it in.
    assert any(url.endswith("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") for url in apply_urls)
    assert any(url.endswith("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb") for url in apply_urls)
    # The page-2 overlap card points at the same apply_url as a page-1 card,
    # so it appears exactly once (cross-page de-dup).
    overlap_url = f"https://au.jora.com/job/Quality-Manager-{_OVERLAP_ID}"
    assert overlap_url in apply_urls
    assert apply_urls.count(overlap_url) == 1


def test_fetch_stops_when_page_adds_no_new_urls(jora_html: str) -> None:
    # Page 2 identical to page 1 => no new apply_urls => stop before page 3.
    client = _FakeJoraClient({1: jora_html, 2: jora_html, 3: _JORA_PAGE2_HTML})
    adapter = _adapter_with_client(client, max_pages=3)

    jobs = adapter.fetch(["quality manager"])

    assert client.requested_pages == [1, 2]
    assert len(jobs) == len(adapter.parse(jora_html))
