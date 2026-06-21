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
