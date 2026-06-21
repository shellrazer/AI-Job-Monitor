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
