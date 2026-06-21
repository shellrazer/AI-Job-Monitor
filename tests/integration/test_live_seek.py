"""Live integration test for the SEEK adapter (network, deselected by default)."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import SourceBlocked
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.seek import SeekAdapter


@pytest.mark.integration
def test_live_fetch_returns_jobs(tmp_path: Path) -> None:
    company = CompanyConfig(
        company_id="seek1",
        name="SEEK",
        adapter="seek",
        search={"keywords": "quality manager", "where": "Sydney NSW"},
    )
    settings = Settings()
    http = PoliteClient(HttpSettings(cache_dir=str(tmp_path / "cache")))
    adapter = SeekAdapter(http=http, company=company, settings=settings)

    try:
        jobs = adapter.fetch(["quality manager"])
    except SourceBlocked:
        pytest.skip("SEEK blocked")

    assert len(jobs) >= 1
    for job in jobs:
        assert job.title.strip()
        assert job.apply_url
