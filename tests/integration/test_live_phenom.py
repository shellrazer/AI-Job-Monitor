"""Live (network) integration test for the Phenom source adapter.

Deselected by default (``-m 'not integration'`` in pyproject). Run with::

    uv run pytest tests/integration/test_live_phenom.py -q -m integration

Hits the verified PepsiCo "api_jobs" tenant.
"""

from __future__ import annotations

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import Source, SourceBlocked
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.phenom import PhenomAdapter


@pytest.mark.integration
def test_live_pepsico_api_jobs_returns_jobs() -> None:
    company = CompanyConfig(
        company_id="pepsico",
        name="PepsiCo",
        adapter="phenom",
        search={
            "mode": "api_jobs",
            "api_url": "https://www.pepsicojobs.com/api/jobs",
            "careers_base": "https://www.pepsicojobs.com",
            "location": "Australia",
            "keywords": "quality",
            "max_pages": 1,
        },
    )
    http = PoliteClient(HttpSettings())
    adapter = PhenomAdapter(http=http, company=company, settings=Settings())

    try:
        jobs = adapter.fetch(["quality"])
    except SourceBlocked:
        pytest.skip("PepsiCo Phenom blocked")

    assert isinstance(jobs, list)
    if not jobs:
        pytest.skip("PepsiCo Phenom returned no jobs")

    for job in jobs:
        assert job.source is Source.OFFICIAL_ATS
        assert job.title
        assert job.apply_url.startswith("http")
