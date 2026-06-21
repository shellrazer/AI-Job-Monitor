"""Live (network) integration test for the LinkedIn source adapter.

Deselected by default (``-m 'not integration'`` in pyproject). Run with::

    uv run pytest tests/integration/test_live_linkedin.py -q -m integration
"""

from __future__ import annotations

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import Source, SourceBlocked
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.linkedin import LinkedInAdapter


@pytest.mark.integration
def test_live_fetch_returns_jobs() -> None:
    company = CompanyConfig(
        company_id="li1",
        name="LinkedIn",
        adapter="linkedin",
        search={
            "keywords": "Quality Manager food",
            "location": "Sydney NSW",
            "max_pages": 1,
        },
    )
    http = PoliteClient(HttpSettings())
    adapter = LinkedInAdapter(http=http, company=company, settings=Settings())

    try:
        jobs = adapter.fetch(["Quality Manager food"])
    except SourceBlocked:
        pytest.skip("LinkedIn blocked/empty")

    assert isinstance(jobs, list)
    if not jobs:
        pytest.skip("LinkedIn blocked/empty")

    for job in jobs:
        assert job.source is Source.LINKEDIN
        assert job.title
        assert "linkedin.com/jobs/view/" in job.apply_url
