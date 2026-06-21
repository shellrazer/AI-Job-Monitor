"""Live (network) integration test for the Jora source adapter.

Deselected by default (``-m 'not integration'`` in pyproject). Run with::

    uv run pytest tests/integration/test_live_jora.py -q -m integration
"""

from __future__ import annotations

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import Source, SourceBlocked
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.jora import JoraAdapter


@pytest.mark.integration
def test_live_fetch_returns_jobs() -> None:
    company = CompanyConfig(
        company_id="jora1",
        name="Jora",
        adapter="jora",
        search={"q": "quality manager", "l": "Sydney NSW"},
    )
    http = PoliteClient(HttpSettings())
    adapter = JoraAdapter(http=http, company=company, settings=Settings())

    try:
        jobs = adapter.fetch(["quality manager"])
    except SourceBlocked:
        pytest.skip("Jora blocked")

    assert len(jobs) >= 1
    for job in jobs:
        assert job.source is Source.JORA
        assert job.title
        assert "au.jora.com/job/" in job.apply_url
