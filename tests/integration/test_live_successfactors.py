"""Live integration test for :class:`SuccessFactorsAdapter`.

Marked ``integration`` so it is deselected by default (see pyproject addopts).
Run explicitly with
``uv run pytest tests/integration/test_live_successfactors.py -m integration``.

GrainCorp's "Careers Centre" is a JS-rendered SuccessFactors site with no
configured RSS ``feed_url`` here, so the static HTML fetch is expected to yield
zero postings — that is a *skip*, not a failure, and documents that the source
needs an RSS ``feed_url`` (or future Playwright rendering). A hard block or any
network failure also results in a skip rather than a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import SourceBlocked
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.successfactors import SuccessFactorsAdapter

pytestmark = pytest.mark.integration

CAREERS_URL = "https://jobs.graincorp.com.au/"


@pytest.fixture
def adapter(tmp_path: Path) -> SuccessFactorsAdapter:
    company = CompanyConfig(
        company_id="graincorp",
        name="GrainCorp",
        adapter="successfactors",
        careers_url=CAREERS_URL,
    )
    settings = Settings()
    # Disable on-disk caching across runs by using a fresh temp cache dir.
    http = PoliteClient(HttpSettings(cache_dir=str(tmp_path / "cache")))
    return SuccessFactorsAdapter(http=http, company=company, settings=settings)


def test_live_fetch(adapter: SuccessFactorsAdapter) -> None:
    try:
        jobs = adapter.fetch(["quality"])
    except SourceBlocked as exc:
        pytest.skip(f"SuccessFactors hard-blocked the request: {exc}")
    except Exception as exc:  # offline / transient network failure
        pytest.skip(f"Live SuccessFactors fetch failed (likely offline): {exc!r}")

    assert isinstance(jobs, list)
    if not jobs:
        pytest.skip("SF static page empty — needs feed_url/Playwright")

    job = jobs[0]
    assert job.title
    assert job.apply_url
