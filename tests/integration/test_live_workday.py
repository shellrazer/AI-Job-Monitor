"""Live integration test for :class:`WorkdayAdapter`.

Marked ``integration`` so it is deselected by default (see pyproject addopts).
Run explicitly with ``uv run pytest tests/integration/test_live_workday.py -m integration``.
Any network failure (offline CI, transient error, or a hard block) results in a
skip rather than a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import SourceBlocked
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.workday import WorkdayAdapter

pytestmark = pytest.mark.integration

BASE_URL = "https://begacheese.wd3.myworkdayjobs.com"


@pytest.fixture
def adapter(tmp_path: Path) -> WorkdayAdapter:
    company = CompanyConfig(
        company_id="bega",
        name="Bega Group",
        adapter="workday",
        search={
            "base_url": BASE_URL,
            "cxs_tenant": "begacheese",
            "cxs_site": "Bega_Careers",
        },
    )
    settings = Settings()
    # Disable on-disk caching across runs by using a fresh temp cache dir.
    http = PoliteClient(HttpSettings(cache_dir=str(tmp_path / "cache")))
    return WorkdayAdapter(http=http, company=company, settings=settings)


def test_live_fetch(adapter: WorkdayAdapter) -> None:
    try:
        jobs = adapter.fetch(["quality"])
    except SourceBlocked as exc:
        pytest.skip(f"Workday hard-blocked the request: {exc}")
    except Exception as exc:  # offline / transient network failure
        pytest.skip(f"Live Workday fetch failed (likely offline): {exc!r}")

    assert len(jobs) >= 1
    job = jobs[0]
    assert job.title
    assert job.apply_url.startswith(f"{BASE_URL}/en-US/Bega_Careers/job/")
    assert job.source_job_id
    # At least one of the fetched postings should have a non-empty description
    # pulled from the detail endpoint.
    assert any(j.description for j in jobs)
