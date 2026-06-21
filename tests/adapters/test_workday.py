"""Offline tests for :class:`WorkdayAdapter`.

``parse`` and ``description_from_detail`` are pure (no network), so these run
against committed fixtures with a real :class:`PoliteClient` that is never
asked to hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import Source
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.workday import WorkdayAdapter

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
BASE_URL = "https://begacheese.wd3.myworkdayjobs.com"


@pytest.fixture
def bega_company() -> CompanyConfig:
    return CompanyConfig(
        company_id="bega",
        name="Bega Group",
        adapter="workday",
        search={
            "base_url": BASE_URL,
            "cxs_tenant": "begacheese",
            "cxs_site": "Bega_Careers",
        },
    )


@pytest.fixture
def adapter(bega_company: CompanyConfig, tmp_path: Path) -> WorkdayAdapter:
    settings = Settings()
    # Point the on-disk cache at a temp dir so tests never touch the real cache.
    http = PoliteClient(HttpSettings(cache_dir=str(tmp_path / "cache")))
    return WorkdayAdapter(http=http, company=bega_company, settings=settings)


def test_parse_list_payload(adapter: WorkdayAdapter) -> None:
    payload = json.loads((FIXTURES_DIR / "workday_bega.json").read_text(encoding="utf-8"))

    jobs = adapter.parse(payload, base_url=BASE_URL)

    assert len(jobs) >= 20

    first = jobs[0]
    assert first.title  # non-empty
    assert first.source is Source.OFFICIAL_ATS
    assert first.company_name == "Bega Group"
    assert first.company_id == "bega"
    assert first.apply_url.startswith(f"{BASE_URL}/en-US/Bega_Careers/job/")
    assert first.source_job_id is not None
    assert first.source_job_id.startswith("JR-")
    assert first.location  # non-empty for the first posting
    assert first.posted_date_raw is not None
    assert "Posted" in first.posted_date_raw
    assert first.description is None  # filled only via the detail endpoint
    assert first.extra["external_path"].startswith("/job/")


def test_parse_accepts_json_string(adapter: WorkdayAdapter) -> None:
    payload_str = (FIXTURES_DIR / "workday_bega.json").read_text(encoding="utf-8")

    jobs = adapter.parse(payload_str, base_url=BASE_URL)

    assert len(jobs) >= 20


def test_parse_uses_config_base_url_when_no_override(adapter: WorkdayAdapter) -> None:
    payload = json.loads((FIXTURES_DIR / "workday_bega.json").read_text(encoding="utf-8"))

    jobs = adapter.parse(payload)  # no base_url override -> falls back to company.search

    assert jobs[0].apply_url.startswith(f"{BASE_URL}/en-US/Bega_Careers/job/")


def test_source_job_id_falls_back_to_path(adapter: WorkdayAdapter) -> None:
    payload = {
        "total": 1,
        "jobPostings": [
            {
                "title": "No Bullet Fields Role",
                "externalPath": "/job/Somewhere/No-Bullet_JR-0",
                "locationsText": "Somewhere",
                "postedOn": "Posted Today",
            }
        ],
    }

    jobs = adapter.parse(payload, base_url=BASE_URL)

    assert len(jobs) == 1
    assert jobs[0].source_job_id == "/job/Somewhere/No-Bullet_JR-0"


def test_parse_skips_postings_without_external_path(adapter: WorkdayAdapter) -> None:
    payload = {"jobPostings": [{"title": "Bad"}, {"title": "Good", "externalPath": "/job/x_JR-1"}]}

    jobs = adapter.parse(payload, base_url=BASE_URL)

    assert len(jobs) == 1
    assert jobs[0].title == "Good"


def test_description_from_detail_real_fixture(adapter: WorkdayAdapter) -> None:
    detail = json.loads((FIXTURES_DIR / "workday_bega_detail.json").read_text(encoding="utf-8"))

    description = adapter.description_from_detail(detail)

    assert description is not None
    # HTML tags stripped and whitespace collapsed.
    assert "<p>" not in description
    assert "<" not in description
    assert "&amp;" not in description  # entities decoded
    assert "curious" in description.lower()


def test_description_from_detail_minimal() -> None:
    detail = {"jobPostingInfo": {"jobDescription": "<p>HACCP &amp; GMP audit readiness</p>"}}

    description = WorkdayAdapter.description_from_detail(detail)

    assert description == "HACCP & GMP audit readiness"


def test_description_from_detail_handles_missing() -> None:
    assert WorkdayAdapter.description_from_detail({}) is None
    assert WorkdayAdapter.description_from_detail({"jobPostingInfo": {}}) is None
    assert WorkdayAdapter.description_from_detail({"jobPostingInfo": {"jobDescription": ""}}) is None
    assert WorkdayAdapter.description_from_detail("not json {") is None
