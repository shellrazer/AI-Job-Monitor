"""Offline (fixture-driven) tests for :class:`SmartRecruitersAdapter`.

``parse`` and ``description_from_detail`` are pure (no network), so these run
against an inline JSON fixture with a real :class:`PoliteClient` that is never
asked to hit the network.
"""

from __future__ import annotations

import json

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import RawJob, Source
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.smartrecruiters import SmartRecruitersAdapter

COMPANY_SLUG = "AcmeFoods"

# A representative list-endpoint payload: a quality role plus two others.
LIST_PAYLOAD = json.dumps(
    {
        "totalFound": 3,
        "content": [
            {
                "id": "743999000000111",
                "name": "Quality Assurance Manager",
                "ref": "https://api.smartrecruiters.com/v1/companies/AcmeFoods/postings/743999000000111",
                "releasedDate": "2026-06-15T09:30:00.000Z",
                "location": {"city": "Melbourne", "region": "VIC", "country": "au"},
                "company": {"name": "Acme Foods Pty Ltd"},
            },
            {
                "id": "743999000000222",
                "name": "Food Safety Specialist",
                "ref": "https://api.smartrecruiters.com/v1/companies/AcmeFoods/postings/743999000000222",
                "releasedDate": "2026-06-10T00:00:00.000Z",
                "location": {"city": "Sydney", "region": "NSW", "country": "au"},
                "company": {"name": "Acme Foods Pty Ltd"},
            },
            {
                # Missing id -> must be skipped defensively.
                "name": "Broken Posting",
                "location": {"city": "Brisbane"},
            },
        ],
    }
)


@pytest.fixture
def company() -> CompanyConfig:
    return CompanyConfig(
        company_id="acme",
        name="Acme Foods",
        adapter="smartrecruiters",
        search={"company_slug": COMPANY_SLUG, "search_terms": ["quality manager"]},
    )


@pytest.fixture
def adapter(company: CompanyConfig) -> SmartRecruitersAdapter:
    http = PoliteClient(HttpSettings())
    return SmartRecruitersAdapter(http=http, company=company, settings=Settings())


def test_parse_yields_correct_rawjobs(adapter: SmartRecruitersAdapter) -> None:
    jobs = adapter.parse(LIST_PAYLOAD)

    # The malformed (id-less) record is dropped.
    assert len(jobs) == 2
    for job in jobs:
        assert isinstance(job, RawJob)
        assert job.source is Source.OFFICIAL_ATS
        assert job.company_id == "acme"

    first = jobs[0]
    assert first.title == "Quality Assurance Manager"
    assert first.company_name == "Acme Foods Pty Ltd"  # posting.company.name wins
    assert first.source_job_id == "743999000000111"
    assert first.apply_url == f"https://jobs.smartrecruiters.com/{COMPANY_SLUG}/743999000000111"
    assert first.location == "Melbourne, VIC, au"
    assert first.posted_date_raw == "2026-06-15T09:30:00.000Z"
    # Description is only filled via the detail endpoint.
    assert first.description is None


def test_parse_accepts_dict_payload(adapter: SmartRecruitersAdapter) -> None:
    jobs = adapter.parse(json.loads(LIST_PAYLOAD))
    assert len(jobs) == 2
    assert jobs[1].title == "Food Safety Specialist"


def test_company_name_falls_back_to_config(adapter: SmartRecruitersAdapter) -> None:
    payload = {
        "totalFound": 1,
        "content": [{"id": "x1", "name": "Plant Manager", "location": {"city": "Perth"}}],
    }
    jobs = adapter.parse(payload)
    assert len(jobs) == 1
    assert jobs[0].company_name == "Acme Foods"  # no posting.company -> config name
    assert jobs[0].location == "Perth"


def test_description_from_detail() -> None:
    detail = {
        "jobAd": {
            "sections": {
                "jobDescription": {"text": "<p>Lead the <b>QA</b> team.</p>"},
                "qualifications": {"text": "<p>HACCP &amp; GMP experience required.</p>"},
            }
        }
    }
    description = SmartRecruitersAdapter.description_from_detail(detail)
    assert description is not None
    assert "<" not in description  # tags stripped
    assert "&amp;" not in description  # entities decoded
    assert "QA" in description
    assert "HACCP & GMP" in description


def test_description_from_detail_handles_missing() -> None:
    assert SmartRecruitersAdapter.description_from_detail({}) is None
    assert SmartRecruitersAdapter.description_from_detail({"jobAd": {}}) is None
    assert SmartRecruitersAdapter.description_from_detail({"jobAd": {"sections": {}}}) is None
    assert SmartRecruitersAdapter.description_from_detail("not json {") is None


def test_parse_empty_payload_is_safe(adapter: SmartRecruitersAdapter) -> None:
    assert adapter.parse("{}") == []
    assert adapter.parse({"content": []}) == []
    assert adapter.parse("not a json object") == []
