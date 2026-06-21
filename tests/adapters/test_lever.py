"""Offline (fixture-driven) tests for :class:`LeverAdapter`.

``parse`` is pure (no network), so these run against an inline JSON fixture with
a real :class:`PoliteClient` that is never asked to hit the network.
"""

from __future__ import annotations

import json

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import RawJob, Source
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.lever import LeverAdapter

COMPANY_SLUG = "acmefoods"

# Representative postings list (Lever returns a top-level JSON array).
# createdAt is epoch milliseconds: 1750118400000 -> 2025-06-17 (UTC).
LIST_PAYLOAD = json.dumps(
    [
        {
            "id": "f1a2b3c4-0001",
            "text": "Quality Manager",
            "hostedUrl": "https://jobs.lever.co/acmefoods/f1a2b3c4-0001",
            "categories": {"location": "Melbourne", "team": "Quality", "commitment": "Full-time"},
            "createdAt": 1750118400000,
            "descriptionPlain": "Lead the quality and food safety program across our plants.",
        },
        {
            "id": "f1a2b3c4-0002",
            "text": "Food Safety Coordinator",
            "hostedUrl": "https://jobs.lever.co/acmefoods/f1a2b3c4-0002",
            "categories": {"location": "Sydney", "team": "Quality", "commitment": "Full-time"},
            "createdAt": 1749513600000,
            "descriptionPlain": "Coordinate HACCP and GMP compliance audits.",
        },
        {
            # Missing hostedUrl -> must be skipped defensively.
            "id": "f1a2b3c4-0003",
            "text": "Broken Posting",
            "categories": {"location": "Nowhere"},
        },
    ]
)


@pytest.fixture
def company() -> CompanyConfig:
    return CompanyConfig(
        company_id="acme",
        name="Acme Foods",
        adapter="lever",
        search={"company_slug": COMPANY_SLUG},
    )


@pytest.fixture
def adapter(company: CompanyConfig) -> LeverAdapter:
    http = PoliteClient(HttpSettings())
    return LeverAdapter(http=http, company=company, settings=Settings())


def test_parse_yields_correct_rawjobs(adapter: LeverAdapter) -> None:
    jobs = adapter.parse(LIST_PAYLOAD)

    # The url-less record is dropped.
    assert len(jobs) == 2
    for job in jobs:
        assert isinstance(job, RawJob)
        assert job.source is Source.OFFICIAL_ATS
        assert job.company_name == "Acme Foods"
        assert job.company_id == "acme"

    first = jobs[0]
    assert first.title == "Quality Manager"
    assert first.apply_url == "https://jobs.lever.co/acmefoods/f1a2b3c4-0001"
    assert first.location == "Melbourne"
    assert first.source_job_id == "f1a2b3c4-0001"
    assert first.description == "Lead the quality and food safety program across our plants."
    # Epoch-millis createdAt converted to an ISO date string.
    assert first.posted_date_raw == "2025-06-17"


def test_parse_accepts_list_payload(adapter: LeverAdapter) -> None:
    jobs = adapter.parse(json.loads(LIST_PAYLOAD))
    assert len(jobs) == 2
    assert jobs[1].title == "Food Safety Coordinator"
    assert jobs[1].posted_date_raw == "2025-06-10"


def test_parse_empty_or_wrong_shape_is_safe(adapter: LeverAdapter) -> None:
    assert adapter.parse("[]") == []
    assert adapter.parse([]) == []
    assert adapter.parse("{}") == []  # object instead of list -> empty


def test_missing_created_at_leaves_date_none(adapter: LeverAdapter) -> None:
    payload = [
        {
            "id": "z1",
            "text": "Role",
            "hostedUrl": "https://jobs.lever.co/acmefoods/z1",
            "categories": {},
        }
    ]
    jobs = adapter.parse(payload)
    assert len(jobs) == 1
    assert jobs[0].posted_date_raw is None
    assert jobs[0].location is None
