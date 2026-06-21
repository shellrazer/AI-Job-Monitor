"""Offline (fixture-driven) tests for :class:`GreenhouseAdapter`.

``parse`` is pure (no network), so these run against an inline JSON fixture with
a real :class:`PoliteClient` that is never asked to hit the network.
"""

from __future__ import annotations

import json

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import RawJob, Source
from job_monitor.sources.greenhouse import GreenhouseAdapter
from job_monitor.sources.http import PoliteClient

BOARD_TOKEN = "acmefoods"

# Representative board payload. ``content`` is HTML-escaped, as the real API
# returns it.
LIST_PAYLOAD = json.dumps(
    {
        "jobs": [
            {
                "id": 4012345,
                "title": "Quality Manager",
                "absolute_url": "https://boards.greenhouse.io/acmefoods/jobs/4012345",
                "location": {"name": "Melbourne, VIC"},
                "updated_at": "2026-06-18T12:00:00-04:00",
                "content": "&lt;p&gt;Own our &lt;b&gt;HACCP&lt;/b&gt; &amp;amp; GMP programs.&lt;/p&gt;",
            },
            {
                "id": 4012346,
                "title": "Food Safety Lead",
                "absolute_url": "https://boards.greenhouse.io/acmefoods/jobs/4012346",
                "location": {"name": "Sydney, NSW"},
                "updated_at": "2026-06-12T08:00:00-04:00",
                "content": "&lt;p&gt;Drive food safety audits.&lt;/p&gt;",
            },
            {
                # No absolute_url -> must be skipped defensively.
                "id": 4012347,
                "title": "Broken Posting",
                "location": {"name": "Nowhere"},
            },
        ]
    }
)


@pytest.fixture
def company() -> CompanyConfig:
    return CompanyConfig(
        company_id="acme",
        name="Acme Foods",
        adapter="greenhouse",
        search={"board_token": BOARD_TOKEN},
    )


@pytest.fixture
def adapter(company: CompanyConfig) -> GreenhouseAdapter:
    http = PoliteClient(HttpSettings())
    return GreenhouseAdapter(http=http, company=company, settings=Settings())


def test_parse_yields_correct_rawjobs(adapter: GreenhouseAdapter) -> None:
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
    assert first.apply_url == "https://boards.greenhouse.io/acmefoods/jobs/4012345"
    assert first.location == "Melbourne, VIC"
    assert first.source_job_id == "4012345"
    assert first.posted_date_raw == "2026-06-18T12:00:00-04:00"
    # HTML-escaped content is unescaped and cleaned to plain text.
    assert first.description
    assert "<" not in first.description  # tags stripped after unescape
    assert "&amp;" not in first.description  # entities decoded
    assert "HACCP" in first.description
    assert "GMP" in first.description


def test_parse_accepts_dict_payload(adapter: GreenhouseAdapter) -> None:
    jobs = adapter.parse(json.loads(LIST_PAYLOAD))
    assert len(jobs) == 2
    assert jobs[1].title == "Food Safety Lead"
    assert jobs[1].description == "Drive food safety audits."


def test_parse_empty_payload_is_safe(adapter: GreenhouseAdapter) -> None:
    assert adapter.parse("{}") == []
    assert adapter.parse({"jobs": []}) == []
    assert adapter.parse("[]") == []  # wrong shape (list, not object)


def test_missing_content_leaves_description_none(adapter: GreenhouseAdapter) -> None:
    payload = {
        "jobs": [
            {
                "id": 1,
                "title": "Role",
                "absolute_url": "https://boards.greenhouse.io/acmefoods/jobs/1",
            }
        ]
    }
    jobs = adapter.parse(payload)
    assert len(jobs) == 1
    assert jobs[0].description is None
    assert jobs[0].location is None
