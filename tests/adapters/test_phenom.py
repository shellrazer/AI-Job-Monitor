"""Offline (fixture-driven) tests for :class:`PhenomAdapter`.

``parse`` is pure (no network) so these run against committed JSON fixtures
captured from live PepsiCo (``api_jobs``) and Mars (``widgets``) probes, plus a
few synthetic edge cases. A real :class:`PoliteClient` is constructed but never
asked to hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import RawJob, Source
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.phenom import PhenomAdapter

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _adapter(company: CompanyConfig) -> PhenomAdapter:
    http = PoliteClient(HttpSettings())
    return PhenomAdapter(http=http, company=company, settings=Settings())


# --------------------------------------------------------------------------- #
# api_jobs mode (PepsiCo)                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def pepsico() -> PhenomAdapter:
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
        },
    )
    return _adapter(company)


def test_api_jobs_parse_yields_rawjobs(pepsico: PhenomAdapter) -> None:
    jobs = pepsico.parse(_load("phenom_pepsico.json"))

    assert len(jobs) == 3
    for job in jobs:
        assert isinstance(job, RawJob)
        assert job.source is Source.OFFICIAL_ATS
        assert job.company_name == "PepsiCo"
        assert job.company_id == "pepsico"
        assert job.title
        assert job.apply_url

    first = jobs[0]
    assert first.title.startswith("Supplier Quality Assurance Associate Scientist")
    assert first.source_job_id == "453428"  # req_id wins
    # This record carries an explicit apply_url -> used verbatim.
    assert first.apply_url == "https://globalcareers-pepsico.icims.com/jobs/453428/login"
    assert first.location == "Chatswood, Australia"  # full_location preferred
    assert first.posted_date_raw == "2026-05-27T12:38:00+0000"
    assert first.description is None  # api_jobs leaves description unset


def test_api_jobs_parse_accepts_json_string(pepsico: PhenomAdapter) -> None:
    jobs = pepsico.parse(json.dumps(_load("phenom_pepsico.json")))
    assert len(jobs) == 3
    assert jobs[1].title == "Production Team Leader - Afternoon Shift"


def test_api_jobs_builds_apply_url_from_slug(pepsico: PhenomAdapter) -> None:
    payload = {
        "jobs": [
            {"data": {"slug": "999111", "req_id": "999111", "title": "QA Lead", "city": "Perth"}},
        ],
        "totalCount": 1,
    }
    jobs = pepsico.parse(payload)
    assert len(jobs) == 1
    # No apply_url in the record -> built from careers_base + slug.
    assert jobs[0].apply_url == "https://www.pepsicojobs.com/job/999111"
    assert jobs[0].location == "Perth"


def test_api_jobs_joins_location_parts(pepsico: PhenomAdapter) -> None:
    payload = {
        "jobs": [
            {
                "data": {
                    "slug": "1",
                    "req_id": "1",
                    "title": "x",
                    "city": "Sydney",
                    "state": "NSW",
                    "country": "Australia",
                }
            }
        ]
    }
    jobs = pepsico.parse(payload)
    assert jobs[0].location == "Sydney, NSW, Australia"


def test_api_jobs_skips_malformed_records(pepsico: PhenomAdapter) -> None:
    payload = {
        "jobs": [
            {"data": {"title": "no id no slug no url"}},  # cannot build apply_url -> skip
            "not a dict",
            {"no_data_key": True},
            {"data": {"slug": "ok1", "title": "kept"}},
        ]
    }
    jobs = pepsico.parse(payload)
    assert len(jobs) == 1
    assert jobs[0].title == "kept"
    assert jobs[0].apply_url == "https://www.pepsicojobs.com/job/ok1"


# --------------------------------------------------------------------------- #
# widgets mode (Mars)                                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mars() -> PhenomAdapter:
    company = CompanyConfig(
        company_id="mars",
        name="Mars",
        adapter="phenom",
        search={
            "mode": "widgets",
            "widgets_url": "https://careers.mars.com/widgets",
            "lang": "en_au",
            "country": "au",
            "keywords": "quality",
        },
    )
    return _adapter(company)


def test_widgets_parse_yields_rawjobs(mars: PhenomAdapter) -> None:
    jobs = mars.parse(_load("phenom_mars.json"))

    # No AU filter configured -> all 3 (incl. the NZ row) are returned.
    assert len(jobs) == 3
    for job in jobs:
        assert isinstance(job, RawJob)
        assert job.source is Source.OFFICIAL_ATS
        assert job.company_name == "Mars"
        assert job.company_id == "mars"
        assert job.title
        assert job.apply_url.startswith("https://")

    first = jobs[0]
    assert first.title == "Logistics Specialist"
    assert first.source_job_id == "R149968"  # jobId
    assert first.location == "Auckland, Auckland"  # cityState preferred
    assert "myworkdayjobs.com" in first.apply_url  # applyUrl used verbatim
    assert first.description is not None  # from descriptionTeaser
    assert "<" not in first.description  # cleaned


def test_widgets_parse_accepts_json_string(mars: PhenomAdapter) -> None:
    jobs = mars.parse(json.dumps(_load("phenom_mars.json")))
    assert len(jobs) == 3


def test_widgets_au_filter_drops_non_australian() -> None:
    company = CompanyConfig(
        company_id="kerry",
        name="Kerry",
        adapter="phenom",
        search={
            "mode": "widgets",
            "widgets_url": "https://jobs.kerry.com/widgets",
            "country": "au",
            "au_filter": True,
        },
    )
    adapter = _adapter(company)
    jobs = adapter.parse(_load("phenom_mars.json"))

    # The first fixture row is New Zealand -> dropped by the AU filter.
    assert len(jobs) == 2
    titles = {j.title for j in jobs}
    assert "Logistics Specialist" not in titles  # NZ row removed
    # The two kept rows are the Australian ones (matched on country, even though
    # their cityState location strings don't literally contain "australia").
    assert titles == {"Production Operator", "Production Operators - Multiple Positions"}


def test_widgets_apply_url_fallbacks() -> None:
    company = CompanyConfig(
        company_id="w",
        name="Widget Co",
        adapter="phenom",
        search={"mode": "widgets", "widgets_url": "https://x/widgets"},
    )
    adapter = _adapter(company)
    payload = {
        "refineSearch": {
            "totalHits": 3,
            "data": {
                "jobs": [
                    {"title": "uses jobUrl", "jobUrl": "https://x/job/1", "jobId": "1"},
                    {"title": "uses reqId only", "reqId": "REQ-2", "jobId": "2"},
                    {"title": "no url at all"},  # dropped
                ]
            },
        }
    }
    jobs = adapter.parse(payload)
    assert [j.apply_url for j in jobs] == ["https://x/job/1", "REQ-2"]
    assert jobs[0].source_job_id == "1"


# --------------------------------------------------------------------------- #
# mode auto-detection in parse()                                               #
# --------------------------------------------------------------------------- #
def test_parse_autodetects_shape_regardless_of_config() -> None:
    # An adapter configured for api_jobs still parses a widgets payload, and
    # vice versa, because parse() detects the shape from the payload itself.
    company = CompanyConfig(
        company_id="pepsico",
        name="PepsiCo",
        adapter="phenom",
        search={"mode": "api_jobs", "careers_base": "https://www.pepsicojobs.com"},
    )
    adapter = _adapter(company)

    api_jobs = adapter.parse(_load("phenom_pepsico.json"))
    widgets = adapter.parse(_load("phenom_mars.json"))
    assert len(api_jobs) == 3
    assert len(widgets) == 3
    assert api_jobs[0].title.startswith("Supplier Quality")
    assert widgets[0].title == "Logistics Specialist"


def test_parse_empty_and_malformed_payloads_are_safe(pepsico: PhenomAdapter) -> None:
    assert pepsico.parse("{}") == []
    assert pepsico.parse({"jobs": []}) == []
    assert pepsico.parse({"jobs": "not a list"}) == []
    assert pepsico.parse({"refineSearch": {}}) == []
    assert pepsico.parse({"refineSearch": {"data": {}}}) == []
    assert pepsico.parse("not a json object") == []
    assert pepsico.parse(12345) == []
