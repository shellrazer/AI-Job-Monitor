"""Offline (fixture-driven) tests for the LinkedIn source adapter."""

from __future__ import annotations

import pytest

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import RawJob, Source
from job_monitor.sources.http import PoliteClient
from job_monitor.sources.linkedin import LinkedInAdapter

# A small HTML fragment mirroring the real LinkedIn guest jobs response: three
# usable cards plus a fourth card that is missing its title (must be skipped).
SAMPLE_HTML = """
<!DOCTYPE html>
<li>
  <div class="base-card relative w-full base-card--link base-search-card base-search-card--link job-search-card"
       data-entity-urn="urn:li:jobPosting:4426694701" data-impression-id="jobs-search-result-0">
    <a class="base-card__full-link"
       href="https://au.linkedin.com/jobs/view/quality-assurance-manager-at-eq-food-4426694701?position=1&amp;pageNum=0&amp;refId=abc">
      <span class="sr-only">Quality Assurance Manager</span>
    </a>
    <div class="base-search-card__info">
      <h3 class="base-search-card__title">
        Quality Assurance Manager
      </h3>
      <h4 class="base-search-card__subtitle">
        <a class="hidden-nested-link" href="https://au.linkedin.com/company/eq-food">
          EQ Food
        </a>
      </h4>
      <div class="base-search-card__metadata">
        <span class="job-search-card__location">
          Minto, New South Wales, Australia
        </span>
        <time class="job-search-card__listdate" datetime="2026-06-12">
          1 week ago
        </time>
      </div>
    </div>
  </div>
</li>
<li>
  <div class="base-card relative w-full base-card--link base-search-card base-search-card--link job-search-card"
       data-entity-urn="urn:li:jobPosting:4428774076" data-impression-id="jobs-search-result-1">
    <a class="base-card__full-link"
       href="https://au.linkedin.com/jobs/view/qc-manager-at-david-russo-pty-ltd-4428774076?position=2&amp;pageNum=0">
      <span class="sr-only">QC Manager</span>
    </a>
    <div class="base-search-card__info">
      <h3 class="base-search-card__title">QC Manager</h3>
      <h4 class="base-search-card__subtitle">
        <a class="hidden-nested-link" href="https://au.linkedin.com/company/david-russo">
          David Russo Pty Ltd
        </a>
      </h4>
      <div class="base-search-card__metadata">
        <span class="job-search-card__location">Sydney, New South Wales, Australia</span>
        <time class="job-search-card__listdate" datetime="2026-06-17">4 days ago</time>
      </div>
    </div>
  </div>
</li>
<li>
  <div class="base-card relative w-full base-card--link base-search-card base-search-card--link job-search-card"
       data-entity-urn="urn:li:jobPosting:4422537148" data-impression-id="jobs-search-result-2">
    <a class="base-card__full-link"
       href="https://au.linkedin.com/jobs/view/site-quality-manager-at-manufacturing-4422537148?position=3">
      <span class="sr-only">Site Quality Manager</span>
    </a>
    <div class="base-search-card__info">
      <h3 class="base-search-card__title">Site Quality Manager</h3>
      <h4 class="base-search-card__subtitle">Manufacturing</h4>
      <div class="base-search-card__metadata">
        <span class="job-search-card__location">Rooty Hill, New South Wales, Australia</span>
        <time class="job-search-card__listdate" datetime="2026-05-31">3 weeks ago</time>
      </div>
    </div>
  </div>
</li>
<li>
  <div class="base-card relative w-full base-card--link base-search-card base-search-card--link job-search-card"
       data-entity-urn="urn:li:jobPosting:9999999999" data-impression-id="jobs-search-result-3">
    <a class="base-card__full-link"
       href="https://au.linkedin.com/jobs/view/mystery-role-9999999999?position=4">
      <span class="sr-only"></span>
    </a>
    <div class="base-search-card__info">
      <h4 class="base-search-card__subtitle">No Title Co</h4>
      <div class="base-search-card__metadata">
        <span class="job-search-card__location">Somewhere, NSW</span>
      </div>
    </div>
  </div>
</li>
"""


@pytest.fixture
def adapter() -> LinkedInAdapter:
    company = CompanyConfig(
        company_id="li1",
        name="LinkedIn",
        adapter="linkedin",
        search={"keywords": "Quality Manager food", "location": "Sydney NSW"},
    )
    http = PoliteClient(HttpSettings())
    return LinkedInAdapter(http=http, company=company, settings=Settings())


def test_parse_returns_expected_jobs(adapter: LinkedInAdapter) -> None:
    jobs = adapter.parse(SAMPLE_HTML)

    # Three usable cards; the fourth (missing title) is skipped.
    assert len(jobs) == 3
    for job in jobs:
        assert isinstance(job, RawJob)
        assert job.source is Source.LINKEDIN
        assert job.title
        assert "linkedin.com/jobs/view/" in job.apply_url
        # Tracking query string must have been stripped.
        assert "?" not in job.apply_url


def test_parse_first_card_fields(adapter: LinkedInAdapter) -> None:
    jobs = adapter.parse(SAMPLE_HTML)
    job = next(j for j in jobs if j.source_job_id == "4426694701")

    assert job.title == "Quality Assurance Manager"
    assert job.company_name == "EQ Food"
    assert job.location == "Minto, New South Wales, Australia"
    assert job.apply_url == (
        "https://au.linkedin.com/jobs/view/quality-assurance-manager-at-eq-food-4426694701"
    )
    assert job.posted_date_raw == "2026-06-12"
    assert job.company_id == "li1"


def test_company_from_plain_subtitle(adapter: LinkedInAdapter) -> None:
    # The third card has a bare ``h4`` subtitle (no nested anchor).
    jobs = adapter.parse(SAMPLE_HTML)
    job = next(j for j in jobs if j.source_job_id == "4422537148")
    assert job.company_name == "Manufacturing"


def test_source_job_id_parsed_for_all(adapter: LinkedInAdapter) -> None:
    jobs = adapter.parse(SAMPLE_HTML)
    ids = {job.source_job_id for job in jobs}
    assert ids == {"4426694701", "4428774076", "4422537148"}


def test_cards_missing_title_are_skipped(adapter: LinkedInAdapter) -> None:
    jobs = adapter.parse(SAMPLE_HTML)
    # The "No Title Co" card (urn ...9999999999) had no <h3> title and must
    # not produce a RawJob.
    assert all(job.source_job_id != "9999999999" for job in jobs)


def test_apply_urls_are_deduplicated(adapter: LinkedInAdapter) -> None:
    jobs = adapter.parse(SAMPLE_HTML)
    apply_urls = [job.apply_url for job in jobs]
    assert len(apply_urls) == len(set(apply_urls))


def test_parse_empty_html_is_safe(adapter: LinkedInAdapter) -> None:
    assert adapter.parse("") == []
    assert adapter.parse("<html><body>no jobs here</body></html>") == []
