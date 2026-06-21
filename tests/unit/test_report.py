"""Unit tests for job_monitor.report."""

from __future__ import annotations

import hashlib
from datetime import date, datetime

import pytest

from job_monitor.config import CompanyRatingsConfig
from job_monitor.models import Job, PriorityTier, Source
from job_monitor.report import (
    format_salary,
    location_bucket,
    recruiter_mark,
    render_email_html,
    render_email_text,
    render_report,
    stars_display,
    tier_counts,
)

GENERATED_AT = datetime(2026, 6, 21, 7, 0)

# Ratings keys are pre-normalized lowercase (config normalizes at load time).
RATINGS = CompanyRatingsConfig(default_stars=3, star_boost=2.5, ratings={"woolworths": 5})


def _job(**overrides) -> Job:
    base = {
        "source": Source.SEEK,
        "title": "Test Role",
        "normalized_title": "test role",
        "company_name": "Test Co",
        "apply_url": "https://example.com/test",
        "description_hash": hashlib.sha256(b"test").hexdigest(),
    }
    base.update(overrides)
    return Job(**base)


def _split_jobs() -> list[Job]:
    """Four jobs spanning both sections plus a recruiter posting."""
    return [
        _job(
            title="Sydney NQM",
            company_name="Woolworths",
            location="Sydney NSW",
            apply_url="https://example.com/syd",
            final_score=90.0,
            priority_tier=PriorityTier.A_PLUS,
        ),
        _job(
            title="Melbourne QM",
            company_name="Local Foods",
            location="Melbourne VIC",
            apply_url="https://example.com/mel",
            final_score=70.0,
            priority_tier=PriorityTier.B,
        ),
        _job(
            title="Remote QA Lead",
            company_name="Cloud Foods",
            location="Remote, Australia",
            apply_url="https://example.com/rem",
            final_score=80.0,
            priority_tier=PriorityTier.A,
        ),
        _job(
            title="Recruiter Role",
            company_name="Michael Page",
            location="Sydney NSW",
            apply_url="https://example.com/rec",
            final_score=60.0,
            priority_tier=PriorityTier.B,
        ),
    ]


# --------------------------------------------------------------------------- #
# location_bucket                                                             #
# --------------------------------------------------------------------------- #
def test_location_bucket():
    assert location_bucket("Remote - Anywhere") == "nsw_remote"
    assert location_bucket("Sydney NSW") == "nsw_remote"
    assert location_bucket("Melbourne VIC") == "other_au"
    assert location_bucket("Perth WA") == "other_au"
    # Remote marker wins even for a non-NSW city.
    assert location_bucket("Work from home, Brisbane") == "nsw_remote"
    assert location_bucket(None) == "other_au"


# --------------------------------------------------------------------------- #
# stars_display / recruiter_mark                                              #
# --------------------------------------------------------------------------- #
def test_stars_display_rated_and_default():
    assert stars_display("Woolworths", RATINGS) == "★★★★★ (5/5)"
    bar = stars_display("Some Other Co", RATINGS)
    assert "★" in bar
    assert "(3/5)" in bar


def test_stars_display_recruiter_is_blank():
    assert stars_display("Michael Page", RATINGS) == ""


def test_recruiter_mark():
    assert "Recruiter" in recruiter_mark("Michael Page")
    assert recruiter_mark("Woolworths") == ""


# --------------------------------------------------------------------------- #
# render_report — sections, stars, recruiter mark                            #
# --------------------------------------------------------------------------- #
def test_render_report_two_sections_split_by_location():
    html = render_report(_split_jobs(), generated_at=GENERATED_AT, ratings=RATINGS)

    # Both section headings present.
    assert "NSW & Remote / 新州与远程" in html
    assert "Other Australia / 其他州" in html

    nsw_idx = html.index("NSW & Remote / 新州与远程")
    other_idx = html.index("Other Australia / 其他州")
    assert nsw_idx < other_idx  # NSW section first

    # Sydney + remote jobs live in the NSW section (before the Other heading).
    assert nsw_idx < html.index("Sydney NQM") < other_idx
    assert nsw_idx < html.index("Remote QA Lead") < other_idx
    assert nsw_idx < html.index("Recruiter Role") < other_idx
    # Melbourne job lives in the Other Australia section.
    assert html.index("Melbourne QM") > other_idx


def test_render_report_stars_and_recruiter_marks():
    html = render_report(_split_jobs(), generated_at=GENERATED_AT, ratings=RATINGS)
    # Rated company shows a star bar.
    assert "★" in html
    assert "(5/5)" in html
    # Bilingual rating row label present.
    assert "评分" in html
    # Recruiter job shows the recruiter mark.
    assert "Recruiter" in html
    assert "招聘中介" in html
    # Recruiter job carries NO star bar — only one job (Woolworths) is rated,
    # so exactly one "(5/5)" and no "/5" entry attributable to Michael Page.
    recruiter_only = [
        _job(company_name="Michael Page", location="Sydney NSW", final_score=50.0)
    ]
    html2 = render_report(recruiter_only, generated_at=GENERATED_AT, ratings=RATINGS)
    assert "★" not in html2  # no star bar at all for a recruiter-only page
    assert "招聘中介" in html2


def test_render_report_section_empty_placeholder():
    nsw_only = [_job(location="Sydney NSW", final_score=50.0)]
    html = render_report(nsw_only, generated_at=GENERATED_AT, ratings=RATINGS)
    # Other Australia section is empty -> shows the muted placeholder.
    assert "No roles in this section" in html


def test_render_report_contains_titles_urls_and_scores(sample_jobs):
    jobs = list(sample_jobs)
    jobs[0].final_score = 88.5
    jobs[0].priority_tier = PriorityTier.A_PLUS
    jobs[0].strong_alert = True
    jobs[0].match_reasons = ["Senior multi-site leadership"]
    jobs[0].resume_tips = ["Lead with FSSC 22000 experience"]
    jobs[0].posted_date = date(2026, 6, 18)
    jobs[1].final_score = 42.0
    jobs[1].priority_tier = PriorityTier.C

    html = render_report(jobs, generated_at=GENERATED_AT, ratings=RATINGS, subtitle="Daily run")

    for job in jobs:
        assert job.title in html
        assert job.apply_url in html
    # Bilingual labels present.
    assert "公司" in html
    assert "Company" in html
    assert "职位" in html
    assert "申请链接" in html
    # Score and reasons surfaced.
    assert "88.5" in html
    assert "Senior multi-site leadership" in html
    assert "Lead with FSSC 22000 experience" in html
    assert "Daily run" in html


def test_render_report_alert_class_for_strong_or_a_plus(sample_jobs):
    jobs = list(sample_jobs)
    jobs[0].priority_tier = PriorityTier.A_PLUS
    jobs[0].final_score = 90.0
    jobs[1].final_score = 30.0
    html = render_report(jobs, generated_at=GENERATED_AT, ratings=RATINGS)
    assert "card--alert" in html

    # Strong alert (without A+) should also trigger the highlighted class.
    only_strong = [_job(strong_alert=True, final_score=70.0, priority_tier=PriorityTier.B)]
    html2 = render_report(only_strong, generated_at=GENERATED_AT, ratings=RATINGS)
    assert "card--alert" in html2


def test_render_report_empty():
    html = render_report([], generated_at=GENERATED_AT, ratings=RATINGS)
    assert "No matching jobs" in html


def test_tier_counts():
    jobs = [
        _job(priority_tier=PriorityTier.A_PLUS),
        _job(priority_tier=PriorityTier.A_PLUS),
        _job(priority_tier=PriorityTier.B),
        _job(priority_tier=None),
    ]
    counts = tier_counts(jobs)
    assert counts["A+"] == 2
    assert counts["B"] == 1
    assert counts["Unscored"] == 1
    assert "A" not in counts  # absent tiers are omitted


def test_format_salary_disclosed_range():
    job = _job(salary_min=160000, salary_max=190000, salary_currency="AUD", salary_period="year")
    s = format_salary(job)
    assert "160,000" in s
    assert "190,000" in s
    assert "AUD" in s
    assert "year" in s


def test_format_salary_single_value():
    job = _job(salary_min=120000, salary_max=120000, salary_currency="AUD", salary_period="year")
    s = format_salary(job)
    assert "120,000" in s
    assert " - " not in s


def test_format_salary_raw_fallback():
    job = _job(salary_raw="$140k - $160k + super")
    assert format_salary(job) == "$140k - $160k + super"


def test_format_salary_not_disclosed():
    job = _job()
    assert format_salary(job) == "Not disclosed / 未披露"


def test_render_email_text_contains_apply_urls(sample_jobs):
    jobs = list(sample_jobs)
    jobs[0].final_score = 88.0
    jobs[0].priority_tier = PriorityTier.A_PLUS
    jobs[1].final_score = 40.0
    text = render_email_text(jobs, generated_at=GENERATED_AT, ratings=RATINGS)
    for job in jobs:
        assert job.apply_url in text
        assert job.title in text
    assert "Job Monitor Digest" in text
    # Section headings appear in the plaintext digest too.
    assert "NSW & Remote / 新州与远程" in text


def test_render_email_text_sections_and_marks():
    text = render_email_text(_split_jobs(), generated_at=GENERATED_AT, ratings=RATINGS)
    assert "NSW & Remote / 新州与远程" in text
    assert "Other Australia / 其他州" in text
    # Rated star bar + recruiter mark surface in the digest.
    assert "★★★★★ (5/5)" in text
    assert "招聘中介" in text


def test_render_email_html(sample_jobs):
    jobs = list(sample_jobs)
    jobs[0].final_score = 88.0
    jobs[0].priority_tier = PriorityTier.A_PLUS
    jobs[0].strong_alert = True
    html = render_email_html(jobs, generated_at=GENERATED_AT, ratings=RATINGS)
    assert jobs[0].apply_url in html
    assert jobs[0].title in html
    # Inline-styled alert card present for the A+/strong job.
    assert "#d92d20" in html
    # Section headings present in the email body.
    assert "NSW & Remote / 新州与远程" in html


def test_render_report_sorts_by_score_desc():
    # All NSW so they share a section; ordering within the section is by score.
    low = _job(title="LowScoreJob", location="Sydney NSW", final_score=10.0)
    high = _job(title="HighScoreJob", location="Sydney NSW", final_score=99.0)
    none = _job(title="NoScoreJob", location="Sydney NSW", final_score=None)
    html = render_report([low, none, high], generated_at=GENERATED_AT, ratings=RATINGS)
    # Highest score first, then lower, then unscored last.
    assert html.index("HighScoreJob") < html.index("LowScoreJob") < html.index("NoScoreJob")


@pytest.mark.parametrize("subtitle", ["", "Morning digest"])
def test_render_report_subtitle_optional(sample_jobs, subtitle):
    html = render_report(
        sample_jobs, generated_at=GENERATED_AT, ratings=RATINGS, subtitle=subtitle
    )
    if subtitle:
        assert subtitle in html
