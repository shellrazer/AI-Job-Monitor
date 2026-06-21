"""Unit tests for job_monitor.report."""

from __future__ import annotations

import hashlib
from datetime import date, datetime

import pytest

from job_monitor.models import Job, PriorityTier, Source
from job_monitor.report import (
    format_salary,
    render_email_html,
    render_email_text,
    render_report,
    tier_counts,
)

GENERATED_AT = datetime(2026, 6, 21, 7, 0)


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

    html = render_report(jobs, generated_at=GENERATED_AT, subtitle="Daily run")

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
    html = render_report(jobs, generated_at=GENERATED_AT)
    assert "card--alert" in html

    # Strong alert (without A+) should also trigger the highlighted class.
    only_strong = [_job(strong_alert=True, final_score=70.0, priority_tier=PriorityTier.B)]
    html2 = render_report(only_strong, generated_at=GENERATED_AT)
    assert "card--alert" in html2


def test_render_report_empty():
    html = render_report([], generated_at=GENERATED_AT)
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
    text = render_email_text(jobs, generated_at=GENERATED_AT)
    for job in jobs:
        assert job.apply_url in text
        assert job.title in text
    assert "Job Monitor Digest" in text


def test_render_email_html(sample_jobs):
    jobs = list(sample_jobs)
    jobs[0].final_score = 88.0
    jobs[0].priority_tier = PriorityTier.A_PLUS
    jobs[0].strong_alert = True
    html = render_email_html(jobs, generated_at=GENERATED_AT)
    assert jobs[0].apply_url in html
    assert jobs[0].title in html
    # Inline-styled alert card present for the A+/strong job.
    assert "#d92d20" in html


def test_render_report_sorts_by_score_desc():
    low = _job(title="LowScoreJob", final_score=10.0)
    high = _job(title="HighScoreJob", final_score=99.0)
    none = _job(title="NoScoreJob", final_score=None)
    html = render_report([low, none, high], generated_at=GENERATED_AT)
    # Highest score first, then lower, then unscored last.
    assert html.index("HighScoreJob") < html.index("LowScoreJob") < html.index("NoScoreJob")


@pytest.mark.parametrize("subtitle", ["", "Morning digest"])
def test_render_report_subtitle_optional(sample_jobs, subtitle):
    html = render_report(sample_jobs, generated_at=GENERATED_AT, subtitle=subtitle)
    if subtitle:
        assert subtitle in html
