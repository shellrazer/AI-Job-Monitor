"""Unit tests for job_monitor.report."""

from __future__ import annotations

import hashlib
from datetime import date, datetime

import pytest

from job_monitor.config import CompanyRatingsConfig
from job_monitor.models import Job, PriorityTier, Source
from job_monitor.report import (
    _tab_groups,
    format_salary,
    is_aggregator,
    is_other_region,
    is_preferred,
    location_bucket,
    recruiter_mark,
    render_email_html,
    render_email_text,
    render_report,
    stars_display,
    tier_counts,
)

# Tab headings (bilingual), in render order.
TAB_PREFERRED = "优选公司 / Preferred Companies"
TAB_NSW = "综合推荐 / NSW & Remote Picks"
TAB_OTHER = "其他地区 / Other Regions"

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
    """Jobs spanning all three tabs plus a recruiter and an unknown-region posting.

    * Preferred (OFFICIAL_ATS) Sydney role -> Tab 1 only.
    * Preferred (OFFICIAL_ATS) Brisbane role -> Tab 1 AND Tab 3.
    * Aggregator (SEEK) Sydney role -> Tab 2.
    * Aggregator (Indeed) unknown-region role (location=None) -> Tab 2 (NOT Tab 3).
    * Aggregator (Jora) Melbourne role -> Tab 3.
    * Recruiter (Michael Page, LinkedIn) Sydney role -> Tab 2.
    """
    return [
        _job(
            source=Source.OFFICIAL_ATS,
            title="Preferred Sydney NQM",
            company_name="Woolworths",
            location="Sydney NSW",
            apply_url="https://example.com/pref-syd",
            final_score=90.0,
            priority_tier=PriorityTier.A_PLUS,
        ),
        _job(
            source=Source.OFFICIAL_ATS,
            title="Preferred Brisbane QM",
            company_name="Local Foods",
            location="Brisbane QLD",
            apply_url="https://example.com/pref-bne",
            final_score=85.0,
            priority_tier=PriorityTier.A,
        ),
        _job(
            source=Source.SEEK,
            title="Aggregator Sydney Lead",
            company_name="Cloud Foods",
            location="Sydney NSW",
            apply_url="https://example.com/agg-syd",
            final_score=80.0,
            priority_tier=PriorityTier.A,
        ),
        _job(
            source=Source.INDEED,
            title="Aggregator Unknown Region",
            company_name="Mystery Foods",
            location=None,
            apply_url="https://example.com/agg-unknown",
            final_score=72.0,
            priority_tier=PriorityTier.B,
        ),
        _job(
            source=Source.JORA,
            title="Aggregator Melbourne QM",
            company_name="Other Foods",
            location="Melbourne VIC",
            apply_url="https://example.com/agg-mel",
            final_score=70.0,
            priority_tier=PriorityTier.B,
        ),
        _job(
            source=Source.LINKEDIN,
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
# is_preferred / is_aggregator                                                #
# --------------------------------------------------------------------------- #
def test_is_preferred_and_is_aggregator():
    preferred = _job(source=Source.OFFICIAL_ATS)
    assert is_preferred(preferred) is True
    assert is_aggregator(preferred) is False

    for src in (Source.SEEK, Source.JORA, Source.LINKEDIN, Source.INDEED):
        agg = _job(source=src)
        assert is_aggregator(agg) is True
        assert is_preferred(agg) is False


# --------------------------------------------------------------------------- #
# is_other_region                                                             #
# --------------------------------------------------------------------------- #
def test_is_other_region():
    # Clearly another AU state -> True.
    assert is_other_region("Melbourne VIC") is True
    assert is_other_region("Brisbane") is True
    assert is_other_region("Queensland") is True
    # NSW / Sydney / remote / unknown-region / empty -> kept (False).
    assert is_other_region("Chatswood, Australia") is False
    assert is_other_region("Sydney NSW") is False
    assert is_other_region("Remote") is False
    assert is_other_region(None) is False
    assert is_other_region("") is False


# --------------------------------------------------------------------------- #
# _tab_groups — selection logic                                               #
# --------------------------------------------------------------------------- #
def test_tab_groups_selection():
    groups = _tab_groups(_split_jobs(), RATINGS)
    assert [g.heading for g in groups] == [TAB_PREFERRED, TAB_NSW, TAB_OTHER]
    preferred, nsw, other = groups

    def titles(section):
        return [item.job.title for item in section.items]

    # Tab 1: both OFFICIAL_ATS roles (all regions), score desc.
    assert titles(preferred) == ["Preferred Sydney NQM", "Preferred Brisbane QM"]

    # Tab 2: aggregator roles NOT clearly in another state, score desc. The
    # unknown-region aggregator (location=None) lands here, NOT in Tab 3.
    assert titles(nsw) == [
        "Aggregator Sydney Lead",
        "Aggregator Unknown Region",
        "Recruiter Role",
    ]
    assert "Preferred Sydney NQM" not in titles(nsw)  # preferred never lands in Tab 2
    assert "Aggregator Melbourne QM" not in titles(nsw)  # clear other-state excluded

    # Tab 3: clearly-other-state roles (any source), score desc. The Brisbane
    # preferred (85) outranks the Melbourne aggregator (70).
    assert titles(other) == ["Preferred Brisbane QM", "Aggregator Melbourne QM"]


def test_tab_groups_other_regions_dedupes_by_apply_url():
    dup_url = "https://example.com/dup"
    jobs = [
        _job(
            source=Source.JORA,
            title="Melbourne A",
            location="Melbourne VIC",
            apply_url=dup_url,
            final_score=70.0,
        ),
        _job(
            source=Source.SEEK,
            title="Melbourne B",
            location="Melbourne VIC",
            apply_url=dup_url,
            final_score=50.0,
        ),
    ]
    other = _tab_groups(jobs, RATINGS)[2]
    urls = [item.job.apply_url for item in other.items]
    assert urls.count(dup_url) == 1  # duplicate apply_url collapsed
    # First occurrence (higher score) is kept.
    assert other.items[0].job.title == "Melbourne A"


# --------------------------------------------------------------------------- #
# render_report — tabs, stars, recruiter mark                                #
# --------------------------------------------------------------------------- #
def test_render_report_three_tabs():
    html = render_report(_split_jobs(), generated_at=GENERATED_AT, ratings=RATINGS)

    # All three tab headings render (with counts).
    assert TAB_PREFERRED in html
    assert TAB_NSW in html
    assert TAB_OTHER in html
    assert f"{TAB_PREFERRED} (2)" in html
    assert f"{TAB_NSW} (3)" in html
    assert f"{TAB_OTHER} (2)" in html

    # Tabs ordered: Preferred, then NSW, then Other.
    pref_idx = html.index(TAB_PREFERRED)
    nsw_idx = html.index(TAB_NSW)
    other_idx = html.index(TAB_OTHER)
    assert pref_idx < nsw_idx < other_idx

    # Self-contained tab machinery present (no external libs).
    assert 'class="tab-btn' in html
    assert 'class="tab-panel' in html
    assert "addEventListener" in html


def test_render_report_tab_membership():
    groups = _tab_groups(_split_jobs(), RATINGS)
    preferred, nsw, other = groups

    def titles(section):
        return [item.job.title for item in section.items]

    # Preferred Sydney role is in Tab 1 and NOT in Tab 2.
    assert "Preferred Sydney NQM" in titles(preferred)
    assert "Preferred Sydney NQM" not in titles(nsw)

    # Aggregator Sydney role is in Tab 2.
    assert "Aggregator Sydney Lead" in titles(nsw)
    # Unknown-region aggregator stays in Tab 2 (kept), not Tab 3.
    assert "Aggregator Unknown Region" in titles(nsw)
    assert "Aggregator Unknown Region" not in titles(other)

    # Tab 3 has the Melbourne aggregator and the Brisbane preferred (preferred
    # ordered first here because it scores higher, not by source).
    other_titles = titles(other)
    assert "Aggregator Melbourne QM" in other_titles
    assert "Preferred Brisbane QM" in other_titles
    assert other_titles.index("Preferred Brisbane QM") < other_titles.index(
        "Aggregator Melbourne QM"
    )


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


def test_render_report_empty_tab_is_skipped():
    # No clearly-other-state jobs -> the 其他地区 tab/heading is dropped entirely,
    # leaving a clean two-tab (preferred + NSW) view.
    nsw_jobs = [
        _job(
            source=Source.OFFICIAL_ATS,
            title="Preferred Sydney",
            company_name="Woolworths",
            location="Sydney NSW",
            apply_url="https://example.com/p",
            final_score=80.0,
            priority_tier=PriorityTier.A,
        ),
        _job(
            source=Source.SEEK,
            title="Aggregator Sydney",
            company_name="Cloud Foods",
            location="Sydney NSW",
            apply_url="https://example.com/a",
            final_score=70.0,
            priority_tier=PriorityTier.B,
        ),
    ]
    html = render_report(nsw_jobs, generated_at=GENERATED_AT, ratings=RATINGS)
    assert TAB_PREFERRED in html
    assert TAB_NSW in html
    # Empty Other Regions tab disappears: no heading, no leftover placeholder.
    assert TAB_OTHER not in html
    assert "No roles in this section" not in html


def test_render_report_all_empty_shows_no_matches():
    # All three groups empty -> single muted "No matches" note, no tabs.
    html = render_report([], generated_at=GENERATED_AT, ratings=RATINGS)
    assert "无匹配职位 / No matches" in html
    assert 'class="tab-btn' not in html


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
    assert "No matches" in html


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
    # sample_jobs are all Sydney -> only the preferred + NSW groups render; the
    # empty 其他地区 group is skipped.
    assert TAB_PREFERRED in text
    assert TAB_NSW in text
    assert TAB_OTHER not in text


def test_render_email_text_sections_and_marks():
    text = render_email_text(_split_jobs(), generated_at=GENERATED_AT, ratings=RATINGS)
    # Three stacked groups in order (优选公司 → 综合推荐 → 其他地区).
    assert TAB_PREFERRED in text
    assert TAB_NSW in text
    assert TAB_OTHER in text
    assert text.index(TAB_PREFERRED) < text.index(TAB_NSW) < text.index(TAB_OTHER)
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
    # sample_jobs are all Sydney -> preferred + NSW groups render; the empty
    # 其他地区 group is skipped.
    assert TAB_PREFERRED in html
    assert TAB_NSW in html
    assert TAB_OTHER not in html


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
