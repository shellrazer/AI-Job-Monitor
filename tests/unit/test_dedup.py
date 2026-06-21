"""Tests for job_monitor.dedup."""

from __future__ import annotations

from datetime import date

from job_monitor.dedup import (
    _city,
    _dates_within,
    _locations_compatible,
    _normalize_company,
    _title_core,
    dedup_key,
    deduplicate,
    description_hash,
    group_duplicates,
    normalize_title,
    pick_best,
)
from job_monitor.models import Job, JobStatus, RawJob, Source
from job_monitor.normalize import parse_posted_date, parse_salary


def _to_job(raw: RawJob) -> Job:
    """Minimal RawJob -> Job conversion for dedup tests."""
    salary = parse_salary(raw.salary_raw)
    return Job(
        source=raw.source,
        title=raw.title,
        normalized_title=normalize_title(raw.title),
        company_name=raw.company_name,
        apply_url=raw.apply_url,
        description_hash=description_hash(raw.description),
        source_job_id=raw.source_job_id,
        company_id=raw.company_id,
        location=raw.location,
        posted_date=parse_posted_date(raw.posted_date_raw),
        description=raw.description,
        salary_min=salary.min,
        salary_max=salary.max,
        salary_currency=salary.currency,
        salary_period=salary.period,
        salary_raw=raw.salary_raw,
    )


def test_pick_best_prefers_official_ats_over_seek() -> None:
    official = Job(
        source=Source.OFFICIAL_ATS,
        title="Site Quality Manager",
        normalized_title="site quality manager",
        company_name="Bega Group",
        apply_url="https://example.com/jobs/sqm-1",
        description_hash=description_hash("short"),
        description="short",
    )
    seek = Job(
        source=Source.SEEK,
        title="Site Quality Manager",
        normalized_title="site quality manager",
        company_name="Bega Group",
        apply_url="https://seek.com.au/job/12345",
        description_hash=description_hash("a much longer description here"),
        description="a much longer description here",
    )
    # SEEK has a longer description but OFFICIAL_ATS still wins on source rank.
    canonical, losers = pick_best([seek, official])
    assert canonical is official
    assert losers == [seek]


def test_pick_best_tie_break_by_longest_description() -> None:
    a = Job(
        source=Source.SEEK,
        title="X",
        normalized_title="x",
        company_name="Co",
        apply_url="https://seek.com.au/a",
        description_hash=description_hash("short"),
        description="short",
    )
    b = Job(
        source=Source.SEEK,
        title="X",
        normalized_title="x",
        company_name="Co",
        apply_url="https://seek.com.au/b",
        description_hash=description_hash("a longer description wins the tie"),
        description="a longer description wins the tie",
    )
    canonical, losers = pick_best([a, b])
    assert canonical is b
    assert losers == [a]


def test_deduplicate_collapses_cross_source_duplicate(sample_raw_jobs: list[RawJob]) -> None:
    jobs = [_to_job(r) for r in sample_raw_jobs]
    result = deduplicate(jobs)

    # All jobs returned.
    assert len(result) == len(jobs)

    # Two Bega "Site Quality Manager" postings collapse to one canonical + one duplicate.
    bega = [j for j in result if j.company_name == "Bega Group"]
    assert len(bega) == 2
    canonical = [j for j in bega if j.status != JobStatus.DUPLICATE]
    losers = [j for j in bega if j.status == JobStatus.DUPLICATE]
    assert len(canonical) == 1
    assert len(losers) == 1
    # Canonical is the official ATS posting.
    assert canonical[0].source is Source.OFFICIAL_ATS
    # Loser is the SEEK posting, marked DUPLICATE.
    assert losers[0].source is Source.SEEK
    assert losers[0].status is JobStatus.DUPLICATE

    # The software QA job stays separate and is NOT marked duplicate.
    qa = [j for j in result if j.company_name == "Tech Co"]
    assert len(qa) == 1
    assert qa[0].status is JobStatus.NEW


def test_deduplicate_does_not_set_duplicate_of(sample_raw_jobs: list[RawJob]) -> None:
    jobs = [_to_job(r) for r in sample_raw_jobs]
    result = deduplicate(jobs)
    # duplicate_of is assigned by the pipeline (needs DB ids), never here.
    assert all(j.duplicate_of is None for j in result)


def test_group_duplicates_returns_canonical_and_losers(sample_raw_jobs: list[RawJob]) -> None:
    jobs = [_to_job(r) for r in sample_raw_jobs]
    groups = group_duplicates(jobs)
    # Two distinct groups: the Bega pair and the lone Tech Co job.
    assert len(groups) == 2
    sizes = sorted(1 + len(losers) for _, losers in groups)
    assert sizes == [1, 2]
    for canonical, losers in groups:
        if canonical.company_name == "Bega Group":
            assert canonical.source is Source.OFFICIAL_ATS
            assert len(losers) == 1


def test_dedup_key_mirrors_db_index() -> None:
    job = _to_job(
        RawJob(
            source=Source.SEEK,
            title="Site Quality Manager",
            company_name="Bega Group",
            apply_url="https://seek.com.au/job/12345",
            source_job_id="12345",
            location="Western Sydney NSW",
            description="x",
            posted_date_raw="2026-06-18",
        )
    )
    assert dedup_key(job) == (
        "site quality manager",
        "bega group",
        "Western Sydney NSW",
        "2026-06-18",
        "12345",
    )


def test_dedup_key_falls_back_to_apply_url_when_no_source_id() -> None:
    job = Job(
        source=Source.JORA,
        title="QA Lead",
        normalized_title="qa lead",
        company_name="Co",
        apply_url="https://jora.com/x",
        description_hash=description_hash("d"),
    )
    assert dedup_key(job)[-1] == "https://jora.com/x"


# --- helper unit asserts --------------------------------------------------


def test_normalize_company_examples() -> None:
    assert _normalize_company("PepsiCo Australia") == "pepsico"
    assert _normalize_company("Saputo Dairy Australia") == "saputo dairy"
    assert _normalize_company("The Arnott's Group") == "arnotts"
    # & expands, legal tokens drop.
    assert _normalize_company("Mondelez Pty Ltd") == "mondelez"
    # Empty / garbage collapses to "".
    assert _normalize_company("") == ""
    assert _normalize_company("Ltd Pty The") == ""


def test_title_core_examples() -> None:
    assert (
        _title_core("supplier quality assurance associate scientist 12 month fixed term contract")
        == "supplier quality assurance associate scientist"
    )
    # Order preserved, noise + standalone digits stripped.
    assert _title_core("quality manager full time permanent") == "quality manager"
    assert _title_core("night shift warehouse operator 2") == "warehouse operator"
    assert _title_core("") == ""


def test_city_and_location_helpers() -> None:
    assert _city("Chatswood, Australia") == "chatswood"
    assert _city("Sydney NSW") == "sydney nsw"
    assert _city(None) == ""
    assert _city("") == ""

    # Empty city on either side -> compatible (unknown).
    assert _locations_compatible("", "Sydney NSW") is True
    assert _locations_compatible(None, "Sydney NSW") is True
    # Shared token -> compatible.
    assert _locations_compatible("Sydney", "Sydney NSW") is True
    # No shared token -> incompatible.
    assert _locations_compatible("Sydney NSW", "Melbourne VIC") is False


def test_dates_within_window() -> None:
    assert _dates_within(None, date(2026, 6, 1)) is True
    assert _dates_within(date(2026, 6, 1), None) is True
    assert _dates_within(date(2026, 6, 1), date(2026, 6, 22)) is True  # 21 days (boundary)
    assert _dates_within(date(2026, 6, 1), date(2026, 6, 23)) is False  # 22 days
    assert _dates_within(date(2026, 6, 1), date(2026, 7, 1)) is False  # 30 days
    assert _dates_within(date(2026, 6, 1), date(2026, 7, 1), days=30) is True


# --- cross-source merge pass ---------------------------------------------


def test_cross_source_pepsico_collapse() -> None:
    """Same role on official_ats + jora + linkedin collapses to one cluster."""
    official = Job(
        source=Source.OFFICIAL_ATS,
        title="Supplier Quality Assurance Associate Scientist",
        normalized_title=normalize_title("Supplier Quality Assurance Associate Scientist"),
        company_name="PepsiCo Australia",
        apply_url="https://pepsico.com/jobs/JR1",
        description_hash=description_hash("A long official description of the role " * 10),
        source_job_id="JR1",
        location="Chatswood NSW",
        posted_date=date(2026, 6, 18),
        description="A long official description of the role " * 10,
    )
    jora_title = "Supplier Quality Assurance Associate Scientist (12-Month Fixed Term Contract)"
    jora = Job(
        source=Source.JORA,
        title=jora_title,
        normalized_title=normalize_title(jora_title),
        company_name="PepsiCo",
        apply_url="https://jora.com/job/jora9",
        description_hash=description_hash("totally different jora blurb"),
        source_job_id="jora9",
        location="",
        posted_date=date(2026, 6, 19),
        description="totally different jora blurb",
    )
    linkedin = Job(
        source=Source.LINKEDIN,
        title="Supplier Quality Assurance Associate Scientist",
        normalized_title=normalize_title("Supplier Quality Assurance Associate Scientist"),
        company_name="PepsiCo",
        apply_url="https://linkedin.com/jobs/li5",
        description_hash=description_hash("linkedin teaser text"),
        source_job_id="li5",
        location="Chatswood, Australia",
        posted_date=date(2026, 6, 20),
        description="linkedin teaser text",
    )

    groups = group_duplicates([official, jora, linkedin])
    assert len(groups) == 1
    canonical, losers = groups[0]
    assert canonical.source is Source.OFFICIAL_ATS
    assert len(losers) == 2
    assert {loser.source for loser in losers} == {Source.JORA, Source.LINKEDIN}

    deduplicate([official, jora, linkedin])
    assert official.status is JobStatus.NEW
    assert jora.status is JobStatus.DUPLICATE
    assert linkedin.status is JobStatus.DUPLICATE


def test_cross_source_recruiter_jora_seek_prefers_seek() -> None:
    """Same recruiter role on jora + seek merges; seek wins (preferred over jora)."""
    jora = Job(
        source=Source.JORA,
        title="Quality Assurance Officer (Casual)",
        normalized_title=normalize_title("Quality Assurance Officer (Casual)"),
        company_name="Hays Recruitment",
        apply_url="https://jora.com/job/abc",
        description_hash=description_hash("jora copy"),
        source_job_id="abc",
        location="Brisbane QLD",
        posted_date=date(2026, 6, 10),
        description="jora copy",
    )
    seek = Job(
        source=Source.SEEK,
        title="Quality Assurance Officer",
        normalized_title=normalize_title("Quality Assurance Officer"),
        company_name="Hays Recruitment",
        apply_url="https://seek.com.au/job/777",
        description_hash=description_hash("seek copy different"),
        source_job_id="777",
        location="Brisbane",
        posted_date=date(2026, 6, 12),
        description="seek copy different",
    )

    groups = group_duplicates([jora, seek])
    assert len(groups) == 1
    canonical, losers = groups[0]
    assert canonical.source is Source.SEEK
    assert [loser.source for loser in losers] == [Source.JORA]


def test_cross_source_negative_different_city_does_not_merge() -> None:
    sydney = Job(
        source=Source.SEEK,
        title="Quality Manager",
        normalized_title=normalize_title("Quality Manager"),
        company_name="Bega Group",
        apply_url="https://seek.com.au/job/1",
        description_hash=description_hash("syd"),
        source_job_id="1",
        location="Sydney NSW",
        posted_date=date(2026, 6, 1),
        description="syd",
    )
    melbourne = Job(
        source=Source.LINKEDIN,
        title="Quality Manager",
        normalized_title=normalize_title("Quality Manager"),
        company_name="Bega Group",
        apply_url="https://linkedin.com/jobs/2",
        description_hash=description_hash("mel"),
        source_job_id="2",
        location="Melbourne VIC",
        posted_date=date(2026, 6, 1),
        description="mel",
    )

    groups = group_duplicates([sydney, melbourne])
    assert len(groups) == 2


def test_cross_source_negative_different_role_does_not_merge() -> None:
    manager = Job(
        source=Source.SEEK,
        title="Quality Manager",
        normalized_title=normalize_title("Quality Manager"),
        company_name="Bega Group",
        apply_url="https://seek.com.au/job/1",
        description_hash=description_hash("qm"),
        source_job_id="1",
        location="Sydney NSW",
        posted_date=date(2026, 6, 1),
        description="qm",
    )
    operator = Job(
        source=Source.LINKEDIN,
        title="Warehouse Operator",
        normalized_title=normalize_title("Warehouse Operator"),
        company_name="Bega Group",
        apply_url="https://linkedin.com/jobs/2",
        description_hash=description_hash("wo"),
        source_job_id="2",
        location="Sydney NSW",
        posted_date=date(2026, 6, 1),
        description="wo",
    )

    groups = group_duplicates([manager, operator])
    assert len(groups) == 2
