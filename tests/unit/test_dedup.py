"""Tests for job_monitor.dedup."""

from __future__ import annotations

from job_monitor.dedup import (
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
