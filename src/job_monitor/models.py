"""Shared data contracts: enums, RawJob, Job, and scoring result types.

Pure data, no I/O. Every other module depends on this one, so keep it stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    """Lifecycle status of a job row (spec §14.2)."""

    NEW = "new"
    REVIEWED = "reviewed"
    APPLY = "apply"
    APPLIED = "applied"
    REJECTED = "rejected"
    IRRELEVANT = "irrelevant"
    DUPLICATE = "duplicate"
    EXPIRED = "expired"


class PriorityTier(str, Enum):
    """Final-score priority band (spec §11)."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Source(str, Enum):
    """Source category, used for dedup source-preference ordering (spec §13)."""

    OFFICIAL_ATS = "official_ats"
    LINKEDIN = "linkedin"
    SEEK = "seek"
    INDEED = "indeed"
    JORA = "jora"


# Lower rank == more preferred when collapsing duplicates (spec §13).
SOURCE_PREFERENCE: dict[Source, int] = {
    Source.OFFICIAL_ATS: 0,
    Source.LINKEDIN: 1,
    Source.SEEK: 2,
    Source.INDEED: 3,
    Source.JORA: 4,
}


@dataclass(slots=True)
class RawJob:
    """Loosely shaped posting emitted by an adapter's ``parse()``.

    Adapters do as little normalization as possible; the pipeline turns this
    into a :class:`Job`.
    """

    source: Source
    title: str
    company_name: str
    apply_url: str
    source_job_id: str | None = None
    location: str | None = None
    description: str | None = None
    posted_date_raw: str | None = None
    salary_raw: str | None = None
    company_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Job:
    """Normalized, scorable, persistable posting (spec §14.2)."""

    # identity / dedup inputs
    source: Source
    title: str
    normalized_title: str
    company_name: str
    apply_url: str
    description_hash: str
    source_job_id: str | None = None
    company_id: str | None = None
    location: str | None = None
    posted_date: date | None = None
    description: str | None = None

    # parsed salary
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    salary_raw: str | None = None

    # matching / scoring (filled by pipeline stages)
    embedding: Any | None = None  # np.ndarray (float32)
    similarity: float | None = None
    semantic_score: float | None = None
    seniority_score: float | None = None
    industry_score: float | None = None
    company_score: float | None = None
    location_score: float | None = None
    salary_score: float | None = None
    final_score: float | None = None
    priority_tier: PriorityTier | None = None
    strong_alert: bool = False
    match_reasons: list[str] = field(default_factory=list)
    resume_tips: list[str] = field(default_factory=list)

    # lifecycle
    status: JobStatus = JobStatus.NEW
    duplicate_of: int | None = None
    db_id: int | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None


@dataclass(frozen=True)
class ScoreComponents:
    """The four weighted components of the final score (each 0-100)."""

    semantic: float
    seniority: float
    industry_company: float
    location_salary: float


@dataclass(slots=True)
class ScoreResult:
    """Output of the scorer for a single job."""

    final_score: float
    priority_tier: PriorityTier
    strong_alert: bool
    components: ScoreComponents
    match_reasons: list[str] = field(default_factory=list)
    resume_tips: list[str] = field(default_factory=list)
    category: str = "other"
    signals: set[str] = field(default_factory=set)
    hard_excluded: bool = False


@dataclass(slots=True)
class SourceHealth:
    """Result of an adapter healthcheck, used by the ``validate`` CLI command."""

    name: str
    source: Source
    ok: bool
    status: str  # "ok" | "blocked" | "empty" | "error"
    job_count: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class SourceBlocked(Exception):
    """Raised when a source hard-blocks the request (e.g. Cloudflare 403)."""
