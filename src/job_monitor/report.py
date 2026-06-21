"""Render the HTML report and the email bodies from scored :class:`Job` objects.

Pure presentation: takes already-scored jobs and turns them into a standalone
HTML page (for the local report file), a compact inline-CSS HTML email body, and
a plaintext fallback digest. Templates live in ``templates/`` and are loaded from
the filesystem relative to this module (works under an editable ``src/`` install).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import scorer
from .config import CompanyRatingsConfig
from .models import Job, PriorityTier

__all__ = [
    "format_salary",
    "location_bucket",
    "recruiter_mark",
    "render_email_html",
    "render_email_text",
    "render_report",
    "stars_display",
    "tier_counts",
]

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Top-N jobs included in the email digest (full report includes everything).
EMAIL_TOP_N = 10

# Tier order for the summary header, highest priority first.
_TIER_ORDER = [PriorityTier.A_PLUS, PriorityTier.A, PriorityTier.B, PriorityTier.C, PriorityTier.D]

_UNDISCLOSED = "Not disclosed / 未披露"

# Location-split section headings, in render order.
_NSW_HEADING = "NSW & Remote / 新州与远程"
_OTHER_HEADING = "Other Australia / 其他州"

# Remote markers that, when present in a location string, always land a job in
# the NSW & Remote section regardless of the geographic classification.
_REMOTE_MARKERS = (
    "remote",
    "work from home",
    "work-from-home",
    "wfh",
    "anywhere",
    "australia wide",
    "australia-wide",
    "fully remote",
)


def _env() -> Environment:
    """Build a Jinja2 environment bound to the bundled template directory."""
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(),
    )
    # Expose the salary formatter to templates so labels stay in one place.
    env.globals["salary_of"] = format_salary
    return env


def location_bucket(location: str | None) -> str:
    """Coarse two-way split for the report sections.

    Returns ``"nsw_remote"`` for NSW/greater-Sydney/regional-NSW roles and any
    role whose location advertises remote work; otherwise ``"other_au"``.
    """
    loc = (location or "").lower()
    if any(marker in loc for marker in _REMOTE_MARKERS):
        return "nsw_remote"
    if scorer.classify_location(location) in ("sydney_greater", "nsw_regional"):
        return "nsw_remote"
    return "other_au"


def stars_display(company_name: str | None, ratings: CompanyRatingsConfig) -> str:
    """A 1-5 star bar for a real employer, e.g. ``"★★★★☆ (4/5)"``.

    Recruiter/agency postings return ``""`` — the recruiter mark stands in for
    the rating there since the true employer is undisclosed.
    """
    if scorer.is_recruiter_company(company_name):
        return ""
    stars = ratings.stars_for(company_name)
    bar = "★" * stars + "☆" * (5 - stars)
    return f"{bar} ({stars}/5)"


def recruiter_mark(company_name: str | None) -> str:
    """Bilingual recruiter/agency marker, or ``""`` for a named employer."""
    if scorer.is_recruiter_company(company_name):
        return "🕵 Recruiter — employer undisclosed / 招聘中介 (雇主未披露)"
    return ""


def format_salary(job: Job) -> str:
    """Human-readable salary string for a job.

    Prefers the parsed ``salary_min/max/currency/period`` fields, falls back to
    the raw string the adapter captured, and finally to a bilingual
    "not disclosed" marker.
    """
    currency = job.salary_currency or ""
    period = f" / {job.salary_period}" if job.salary_period else ""

    def _fmt(amount: int) -> str:
        prefix = f"{currency} " if currency else ""
        return f"{prefix}{amount:,}"

    if job.salary_min is not None and job.salary_max is not None:
        if job.salary_min == job.salary_max:
            return f"{_fmt(job.salary_min)}{period}"
        return f"{_fmt(job.salary_min)} - {_fmt(job.salary_max)}{period}"
    if job.salary_min is not None:
        return f"{_fmt(job.salary_min)}{period}"
    if job.salary_max is not None:
        return f"{_fmt(job.salary_max)}{period}"
    if job.salary_raw and job.salary_raw.strip():
        return job.salary_raw.strip()
    return _UNDISCLOSED


def tier_counts(jobs: list[Job]) -> dict[str, int]:
    """Count jobs per priority tier (plus an ``"Unscored"`` bucket).

    Only tiers that actually occur are included, in highest-to-lowest order, so
    the summary header stays compact.
    """
    counts: dict[str, int] = {}
    for tier in _TIER_ORDER:
        n = sum(1 for j in jobs if j.priority_tier is tier)
        if n:
            counts[tier.value] = n
    unscored = sum(1 for j in jobs if j.priority_tier is None)
    if unscored:
        counts["Unscored"] = unscored
    return counts


def _sorted_by_score(jobs: list[Job]) -> list[Job]:
    """Sort jobs by ``final_score`` descending, with unscored jobs (None) last."""
    return sorted(jobs, key=lambda j: (j.final_score is None, -(j.final_score or 0.0)))


@dataclass(frozen=True)
class _DecoratedJob:
    """A job paired with its precomputed stars / recruiter mark for templates."""

    job: Job
    stars: str
    recruiter: str


@dataclass(frozen=True)
class _Section:
    """A location section: a bilingual heading and its decorated jobs."""

    heading: str
    items: list[_DecoratedJob]


def _decorate(job: Job, ratings: CompanyRatingsConfig) -> _DecoratedJob:
    """Bundle a job with its precomputed stars / recruiter mark for templates."""
    return _DecoratedJob(
        job=job,
        stars=stars_display(job.company_name, ratings),
        recruiter=recruiter_mark(job.company_name),
    )


def _sections(jobs: list[Job], ratings: CompanyRatingsConfig) -> list[_Section]:
    """Split jobs into the NSW & Remote / Other Australia sections.

    Each section is sorted by ``final_score`` descending and its jobs are
    decorated with stars / recruiter marks so templates stay logic-light.
    """
    buckets: dict[str, list[Job]] = {"nsw_remote": [], "other_au": []}
    for job in jobs:
        buckets[location_bucket(job.location)].append(job)
    return [
        _Section(
            heading=_NSW_HEADING,
            items=[_decorate(j, ratings) for j in _sorted_by_score(buckets["nsw_remote"])],
        ),
        _Section(
            heading=_OTHER_HEADING,
            items=[_decorate(j, ratings) for j in _sorted_by_score(buckets["other_au"])],
        ),
    ]


def render_report(
    jobs: list[Job],
    *,
    generated_at: datetime,
    ratings: CompanyRatingsConfig,
    subtitle: str = "",
) -> str:
    """Render the full standalone HTML report page (two location sections)."""
    template = _env().get_template("report.html.j2")
    return template.render(
        sections=_sections(jobs, ratings),
        total=len(jobs),
        counts=tier_counts(jobs),
        generated_at=generated_at,
        subtitle=subtitle,
    )


def render_email_html(
    jobs: list[Job], *, generated_at: datetime, ratings: CompanyRatingsConfig
) -> str:
    """Render a compact, inline-CSS HTML email body (two location sections)."""
    ordered = _sorted_by_score(jobs)[:EMAIL_TOP_N]
    template = _env().get_template("email.html.j2")
    return template.render(
        sections=_sections(ordered, ratings),
        total=len(ordered),
        generated_at=generated_at,
    )


def render_email_text(
    jobs: list[Job], *, generated_at: datetime, ratings: CompanyRatingsConfig
) -> str:
    """Render the plaintext fallback digest: two location sections of top jobs."""
    ordered = _sorted_by_score(jobs)[:EMAIL_TOP_N]
    lines = [
        "职位监控摘要 / Job Monitor Digest",
        f"生成时间 / Generated: {generated_at:%Y-%m-%d %H:%M}",
        "",
    ]
    if not ordered:
        lines.append("没有匹配的职位 / No matching jobs.")
    for section in _sections(ordered, ratings):
        lines.append(section.heading)
        if not section.items:
            lines.append("  本节暂无职位 / No roles in this section.")
            lines.append("")
            continue
        for item in section.items:
            job = item.job
            tier = job.priority_tier.value if job.priority_tier else "—"
            score = f"{job.final_score:.1f}" if job.final_score is not None else "—"
            location = job.location or "—"
            lines.append(f"{job.title} — {job.company_name} — {location} — {score}/{tier}")
            if item.recruiter:
                lines.append(f"  {item.recruiter}")
            if item.stars:
                lines.append(f"  评分 / Rating: {item.stars}")
            lines.append(f"  薪资 / Salary: {format_salary(job)}")
            lines.append(f"  申请链接 / Apply: {job.apply_url}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
