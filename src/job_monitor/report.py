"""Render the HTML report and the email bodies from scored :class:`Job` objects.

Pure presentation: takes already-scored jobs and turns them into a standalone
HTML page (for the local report file), a compact inline-CSS HTML email body, and
a plaintext fallback digest. Templates live in ``templates/`` and are loaded from
the filesystem relative to this module (works under an editable ``src/`` install).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import Job, PriorityTier

__all__ = [
    "format_salary",
    "render_email_html",
    "render_email_text",
    "render_report",
    "tier_counts",
]

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Top-N jobs included in the email digest (full report includes everything).
EMAIL_TOP_N = 10

# Tier order for the summary header, highest priority first.
_TIER_ORDER = [PriorityTier.A_PLUS, PriorityTier.A, PriorityTier.B, PriorityTier.C, PriorityTier.D]

_UNDISCLOSED = "Not disclosed / 未披露"


def _env() -> Environment:
    """Build a Jinja2 environment bound to the bundled template directory."""
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(),
    )
    # Expose the salary formatter to templates so labels stay in one place.
    env.globals["salary_of"] = format_salary
    return env


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


def render_report(jobs: list[Job], *, generated_at: datetime, subtitle: str = "") -> str:
    """Render the full standalone HTML report page."""
    ordered = _sorted_by_score(jobs)
    template = _env().get_template("report.html.j2")
    return template.render(
        jobs=ordered,
        counts=tier_counts(ordered),
        generated_at=generated_at,
        subtitle=subtitle,
    )


def render_email_html(jobs: list[Job], *, generated_at: datetime) -> str:
    """Render a compact, inline-CSS HTML email body of the top-scoring jobs."""
    ordered = _sorted_by_score(jobs)[:EMAIL_TOP_N]
    template = _env().get_template("email.html.j2")
    return template.render(jobs=ordered, generated_at=generated_at)


def render_email_text(jobs: list[Job], *, generated_at: datetime) -> str:
    """Render the plaintext fallback digest: one block per top-scoring job."""
    ordered = _sorted_by_score(jobs)[:EMAIL_TOP_N]
    lines = [
        "职位监控摘要 / Job Monitor Digest",
        f"生成时间 / Generated: {generated_at:%Y-%m-%d %H:%M}",
        "",
    ]
    if not ordered:
        lines.append("没有匹配的职位 / No matching jobs.")
    for job in ordered:
        tier = job.priority_tier.value if job.priority_tier else "—"
        score = f"{job.final_score:.1f}" if job.final_score is not None else "—"
        location = job.location or "—"
        lines.append(f"{job.title} — {job.company_name} — {location} — {score}/{tier}")
        lines.append(f"  薪资 / Salary: {format_salary(job)}")
        lines.append(f"  申请链接 / Apply: {job.apply_url}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
