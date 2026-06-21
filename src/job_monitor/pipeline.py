"""Pipeline orchestration: fetch -> normalize -> dedup -> embed -> score -> persist -> report/notify.

This module wires the stages together but owns no parsing or SQL itself. Key seams
(http client, embedder, db connection, fetch function, clock) are injectable so the
end-to-end test can run fully offline.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from job_monitor import db as dbmod
from job_monitor import dedup, normalize, report, scorer
from job_monitor.config import AppConfig, CompanyConfig, expand_path
from job_monitor.logging import RunStats, SourceStat, setup_logging
from job_monitor.models import Job, JobStatus, RawJob
from job_monitor.models import SourceBlocked as SourceBlocked
from job_monitor.notify import send_digest
from job_monitor.sources import build_adapter
from job_monitor.sources.http import PoliteClient

_TIER_RANK = {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4}


@dataclass
class RunResult:
    """Outcome of a pipeline run."""

    stats: RunStats
    top_jobs: list[Job] = field(default_factory=list)
    report_path: Path | None = None
    emailed: bool = False


# --------------------------------------------------------------------------- #
# Stage 1: fetch                                                              #
# --------------------------------------------------------------------------- #
def fetch_all(cfg: AppConfig, http: PoliteClient, stats: RunStats) -> list[RawJob]:
    """Fetch from every active company/source, capturing per-source outcomes.

    One source failing never aborts the run.
    """
    logger = setup_logging()
    raw: list[RawJob] = []
    for company in cfg.companies.active():
        stat = SourceStat(company_id=company.company_id, name=company.name)
        try:
            adapter = build_adapter(company, cfg.settings, http)
            terms = adapter.search_terms_from_company()
            jobs = adapter.fetch(terms)
            stat.fetched = len(jobs)
            stat.status = "ok" if jobs else "empty"
            raw.extend(jobs)
        except SourceBlocked as exc:
            stat.status = "blocked"
            stat.error = str(exc)
            logger.warning(f"[yellow]{company.name}: blocked[/] ({exc})")
        except Exception as exc:
            stat.status = "error"
            stat.error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"[red]{company.name}: error[/] ({stat.error})")
        else:
            logger.info(f"{company.name}: {stat.fetched} jobs ({stat.status})")
        stats.add_source(stat)
    return raw


# --------------------------------------------------------------------------- #
# Stage 2: normalize                                                          #
# --------------------------------------------------------------------------- #
def normalize_raw(raw: RawJob, *, today: date) -> Job:
    """Turn a loosely-shaped RawJob into a normalized, persistable Job."""
    sal = normalize.parse_salary(raw.salary_raw)
    posted = normalize.parse_posted_date(raw.posted_date_raw, today=today)
    desc = normalize.clean_text(raw.description) if raw.description else None
    return Job(
        source=raw.source,
        title=raw.title.strip(),
        normalized_title=dedup.normalize_title(raw.title),
        company_name=raw.company_name.strip() or "(unknown)",
        apply_url=raw.apply_url,
        description_hash=dedup.description_hash(desc),
        source_job_id=raw.source_job_id,
        company_id=raw.company_id,
        location=raw.location,
        posted_date=posted,
        description=desc,
        salary_min=sal.min,
        salary_max=sal.max,
        salary_currency=sal.currency,
        salary_period=sal.period,
        salary_raw=raw.salary_raw,
    )


def normalize_all(raws: Sequence[RawJob], *, today: date) -> list[Job]:
    jobs: list[Job] = []
    for r in raws:
        if not r.title or not r.apply_url:
            continue
        jobs.append(normalize_raw(r, today=today))
    return jobs


# --------------------------------------------------------------------------- #
# Stage 4+5: embed + score (canonical jobs only)                              #
# --------------------------------------------------------------------------- #
def _sector_for(cfg: AppConfig, job: Job) -> tuple[str, bool, bool]:
    """Return (sector, is_tier1, is_recruiter) for a job, from its company config."""
    company = _company_index(cfg).get(job.company_id or "")
    if company is None:
        return "", False, ("recruit" in job.company_name.lower() or "agency" in job.company_name.lower())
    is_recruiter = company.sector.lower() == "aggregator"
    return company.sector, company.priority_tier.upper() == "P1", is_recruiter


_COMPANY_INDEX_CACHE: dict[int, dict[str, CompanyConfig]] = {}


def _company_index(cfg: AppConfig) -> dict[str, CompanyConfig]:
    key = id(cfg)
    if key not in _COMPANY_INDEX_CACHE:
        _COMPANY_INDEX_CACHE[key] = {c.company_id: c for c in cfg.companies.companies}
    return _COMPANY_INDEX_CACHE[key]


def embed_and_score(jobs: list[Job], cfg: AppConfig, *, embedder: Any, today: date) -> None:
    """Embed each job, compute similarity vs the profile, and score. Mutates jobs."""
    if not jobs:
        return
    from job_monitor.embeddings import cosine

    profile_vec = embedder.encode(cfg.profile.full_text)
    texts = [f"{j.title}. {j.description or ''}" for j in jobs]
    vecs = embedder.encode(texts)
    for i, job in enumerate(jobs):
        vec = vecs[i]
        job.embedding = vec
        similarity = cosine(vec, profile_vec)
        job.similarity = float(similarity)
        sector, is_tier1, is_recruiter = _sector_for(cfg, job)
        posted_days = (today - job.posted_date).days if job.posted_date else None
        result = scorer.score_job(
            job,
            similarity=similarity,
            scoring=cfg.scoring,
            profile=cfg.profile,
            sector=sector,
            is_tier1=is_tier1,
            is_recruiter=is_recruiter,
            posted_days_ago=posted_days,
        )
        job.semantic_score = result.components.semantic
        job.seniority_score = result.components.seniority
        job.industry_score = result.components.industry_company
        job.location_score = result.components.location_salary
        job.final_score = result.final_score
        job.priority_tier = result.priority_tier
        job.strong_alert = result.strong_alert
        job.match_reasons = result.match_reasons
        job.resume_tips = result.resume_tips
        if result.hard_excluded:
            job.status = JobStatus.IRRELEVANT


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def run_pipeline(
    cfg: AppConfig,
    *,
    http: PoliteClient | None = None,
    embedder: Any = None,
    conn: sqlite3.Connection | None = None,
    today: date | None = None,
    fetch_fn: Callable[[AppConfig, PoliteClient, RunStats], list[RawJob]] | None = None,
    write_report: bool = True,
    send_email: bool | None = None,
) -> RunResult:
    """Run the full pipeline. Returns a :class:`RunResult`.

    Injectable seams (http/embedder/conn/today/fetch_fn) let the e2e test run offline.
    """
    logger = setup_logging()
    now = datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    today = today or now.date()
    stats = RunStats(run_id=run_id)

    owns_conn = conn is None
    conn = conn or dbmod.connect(cfg.settings.db_file)
    http = http or PoliteClient(cfg.settings.http)
    fetch_fn = fetch_fn or fetch_all

    try:
        # 1. fetch
        raws = fetch_fn(cfg, http, stats)
        # 2. normalize
        jobs = normalize_all(raws, today=today)
        # 3. dedup
        groups = dedup.group_duplicates(jobs)
        canonicals = [g[0] for g in groups]
        stats.canonical = len(canonicals)
        stats.duplicates = sum(len(g[1]) for g in groups)
        # 4+5. embed + score canonicals
        if embedder is None:
            from job_monitor.embeddings import get_embedder

            embedder = get_embedder(cfg.settings)
        embed_and_score(canonicals, cfg, embedder=embedder, today=today)
        # 6. persist canonicals, then their duplicates linked back
        for canonical, losers in groups:
            cid = dbmod.upsert_job(conn, canonical)
            for loser in losers:
                loser.status = JobStatus.DUPLICATE
                lid = dbmod.upsert_job(conn, loser)
                if lid:
                    dbmod.mark_duplicate(conn, lid, cid)
            stats.persisted += 1 + len(losers)
        # 7. report + notify
        result = RunResult(stats=stats)
        top = _select_for_digest(canonicals, cfg)
        result.top_jobs = top
        if write_report:
            result.report_path = _write_report(top, cfg, now)
        emit = cfg.settings.email.enabled if send_email is None else send_email
        if emit and top:
            html = report.render_email_html(top, generated_at=now)
            text = report.render_email_text(top, generated_at=now)
            n_alert = sum(1 for j in top if j.strong_alert or (j.priority_tier and j.priority_tier.value == "A+"))
            subject = f"[Job Monitor] {len(top)} roles ({n_alert} high-priority) — {today.isoformat()}"
            result.emailed = send_digest(
                cfg.settings.email, subject=subject, html_body=html, text_body=text
            )
            for job in top:
                if job.db_id:
                    dbmod.record_alert(
                        conn, job_id=job.db_id, run_id=run_id, channel="email",
                        priority_tier=job.priority_tier.value if job.priority_tier else None,
                        strong_alert=job.strong_alert,
                    )
                    stats.alerts += 1
        logger.info(stats.summary())
        return result
    finally:
        if owns_conn:
            conn.close()


def _select_for_digest(jobs: list[Job], cfg: AppConfig) -> list[Job]:
    """Filter to report-worthy jobs (>= min_tier, not irrelevant) sorted by score desc."""
    min_rank = _TIER_RANK.get(cfg.settings.report.min_tier.upper(), 3)
    eligible = [
        j for j in jobs
        if j.status != JobStatus.IRRELEVANT
        and j.priority_tier is not None
        and _TIER_RANK.get(j.priority_tier.value, 4) <= min_rank
    ]
    eligible.sort(key=lambda j: j.final_score or 0.0, reverse=True)
    return eligible


def _write_report(jobs: list[Job], cfg: AppConfig, now: datetime) -> Path:
    out_dir = expand_path(cfg.settings.report.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = cfg.settings.report.filename_template.format(date=now.strftime("%Y%m%d-%H%M%S"))
    path = out_dir / fname
    html = report.render_report(jobs, generated_at=now, subtitle=f"{len(jobs)} roles")
    path.write_text(html, encoding="utf-8")
    return path
