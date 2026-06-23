"""Command-line interface (typer): run / validate / init-db / report / list-sources."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from job_monitor import db as dbmod
from job_monitor.config import DEFAULT_CONFIG_DIR, AppConfig, load_config
from job_monitor.logging import setup_logging
from job_monitor.sources import build_adapter
from job_monitor.sources.http import PoliteClient

app = typer.Typer(add_completion=False, help="Australian food-industry senior quality job monitor.")
console = Console()


def _load(config_dir: str | None) -> AppConfig:
    return load_config(config_dir or DEFAULT_CONFIG_DIR)


@app.command("init-db")
def init_db(config_dir: str | None = typer.Option(None, help="Path to config dir")) -> None:
    """Create the SQLite schema and sync the company master list."""
    cfg = _load(config_dir)
    conn = dbmod.connect(cfg.settings.db_file)
    for c in cfg.companies.companies:
        dbmod.upsert_company(
            conn, company_id=c.company_id, name=c.name, sector=c.sector,
            priority_tier=c.priority_tier, careers_url=c.careers_url,
            ats_platform=c.ats_platform, adapter=c.adapter, active=c.active,
        )
    conn.close()
    console.print(f"[green]Initialised DB[/] at {cfg.settings.db_file} with {len(cfg.companies.companies)} companies.")


@app.command("list-sources")
def list_sources(config_dir: str | None = typer.Option(None)) -> None:
    """Print the configured sources."""
    cfg = _load(config_dir)
    table = Table(title="Sources")
    for col in ("company_id", "name", "tier", "adapter", "active"):
        table.add_column(col)
    for c in cfg.companies.companies:
        table.add_row(c.company_id, c.name, c.priority_tier, c.adapter,
                      "[green]yes" if c.active else "[dim]no")
    console.print(table)


@app.command("validate")
def validate(config_dir: str | None = typer.Option(None)) -> None:
    """Healthcheck every active source; report which currently return data.

    Exits non-zero if any P1 source is not OK (early adapter-rot warning).
    """
    cfg = _load(config_dir)
    setup_logging()
    http = PoliteClient(cfg.settings.http)
    table = Table(title="Source health")
    for col in ("company", "adapter", "tier", "status", "jobs", "latency", "error"):
        table.add_column(col)
    p1_broken = 0
    for company in cfg.companies.active():
        adapter = build_adapter(company, cfg.settings, http)
        health = adapter.healthcheck()
        colour = {"ok": "green", "empty": "yellow", "blocked": "red", "error": "red"}.get(health.status, "white")
        if company.priority_tier.upper() == "P1" and health.status != "ok":
            p1_broken += 1
        table.add_row(
            company.name, company.adapter, company.priority_tier,
            f"[{colour}]{health.status}[/]", str(health.job_count),
            f"{health.latency_ms:.0f}ms", (health.error or "")[:50],
        )
    console.print(table)
    if p1_broken:
        console.print(f"[red]{p1_broken} P1 source(s) not OK.[/]")
        raise typer.Exit(code=1)
    console.print("[green]All P1 sources OK.[/]")


@app.command("run")
def run(
    config_dir: str | None = typer.Option(None),
    no_email: bool = typer.Option(False, "--no-email", help="Skip the email digest"),
) -> None:
    """Run the full pipeline: fetch, score, persist, report (+ email)."""
    from job_monitor.pipeline import run_pipeline

    cfg = _load(config_dir)
    cfg.validate_email_ready()
    setup_logging()
    result = run_pipeline(cfg, send_email=False if no_email else None)
    console.print(f"[green]{result.stats.summary()}[/]")
    if result.report_path:
        console.print(f"Report: {result.report_path}")
        if cfg.settings.report.open_after_run:
            import webbrowser

            webbrowser.open(result.report_path.as_uri())
    if result.emailed:
        console.print("[green]Email digest sent.[/]")


@app.command("report")
def report_cmd(
    config_dir: str | None = typer.Option(None),
    limit: int = typer.Option(50, help="Max jobs"),
    min_score: float = typer.Option(0.0, help="Minimum final score"),
) -> None:
    """Regenerate an HTML report from already-persisted jobs (no fetching)."""
    from datetime import datetime

    from job_monitor import report as report_mod
    from job_monitor.config import expand_path
    from job_monitor.db import unpack_embedding  # noqa: F401  (ensures db import side-effects)
    from job_monitor.models import Job, PriorityTier, Source

    cfg = _load(config_dir)
    conn = dbmod.connect(cfg.settings.db_file)
    rows = dbmod.top_jobs(conn, limit=limit, min_score=min_score)
    conn.close()
    jobs: list[Job] = []
    for r in rows:
        jobs.append(Job(
            source=Source(r["source"]), title=r["title"], normalized_title=r["normalized_title"],
            company_name=r["company_name"], apply_url=r["apply_url"], description_hash=r["description_hash"],
            location=r["location"], description=r["description"], salary_min=r["salary_min"],
            salary_max=r["salary_max"], salary_currency=r["salary_currency"], salary_raw=r["salary_raw"],
            final_score=r["final_score"], strong_alert=bool(r["strong_alert"]),
            priority_tier=PriorityTier(r["priority_tier"]) if r["priority_tier"] else None,
        ))
    # NSW-only view (view-level, toggleable): hide roles clearly in another state;
    # keep NSW/Sydney + remote + unknown-region.
    if cfg.settings.report.nsw_only:
        jobs = [j for j in jobs if not report_mod.is_other_region(j.location, j.title)]
    out_dir = expand_path(cfg.settings.report.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = out_dir / cfg.settings.report.filename_template.format(date=now.strftime("%Y%m%d-%H%M%S"))
    path.write_text(
        report_mod.render_report(jobs, generated_at=now, ratings=cfg.ratings), encoding="utf-8"
    )
    console.print(f"[green]Wrote {len(jobs)} jobs[/] to {path}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
