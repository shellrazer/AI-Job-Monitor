# Job Monitor — Australian Food-Industry Senior Quality Roles

A low-maintenance system that monitors company ATS/careers feeds and job boards (SEEK, Jora),
matches postings against a fixed senior food-quality candidate profile using local embeddings +
rule-based scoring, deduplicates, and delivers a daily ranked HTML report + email digest.

See the design plan for full context. Target roles: Senior/Site/National/Group Quality Manager,
Food Safety & Quality Manager, Supplier Quality / Vendor Assurance Manager, etc. (Sydney/NSW first,
AUD 130k–200k).

## Quick start

```bash
uv python install 3.12      # one-time
uv sync                     # install deps into .venv
uv run job-monitor init-db  # create the SQLite schema
uv run job-monitor validate # check which live sources currently return data
uv run job-monitor run      # full pipeline -> HTML report (+ email if configured)
```

## Email (optional)

```bash
export JOB_MONITOR_GMAIL_USER="you@gmail.com"
export JOB_MONITOR_GMAIL_APP_PASSWORD="<16-char app password>"
```

## Tests

```bash
uv run pytest                                   # unit + adapter + e2e (offline)
uv run pytest -m integration --reruns 3         # live network tests (opt-in)
uv run ruff check . && uv run mypy src
```

## Config

All tuning lives in `config/*.yaml` — company list, keyword sets, scoring weights/tables,
runtime settings, and the candidate profile. No code changes needed to retune.

## Scheduling (macOS)

Install `deploy/com.user.jobmonitor.plist` via `launchctl load` for a daily run.
