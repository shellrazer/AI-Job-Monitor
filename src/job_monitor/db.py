"""All SQLite access: schema bootstrap, upserts, and queries (spec §14).

This is the only module that issues SQL. Embeddings are stored as float32 BLOBs;
list fields (match_reasons, resume_tips) are stored as JSON text.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from job_monitor.models import Job, JobStatus, PriorityTier, Source

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    company_id     TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    sector         TEXT,
    priority_tier  TEXT,
    careers_url    TEXT,
    ats_platform   TEXT,
    adapter        TEXT,
    active         INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source            TEXT NOT NULL,
    source_job_id     TEXT,
    company_id        TEXT,
    company_name      TEXT NOT NULL,
    title             TEXT NOT NULL,
    normalized_title  TEXT NOT NULL,
    location          TEXT,
    posted_date       TEXT,
    apply_url         TEXT NOT NULL,
    description       TEXT,
    description_hash  TEXT NOT NULL,
    salary_min        INTEGER,
    salary_max        INTEGER,
    salary_currency   TEXT,
    salary_period     TEXT,
    salary_raw        TEXT,
    embedding         BLOB,
    similarity        REAL,
    semantic_score    REAL,
    seniority_score   REAL,
    industry_score    REAL,
    company_score     REAL,
    location_score    REAL,
    salary_score      REAL,
    final_score       REAL,
    priority_tier     TEXT,
    strong_alert      INTEGER NOT NULL DEFAULT 0,
    match_reasons     TEXT,
    resume_tips       TEXT,
    status            TEXT NOT NULL DEFAULT 'new',
    duplicate_of      INTEGER REFERENCES jobs(id),
    first_seen        TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_dedup ON jobs (
    normalized_title,
    company_name,
    COALESCE(location, ''),
    COALESCE(posted_date, ''),
    COALESCE(source_job_id, apply_url)
);
CREATE INDEX IF NOT EXISTS ix_jobs_desc_hash ON jobs (description_hash);
CREATE INDEX IF NOT EXISTS ix_jobs_status    ON jobs (status);
CREATE INDEX IF NOT EXISTS ix_jobs_score     ON jobs (final_score DESC);
CREATE INDEX IF NOT EXISTS ix_jobs_company   ON jobs (company_id);
CREATE INDEX IF NOT EXISTS ix_jobs_apply_url ON jobs (apply_url);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        INTEGER NOT NULL REFERENCES jobs(id),
    run_id        TEXT NOT NULL,
    channel       TEXT NOT NULL,
    priority_tier TEXT,
    strong_alert  INTEGER NOT NULL DEFAULT 0,
    sent_at       TEXT NOT NULL DEFAULT (datetime('now')),
    payload_ref   TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_job ON alerts (job_id);
CREATE INDEX IF NOT EXISTS ix_alerts_run ON alerts (run_id);

CREATE TABLE IF NOT EXISTS applications (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         INTEGER NOT NULL REFERENCES jobs(id),
    status         TEXT NOT NULL,
    applied_at     TEXT,
    resume_version TEXT,
    notes          TEXT,
    follow_up_date TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_applications_job ON applications (job_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection, ensure the parent dir exists, and apply the schema."""
    path = Path(db_path)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def init_db(db_path: str | Path) -> None:
    """Create the schema (idempotent)."""
    conn = connect(db_path)
    conn.close()


# --------------------------------------------------------------------------- #
# Embedding (de)serialization                                                 #
# --------------------------------------------------------------------------- #
def pack_embedding(vec: Any | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_embedding(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


# --------------------------------------------------------------------------- #
# Companies                                                                   #
# --------------------------------------------------------------------------- #
def upsert_company(conn: sqlite3.Connection, *, company_id: str, name: str, sector: str = "",
                   priority_tier: str = "", careers_url: str = "", ats_platform: str = "",
                   adapter: str = "", active: bool = True) -> None:
    conn.execute(
        """
        INSERT INTO companies (company_id, name, sector, priority_tier, careers_url,
                               ats_platform, adapter, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_id) DO UPDATE SET
            name=excluded.name, sector=excluded.sector, priority_tier=excluded.priority_tier,
            careers_url=excluded.careers_url, ats_platform=excluded.ats_platform,
            adapter=excluded.adapter, active=excluded.active, updated_at=datetime('now')
        """,
        (company_id, name, sector, priority_tier, careers_url, ats_platform, adapter, int(active)),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# Jobs                                                                        #
# --------------------------------------------------------------------------- #
def upsert_job(conn: sqlite3.Connection, job: Job) -> int:
    """Insert or update a job by its natural dedup key. Returns the row id.

    On conflict we refresh last_seen + scores/status but keep first_seen.
    """
    params = {
        "source": job.source.value,
        "source_job_id": job.source_job_id,
        "company_id": job.company_id,
        "company_name": job.company_name,
        "title": job.title,
        "normalized_title": job.normalized_title,
        "location": job.location,
        "posted_date": _iso(job.posted_date),
        "apply_url": job.apply_url,
        "description": job.description,
        "description_hash": job.description_hash,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "salary_currency": job.salary_currency,
        "salary_period": job.salary_period,
        "salary_raw": job.salary_raw,
        "embedding": pack_embedding(job.embedding),
        "similarity": job.similarity,
        "semantic_score": job.semantic_score,
        "seniority_score": job.seniority_score,
        "industry_score": job.industry_score,
        "company_score": job.company_score,
        "location_score": job.location_score,
        "salary_score": job.salary_score,
        "final_score": job.final_score,
        "priority_tier": job.priority_tier.value if job.priority_tier else None,
        "strong_alert": int(job.strong_alert),
        "match_reasons": json.dumps(job.match_reasons, ensure_ascii=False),
        "resume_tips": json.dumps(job.resume_tips, ensure_ascii=False),
        "status": job.status.value,
        "duplicate_of": job.duplicate_of,
    }
    cols = ", ".join(params)
    placeholders = ", ".join(f":{k}" for k in params)
    update_cols = ", ".join(
        f"{k}=excluded.{k}" for k in params if k not in ("source_job_id",)
    )
    cur = conn.execute(
        f"""
        INSERT INTO jobs ({cols}) VALUES ({placeholders})
        ON CONFLICT(normalized_title, company_name,
                    COALESCE(location, ''), COALESCE(posted_date, ''),
                    COALESCE(source_job_id, apply_url))
        DO UPDATE SET {update_cols}, last_seen=datetime('now')
        """,
        params,
    )
    conn.commit()
    if cur.lastrowid:
        job.db_id = cur.lastrowid
        return cur.lastrowid
    # On UPDATE, lastrowid may be 0 — look up the existing id.
    row = conn.execute(
        """
        SELECT id FROM jobs
        WHERE normalized_title=? AND company_name=?
          AND COALESCE(location,'')=COALESCE(?, '')
          AND COALESCE(posted_date,'')=COALESCE(?, '')
          AND COALESCE(source_job_id, apply_url)=COALESCE(?, ?)
        """,
        (job.normalized_title, job.company_name, job.location, _iso(job.posted_date),
         job.source_job_id, job.apply_url),
    ).fetchone()
    job.db_id = int(row["id"]) if row else None
    return job.db_id or 0


def upsert_jobs(conn: sqlite3.Connection, jobs: Iterable[Job]) -> list[int]:
    return [upsert_job(conn, j) for j in jobs]


def mark_duplicate(conn: sqlite3.Connection, job_id: int, canonical_id: int) -> None:
    conn.execute(
        "UPDATE jobs SET status=?, duplicate_of=?, last_seen=datetime('now') WHERE id=?",
        (JobStatus.DUPLICATE.value, canonical_id, job_id),
    )
    conn.commit()


def get_job(conn: sqlite3.Connection, job_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def find_job_id(conn: sqlite3.Connection, job: Job) -> int | None:
    """Return the id of an already-persisted row matching this job's dedup key, else None.

    Timing-independent newness check: call BEFORE upserting to know whether a job
    is genuinely new (not previously seen) regardless of clock granularity.
    """
    row = conn.execute(
        """
        SELECT id FROM jobs
        WHERE normalized_title=? AND company_name=?
          AND COALESCE(location,'')=COALESCE(?, '')
          AND COALESCE(posted_date,'')=COALESCE(?, '')
          AND COALESCE(source_job_id, apply_url)=COALESCE(?, ?)
        """,
        (job.normalized_title, job.company_name, job.location, _iso(job.posted_date),
         job.source_job_id, job.apply_url),
    ).fetchone()
    return int(row["id"]) if row else None


def top_jobs(conn: sqlite3.Connection, *, limit: int = 50, min_score: float = 0.0,
             exclude_status: Iterable[str] = (JobStatus.DUPLICATE.value, JobStatus.IRRELEVANT.value,
                                              JobStatus.EXPIRED.value)) -> list[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in exclude_status)
    sql = (
        "SELECT * FROM jobs WHERE final_score >= ? "
        f"AND status NOT IN ({placeholders}) "
        "ORDER BY final_score DESC LIMIT ?"
    )
    return conn.execute(sql, (min_score, *exclude_status, limit)).fetchall()


def jobs_seen_since(conn: sqlite3.Connection, since: datetime) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM jobs WHERE last_seen >= ? ORDER BY final_score DESC",
        (since.isoformat(),),
    ).fetchall()


def embedding_for(conn: sqlite3.Connection, *, normalized_title: str, company_name: str,
                  description_hash: str) -> np.ndarray | None:
    """Return a previously computed embedding for an identical posting, if any."""
    row = conn.execute(
        "SELECT embedding FROM jobs WHERE normalized_title=? AND company_name=? "
        "AND description_hash=? AND embedding IS NOT NULL LIMIT 1",
        (normalized_title, company_name, description_hash),
    ).fetchone()
    return unpack_embedding(row["embedding"]) if row else None


# --------------------------------------------------------------------------- #
# Alerts                                                                      #
# --------------------------------------------------------------------------- #
def record_alert(conn: sqlite3.Connection, *, job_id: int, run_id: str, channel: str,
                 priority_tier: str | None, strong_alert: bool,
                 payload_ref: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO alerts (job_id, run_id, channel, priority_tier, strong_alert, payload_ref) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, run_id, channel, priority_tier, int(strong_alert), payload_ref),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


# Re-export for convenience.
__all__ = [
    "SCHEMA",
    "Job",
    "JobStatus",
    "PriorityTier",
    "Source",
    "connect",
    "embedding_for",
    "get_job",
    "init_db",
    "jobs_seen_since",
    "mark_duplicate",
    "pack_embedding",
    "record_alert",
    "top_jobs",
    "unpack_embedding",
    "upsert_company",
    "upsert_job",
    "upsert_jobs",
]
