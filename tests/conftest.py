"""Shared pytest fixtures. Keep these deterministic and offline."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from job_monitor import db as dbmod
from job_monitor.models import Job, RawJob, Source

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
EMBED_DIM = 384


@pytest.fixture
def tmp_db(tmp_path: Path) -> sqlite3.Connection:
    """A fresh SQLite connection with the schema applied, in a temp dir."""
    conn = dbmod.connect(tmp_path / "test.sqlite")
    yield conn
    conn.close()


def _hash_vector(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic pseudo-embedding from text — no torch, stable across runs.

    Seeds a PRNG from the sha256 of the text so identical text -> identical
    vector and different text -> different vector. Returned L2-normalized.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


@pytest.fixture
def fake_embedder():
    """A callable that maps text -> deterministic float32 vector (offline)."""

    def _embed(text: str) -> np.ndarray:
        return _hash_vector(text)

    return _embed


@pytest.fixture
def fixture_bytes():
    """Read a committed fixture file's bytes by name from tests/fixtures/."""

    def _read(name: str) -> bytes:
        return (FIXTURES_DIR / name).read_bytes()

    return _read


@pytest.fixture
def fixture_text():
    def _read(name: str) -> str:
        return (FIXTURES_DIR / name).read_text(encoding="utf-8")

    return _read


@pytest.fixture
def sample_raw_jobs() -> list[RawJob]:
    """A small mixed set covering dupes, exclusions, missing fields."""
    return [
        RawJob(
            source=Source.OFFICIAL_ATS,
            title="Site Quality Manager",
            company_name="Bega Group",
            apply_url="https://example.com/jobs/sqm-1",
            source_job_id="JR-1001",
            location="Western Sydney NSW",
            description="Lead the site quality team. HACCP, GMP, audit readiness, site leadership.",
            posted_date_raw="2026-06-18",
            salary_raw="$140,000 - $160,000 + super",
            company_id="bega",
        ),
        RawJob(
            source=Source.SEEK,
            title="Site Quality Manager",  # cross-source duplicate of the above
            company_name="Bega Group",
            apply_url="https://seek.com.au/job/12345",
            source_job_id="12345",
            location="Western Sydney NSW",
            description="Lead the site quality team. HACCP, GMP, audit readiness, site leadership.",
            posted_date_raw="2026-06-18",
            salary_raw="$140k - $160k",
        ),
        RawJob(
            source=Source.SEEK,
            title="Software QA Engineer",  # out-of-domain, should be down-weighted
            company_name="Tech Co",
            apply_url="https://seek.com.au/job/99999",
            source_job_id="99999",
            location="Sydney NSW",
            description="Automated test analyst for our SaaS platform. Selenium, CI/CD.",
            posted_date_raw="2026-06-20",
            salary_raw=None,
        ),
    ]


@pytest.fixture
def sample_jobs() -> list[Job]:
    """Pre-normalized Job objects for report / db / pipeline tests."""
    return [
        Job(
            source=Source.OFFICIAL_ATS,
            title="National Quality Manager",
            normalized_title="national quality manager",
            company_name="GrainCorp",
            apply_url="https://example.com/nqm",
            description_hash=hashlib.sha256(b"nqm").hexdigest(),
            location="Sydney NSW",
            description="National multi-site quality leadership, FSSC 22000, supplier audits.",
            salary_min=160000,
            salary_max=190000,
            salary_currency="AUD",
            salary_period="year",
        ),
        Job(
            source=Source.SEEK,
            title="QA Officer",
            normalized_title="qa officer",
            company_name="Small Foods",
            apply_url="https://example.com/qao",
            description_hash=hashlib.sha256(b"qao").hexdigest(),
            location="Sydney NSW",
            description="Entry-level QA officer, food factory.",
            salary_min=75000,
            salary_max=85000,
            salary_currency="AUD",
            salary_period="year",
        ),
    ]


@pytest.fixture
def mock_smtp(monkeypatch):
    """Capture the MIME message that notify.py would send via SMTP.

    Returns a list that will hold (sender, recipients, message_string) tuples.
    """
    sent: list[tuple] = []

    class _FakeSMTP:
        def __init__(self, host, port, *a, **k):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, *a, **k):
            return None

        def login(self, *a, **k):
            return None

        def send_message(self, msg, *a, **k):
            sent.append((msg.get("From"), msg.get("To"), msg.as_string()))

        def sendmail(self, sender, recipients, message):
            sent.append((sender, recipients, message))

        def quit(self):
            return None

    monkeypatch.setattr("smtplib.SMTP", _FakeSMTP)
    return sent
