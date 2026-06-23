"""End-to-end pipeline test — fully offline (fake embedder, injected fetch, mocked SMTP)."""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np

from job_monitor.config import load_config
from job_monitor.models import Job, JobStatus, PriorityTier, Source
from job_monitor.pipeline import _select_for_digest, regate_existing, run_pipeline

TODAY = date(2026, 6, 21)


def _vec(text: str, dim: int = 384) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEmbedder:
    """Deterministic offline embedder matching the real Embedder.encode contract."""

    def encode(self, texts):
        if isinstance(texts, str):
            return _vec(texts)
        return np.vstack([_vec(t) for t in texts])


def _make_cfg(tmp_path):
    cfg = load_config()
    cfg.settings.db_path = str(tmp_path / "jobs.sqlite")
    cfg.settings.report.output_dir = str(tmp_path / "reports")
    cfg.settings.email.enabled = True
    cfg.settings.email.sender = "you@gmail.com"
    cfg.settings.email.app_password = "app-pw"
    cfg.settings.email.recipients = ["you@gmail.com"]
    cfg.settings.report.min_tier = "D"  # include everything scorable for the assertion
    return cfg


def test_pipeline_end_to_end(tmp_path, tmp_db, sample_raw_jobs, mock_smtp):
    cfg = _make_cfg(tmp_path)

    def fake_fetch(_cfg, _http, stats):
        from job_monitor.logging import SourceStat

        stats.add_source(SourceStat(company_id="bega", name="Bega Group", fetched=len(sample_raw_jobs)))
        return list(sample_raw_jobs)

    result = run_pipeline(
        cfg,
        embedder=FakeEmbedder(),
        conn=tmp_db,
        today=TODAY,
        fetch_fn=fake_fetch,
        write_report=True,
        send_email=None,  # follow cfg.email.enabled (True)
    )

    # dedup: two identical Site Quality Manager postings collapse, software QA stays
    assert result.stats.canonical == 2
    assert result.stats.duplicates == 1
    assert result.stats.persisted == 3

    # DB state
    rows = tmp_db.execute("SELECT title, status, priority_tier, final_score FROM jobs ORDER BY id").fetchall()
    assert len(rows) == 3
    statuses = {r["title"]: r["status"] for r in rows}
    assert statuses["Software QA Engineer"] == JobStatus.IRRELEVANT.value  # hard-excluded
    assert JobStatus.DUPLICATE.value in [r["status"] for r in rows]  # the SEEK dup

    # report written and contains the senior role, not the excluded one
    assert result.report_path is not None
    assert result.report_path.exists()
    html = result.report_path.read_text(encoding="utf-8")
    assert "Site Quality Manager" in html
    titles = [j.title for j in result.top_jobs]
    assert "Site Quality Manager" in titles
    assert "Software QA Engineer" not in titles

    # email captured by the mocked SMTP
    assert result.emailed is True
    assert len(mock_smtp) == 1
    _from, _to, body = mock_smtp[0]
    assert "Site Quality Manager" in body

    # alerts recorded
    alert_count = tmp_db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    assert alert_count >= 1


def test_pipeline_no_email_when_disabled(tmp_path, tmp_db, sample_raw_jobs, mock_smtp):
    cfg = _make_cfg(tmp_path)

    def fake_fetch(_cfg, _http, stats):
        return list(sample_raw_jobs)

    result = run_pipeline(
        cfg, embedder=FakeEmbedder(), conn=tmp_db, today=TODAY,
        fetch_fn=fake_fetch, write_report=True, send_email=False,
    )
    assert result.emailed is False
    assert len(mock_smtp) == 0
    assert result.report_path is not None
    assert result.report_path.exists()


def test_daily_digest_emails_only_new_matches(tmp_path, tmp_db, sample_raw_jobs, mock_smtp):
    """Daily newsletter: first run emails matches; an identical re-run sends nothing new."""
    cfg = _make_cfg(tmp_path)
    assert cfg.settings.report.email_new_only is True

    def fake_fetch(_cfg, _http, stats):
        return list(sample_raw_jobs)

    run1 = run_pipeline(cfg, embedder=FakeEmbedder(), conn=tmp_db, today=TODAY,
                        fetch_fn=fake_fetch, write_report=False, send_email=None)
    assert run1.emailed is True
    assert len(run1.digest_jobs) >= 1  # the senior role is new

    # Identical re-run: nothing newly discovered -> no email, empty digest.
    run2 = run_pipeline(cfg, embedder=FakeEmbedder(), conn=tmp_db, today=TODAY,
                        fetch_fn=fake_fetch, write_report=False, send_email=None)
    assert run2.digest_jobs == []
    assert run2.emailed is False
    assert len(mock_smtp) == 1  # only run1 sent


def _stored_job(tmp_db, title, location):
    from job_monitor import db as dbmod

    job = Job(
        source=Source.OFFICIAL_ATS, title=title, normalized_title=title.lower(),
        company_name="Bega Group", apply_url=f"https://x/{hashlib.sha256(title.encode()).hexdigest()[:8]}",
        description_hash=hashlib.sha256(title.encode()).hexdigest(), location=location,
        description="", final_score=50.0,
    )
    return dbmod.upsert_job(tmp_db, job)


def _scored_job(title, location, score) -> Job:
    """A minimal already-scored, report-eligible Job for digest-selection tests."""
    return Job(
        source=Source.SEEK,
        title=title,
        normalized_title=title.lower(),
        company_name="Cloud Foods",
        apply_url=f"https://x/{hashlib.sha256(title.encode()).hexdigest()[:8]}",
        description_hash=hashlib.sha256(title.encode()).hexdigest(),
        location=location,
        final_score=score,
        priority_tier=PriorityTier.A,
        status=JobStatus.NEW,
    )


def test_select_for_digest_nsw_only_filters_other_state(tmp_path):
    """nsw_only drops a clearly-other-state role but keeps Sydney + unknown-region."""
    cfg = _make_cfg(tmp_path)
    sydney = _scored_job("Sydney QM", "Sydney NSW", 80.0)
    melbourne = _scored_job("Melbourne QM", "Melbourne VIC", 75.0)
    unknown = _scored_job("Unknown Region QM", None, 70.0)
    # Location is bare/ambiguous but the TITLE names another state -> other-region.
    title_leak = _scored_job(
        "National Quality and Maintenance Coordinator - Moorabbin, VIC",
        "Moorabbin, Australia",
        73.0,
    )
    jobs = [sydney, melbourne, unknown, title_leak]

    cfg.settings.report.nsw_only = True
    selected = {j.title for j in _select_for_digest(jobs, cfg)}
    assert "Sydney QM" in selected
    assert "Unknown Region QM" in selected  # unknown-region kept
    assert "Melbourne QM" not in selected  # clearly other-state dropped
    # Other-state signal only in the title is also dropped under nsw_only.
    assert "National Quality and Maintenance Coordinator - Moorabbin, VIC" not in selected

    cfg.settings.report.nsw_only = False
    selected_all = {j.title for j in _select_for_digest(jobs, cfg)}
    assert "Melbourne QM" in selected_all  # retained when the view is off
    # Title-leak role is kept when the NSW-only view is off.
    assert "National Quality and Maintenance Coordinator - Moorabbin, VIC" in selected_all


def test_regate_existing_cleans_stale_rows(tmp_db):
    """Stale 'new' rows that are overseas or non-quality get marked irrelevant; good ones stay."""
    good = _stored_job(tmp_db, "Quality Manager", "Sydney NSW")
    overseas = _stored_job(tmp_db, "Quality Manager", "Franklin, WI")
    nonquality = _stored_job(tmp_db, "Production Operator", "Melbourne VIC")

    changed = regate_existing(tmp_db)
    assert changed == 2

    def status(jid):
        return tmp_db.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()[0]

    assert status(good) == JobStatus.NEW.value
    assert status(overseas) == JobStatus.IRRELEVANT.value
    assert status(nonquality) == JobStatus.IRRELEVANT.value


def test_upsert_returns_affected_row_not_stale_lastrowid(tmp_db):
    """Updating an existing row must return ITS id, not a stale lastrowid from a
    prior insert (the bug that orphaned duplicates)."""
    from job_monitor import db as dbmod

    id_a = _stored_job(tmp_db, "Quality Manager", "Sydney NSW")  # insert -> id_a
    _stored_job(tmp_db, "QA Manager", "Sydney NSW")              # insert -> id_b (lastrowid now id_b)
    again = Job(
        source=Source.OFFICIAL_ATS, title="Quality Manager", normalized_title="quality manager",
        company_name="Bega Group",
        apply_url=f"https://x/{hashlib.sha256(b'Quality Manager').hexdigest()[:8]}",
        description_hash=hashlib.sha256(b"Quality Manager").hexdigest(), location="Sydney NSW",
        final_score=61.0,
    )
    assert dbmod.upsert_job(tmp_db, again) == id_a  # not the stale id_b


def test_repair_orphan_duplicates_promotes_when_no_canonical(tmp_db):
    """A role whose every copy is orphaned (duplicate + duplicate_of NULL) gets one
    copy promoted back to 'new' so it isn't permanently hidden."""
    from job_monitor.pipeline import repair_orphan_duplicates

    a = _stored_job(tmp_db, "Production & QA lead", "Sydney NSW")
    b = _stored_job(tmp_db, "Production & QA lead", "Sydney CBD NSW")
    for jid in (a, b):
        tmp_db.execute("UPDATE jobs SET status='duplicate', duplicate_of=NULL WHERE id=?", (jid,))
    tmp_db.commit()

    relinked, promoted = repair_orphan_duplicates(tmp_db)
    assert promoted == 1
    assert relinked == 1
    statuses = sorted(r["status"] for r in tmp_db.execute("SELECT status FROM jobs WHERE id IN (?,?)", (a, b)))
    assert statuses == ["duplicate", "new"]  # one resurfaced, one linked
