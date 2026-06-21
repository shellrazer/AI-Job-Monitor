"""Deduplication of normalized jobs (spec §13).

Groups jobs that refer to the same posting, picks a canonical per group using
the source preference order, and marks the rest as duplicates. This module does
NOT touch the DB: it only sets ``status`` on losers; the pipeline assigns
``duplicate_of`` after it has the canonical's row id.
"""

from __future__ import annotations

import hashlib
import re

from job_monitor.models import SOURCE_PREFERENCE, Job, JobStatus, Source

__all__ = [
    "dedup_key",
    "deduplicate",
    "description_hash",
    "group_duplicates",
    "normalize_title",
    "pick_best",
]


# Trailing parenthetical/bracketed noise like "(m/f/d)" or "(Maternity Cover)".
_TRAILING_PAREN = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]\s*$")
# Anything that is not a word character or whitespace (after & -> and).
_PUNCT = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Canonical display/dedup title STRING (not a seniority category).

    Lowercases, expands ``&`` to ``and``, drops trailing parenthetical noise,
    strips punctuation, and squashes whitespace.
    """
    if not title:
        return ""
    text = title.lower().replace("&", " and ")
    # Repeatedly strip trailing parenthetical groups, e.g. "x (a) (b)".
    prev = None
    while prev != text:
        prev = text
        text = _TRAILING_PAREN.sub("", text).strip()
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _normalize_description(description: str | None) -> str:
    """Lowercase + whitespace-collapse a description for hashing/comparison."""
    if not description:
        return ""
    return _WHITESPACE.sub(" ", description.lower()).strip()


def description_hash(description: str | None) -> str:
    """SHA-256 hex of the whitespace-normalized, lowercased description ("" if None)."""
    normalized = _normalize_description(description)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedup_key(job: Job) -> tuple:
    """Natural dedup key mirroring the DB unique index in db.py.

    (normalized_title, company_name lower/strip, location or '',
     posted_date iso or '', source_job_id or apply_url)
    """
    posted = job.posted_date.isoformat() if job.posted_date else ""
    identity = job.source_job_id or job.apply_url
    return (
        job.normalized_title,
        job.company_name.lower().strip(),
        job.location or "",
        posted,
        identity,
    )


def _preference_rank(source: Source) -> int:
    """Lower == more preferred. Unknown sources sort last."""
    return SOURCE_PREFERENCE.get(source, len(SOURCE_PREFERENCE))


def pick_best(jobs: list[Job]) -> tuple[Job, list[Job]]:
    """Choose the canonical job from a group of duplicates.

    Canonical = most-preferred source (lowest rank); tie-break by longest
    description, then by apply_url for determinism. Returns (canonical, losers).
    """
    if not jobs:
        raise ValueError("pick_best requires at least one job")

    def sort_key(job: Job) -> tuple:
        return (
            _preference_rank(job.source),
            -len(job.description or ""),
            job.apply_url or "",
        )

    ordered = sorted(jobs, key=sort_key)
    canonical, *losers = ordered
    return canonical, losers


def group_duplicates(jobs: list[Job]) -> list[tuple[Job, list[Job]]]:
    """Group jobs into duplicate clusters and pick a canonical for each.

    Grouping is by :func:`dedup_key`, then clusters that share the same
    (description_hash, normalized_title, company) are merged even when their
    dedup keys differ (e.g. different source_job_id / apply_url / posted_date).
    Returns one (canonical, losers) tuple per group, preserving first-seen order.
    """
    # First pass: group by the natural dedup key.
    key_groups: dict[tuple, list[Job]] = {}
    key_order: list[tuple] = []
    for job in jobs:
        key = dedup_key(job)
        if key not in key_groups:
            key_groups[key] = []
            key_order.append(key)
        key_groups[key].append(job)

    # Second pass: merge groups that share content identity.
    content_to_group: dict[tuple, int] = {}
    merged: list[list[Job]] = []
    for key in key_order:
        members = key_groups[key]
        rep = members[0]
        content_id = (rep.description_hash, rep.normalized_title, rep.company_name.lower().strip())
        if content_id in content_to_group:
            merged[content_to_group[content_id]].extend(members)
        else:
            content_to_group[content_id] = len(merged)
            merged.append(list(members))

    return [pick_best(group) for group in merged]


def deduplicate(jobs: list[Job]) -> list[Job]:
    """Collapse duplicate jobs, marking losers ``status = DUPLICATE``.

    Returns ALL jobs, canonical first within each group, so the pipeline can
    persist canonicals and then link losers via ``duplicate_of`` (set by the
    pipeline once it has DB ids — not here).
    """
    result: list[Job] = []
    for canonical, losers in group_duplicates(jobs):
        result.append(canonical)
        for loser in losers:
            loser.status = JobStatus.DUPLICATE
            result.append(loser)
    return result
