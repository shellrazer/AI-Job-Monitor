"""Deduplication of normalized jobs (spec §13).

Groups jobs that refer to the same posting, picks a canonical per group using
the source preference order, and marks the rest as duplicates. This module does
NOT touch the DB: it only sets ``status`` on losers; the pipeline assigns
``duplicate_of`` after it has the canonical's row id.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date

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


# Standalone legal/region tokens dropped from a company name during cross-source
# matching. These only count when they appear as whole words.
_COMPANY_STOP_TOKENS = frozenset(
    {
        "pty",
        "ltd",
        "limited",
        "inc",
        "co",
        "group",
        "holdings",
        "australia",
        "australian",
        "aust",
        "anz",
        "oceania",
        "pacific",
        "nz",
        "the",
    }
)

# Descriptor / employment-arrangement noise dropped from a title to leave the
# core role words.
_TITLE_NOISE_TOKENS = frozenset(
    {
        "full",
        "time",
        "part",
        "casual",
        "permanent",
        "ongoing",
        "contract",
        "fixed",
        "term",
        "temp",
        "temporary",
        "month",
        "months",
        "day",
        "night",
        "afternoon",
        "morning",
        "weekend",
        "shift",
        "ref",
        "fte",
        "snr",
    }
)


def _normalize_company(name: str) -> str:
    """Aggressive company-name normalization for cross-source matching.

    Lowercases, expands ``&`` to ``and``, strips punctuation (so apostrophes in
    "Arnott's" collapse to "arnotts"), removes standalone legal/region tokens
    (pty, ltd, limited, australia, the, ...), and squashes whitespace. Returns
    ``""`` for empty/garbage input.

    Examples:
        "PepsiCo Australia"     -> "pepsico"
        "Saputo Dairy Australia"-> "saputo dairy"
        "The Arnott's Group"    -> "arnotts"
    """
    if not name:
        return ""
    text = name.lower().replace("&", " and ")
    # Strip punctuation but keep word characters touching (Arnott's -> arnotts).
    text = _PUNCT.sub("", text)
    tokens = [tok for tok in text.split() if tok and tok not in _COMPANY_STOP_TOKENS]
    return " ".join(tokens)


def _title_core(normalized_title: str) -> str:
    """Reduce an already-normalized title to its core role words.

    Drops employment-arrangement / descriptor noise tokens and standalone
    digits, preserving the order of the remaining words. Input is assumed to
    already be lowercased and punctuation-stripped (i.e. the output of
    :func:`normalize_title`).

    Example:
        "supplier quality assurance associate scientist 12 month fixed term contract"
        -> "supplier quality assurance associate scientist"
    """
    if not normalized_title:
        return ""
    tokens = [
        tok
        for tok in normalized_title.split()
        if tok and tok not in _TITLE_NOISE_TOKENS and not tok.isdigit()
    ]
    return " ".join(tokens)


def _city(location: str | None) -> str:
    """First comma-separated chunk of a location, lowercased and trimmed.

    Example: "Chatswood, Australia" -> "chatswood"; ``None``/"" -> "".
    """
    if not location:
        return ""
    return location.split(",", 1)[0].strip().lower()


def _locations_compatible(a: str | None, b: str | None) -> bool:
    """Whether two locations are plausibly the same place.

    Compatible when either city is empty (unknown), or the two cities share at
    least one whitespace token. So "Sydney" vs "Sydney NSW" -> True, but
    "Sydney" vs "Melbourne" -> False.
    """
    city_a = _city(a)
    city_b = _city(b)
    if not city_a or not city_b:
        return True
    return bool(set(city_a.split()) & set(city_b.split()))


def _dates_within(a: date | None, b: date | None, days: int = 21) -> bool:
    """Whether two posted dates are close enough to be the same posting.

    True when either date is unknown (``None``) or they are within ``days`` of
    each other.
    """
    if a is None or b is None:
        return True
    return abs((a - b).days) <= days


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


def _clusters_match(rep_a: Job, rep_b: Job) -> bool:
    """Whether two cluster representatives are the SAME role across sources.

    Conservative: ALL of the following must hold —
    * normalized companies are equal and non-empty,
    * core title role words are equal and non-empty,
    * locations are compatible (shared city token / unknown), and
    * posted dates are within the default window.
    """
    company_a = _normalize_company(rep_a.company_name)
    company_b = _normalize_company(rep_b.company_name)
    if not company_a or company_a != company_b:
        return False
    core_a = _title_core(rep_a.normalized_title)
    core_b = _title_core(rep_b.normalized_title)
    if not core_a or core_a != core_b:
        return False
    if not _locations_compatible(rep_a.location, rep_b.location):
        return False
    return _dates_within(rep_a.posted_date, rep_b.posted_date)


def group_duplicates(jobs: list[Job]) -> list[tuple[Job, list[Job]]]:
    """Group jobs into duplicate clusters and pick a canonical for each.

    Three passes:

    1. Group by the natural :func:`dedup_key`.
    2. Merge clusters that share content identity
       (description_hash, normalized_title, company) even when their dedup keys
       differ (e.g. different source_job_id / apply_url / posted_date).
    3. Cross-source merge: collapse the SAME role appearing on different sources
       (e.g. LinkedIn + SEEK + the official site) when company, core title,
       location, and posted date all agree (see :func:`_clusters_match`).

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

    # Third pass: cross-source merge over cluster representatives via union-find.
    # parent[i] points toward the representative cluster index for cluster i.
    parent = list(range(len(merged)))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:  # path compression
            parent[i], i = root, parent[i]
        return root

    def union(i: int, j: int) -> None:
        # Keep the earlier (first-seen) cluster as the root to preserve order.
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            return
        lo, hi = (root_i, root_j) if root_i < root_j else (root_j, root_i)
        parent[hi] = lo

    reps = [group[0] for group in merged]
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            if find(i) == find(j):
                continue
            if _clusters_match(reps[i], reps[j]):
                union(i, j)

    # Collect members per root, preserving the first-seen order of roots and of
    # members within each root.
    final_groups: list[list[Job]] = []
    root_to_index: dict[int, int] = {}
    for i, group in enumerate(merged):
        root = find(i)
        if root not in root_to_index:
            root_to_index[root] = len(final_groups)
            final_groups.append([])
        final_groups[root_to_index[root]].extend(group)

    return [pick_best(group) for group in final_groups]


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
