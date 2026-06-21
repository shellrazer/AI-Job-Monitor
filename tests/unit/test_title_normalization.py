"""Tests for normalize_title, description_hash, and clean_text."""

from __future__ import annotations

import pytest

from job_monitor.dedup import description_hash, normalize_title
from job_monitor.normalize import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # case folding
        ("Site Quality Manager", "site quality manager"),
        ("SITE QUALITY MANAGER", "site quality manager"),
        # ampersand -> and
        ("Health & Safety Lead", "health and safety lead"),
        ("R&D Manager", "r and d manager"),
        # punctuation stripped
        ("Quality Manager, Senior", "quality manager senior"),
        ("QA/QC Lead", "qa qc lead"),
        # trailing parenthetical noise removed
        ("Quality Manager (m/f/d)", "quality manager"),
        ("Quality Manager (Maternity Cover)", "quality manager"),
        ("Quality Manager [Remote]", "quality manager"),
        ("Quality Manager (12 month contract) (m/f/d)", "quality manager"),
        # whitespace squashed
        ("  Quality    Manager  ", "quality manager"),
        # empty
        ("", ""),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


def test_normalize_title_keeps_internal_parens_meaning_via_strip() -> None:
    # Only TRAILING parentheticals are dropped; the rest is punctuation-stripped.
    assert normalize_title("Manager (APAC) Quality") == "manager apac quality"


def test_description_hash_stable_across_whitespace() -> None:
    a = description_hash("Lead the site quality team.")
    b = description_hash("Lead   the\nsite  quality team.")
    c = description_hash("  Lead the site quality team.  ")
    assert a == b == c


def test_description_hash_case_insensitive() -> None:
    assert description_hash("HACCP GMP Audit") == description_hash("haccp gmp audit")


def test_description_hash_differs_for_different_text() -> None:
    assert description_hash("alpha") != description_hash("beta")


def test_description_hash_none_and_empty() -> None:
    assert description_hash(None) == description_hash("")
    # Stable, deterministic sha256 hex length.
    assert len(description_hash(None)) == 64


def test_clean_text_strips_html_and_collapses() -> None:
    html = "<p>Lead   the <b>site</b> quality team.</p>\n<p>HACCP &amp; GMP</p>"
    assert clean_text(html) == "Lead the site quality team. HACCP & GMP"


def test_clean_text_decodes_entities_in_plain_text() -> None:
    assert clean_text("Food &amp; Beverage") == "Food & Beverage"


def test_clean_text_none_returns_empty() -> None:
    assert clean_text(None) == ""
