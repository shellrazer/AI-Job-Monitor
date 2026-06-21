"""Pure normalization helpers used by the pipeline.

No I/O, no DB, no network. Turns the loose strings emitted by adapters into
structured fields: parsed salary, parsed posted date, and cleaned text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from dateutil import parser as date_parser
from selectolax.parser import HTMLParser

__all__ = ["SalaryParse", "clean_text", "parse_posted_date", "parse_salary"]


@dataclass(slots=True)
class SalaryParse:
    """Structured result of :func:`parse_salary`."""

    min: int | None = None
    max: int | None = None
    currency: str | None = None
    period: str | None = None
    disclosed: bool = False


# Phrases that mean "no salary disclosed".
_UNDISCLOSED = re.compile(
    r"\b(competitive|negotiable|not\s+disclosed|undisclosed|tbc|tba|doe|"
    r"depends?\s+on\s+experience|attractive\s+package|market\s+rate)\b",
    re.IGNORECASE,
)

# A salary figure, optionally with a $ prefix, thousands separators, and a k suffix.
# Groups: (number, k-suffix?)
_FIGURE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k)?",
    re.IGNORECASE,
)


def _period_from_text(text: str) -> str:
    """Infer the pay period from free text. Defaults to 'year'."""
    lowered = text.lower()
    if re.search(r"\b(?:p\.?h\.?|per\s*hour|hour|hourly|/\s*hr|/\s*hour)\b", lowered):
        return "hour"
    if re.search(r"\b(?:per\s*day|daily|/\s*day|day\s*rate)\b", lowered):
        return "day"
    return "year"


def _to_amount(number: str, k_suffix: str | None) -> int:
    """Convert a matched number (with optional 'k') into an int dollar amount."""
    value = float(number.replace(",", ""))
    if k_suffix:
        value *= 1000
    return round(value)


def parse_salary(raw: str | None) -> SalaryParse:
    """Parse an AUD salary string into a :class:`SalaryParse`.

    Handles ranges, single figures, k-suffixes, hourly/daily rates, and
    undisclosed markers. When nothing usable is present, returns an all-None,
    ``disclosed=False`` result.
    """
    if raw is None:
        return SalaryParse()
    text = raw.strip()
    if not text or _UNDISCLOSED.search(text):
        return SalaryParse()

    matches = _FIGURE.findall(text)
    # Drop bare digit groups that are clearly not money (e.g. "super" has none;
    # but a stray "12 months" should not count). We keep it simple: require a
    # $ sign somewhere OR a k-suffix OR a value >= 100 to treat as salary.
    amounts: list[int] = []
    for number, k in matches:
        amount = _to_amount(number, k)
        amounts.append(amount)

    if not amounts:
        return SalaryParse()

    has_dollar = "$" in text
    # If there is no currency marker and the only numbers are tiny (e.g. "3 days"),
    # they were probably not salary. Filter out values under 100 unless k-suffixed
    # or dollar-prefixed.
    filtered: list[int] = []
    for (_number, k), amount in zip(matches, amounts, strict=True):
        if k or has_dollar or amount >= 1000:
            filtered.append(amount)
    amounts = filtered or amounts

    period = _period_from_text(text)

    if len(amounts) >= 2:
        lo, hi = min(amounts[0], amounts[1]), max(amounts[0], amounts[1])
        salary_min, salary_max = lo, hi
    else:
        salary_min = salary_max = amounts[0]

    return SalaryParse(
        min=salary_min,
        max=salary_max,
        currency="AUD",
        period=period,
        disclosed=True,
    )


# --------------------------------------------------------------------------- #
# Posted date                                                                 #
# --------------------------------------------------------------------------- #
_RELATIVE_DAYS = re.compile(r"(\d+)\s*\+?\s*day", re.IGNORECASE)
_RELATIVE_WEEKS = re.compile(r"(\d+)\s*\+?\s*week", re.IGNORECASE)
_RELATIVE_MONTHS = re.compile(r"(\d+)\s*\+?\s*month", re.IGNORECASE)
_RELATIVE_HOURS = re.compile(r"(\d+)\s*\+?\s*(?:hour|hr|minute|min)", re.IGNORECASE)


def parse_posted_date(raw: str | None, *, today: date | None = None) -> date | None:
    """Parse an absolute or relative posted-date string into a :class:`date`.

    Returns ``None`` when the string cannot be parsed. ``today`` is injectable
    for deterministic tests; it defaults to :func:`date.today`.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    base = today or date.today()
    lowered = text.lower()

    if "today" in lowered or "just posted" in lowered or "just now" in lowered:
        return base
    if "yesterday" in lowered:
        return base - timedelta(days=1)

    # Relative "N days/weeks/months/hours ago" (also handles "30+ days ago").
    if "ago" in lowered or "posted" in lowered:
        if m := _RELATIVE_HOURS.search(lowered):
            return base  # less than a day -> today
        if m := _RELATIVE_DAYS.search(lowered):
            return base - timedelta(days=int(m.group(1)))
        if m := _RELATIVE_WEEKS.search(lowered):
            return base - timedelta(weeks=int(m.group(1)))
        if m := _RELATIVE_MONTHS.search(lowered):
            return base - timedelta(days=int(m.group(1)) * 30)

    # Absolute formats: ISO, "18 Jun 2026", "18/06/2026", etc.
    cleaned = re.sub(r"(?i)\bposted\b[:\s]*", "", text).strip()
    try:
        parsed = date_parser.parse(cleaned, dayfirst=True, default=None)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.date()


# --------------------------------------------------------------------------- #
# Text cleaning                                                               #
# --------------------------------------------------------------------------- #
_WHITESPACE = re.compile(r"\s+")


def clean_text(html_or_text: str | None) -> str:
    """Strip HTML tags, decode entities, and collapse whitespace.

    Returns an empty string for ``None``.
    """
    if html_or_text is None:
        return ""
    text = html_or_text
    if "<" in text and ">" in text:
        tree = HTMLParser(text)
        text = tree.text(separator=" ", strip=False)
    else:
        # Still decode any stray entities (e.g. "&amp;") in plain text.
        text = HTMLParser(f"<span>{text}</span>").text(separator=" ", strip=False)
    return _WHITESPACE.sub(" ", text).strip()
