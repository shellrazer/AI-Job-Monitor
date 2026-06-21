"""Tests for job_monitor.normalize.parse_salary and parse_posted_date."""

from __future__ import annotations

from datetime import date

import pytest

from job_monitor.normalize import SalaryParse, parse_posted_date, parse_salary


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # full range with super
        (
            "$140,000 - $160,000 + super",
            SalaryParse(min=140000, max=160000, currency="AUD", period="year", disclosed=True),
        ),
        # k-suffix range
        (
            "$140k - $160k",
            SalaryParse(min=140000, max=160000, currency="AUD", period="year", disclosed=True),
        ),
        # bare range with p.a.
        (
            "120000-140000 p.a.",
            SalaryParse(min=120000, max=140000, currency="AUD", period="year", disclosed=True),
        ),
        # hourly single figure
        (
            "$55/hr",
            SalaryParse(min=55, max=55, currency="AUD", period="hour", disclosed=True),
        ),
        # daily single figure
        (
            "$700/day",
            SalaryParse(min=700, max=700, currency="AUD", period="day", disclosed=True),
        ),
        # exact single annual figure -> both min and max set
        (
            "130000",
            SalaryParse(min=130000, max=130000, currency="AUD", period="year", disclosed=True),
        ),
        ("$130,000", SalaryParse(min=130000, max=130000, currency="AUD", period="year", disclosed=True)),
        ("$130k", SalaryParse(min=130000, max=130000, currency="AUD", period="year", disclosed=True)),
        # undisclosed variants -> all None, disclosed False
        ("Competitive", SalaryParse()),
        ("Not disclosed", SalaryParse()),
        ("Negotiable", SalaryParse()),
        (None, SalaryParse()),
        ("", SalaryParse()),
    ],
)
def test_parse_salary(raw: str | None, expected: SalaryParse) -> None:
    assert parse_salary(raw) == expected


def test_parse_salary_super_does_not_add_a_figure() -> None:
    """'+ super' must not introduce a stray third figure or alter the range."""
    result = parse_salary("$140,000 - $160,000 + super")
    assert (result.min, result.max) == (140000, 160000)


def test_parse_salary_k_decimal() -> None:
    result = parse_salary("$95.5k")
    assert result.min == 95500
    assert result.max == 95500
    assert result.disclosed is True


def test_parse_salary_disclosed_flag() -> None:
    assert parse_salary("$120k").disclosed is True
    assert parse_salary("Competitive").disclosed is False
    assert parse_salary(None).disclosed is False


# --------------------------------------------------------------------------- #
# parse_posted_date                                                           #
# --------------------------------------------------------------------------- #
TODAY = date(2026, 6, 21)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-06-18", date(2026, 6, 18)),
        ("18 Jun 2026", date(2026, 6, 18)),
        ("18/06/2026", date(2026, 6, 18)),
        ("3 days ago", date(2026, 6, 18)),
        ("Posted 2 days ago", date(2026, 6, 19)),
        ("30+ days ago", date(2026, 5, 22)),
        ("today", TODAY),
        ("yesterday", date(2026, 6, 20)),
        ("Posted today", TODAY),
        ("1 week ago", date(2026, 6, 14)),
        ("5 hours ago", TODAY),
        (None, None),
        ("", None),
        ("garbage that is not a date", None),
    ],
)
def test_parse_posted_date(raw: str | None, expected: date | None) -> None:
    assert parse_posted_date(raw, today=TODAY) == expected


def test_parse_posted_date_defaults_to_today() -> None:
    assert parse_posted_date("today") == date.today()
