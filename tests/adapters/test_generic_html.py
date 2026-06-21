"""Offline tests for :class:`job_monitor.sources.generic_html.GenericHtmlAdapter`.

All tests drive the pure ``parse`` path against a synthetic HTML fixture; no
network access. They cover the documented selector syntax, relative-link
resolution, the company-name fallback, and defensive skipping of malformed
items.
"""

from __future__ import annotations

from typing import Any

from selectolax.parser import HTMLParser

from job_monitor.config import CompanyConfig, HttpSettings, Settings
from job_monitor.models import Source
from job_monitor.sources.generic_html import GenericHtmlAdapter
from job_monitor.sources.http import PoliteClient

# Three job cards with a known structure. The company on card 2 is intentionally
# omitted to exercise the fallback to the company config name.
SAMPLE_HTML = """
<html><body>
  <div class="job-card">
    <a class="title" href="/jobs/1">Site Quality Manager</a>
    <span class="loc">Sydney NSW</span>
    <span class="co">ACME Foods</span>
    <span class="sal">$140k-$160k</span>
    <span class="date">Posted 18 Jun 2026</span>
    <span class="desc">HACCP, FSSC 22000, multi-site</span>
  </div>
  <div class="job-card">
    <a class="title" href="/jobs/2">National Quality Lead</a>
    <span class="loc">Melbourne VIC</span>
    <span class="sal">$170k-$190k</span>
    <span class="desc">Lead multi-site GMP &amp; audit programs</span>
  </div>
  <div class="job-card">
    <a class="title" href="https://other.example.com/jobs/3">Quality Coordinator</a>
    <span class="loc">Brisbane QLD</span>
    <span class="co">ACME Foods</span>
    <span class="desc">Supplier quality, CAPA</span>
  </div>
</body></html>
"""

SELECTORS = {
    "item": "div.job-card",
    "title": "a.title",
    "link": "a.title@href",
    "location": "span.loc",
    "company": "span.co",
    "salary": "span.sal",
    "date": "span.date",
    "snippet": "span.desc",
}


def _make_adapter(search: dict[str, Any]) -> GenericHtmlAdapter:
    company = CompanyConfig(
        company_id="acme",
        name="ACME Foods",
        adapter="generic_html",
        search=search,
    )
    http = PoliteClient(HttpSettings())
    return GenericHtmlAdapter(http=http, company=company, settings=Settings())


def _default_search(**overrides: Any) -> dict[str, Any]:
    search: dict[str, Any] = {
        "base_url": "https://careers.acme.com",
        "selectors": dict(SELECTORS),
    }
    search.update(overrides)
    return search


# --------------------------------------------------------------------------- #
# parse()                                                                     #
# --------------------------------------------------------------------------- #
def test_parse_extracts_three_jobs_with_all_fields() -> None:
    adapter = _make_adapter(_default_search())

    jobs = adapter.parse(SAMPLE_HTML)

    assert len(jobs) == 3

    first = jobs[0]
    assert first.source is Source.OFFICIAL_ATS
    assert first.title == "Site Quality Manager"
    # Relative link resolved against base_url.
    assert first.apply_url == "https://careers.acme.com/jobs/1"
    assert first.location == "Sydney NSW"
    assert first.company_name == "ACME Foods"
    assert first.salary_raw == "$140k-$160k"
    assert first.posted_date_raw == "Posted 18 Jun 2026"
    assert first.description is not None
    assert "HACCP" in first.description
    assert first.company_id == "acme"


def test_parse_company_selector_absent_falls_back_to_config_name() -> None:
    adapter = _make_adapter(_default_search())

    jobs = adapter.parse(SAMPLE_HTML)

    # Card 2 has no <span class="co">, so it must fall back to the config name.
    second = jobs[1]
    assert second.title == "National Quality Lead"
    assert second.company_name == "ACME Foods"
    # Entity in the snippet is decoded by clean_text.
    assert second.description is not None
    assert "GMP & audit" in second.description


def test_parse_absolute_link_is_left_untouched() -> None:
    adapter = _make_adapter(_default_search())

    jobs = adapter.parse(SAMPLE_HTML)

    third = jobs[2]
    assert third.apply_url == "https://other.example.com/jobs/3"


def test_parse_skips_item_without_title() -> None:
    html = """
    <div class="job-card">
      <a class="title" href="/jobs/ok">Has Title</a>
    </div>
    <div class="job-card">
      <a class="title" href="/jobs/no-title"></a>
      <span class="loc">Nowhere</span>
    </div>
    """
    adapter = _make_adapter(_default_search())

    jobs = adapter.parse(html)

    assert len(jobs) == 1
    assert jobs[0].title == "Has Title"


def test_parse_skips_item_without_link() -> None:
    html = """
    <div class="job-card">
      <a class="title" href="/jobs/ok">Has Link</a>
    </div>
    <div class="job-card">
      <a class="title">No Link Here</a>
    </div>
    """
    adapter = _make_adapter(_default_search())

    jobs = adapter.parse(html)

    assert len(jobs) == 1
    assert jobs[0].title == "Has Link"


def test_parse_returns_empty_when_no_items_match() -> None:
    adapter = _make_adapter(_default_search())
    # Valid HTML, but nothing matches the configured item selector.
    assert adapter.parse("<html><body><p>No jobs today</p></body></html>") == []


def test_parse_returns_empty_without_item_selector() -> None:
    search = _default_search(selectors={"title": "a", "link": "a@href"})
    adapter = _make_adapter(search)
    assert adapter.parse(SAMPLE_HTML) == []


def test_parse_handles_none_and_bytes_payloads() -> None:
    adapter = _make_adapter(_default_search())
    assert adapter.parse(None) == []
    jobs = adapter.parse(SAMPLE_HTML.encode("utf-8"))
    assert len(jobs) == 3


def test_parse_resolves_relative_against_careers_url_when_no_base_url() -> None:
    search = {"selectors": dict(SELECTORS)}  # no base_url
    company = CompanyConfig(
        company_id="acme",
        name="ACME Foods",
        adapter="generic_html",
        careers_url="https://jobs.acme.com",
        search=search,
    )
    adapter = GenericHtmlAdapter(
        http=PoliteClient(HttpSettings()),
        company=company,
        settings=Settings(),
    )

    jobs = adapter.parse(SAMPLE_HTML)

    assert jobs[0].apply_url == "https://jobs.acme.com/jobs/1"


def test_parse_base_url_override_argument_wins() -> None:
    adapter = _make_adapter(_default_search())

    jobs = adapter.parse(SAMPLE_HTML, base_url="https://override.example.com")

    assert jobs[0].apply_url == "https://override.example.com/jobs/1"


# --------------------------------------------------------------------------- #
# _select() helper                                                            #
# --------------------------------------------------------------------------- #
def _node(html: str):
    return HTMLParser(html).css_first("div")


def test_select_text_vs_attr() -> None:
    node = _node('<div><a class="title" href="/jobs/9">Quality Manager</a></div>')

    # No "@attr" -> element text.
    assert GenericHtmlAdapter._select(node, "a.title") == "Quality Manager"
    # "@attr" -> the named attribute value.
    assert GenericHtmlAdapter._select(node, "a.title@href") == "/jobs/9"


def test_select_returns_none_for_missing_element_or_attr() -> None:
    node = _node('<div><a class="title" href="/jobs/9">Quality Manager</a></div>')

    assert GenericHtmlAdapter._select(node, "span.missing") is None  # no element
    assert GenericHtmlAdapter._select(node, "a.title@data-id") is None  # no attr
    assert GenericHtmlAdapter._select(node, None) is None  # no spec
    assert GenericHtmlAdapter._select(node, "") is None  # empty spec
    assert GenericHtmlAdapter._select(node, "@href") is None  # no css part


def test_select_blank_text_returns_none() -> None:
    node = _node('<div><span class="loc">   </span></div>')
    assert GenericHtmlAdapter._select(node, "span.loc") is None


# --------------------------------------------------------------------------- #
# fetch()                                                                     #
# --------------------------------------------------------------------------- #
class _FakeHttp:
    """Records which transport was used and returns canned HTML."""

    def __init__(self, html: str) -> None:
        self.html = html
        self.calls: list[tuple[str, str]] = []

    def get_text(self, url: str, **_: Any) -> str:
        self.calls.append(("httpx", url))
        return self.html

    def get_text_impersonate(self, url: str, **_: Any) -> str:
        self.calls.append(("cffi", url))
        return self.html


def _adapter_with_http(http: Any, search: dict[str, Any]) -> GenericHtmlAdapter:
    company = CompanyConfig(
        company_id="acme",
        name="ACME Foods",
        adapter="generic_html",
        search=search,
    )
    return GenericHtmlAdapter(http=http, company=company, settings=Settings())


def test_fetch_uses_httpx_and_parses() -> None:
    http = _FakeHttp(SAMPLE_HTML)
    search = _default_search(list_url="https://careers.acme.com/jobs")
    adapter = _adapter_with_http(http, search)

    jobs = adapter.fetch(["quality manager"])

    assert len(jobs) == 3
    assert http.calls == [("httpx", "https://careers.acme.com/jobs")]


def test_fetch_uses_impersonate_when_configured() -> None:
    http = _FakeHttp(SAMPLE_HTML)
    search = _default_search(
        list_url="https://careers.acme.com/jobs",
        impersonate=True,
    )
    adapter = _adapter_with_http(http, search)

    jobs = adapter.fetch(["quality manager"])

    assert len(jobs) == 3
    assert http.calls == [("cffi", "https://careers.acme.com/jobs")]


def test_fetch_returns_empty_without_list_url() -> None:
    http = _FakeHttp(SAMPLE_HTML)
    adapter = _adapter_with_http(http, _default_search())

    assert adapter.fetch(["quality manager"]) == []
    assert http.calls == []
