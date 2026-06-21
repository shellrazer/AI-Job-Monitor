"""SEEK (seek.com.au) source adapter.

SEEK is a Cloudflare-fronted Australian job board, so live fetches go through
:meth:`PoliteClient.get_text_impersonate` (curl_cffi browser impersonation).
:meth:`SeekAdapter.parse` is pure and offline-testable against a captured
search-results HTML fixture.

Two extraction paths are implemented, defensively, so a markup change on either
side degrades gracefully rather than crashing:

* PRIMARY: the ``window.SEEK_REDUX_DATA = {...};`` JSON blob embedded in the
  page. Its ``results.results.jobs`` array carries richly structured records
  (id / title / companyName / advertiser / locations / salaryLabel /
  listingDateDisplay / teaser). This is the most robust source and is tried
  first.
* FALLBACK: HTML job cards selected via SEEK's ``data-automation`` attributes
  (``normalJob`` articles with ``jobTitle`` / ``jobCompany`` / ``jobLocation`` /
  ``jobSalary`` / ``jobListingDate`` / ``jobShortDescription`` children). Used
  only when the redux blob is absent or yields nothing.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar
from urllib.parse import quote, urlsplit

from selectolax.parser import HTMLParser, Node

from job_monitor.models import RawJob, Source, SourceBlocked
from job_monitor.normalize import clean_text
from job_monitor.sources.base import BaseAdapter

DEFAULT_BASE_URL = "https://www.seek.com.au"

# Default number of search-results pages to walk when the company config does
# not specify ``max_pages``.
DEFAULT_MAX_PAGES = 3

# Locates the start of the redux assignment; the object itself is extracted by
# brace-matching from the first ``{`` (regex alone can't balance nested braces).
_REDUX_MARKER = re.compile(r"window\.SEEK_REDUX_DATA\s*=\s*")

# Pulls the numeric job id out of an apply href such as
# ``/job/92751904?type=standard&ref=...`` (also tolerates absolute URLs).
_JOB_ID_RE = re.compile(r"/job/(\d+)")

# Company names SEEK uses for hidden advertisers; treated as "no company".
_HIDDEN_COMPANIES = {"private advertiser", "private advertisor"}


class SeekAdapter(BaseAdapter):
    """Adapter for SEEK search-results pages."""

    name: ClassVar[str] = "seek"
    source: ClassVar[Source] = Source.SEEK

    # ------------------------------------------------------------------ #
    # Fetch                                                              #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch SEEK search results across pages for the company's keywords/where.

        Builds the query-form URL ``.../jobs?keywords=...&where=...&page={n}``
        and walks pages ``1..max_pages`` (``max_pages`` read from the company's
        ``search`` config, default :data:`DEFAULT_MAX_PAGES`), retrieving each via
        curl_cffi impersonation (SEEK sits behind Cloudflare). Results are
        accumulated and de-duplicated by ``apply_url``.

        Stops early when a page yields zero jobs or contributes no new
        ``apply_url`` (end of results). ``SourceBlocked`` from page 1 propagates;
        if a later page blocks, we stop and return what we have so far.
        """
        search = getattr(self.company, "search", None) or {}
        keywords = search.get("keywords")
        if not keywords:
            # Fall back to the resolved search terms (joined as an OR-ish query;
            # SEEK already understands OR syntax inside ``keywords``).
            keywords = " ".join(search_terms) if search_terms else ""
        where = search.get("where", "")
        max_pages = self._max_pages(search)

        by_url: dict[str, RawJob] = {}
        for page in range(1, max_pages + 1):
            url = self._search_url(str(keywords), str(where), page)
            try:
                html = self.http.get_text_impersonate(url)
            except SourceBlocked:
                if page == 1:
                    raise
                # A later page blocked: keep what we already gathered.
                break

            jobs = self.parse(html, base_url=DEFAULT_BASE_URL)
            if not jobs:
                # Empty page => end of results.
                break

            added = False
            for job in jobs:
                if job.apply_url not in by_url:
                    by_url[job.apply_url] = job
                    added = True
            if not added:
                # No new postings on this page => end of results.
                break

        return list(by_url.values())

    @staticmethod
    def _max_pages(search: dict[str, Any]) -> int:
        try:
            value = int(search.get("max_pages", DEFAULT_MAX_PAGES))
        except (TypeError, ValueError):
            return DEFAULT_MAX_PAGES
        return value if value >= 1 else DEFAULT_MAX_PAGES

    def _search_url(self, keywords: str, where: str, page: int) -> str:
        return (
            f"{DEFAULT_BASE_URL}/jobs?keywords={quote(keywords)}"
            f"&where={quote(where)}&page={page}"
        )

    # ------------------------------------------------------------------ #
    # Parse                                                              #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Parse SEEK search-results HTML into :class:`RawJob` records.

        Tries the redux JSON blob first and falls back to ``data-automation``
        HTML cards. Always returns a list; never raises on a missing field.
        """
        base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        html = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)

        jobs = self._parse_redux(html, base)
        if jobs:
            return jobs
        return self._parse_html(html, base)

    # -------------------------- redux path --------------------------- #
    def _parse_redux(self, html: str, base: str) -> list[RawJob]:
        blob = self._extract_redux_blob(html)
        if blob is None:
            return []
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            return []

        records = self._redux_job_records(data)
        jobs: list[RawJob] = []
        for rec in records:
            job = self._redux_job(rec, base)
            if job is not None:
                jobs.append(job)
        return jobs

    @staticmethod
    def _extract_redux_blob(html: str) -> str | None:
        """Return the balanced ``{...}`` object assigned to SEEK_REDUX_DATA."""
        m = _REDUX_MARKER.search(html)
        if m is None:
            return None
        start = m.end()
        if start >= len(html) or html[start] != "{":
            return None

        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(html)):
            c = html[i]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return html[start : i + 1]
        return None

    @staticmethod
    def _redux_job_records(data: Any) -> list[dict[str, Any]]:
        """Pull the list of job dicts out of the redux state, defensively.

        Preferred shape is ``results.results.jobs``; we tolerate either a list
        or a dict of jobs there, and otherwise scan a couple of likely spots.
        """
        if not isinstance(data, dict):
            return []

        results = data.get("results")
        if isinstance(results, dict):
            inner = results.get("results")
            container = inner.get("jobs") if isinstance(inner, dict) else None
            if container is None:
                container = results.get("jobs")
            records = SeekAdapter._as_record_list(container)
            if records:
                return records

        return []

    @staticmethod
    def _as_record_list(container: Any) -> list[dict[str, Any]]:
        if isinstance(container, list):
            return [r for r in container if isinstance(r, dict)]
        if isinstance(container, dict):
            return [r for r in container.values() if isinstance(r, dict)]
        return []

    def _redux_job(self, rec: dict[str, Any], base: str) -> RawJob | None:
        title = clean_text(str(rec.get("title") or "")).strip()
        if not title:
            return None

        job_id = rec.get("id")
        job_id_str = str(job_id) if job_id not in (None, "") else None

        apply_url = self._apply_url_from_id(job_id_str, base)
        if not apply_url:
            return None

        company = self._clean_company(rec.get("companyName") or self._advertiser_name(rec))
        location = self._first_location(rec)
        salary = clean_text(str(rec.get("salaryLabel") or "")).strip() or None
        posted = clean_text(str(rec.get("listingDateDisplay") or "")).strip() or None
        description = clean_text(str(rec.get("teaser") or "")) or None

        return RawJob(
            source=Source.SEEK,
            title=title,
            company_name=company,
            apply_url=apply_url,
            source_job_id=job_id_str,
            location=location,
            description=description,
            posted_date_raw=posted,
            salary_raw=salary,
            company_id=None,
        )

    @staticmethod
    def _advertiser_name(rec: dict[str, Any]) -> str:
        advertiser = rec.get("advertiser")
        if isinstance(advertiser, dict):
            return str(advertiser.get("description") or "")
        return ""

    @staticmethod
    def _first_location(rec: dict[str, Any]) -> str | None:
        locations = rec.get("locations")
        if isinstance(locations, list):
            for loc in locations:
                if isinstance(loc, dict):
                    label = clean_text(str(loc.get("label") or "")).strip()
                    if label:
                        return label
        return None

    # --------------------------- html path --------------------------- #
    def _parse_html(self, html: str, base: str) -> list[RawJob]:
        tree = HTMLParser(html)
        jobs: list[RawJob] = []
        for card in tree.css('[data-automation="normalJob"]'):
            job = self._html_job(card, base)
            if job is not None:
                jobs.append(job)
        return jobs

    def _html_job(self, card: Node, base: str) -> RawJob | None:
        title_el = card.css_first('[data-automation="jobTitle"]')
        if title_el is None:
            return None
        title = clean_text(title_el.text(strip=True)).strip()
        if not title:
            return None

        href = title_el.attributes.get("href") or ""
        apply_url = self._apply_url_from_href(href, base)
        if not apply_url:
            return None
        job_id = self._job_id_from_href(href)

        company = self._clean_company(self._card_text(card, "jobCompany"))
        location = self._card_text(card, "jobLocation") or self._card_text(card, "jobCardLocation") or None
        salary = self._card_text(card, "jobSalary") or None
        posted = self._card_text(card, "jobListingDate") or None
        description = self._card_text(card, "jobShortDescription") or None

        return RawJob(
            source=Source.SEEK,
            title=title,
            company_name=company,
            apply_url=apply_url,
            source_job_id=job_id,
            location=location,
            description=description,
            posted_date_raw=posted,
            salary_raw=salary,
            company_id=None,
        )

    @staticmethod
    def _card_text(card: Node, automation: str) -> str:
        el = card.css_first(f'[data-automation="{automation}"]')
        if el is None:
            return ""
        return clean_text(el.text(strip=True)).strip()

    # --------------------------- url helpers -------------------------- #
    def _apply_url_from_href(self, href: str, base: str) -> str | None:
        job_id = self._job_id_from_href(href)
        if job_id:
            return self._apply_url_from_id(job_id, base)
        # No /job/<id> in the href: best-effort absolute URL with tracking
        # query stripped.
        cleaned = href.split("?", 1)[0].strip()
        if not cleaned:
            return None
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return cleaned
        return f"{base}{cleaned if cleaned.startswith('/') else '/' + cleaned}"

    @staticmethod
    def _apply_url_from_id(job_id: str | None, base: str) -> str | None:
        if not job_id:
            return None
        return f"{base}/job/{job_id}"

    @staticmethod
    def _job_id_from_href(href: str) -> str | None:
        if not href:
            return None
        # Strip query/fragment first so only the path is searched.
        path = urlsplit(href).path or href
        m = _JOB_ID_RE.search(path)
        return m.group(1) if m else None

    # --------------------------- misc helpers ------------------------- #
    @staticmethod
    def _clean_company(value: str | None) -> str:
        company = clean_text(str(value or "")).strip()
        if not company or company.lower() in _HIDDEN_COMPANIES:
            return ""
        return company
