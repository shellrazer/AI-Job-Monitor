"""SAP SuccessFactors Recruiting Marketing source adapter.

SuccessFactors career sites (e.g. a "Careers Centre" branded landing page) are
almost always rendered client-side: the static HTML served to a plain HTTP
client contains the chrome (header, footer, browse-by-category links) but *not*
the individual job postings, which are injected later by JavaScript. A plain
``get_text`` of the landing page therefore yields zero postings.

SuccessFactors Recruiting Marketing does, however, expose a hidden RSS / XML job
feed. The exact per-company URL is not discoverable from the landing page, but
common shapes are::

    <careers_host>/rssfeed/...
    https://<instance>/career?company=<ID>&career_ns=job_listing_summary\
        &navBarLevel=JOB_SEARCH&...&resultType=XML

returning an RSS 2.0 document whose ``<item>`` elements carry ``<title>``,
``<link>``, ``<description>``, a ``<guid>`` and Google-jobs ``<g:...>`` fields
such as ``<g:location>`` / ``<g:job_function>``.

This adapter is deliberately dual-mode and config-driven:

* if ``company.search['feed_url']`` is set, :meth:`fetch` GETs it and parses the
  RSS feed (the reliable path for JS-rendered sites); otherwise
* it GETs ``company.careers_url`` and best-effort scrapes job-posting anchors out
  of the static HTML.

:meth:`parse` auto-detects which of the two payload shapes it was handed, so the
same pure method serves both paths and both can be exercised from fixtures.

NOTE: For a JS-rendered SuccessFactors site with no configured ``feed_url`` the
static HTML will contain no postings and :meth:`fetch` returns ``[]`` (the
healthcheck then reports "empty"). Resolving this needs either the RSS
``feed_url`` or a future Playwright-backed fetch path.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urljoin, urlsplit

from selectolax.parser import HTMLParser

from job_monitor.models import RawJob, Source
from job_monitor.normalize import clean_text
from job_monitor.sources.base import BaseAdapter

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.http import PoliteClient

# Google-jobs namespace used for the <g:location> / <g:job_function> extensions
# in SuccessFactors RSS feeds.
_G_NAMESPACE = "http://base.google.com/ns/1.0"

# Substrings that mark an anchor as pointing at an individual job posting (as
# opposed to a browse-by-category / search / marketing link). SuccessFactors
# "Careers Centre" pages use ``/go/<Category>-Jobs/<id>/`` for *category* browse
# pages, which we deliberately do NOT treat as postings.
_JOB_HREF_MARKERS = ("/job/", "jobdetail", "/jobs/")

# Anchor text shorter than this is almost certainly navigation chrome, not a
# real posting title.
_MIN_TITLE_LEN = 3


class SuccessFactorsAdapter(BaseAdapter):
    """Adapter for SAP SuccessFactors Recruiting Marketing career sites."""

    name: ClassVar[str] = "successfactors"
    source: ClassVar[Source] = Source.OFFICIAL_ATS

    def __init__(
        self,
        *,
        http: PoliteClient,
        company: CompanyConfig | None,
        settings: Settings,
    ) -> None:
        super().__init__(http=http, company=company, settings=settings)

    # ------------------------------------------------------------------ #
    # Config helpers                                                     #
    # ------------------------------------------------------------------ #
    def _search(self) -> dict[str, Any]:
        return getattr(self.company, "search", None) or {}

    def _feed_url(self) -> str:
        return str(self._search().get("feed_url", "") or "")

    def _careers_url(self) -> str:
        # Prefer an explicit search override, else the company's careers_url.
        override = self._search().get("careers_url")
        if override:
            return str(override)
        return getattr(self.company, "careers_url", "") or ""

    def _company_name(self) -> str:
        return getattr(self.company, "name", "") or ""

    def _company_id(self) -> str | None:
        return getattr(self.company, "company_id", None)

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                     #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Turn an RSS-feed or HTML payload into ``RawJob`` records. PURE — no I/O.

        Auto-detects the payload shape: RSS/XML is routed to the feed parser and
        anything else is treated as HTML. Returns ``[]`` (never raises) when the
        payload yields nothing parseable, so a JS-rendered landing page degrades
        gracefully.
        """
        if not isinstance(payload, str):
            return []
        text = payload.strip()
        if not text:
            return []
        if self._looks_like_rss(text):
            return self._parse_rss(text)
        return self._parse_html(text, base_url=base_url)

    @staticmethod
    def _looks_like_rss(text: str) -> bool:
        head = text[:512].lstrip().lower()
        if head.startswith("<?xml"):
            return True
        return "<rss" in head or "<item>" in head or "<item " in head

    # -- RSS / XML -------------------------------------------------------- #
    def _parse_rss(self, text: str) -> list[RawJob]:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []

        company_name = self._company_name()
        company_id = self._company_id()

        jobs: list[RawJob] = []
        for item in root.iter("item"):
            title = self._item_text(item, "title")
            link = self._item_text(item, "link")
            if not title and not link:
                continue

            description_raw = self._item_text(item, "description")
            description = clean_text(description_raw) if description_raw else None

            location = self._g_text(item, "location") or self._location_from_title(title)
            guid = self._item_text(item, "guid")
            pub_date = self._item_text(item, "pubDate")

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=title,
                    company_name=company_name,
                    apply_url=link,
                    source_job_id=guid or None,
                    location=location or None,
                    description=description or None,
                    posted_date_raw=pub_date or None,
                    company_id=company_id,
                )
            )
        return jobs

    @staticmethod
    def _item_text(item: ET.Element, tag: str) -> str:
        child = item.find(tag)
        if child is None or child.text is None:
            return ""
        return child.text.strip()

    @staticmethod
    def _g_text(item: ET.Element, local_name: str) -> str:
        """Read a Google-jobs ``<g:...>`` extension element, namespace-tolerant."""
        # Try the proper namespaced lookup first, then fall back to scanning
        # children by local tag name (handles feeds that omit the declaration).
        child = item.find(f"{{{_G_NAMESPACE}}}{local_name}")
        if child is not None and child.text is not None:
            return child.text.strip()
        suffix = f"}}{local_name}"
        for el in item:
            tag = el.tag
            matches = tag == local_name or tag.endswith(suffix) or tag.endswith(f":{local_name}")
            if matches and el.text is not None:
                return el.text.strip()
        return ""

    @staticmethod
    def _location_from_title(title: str) -> str:
        """Best-effort: many SF titles read 'Job Title - Location'."""
        if " - " in title:
            tail = title.rsplit(" - ", 1)[-1].strip()
            # Only treat the tail as a location if it is short-ish (a city/region),
            # not another clause of the title.
            if 0 < len(tail) <= 40:
                return tail
        return ""

    # -- HTML ------------------------------------------------------------- #
    def _parse_html(self, html: str, *, base_url: str | None) -> list[RawJob]:
        tree = HTMLParser(html)
        company_name = self._company_name()
        company_id = self._company_id()
        resolve_base = base_url or self._careers_url()

        jobs: list[RawJob] = []
        seen: set[str] = set()
        for anchor in tree.css("a[href]"):
            href = anchor.attributes.get("href") or ""
            if not self._is_job_anchor(href):
                continue
            title = clean_text(anchor.text(separator=" ", strip=True))
            if len(title) < _MIN_TITLE_LEN:
                continue

            apply_url = urljoin(resolve_base, href) if resolve_base else href
            if apply_url in seen:
                continue
            seen.add(apply_url)

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=title,
                    company_name=company_name,
                    apply_url=apply_url,
                    source_job_id=None,
                    location=self._location_from_title(title) or None,
                    description=None,
                    company_id=company_id,
                )
            )
        return jobs

    @staticmethod
    def _is_job_anchor(href: str) -> bool:
        lowered = href.lower()
        if not lowered or lowered.startswith(("#", "javascript:", "mailto:")):
            return False
        return any(marker in lowered for marker in _JOB_HREF_MARKERS)

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch postings, preferring the RSS ``feed_url`` over HTML scraping.

        If ``company.search['feed_url']`` is configured we GET and parse that RSS
        feed (the reliable path). Otherwise we GET ``careers_url`` and best-effort
        scrape job-posting anchors out of the static HTML — which, for a
        JS-rendered SuccessFactors site, will typically yield ``[]``. Returning an
        empty list is intentional and acceptable: the healthcheck reports "empty",
        documenting that this site needs either an RSS ``feed_url`` or (future)
        Playwright rendering. ``SourceBlocked`` propagates from the HTTP client.

        ``search_terms`` is accepted for interface parity; SuccessFactors RSS
        feeds are pre-scoped per company, so it is not used to vary the request.
        """
        feed_url = self._feed_url()
        if feed_url:
            payload = self.http.get_text(feed_url)
            return self.parse(payload, base_url=self._feed_base(feed_url))

        careers_url = self._careers_url()
        if not careers_url:
            return []
        payload = self.http.get_text(careers_url)
        return self.parse(payload, base_url=careers_url)

    @staticmethod
    def _feed_base(feed_url: str) -> str:
        """Scheme+host of the feed, used to resolve relative ``<link>`` values."""
        parts = urlsplit(feed_url)
        if parts.scheme and parts.netloc:
            return f"{parts.scheme}://{parts.netloc}"
        return feed_url
