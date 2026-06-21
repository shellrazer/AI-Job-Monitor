"""Jora source adapter.

Jora is a Cloudflare-fronted aggregator job board. The search results page
(``{base_url}/j?q=...&l=...``) renders job cards server-side, so there is no
JSON / ``ld+json`` payload to mine — the postings live in the card markup.

Each card is a ``div`` carrying the classes ``job-card result organic-job`` and
contains:

* a title anchor ``a.job-link`` (there are two such anchors per card — a desktop
  and a mobile variant — pointing at the same ``/job/<slug>-<32-hex-id>`` path
  with differing tracking query params);
* ``.job-company`` / ``.job-location`` / ``.job-listed-date`` metadata;
* a ``.job-abstract`` snippet.

As with every adapter, :meth:`JoraAdapter.parse` is pure (no network) so it can
be exercised against a committed fixture, while :meth:`JoraAdapter.fetch` does
the single Cloudflare-impersonated GET and delegates to :meth:`parse`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import quote, urlsplit, urlunsplit

from job_monitor.models import RawJob, Source, SourceBlocked
from job_monitor.normalize import clean_text
from job_monitor.sources.base import BaseAdapter

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.http import PoliteClient

_DEFAULT_BASE_URL = "https://au.jora.com"

# Default number of search-results pages to walk when the company config does
# not specify ``max_pages``.
_DEFAULT_MAX_PAGES = 3

# The 32-char hex id trailing a job path, e.g. ``/job/Quality-Manager-<hash>``.
# Anchored at the end of the *path* (query already stripped before matching).
_JOB_ID_RE = re.compile(r"-([0-9a-f]{32})$", re.IGNORECASE)

# Card containers, in preference order. Jora tags the same node with all three
# classes; ``job-card`` is the most stable / inclusive (it also covers sponsored
# cards that lack ``organic-job``), with ``result`` as a fallback.
_CARD_SELECTORS = ("div.job-card", "div.result")

# Field selectors, tried in order; the first match wins.
_COMPANY_SELECTORS = (".job-company", "span.company", ".company")
_LOCATION_SELECTORS = (".job-location", ".location")
_SALARY_SELECTORS = (".job-salary", ".salary")
_SNIPPET_SELECTORS = (".job-abstract", ".summary")
_DATE_SELECTORS = (".job-listed-date", ".date")


class JoraAdapter(BaseAdapter):
    """Adapter for the Jora aggregator job board (Cloudflare-fronted)."""

    name: ClassVar[str] = "jora"
    source: ClassVar[Source] = Source.JORA

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

    def _company_id(self) -> str | None:
        return getattr(self.company, "company_id", None)

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                     #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = _DEFAULT_BASE_URL) -> list[RawJob]:
        """Parse a Jora search-results HTML page into ``RawJob`` records.

        PURE — no network. Defensive throughout: a missing field never raises,
        and cards without a usable title or href are skipped. Each card carries
        two anchors to the same posting (differing only in tracking query
        params), so we strip the query string and de-duplicate by the resulting
        absolute ``apply_url``.
        """
        # Local import keeps module import light and mirrors normalize.py's use.
        from selectolax.parser import HTMLParser

        html = payload if isinstance(payload, str) else (payload or b"").decode("utf-8", "replace")
        tree = HTMLParser(html)
        base = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        company_id = self._company_id()

        cards = self._select_cards(tree)

        by_url: dict[str, RawJob] = {}
        for card in cards:
            job = self._parse_card(card, base=base, company_id=company_id)
            if job is None:
                continue
            # De-dup within the page by apply_url (query already stripped).
            by_url.setdefault(job.apply_url, job)
        return list(by_url.values())

    @staticmethod
    def _select_cards(tree: Any) -> list[Any]:
        """Return the job-card nodes using the first selector that matches."""
        for selector in _CARD_SELECTORS:
            nodes = tree.css(selector)
            if nodes:
                return nodes
        return []

    def _parse_card(self, card: Any, *, base: str, company_id: str | None) -> RawJob | None:
        """Extract one ``RawJob`` from a card node, or ``None`` if unusable."""
        anchor = self._title_anchor(card)
        if anchor is None:
            return None

        title = clean_text(anchor.text(strip=True))
        href = (anchor.attributes.get("href") or "").strip()
        if not title or not href:
            return None

        apply_url = self._absolute_no_query(href, base)
        if not apply_url:
            return None

        source_job_id = self._job_id_from_url(apply_url)
        company = self._first_text(card, _COMPANY_SELECTORS)
        location = self._first_text(card, _LOCATION_SELECTORS)
        salary = self._first_text(card, _SALARY_SELECTORS)
        snippet = self._first_text(card, _SNIPPET_SELECTORS)
        posted = self._first_text(card, _DATE_SELECTORS)

        return RawJob(
            source=Source.JORA,
            title=title,
            company_name=company or "",
            apply_url=apply_url,
            source_job_id=source_job_id,
            location=location,
            description=snippet,
            posted_date_raw=posted,
            salary_raw=salary,
            company_id=company_id,
        )

    # ------------------------------------------------------------------ #
    # Extraction helpers                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _title_anchor(card: Any) -> Any:
        """Return the first ``a.job-link`` that has text, else any ``a.job-link``."""
        links = card.css("a.job-link")
        for link in links:
            if link.text(strip=True):
                return link
        return links[0] if links else None

    @staticmethod
    def _first_text(card: Any, selectors: tuple[str, ...]) -> str | None:
        """Return cleaned text from the first matching selector, else ``None``."""
        for selector in selectors:
            node = card.css_first(selector)
            if node is None:
                continue
            text = clean_text(node.text(strip=True))
            if text:
                return text
        return None

    @staticmethod
    def _absolute_no_query(href: str, base: str) -> str:
        """Build an absolute URL from ``href`` and strip its query/fragment.

        Stripping the query collapses the two per-card anchors (which differ only
        in tracking params such as ``fsv=true``) onto a single canonical URL.
        """
        if href.startswith(("http://", "https://")):
            absolute = href
        elif href.startswith("/"):
            absolute = base + href
        else:
            absolute = f"{base}/{href}"
        split = urlsplit(absolute)
        return urlunsplit((split.scheme, split.netloc, split.path, "", ""))

    @staticmethod
    def _job_id_from_url(url: str) -> str | None:
        """Parse the trailing 32-hex job id from a ``/job/<slug>-<hash>`` path."""
        path = urlsplit(url).path
        match = _JOB_ID_RE.search(path)
        return match.group(1) if match else None

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch the Jora search-results pages and parse them.

        Builds the URL from the company's ``search`` config: ``q`` (query) and
        ``l`` (location), with ``search_terms`` overriding ``q`` when provided.
        Walks pages ``1..max_pages`` via the ``&p={n}`` (1-indexed) pagination
        param — ``max_pages`` read from ``search`` (default
        :data:`_DEFAULT_MAX_PAGES`). Each page is fetched via curl_cffi browser
        impersonation (Cloudflare); results accumulate and are de-duplicated by
        ``apply_url``.

        Stops early when a page yields zero jobs or contributes no new
        ``apply_url`` (end of results). A :class:`~job_monitor.models.SourceBlocked`
        from page 1 propagates; if a later page blocks, we stop and return what
        we have so far.
        """
        search = self._search()
        query = search_terms[0] if search_terms else str(search.get("q", ""))
        location = str(search.get("l", ""))
        max_pages = self._max_pages(search)

        by_url: dict[str, RawJob] = {}
        for page in range(1, max_pages + 1):
            url = f"{_DEFAULT_BASE_URL}/j?q={quote(query)}&l={quote(location)}&p={page}"
            try:
                html = self.http.get_text_impersonate(url)
            except SourceBlocked:
                if page == 1:
                    raise
                # A later page blocked: keep what we already gathered.
                break

            jobs = self.parse(html, base_url=_DEFAULT_BASE_URL)
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
            value = int(search.get("max_pages", _DEFAULT_MAX_PAGES))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_PAGES
        return value if value >= 1 else _DEFAULT_MAX_PAGES
