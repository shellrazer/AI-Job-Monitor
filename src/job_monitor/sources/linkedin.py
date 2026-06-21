"""LinkedIn source adapter (semi-public guest jobs endpoint).

LinkedIn does not expose an official jobs API for unauthenticated use, but the
public job-search UI is backed by a "guest" endpoint that returns an HTML
fragment of job cards without requiring a login::

    GET https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search
        ?keywords=<kw>&location=<loc>&start=<n>

LinkedIn aggressively rate-limits / blocks heavy scraping, so the fetch path
goes through curl_cffi browser impersonation (which surfaces a persistent 403
as :class:`~job_monitor.models.SourceBlocked`) and leans on the shared
:class:`~job_monitor.sources.http.PoliteClient` rate limiter. We page politely
in increments of 25 (LinkedIn's page size) up to ``max_pages``.

Each returned ``<li>`` wraps a ``div.base-card`` (also tagged
``base-search-card`` / ``job-search-card``) carrying:

* title  -> ``h3.base-search-card__title`` (text)
* company -> ``h4.base-search-card__subtitle`` (often an ``a.hidden-nested-link``)
* location -> ``span.job-search-card__location``
* link -> ``a.base-card__full-link`` ``href`` (query stripped) -> ``apply_url``
* posted -> ``time`` element ``datetime`` attribute
* job id -> the card ``data-entity-urn`` (``urn:li:jobPosting:<id>``), falling
  back to the ``/jobs/view/<slug>-<id>`` segment of the href.

As with every adapter, :meth:`LinkedInAdapter.parse` is pure (no network) so it
can be exercised against a committed fixture, while :meth:`LinkedInAdapter.fetch`
performs the impersonated GET(s) and delegates to :meth:`parse`.
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

_DEFAULT_BASE_URL = "https://www.linkedin.com"
_GUEST_SEARCH_URL = (
    "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
)

# LinkedIn pages by 25 results; default to 3 pages (75 postings) unless overridden.
_PAGE_SIZE = 25
_DEFAULT_MAX_PAGES = 3

# Card containers, in preference order. LinkedIn tags the same node with all
# three classes; ``base-card`` is the most inclusive, with ``base-search-card``
# as a fallback in case the primary class is dropped.
_CARD_SELECTORS = ("div.base-card", "div.base-search-card")

# Field selectors, tried in order; the first non-empty match wins.
_TITLE_SELECTORS = ("h3.base-search-card__title",)
_COMPANY_SELECTORS = (
    "h4.base-search-card__subtitle a.hidden-nested-link",
    "h4.base-search-card__subtitle",
    "a.hidden-nested-link",
)
_LOCATION_SELECTORS = ("span.job-search-card__location", ".job-search-card__location")
_LINK_SELECTORS = ("a.base-card__full-link", "a.base-search-card--link")

# Job id from a ``urn:li:jobPosting:<id>`` entity urn.
_URN_ID_RE = re.compile(r"urn:li:jobPosting:(\d+)", re.IGNORECASE)
# Job id from a ``/jobs/view/<slug>-<id>`` (or ``/jobs/view/<id>``) path.
_VIEW_ID_RE = re.compile(r"/jobs/view/(?:[^/?#]*?-)?(\d+)", re.IGNORECASE)


class LinkedInAdapter(BaseAdapter):
    """Adapter for the LinkedIn semi-public guest jobs endpoint."""

    name: ClassVar[str] = "linkedin"
    source: ClassVar[Source] = Source.LINKEDIN

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
        """Parse a LinkedIn guest jobs HTML fragment into ``RawJob`` records.

        PURE — no network. Defensive throughout: a missing field never raises,
        and cards without a usable title or link are skipped. The per-card link
        carries tracking query params, so we strip the query string and
        de-duplicate by the resulting absolute ``apply_url``.
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
        title = self._first_text(card, _TITLE_SELECTORS)
        if not title:
            return None

        link = self._first_node(card, _LINK_SELECTORS)
        href = (link.attributes.get("href") or "").strip() if link is not None else ""
        apply_url = self._absolute_no_query(href, base)
        if not apply_url:
            return None

        source_job_id = self._job_id(card, apply_url)
        company = self._first_text(card, _COMPANY_SELECTORS)
        location = self._first_text(card, _LOCATION_SELECTORS)
        posted = self._posted_datetime(card)

        return RawJob(
            source=Source.LINKEDIN,
            title=title,
            company_name=company or "",
            apply_url=apply_url,
            source_job_id=source_job_id,
            location=location,
            description=None,
            posted_date_raw=posted,
            salary_raw=None,
            company_id=company_id,
        )

    # ------------------------------------------------------------------ #
    # Extraction helpers                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_node(card: Any, selectors: tuple[str, ...]) -> Any:
        """Return the first node matching any selector, else ``None``."""
        for selector in selectors:
            node = card.css_first(selector)
            if node is not None:
                return node
        return None

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
    def _posted_datetime(card: Any) -> str | None:
        """Return the ``datetime`` attribute (or text) of the card ``time`` node."""
        node = card.css_first("time")
        if node is None:
            return None
        dt = (node.attributes.get("datetime") or "").strip()
        if dt:
            return dt
        text = clean_text(node.text(strip=True))
        return text or None

    @staticmethod
    def _absolute_no_query(href: str, base: str) -> str:
        """Build an absolute URL from ``href`` and strip its query/fragment.

        Stripping the query collapses the tracking params (``position``,
        ``trackingId``, ...) so two cards pointing at the same posting share one
        canonical ``apply_url``.
        """
        if not href:
            return ""
        if href.startswith(("http://", "https://")):
            absolute = href
        elif href.startswith("/"):
            absolute = base + href
        else:
            absolute = f"{base}/{href}"
        split = urlsplit(absolute)
        if not split.netloc:
            return ""
        return urlunsplit((split.scheme, split.netloc, split.path, "", ""))

    @classmethod
    def _job_id(cls, card: Any, apply_url: str) -> str | None:
        """Parse the numeric job id from the entity urn, falling back to the URL."""
        urn = (card.attributes.get("data-entity-urn") or "").strip()
        if urn:
            match = _URN_ID_RE.search(urn)
            if match:
                return match.group(1)
        match = _VIEW_ID_RE.search(urlsplit(apply_url).path)
        return match.group(1) if match else None

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch LinkedIn guest job postings, paging politely by 25.

        Reads ``company.search`` keys ``keywords`` (overridden by
        ``search_terms`` when provided), ``location``, and ``max_pages``
        (default ``3``). Each page is fetched via curl_cffi browser
        impersonation. Pagination stops when a page returns no cards, ``max_pages``
        is reached, or a later page is blocked. A :class:`SourceBlocked` from the
        *first* page propagates; if a later page blocks we stop gracefully and
        return what we have. De-dups across pages by ``apply_url``.
        """
        search = self._search()
        keywords = search_terms[0] if search_terms else str(search.get("keywords", ""))
        location = str(search.get("location", ""))
        max_pages = self._max_pages(search.get("max_pages"))

        by_url: dict[str, RawJob] = {}
        for page in range(max_pages):
            start = page * _PAGE_SIZE
            url = (
                f"{_GUEST_SEARCH_URL}?keywords={quote(keywords)}"
                f"&location={quote(location)}&start={start}"
            )
            try:
                html = self.http.get_text_impersonate(url)
            except SourceBlocked:
                if page == 0:
                    raise
                break  # later page blocked -> return what we have

            page_jobs = self.parse(html, base_url=_DEFAULT_BASE_URL)
            if not page_jobs:
                break  # no more results

            for job in page_jobs:
                by_url.setdefault(job.apply_url, job)

        return list(by_url.values())

    @staticmethod
    def _max_pages(value: Any) -> int:
        """Coerce the configured ``max_pages`` to a sane positive int."""
        try:
            pages = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_MAX_PAGES
        return pages if pages > 0 else _DEFAULT_MAX_PAGES
