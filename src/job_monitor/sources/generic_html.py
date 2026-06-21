"""Config-driven generic HTML source adapter.

Many smaller careers sites render their job listings as plain server-side HTML
with no usable JSON API. Rather than write a bespoke adapter per site, this one
is driven entirely by a ``selectors`` mapping in the company config, so a new
site can be onboarded by editing YAML.

The config (under ``CompanyConfig.search``) looks like::

    list_url: "https://careers.acme.com/jobs"   # page to fetch
    base_url: "https://careers.acme.com"          # resolves relative links
    selectors:
      item: "div.job-card"      # one element per posting (required)
      title: "h3"               # required per item
      link: "a@href"            # required per item — see selector syntax below
      location: "span.loc"      # optional
      company: ".co"            # optional — falls back to the company name
      salary: ".sal"            # optional
      date: ".date"             # optional
      snippet: ".desc"          # optional

Selector syntax
---------------
Each selector value is a CSS selector, optionally suffixed with ``@attr``:

* ``"a.title"``        -> select ``a.title``, take its **text**.
* ``"a.title@href"``   -> select ``a.title``, take its ``href`` **attribute**.

All per-item selectors are evaluated **relative to** the matched ``item``
element. ``title`` and ``link`` are required for a posting to be emitted;
every other field is optional and silently skipped when absent.

As with every adapter, :meth:`GenericHtmlAdapter.parse` is pure (no network) so
it can be exercised against an HTML fixture, while
:meth:`GenericHtmlAdapter.fetch` does the single list-page GET.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urljoin

from selectolax.parser import HTMLParser, Node

from job_monitor.models import RawJob, Source
from job_monitor.normalize import clean_text
from job_monitor.sources.base import BaseAdapter

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.http import PoliteClient


class GenericHtmlAdapter(BaseAdapter):
    """Adapter for plain server-rendered HTML careers pages, driven by config."""

    name: ClassVar[str] = "generic_html"
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

    def _selectors(self) -> dict[str, str]:
        raw = self._search().get("selectors") or {}
        if not isinstance(raw, dict):
            return {}
        # Coerce to a clean str->str mapping; ignore non-string entries.
        return {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v}

    def _company_name(self) -> str:
        return getattr(self.company, "name", "") or ""

    def _company_id(self) -> str | None:
        return getattr(self.company, "company_id", None)

    def _base_url(self, override: str | None = None) -> str:
        """The base URL used to resolve relative links.

        Priority: explicit ``override`` arg -> ``search["base_url"]`` ->
        the company's ``careers_url``. Returns ``""`` when none are configured.
        """
        if override:
            return override
        search = self._search()
        base = search.get("base_url")
        if base:
            return str(base)
        careers = getattr(self.company, "careers_url", "") or ""
        return str(careers)

    # ------------------------------------------------------------------ #
    # Selector helper                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select(node: Node, spec: str | None) -> str | None:
        """Apply a ``"css"`` or ``"css@attr"`` selector relative to ``node``.

        Returns the matched element's stripped text (no ``@attr``) or the named
        attribute's value (with ``@attr``). Returns ``None`` when ``spec`` is
        empty, the element is not found, the attribute is missing, or the result
        is blank. Fully defensive — never raises on malformed input.
        """
        if not spec:
            return None
        css, sep, attr = spec.partition("@")
        css = css.strip()
        if not css:
            return None
        try:
            found = node.css_first(css)
        except Exception:
            # A malformed CSS selector should never crash a whole run.
            return None
        if found is None:
            return None

        if sep:  # an "@attr" suffix was present
            attr = attr.strip()
            if not attr:
                return None
            value = found.attributes.get(attr)
            if value is None:
                return None
            value = value.strip()
            return value or None

        text = (found.text() or "").strip()
        return text or None

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                     #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Turn a list-page HTML string into ``RawJob`` records. PURE — no network.

        ``payload`` is the page HTML (``str`` or ``bytes``). Items without a
        title or a resolvable link are skipped. Never raises on missing
        selectors or attributes; a site whose HTML matches no items simply
        yields ``[]``.
        """
        html = self._coerce_html(payload)
        if not html:
            return []

        selectors = self._selectors()
        item_sel = selectors.get("item")
        if not item_sel:
            return []

        try:
            tree = HTMLParser(html)
            items = tree.css(item_sel)
        except Exception:
            return []

        base = self._base_url(base_url)
        company_name = self._company_name()
        company_id = self._company_id()

        jobs: list[RawJob] = []
        for item in items:
            title = self._select(item, selectors.get("title"))
            if not title:
                continue  # title is required

            link = self._select(item, selectors.get("link"))
            if not link:
                continue  # link is required

            apply_url = urljoin(base, link) if base else link

            company = self._select(item, selectors.get("company")) or company_name

            snippet = self._select(item, selectors.get("snippet"))
            description = clean_text(snippet) or None if snippet else None

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=title,
                    company_name=company,
                    apply_url=apply_url,
                    source_job_id=None,
                    location=self._select(item, selectors.get("location")),
                    description=description,
                    posted_date_raw=self._select(item, selectors.get("date")),
                    salary_raw=self._select(item, selectors.get("salary")),
                    company_id=company_id,
                )
            )
        return jobs

    @staticmethod
    def _coerce_html(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return str(payload)

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch and parse the configured ``list_url``.

        The list URL is pre-built in config, so ``search_terms`` are not used to
        construct it (the page already encodes the desired query). When
        ``search["impersonate"]`` is truthy the request goes through curl_cffi
        browser impersonation, which some Cloudflare-fronted sites require.
        ``SourceBlocked`` from the HTTP client propagates to the caller.

        TODO: JS-rendered sites (postings injected client-side) return no
        matching items from this static fetch and therefore yield ``[]``. Such
        sites would need an optional Playwright-backed transport to render the
        page before parsing; that is intentionally not implemented here and is
        off by default.
        """
        search = self._search()
        list_url = search.get("list_url")
        if not list_url:
            return []
        list_url = str(list_url)

        html = (
            self.http.get_text_impersonate(list_url)
            if search.get("impersonate")
            else self.http.get_text(list_url)
        )
        return self.parse(html)
