"""Phenom (Phenom People) ATS source adapter.

Phenom-hosted career sites expose two distinct JSON shapes depending on how the
tenant has configured its front-end. This adapter supports both, selected by
``company.search["mode"]``:

* ``"api_jobs"`` (verified for PepsiCo AU) — a simple ``GET {api_url}`` search
  endpoint returning ``{"jobs": [{"data": {...}}, ...], "totalCount": N}``. The
  page size is fixed server-side (10) and pagination uses a 1-based ``page``
  query parameter::

      GET https://www.pepsicojobs.com/api/jobs?location=Australia&keywords=quality&page=2

* ``"widgets"`` (verified for Mars AU; the Kerry tenant is global) — a
  ``POST {widgets_url}?ddoKey=refineSearch`` with a JSON body, returning
  ``{"refineSearch": {"totalHits": N, "data": {"jobs": [...]}}}``. Pagination
  uses ``from`` (offset) + ``size`` in the body.

As with every adapter, :meth:`PhenomAdapter.parse` is pure (no network) and
auto-detects which of the two shapes it has been handed, while
:meth:`PhenomAdapter.fetch` reads the configured mode, paginates, and dedupes by
apply URL. ``SourceBlocked`` propagates from the HTTP client.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlencode

from job_monitor.models import RawJob, Source
from job_monitor.normalize import clean_text
from job_monitor.sources.base import BaseAdapter

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.http import PoliteClient

# Defaults for the verified PepsiCo "api_jobs" tenant.
_DEFAULT_API_URL = "https://www.pepsicojobs.com/api/jobs"
_DEFAULT_CAREERS_BASE = "https://www.pepsicojobs.com"

# Stay a polite neighbour: cap pages pulled per term overall.
_DEFAULT_MAX_PAGES = 5
# "api_jobs" page size is fixed server-side at 10; used to step the offset
# fallback for "widgets" when not configured.
_API_JOBS_PAGE_SIZE = 10
_DEFAULT_WIDGETS_SIZE = 50

_JSON_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


class PhenomAdapter(BaseAdapter):
    """Adapter for Phenom-hosted career sites (api_jobs and widgets modes)."""

    name: ClassVar[str] = "phenom"
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

    def _mode(self) -> str:
        return str(self._search().get("mode", "api_jobs") or "api_jobs")

    def _company_name(self) -> str:
        return getattr(self.company, "name", "") or ""

    def _company_id(self) -> str | None:
        return getattr(self.company, "company_id", None)

    def _api_url(self) -> str:
        return str(self._search().get("api_url", _DEFAULT_API_URL) or _DEFAULT_API_URL)

    def _careers_base(self) -> str:
        base = str(self._search().get("careers_base", _DEFAULT_CAREERS_BASE) or _DEFAULT_CAREERS_BASE)
        return base.rstrip("/")

    def _widgets_url(self) -> str:
        return str(self._search().get("widgets_url", "") or "")

    def _max_pages(self) -> int:
        try:
            value = int(self._search().get("max_pages", _DEFAULT_MAX_PAGES))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_PAGES
        return max(1, value)

    def _au_filter(self) -> bool:
        return bool(self._search().get("au_filter", False))

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                     #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Turn a Phenom payload into ``RawJob`` records. PURE — no network.

        Auto-detects the response shape: a top-level ``refineSearch`` key means
        the "widgets" shape; otherwise a top-level ``jobs`` list is treated as
        the "api_jobs" shape. ``payload`` may be a decoded dict or a JSON string.
        Defensive throughout: malformed records are skipped.
        """
        data = self._coerce_payload(payload)
        if "refineSearch" in data:
            return self._parse_widgets(data)
        return self._parse_api_jobs(data, base_url=base_url)

    # -- api_jobs shape ------------------------------------------------- #
    def _parse_api_jobs(self, data: dict[str, Any], *, base_url: str | None) -> list[RawJob]:
        records = data.get("jobs")
        if not isinstance(records, list):
            return []
        careers_base = (base_url or self._careers_base()).rstrip("/")
        company_name = self._company_name()
        company_id = self._company_id()

        jobs: list[RawJob] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            job_data = record.get("data")
            if not isinstance(job_data, dict):
                continue

            slug = job_data.get("slug")
            req_id = job_data.get("req_id")
            source_job_id = self._stringify(req_id) or self._stringify(slug)

            apply_url = self._stringify(job_data.get("apply_url"))
            if not apply_url and slug:
                apply_url = f"{careers_base}/job/{slug}"
            if not apply_url:
                continue

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=str(job_data.get("title") or ""),
                    company_name=company_name,
                    apply_url=apply_url,
                    source_job_id=source_job_id,
                    location=self._api_location(job_data),
                    description=None,
                    posted_date_raw=self._stringify(job_data.get("posted_date")),
                    company_id=company_id,
                    extra={"slug": self._stringify(slug)},
                )
            )
        return jobs

    @staticmethod
    def _api_location(job_data: dict[str, Any]) -> str | None:
        """Prefer ``full_location``, else join city / state / country."""
        full = job_data.get("full_location")
        if isinstance(full, str) and full.strip():
            return full.strip()
        parts = [
            str(job_data.get(key)).strip()
            for key in ("city", "state", "country")
            if job_data.get(key)
        ]
        joined = ", ".join(p for p in parts if p)
        return joined or None

    # -- widgets shape -------------------------------------------------- #
    def _parse_widgets(self, data: dict[str, Any]) -> list[RawJob]:
        refine = data.get("refineSearch")
        if not isinstance(refine, dict):
            return []
        inner = refine.get("data")
        records = inner.get("jobs") if isinstance(inner, dict) else None
        if not isinstance(records, list):
            return []
        company_name = self._company_name()
        company_id = self._company_id()
        au_only = self._au_filter()

        jobs: list[RawJob] = []
        for record in records:
            if not isinstance(record, dict):
                continue

            apply_url = (
                self._stringify(record.get("applyUrl"))
                or self._stringify(record.get("jobUrl"))
                or self._stringify(record.get("reqId"))
            )
            if not apply_url:
                continue

            location = self._widget_location(record)
            if au_only and not self._is_australian(record, location):
                continue

            source_job_id = self._stringify(record.get("jobId")) or self._stringify(
                record.get("jobSeqNo")
            )
            teaser = record.get("descriptionTeaser")
            description = clean_text(str(teaser)) or None if teaser else None

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=str(record.get("title") or ""),
                    company_name=company_name,
                    apply_url=apply_url,
                    source_job_id=source_job_id,
                    location=location,
                    description=description,
                    posted_date_raw=self._stringify(record.get("postedDate")),
                    company_id=company_id,
                    extra={"job_id": self._stringify(record.get("jobId"))},
                )
            )
        return jobs

    @staticmethod
    def _widget_location(record: dict[str, Any]) -> str | None:
        """Prefer ``cityState``, else join city / state / country."""
        city_state = record.get("cityState")
        if isinstance(city_state, str) and city_state.strip():
            return city_state.strip()
        parts = [
            str(record.get(key)).strip()
            for key in ("city", "state", "country")
            if record.get(key)
        ]
        joined = ", ".join(p for p in parts if p)
        return joined or None

    @staticmethod
    def _is_australian(record: dict[str, Any], location: str | None) -> bool:
        """Heuristic AU check across country / location free text."""
        haystacks = [
            record.get("country"),
            record.get("cityStateCountry"),
            location,
        ]
        for value in haystacks:
            if not isinstance(value, str):
                continue
            lowered = value.lower()
            if "australia" in lowered or lowered.strip() == "au":
                return True
        return False

    # -- shared helpers ------------------------------------------------- #
    @staticmethod
    def _coerce_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    @staticmethod
    def _stringify(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch postings for ``search_terms``, paginating per the active mode.

        Dedupes by ``apply_url`` across all terms and pages. ``SourceBlocked``
        propagates from the HTTP client.
        """
        terms = search_terms or self.search_terms_from_company()
        by_url: dict[str, RawJob] = {}

        for term in terms:
            for job in self._fetch_term(term):
                if job.apply_url in by_url:
                    continue
                by_url[job.apply_url] = job
        return list(by_url.values())

    def _fetch_term(self, term: str) -> list[RawJob]:
        if self._mode() == "widgets":
            return self._fetch_widgets_term(term)
        return self._fetch_api_jobs_term(term)

    # -- api_jobs fetching ---------------------------------------------- #
    def _fetch_api_jobs_term(self, term: str) -> list[RawJob]:
        """Page through the api_jobs endpoint (1-based ``page``) for one term."""
        search = self._search()
        api_url = self._api_url()
        location = self._stringify(search.get("location")) or ""
        configured_keywords = self._stringify(search.get("keywords"))
        keywords = term or configured_keywords or ""
        try:
            num = int(search.get("num", _API_JOBS_PAGE_SIZE))
        except (TypeError, ValueError):
            num = _API_JOBS_PAGE_SIZE
        page_size = max(1, num)

        collected: list[RawJob] = []
        total: int | None = None
        seen = 0
        for page in range(1, self._max_pages() + 1):
            params: dict[str, Any] = {"num": page_size, "page": page}
            if keywords:
                params["keywords"] = keywords
            if location:
                params["location"] = location
            url = f"{api_url}?{urlencode(params)}"

            payload = self.http.get_json(url, headers=_JSON_HEADERS)
            page_jobs = self.parse(payload)
            if not page_jobs:
                break
            collected.extend(page_jobs)
            seen += len(page_jobs)

            if isinstance(payload, dict):
                raw_total = payload.get("totalCount")
                if isinstance(raw_total, int):
                    total = raw_total
            if total is not None and seen >= total:
                break
        return collected

    # -- widgets fetching ----------------------------------------------- #
    def _fetch_widgets_term(self, term: str) -> list[RawJob]:
        """Page through the widgets endpoint (``from`` + ``size``) for one term."""
        widgets_url = self._widgets_url()
        if not widgets_url:
            return []
        search = self._search()
        try:
            size = int(search.get("size", _DEFAULT_WIDGETS_SIZE))
        except (TypeError, ValueError):
            size = _DEFAULT_WIDGETS_SIZE
        size = max(1, size)
        url = f"{widgets_url}?ddoKey=refineSearch"

        collected: list[RawJob] = []
        total: int | None = None
        offset = 0
        for _page in range(self._max_pages()):
            body = self._widgets_body(term, offset=offset, size=size)
            payload = self.http.get_json(
                url,
                method="POST",
                json_body=body,
                headers=_JSON_HEADERS,
            )
            page_jobs = self.parse(payload)
            page_hits = self._widgets_page_hits(payload)
            collected.extend(page_jobs)

            if isinstance(payload, dict):
                refine = payload.get("refineSearch")
                if isinstance(refine, dict) and isinstance(refine.get("totalHits"), int):
                    total = refine["totalHits"]

            # Stop when the server returned fewer than a full page, or we have
            # consumed every hit. ``page_jobs`` may be smaller than ``page_hits``
            # when the client-side AU filter drops rows, so paginate on hits.
            if page_hits == 0:
                break
            offset += size
            if total is not None and offset >= total:
                break
        return collected

    def _widgets_body(self, term: str, *, offset: int, size: int) -> dict[str, Any]:
        """Build the minimal-but-sufficient widgets POST body."""
        search = self._search()
        lang = self._stringify(search.get("lang")) or "en_us"
        country = self._stringify(search.get("country")) or ""
        configured_keywords = self._stringify(search.get("keywords"))
        keywords = term or configured_keywords or ""
        return {
            "lang": lang,
            "deviceType": "desktop",
            "country": country,
            "pageName": "search-results",
            "ddoKey": "refineSearch",
            "stringTypeParameters": [],
            "cTypeOfWdgt": "JobSearch",
            "pageType": "searchResults",
            "siteType": "external",
            "keywords": keywords,
            "globalSearch": True,
            "size": size,
            "from": offset,
            "jdSource": "facets",
            "all_fields": [],
            "jobs": True,
            "counts": False,
            "sortBy": "Most relevant",
        }

    @staticmethod
    def _widgets_page_hits(payload: Any) -> int:
        """Return how many raw job rows the widgets payload carried (pre-filter)."""
        if not isinstance(payload, dict):
            return 0
        refine = payload.get("refineSearch")
        if not isinstance(refine, dict):
            return 0
        inner = refine.get("data")
        rows = inner.get("jobs") if isinstance(inner, dict) else None
        return len(rows) if isinstance(rows, list) else 0


__all__ = ["PhenomAdapter"]
