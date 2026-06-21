"""Workday CXS source adapter.

Workday-hosted careers sites expose a JSON "CXS" API behind the React front-end.
Two endpoints matter:

* the *list* endpoint — ``POST {base_url}/wday/cxs/{tenant}/{site}/jobs`` with a
  small JSON body — returns a page of postings (title, ``externalPath``,
  ``locationsText``, ``postedOn``, ``bulletFields``); and
* the *detail* endpoint — ``GET {base_url}/wday/cxs/{tenant}/{site}/job{path}`` —
  returns ``jobPostingInfo`` with the full ``jobDescription`` HTML.

The human-facing apply URL for a posting is
``{base_url}/en-US/{site}{externalPath}``.

As with every adapter, :meth:`WorkdayAdapter.parse` is pure (no network) so it
can be exercised against a committed fixture, while :meth:`WorkdayAdapter.fetch`
does the paginated list calls plus the per-posting detail enrichment.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from job_monitor.models import RawJob, Source, SourceBlocked
from job_monitor.normalize import clean_text
from job_monitor.sources.base import BaseAdapter

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.http import PoliteClient

# Stay a polite neighbour: cap how much we pull per term and overall.
_PAGE_LIMIT = 20
_MAX_PAGES_PER_TERM = 3
_MAX_DETAIL_FETCHES = 40

_JSON_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


class WorkdayAdapter(BaseAdapter):
    """Adapter for Workday CXS-hosted careers sites."""

    name: ClassVar[str] = "workday"
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

    def _base_url(self, override: str | None = None) -> str:
        base = override or str(self._search().get("base_url", ""))
        return base.rstrip("/")

    def _tenant(self) -> str:
        return str(self._search().get("cxs_tenant", ""))

    def _site(self) -> str:
        return str(self._search().get("cxs_site", ""))

    def _company_name(self) -> str:
        return getattr(self.company, "name", "") or ""

    def _company_id(self) -> str | None:
        return getattr(self.company, "company_id", None)

    def _list_url(self, base_url: str) -> str:
        return f"{base_url}/wday/cxs/{self._tenant()}/{self._site()}/jobs"

    def _detail_url(self, base_url: str, external_path: str) -> str:
        return f"{base_url}/wday/cxs/{self._tenant()}/{self._site()}/job{external_path}"

    def _apply_url(self, base_url: str, external_path: str) -> str:
        return f"{base_url}/en-US/{self._site()}{external_path}"

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                     #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Turn a list-endpoint payload into ``RawJob`` records. PURE — no network.

        ``payload`` may be the already-decoded dict or a JSON string. Descriptions
        are left ``None`` here; :meth:`fetch` fills them from the detail endpoint.
        """
        data = self._coerce_payload(payload)
        postings = data.get("jobPostings") or []
        base = self._base_url(base_url)
        company_name = self._company_name()
        company_id = self._company_id()

        jobs: list[RawJob] = []
        for posting in postings:
            if not isinstance(posting, dict):
                continue
            external_path = posting.get("externalPath")
            if not external_path:
                continue
            external_path = str(external_path)

            bullet_fields = posting.get("bulletFields") or []
            source_job_id = str(bullet_fields[0]) if bullet_fields else external_path

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=str(posting.get("title") or ""),
                    company_name=company_name,
                    apply_url=self._apply_url(base, external_path),
                    source_job_id=source_job_id,
                    location=posting.get("locationsText"),
                    description=None,
                    posted_date_raw=posting.get("postedOn"),
                    company_id=company_id,
                    extra={"external_path": external_path},
                )
            )
        return jobs

    @staticmethod
    def _coerce_payload(payload: Any) -> dict[str, Any]:
        decoded = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(decoded, dict):
            return {}
        return decoded

    @staticmethod
    def description_from_detail(detail: Any) -> str | None:
        """Extract a cleaned plain-text description from a detail payload.

        Returns ``None`` when no usable ``jobDescription`` is present so callers
        can leave :attr:`RawJob.description` unset rather than blank.
        """
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except ValueError:
                return None
        if not isinstance(detail, dict):
            return None
        info = detail.get("jobPostingInfo")
        if not isinstance(info, dict):
            return None
        raw_html = info.get("jobDescription")
        if not raw_html:
            return None
        cleaned = clean_text(str(raw_html))
        return cleaned or None

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch postings for ``search_terms`` and enrich with descriptions.

        Paginates the list endpoint per term (capped), dedupes by
        ``externalPath`` across all terms, then fetches the detail endpoint for
        each unique posting (capped) to populate the description. A failed detail
        fetch leaves the posting with ``description=None`` rather than dropping it.
        ``SourceBlocked`` propagates from the HTTP client.
        """
        base_url = self._base_url()
        terms = search_terms or self.search_terms_from_company()

        by_path: dict[str, RawJob] = {}
        for term in terms:
            for posting in self._fetch_term(base_url, term):
                external_path = posting.extra.get("external_path")
                if not external_path or external_path in by_path:
                    continue
                by_path[external_path] = posting

        jobs = list(by_path.values())
        self._enrich_descriptions(base_url, jobs)
        return jobs

    def _fetch_term(self, base_url: str, term: str) -> list[RawJob]:
        """Page through the list endpoint for one search term."""
        url = self._list_url(base_url)
        collected: list[RawJob] = []
        offset = 0
        for _page in range(_MAX_PAGES_PER_TERM):
            body = {
                "appliedFacets": {},
                "limit": _PAGE_LIMIT,
                "offset": offset,
                "searchText": term,
            }
            payload = self.http.get_json(
                url,
                method="POST",
                json_body=body,
                headers=_JSON_HEADERS,
            )
            page_jobs = self.parse(payload, base_url=base_url)
            if not page_jobs:
                break
            collected.extend(page_jobs)

            total = payload.get("total") if isinstance(payload, dict) else None
            offset += _PAGE_LIMIT
            if isinstance(total, int) and offset >= total:
                break
        return collected

    def _enrich_descriptions(self, base_url: str, jobs: list[RawJob]) -> None:
        """Populate ``description`` from the detail endpoint, resiliently."""
        for job in jobs[:_MAX_DETAIL_FETCHES]:
            external_path = job.extra.get("external_path")
            if not external_path:
                continue
            try:
                detail = self.http.get_json(self._detail_url(base_url, external_path))
            except SourceBlocked:
                # A hard block applies to the whole source: let it propagate.
                raise
            except Exception:
                # Any other detail-fetch failure is non-fatal: keep the posting
                # with description left as None.
                continue
            job.description = self.description_from_detail(detail)
