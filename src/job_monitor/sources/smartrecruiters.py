"""SmartRecruiters Posting API source adapter.

SmartRecruiters exposes a public, unauthenticated JSON API for a company's
public job postings::

    GET https://api.smartrecruiters.com/v1/companies/{company_slug}/postings
        ?limit=100&offset={n}[&q={term}]

The *list* response is paginated by ``offset`` and carries ``totalFound`` plus a
``content`` array of summaries::

    {"totalFound": N,
     "content": [{"id", "name", "ref", "releasedDate",
                  "location": {"city", "region", "country"},
                  "company": {"name"}}, ...]}

Each summary lacks the job description; the *detail* endpoint::

    GET https://api.smartrecruiters.com/v1/companies/{company_slug}/postings/{id}

returns ``jobAd.sections.{jobDescription,qualifications}.text`` which is joined
and cleaned.

As with every adapter, :meth:`SmartRecruitersAdapter.parse` is pure (no network)
so it can be exercised against a fixture, while :meth:`SmartRecruitersAdapter.fetch`
does the paginated list calls plus the capped per-posting detail enrichment.
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

_API_ROOT = "https://api.smartrecruiters.com/v1/companies"
_APPLY_ROOT = "https://jobs.smartrecruiters.com"

# Stay a polite neighbour: cap how much we pull per term and how many detail
# fetches we make overall.
_PAGE_LIMIT = 100
_MAX_PAGES_PER_TERM = 5
_MAX_DETAIL_FETCHES = 30


class SmartRecruitersAdapter(BaseAdapter):
    """Adapter for SmartRecruiters-hosted public job postings."""

    name: ClassVar[str] = "smartrecruiters"
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

    def _company_slug(self) -> str:
        return str(self._search().get("company_slug", "") or "")

    def _company_name(self) -> str:
        return getattr(self.company, "name", "") or ""

    def _company_id(self) -> str | None:
        return getattr(self.company, "company_id", None)

    def _list_url(self, slug: str, *, offset: int, term: str | None) -> str:
        url = f"{_API_ROOT}/{slug}/postings?limit={_PAGE_LIMIT}&offset={offset}"
        if term:
            from urllib.parse import quote

            url += f"&q={quote(term)}"
        return url

    def _detail_url(self, slug: str, posting_id: str) -> str:
        return f"{_API_ROOT}/{slug}/postings/{posting_id}"

    def _apply_url(self, slug: str, posting_id: str) -> str:
        return f"{_APPLY_ROOT}/{slug}/{posting_id}"

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                     #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Turn a list-endpoint payload into ``RawJob`` records. PURE — no network.

        ``payload`` may be the already-decoded dict or a JSON string. Descriptions
        are left ``None`` here; :meth:`fetch` fills them from the detail endpoint.
        ``base_url`` is accepted for interface parity but the company slug comes
        from config.
        """
        data = self._coerce_payload(payload)
        postings = data.get("content") or []
        slug = self._company_slug()
        default_company = self._company_name()
        company_id = self._company_id()

        jobs: list[RawJob] = []
        for posting in postings:
            if not isinstance(posting, dict):
                continue
            posting_id = posting.get("id")
            if not posting_id:
                continue
            posting_id = str(posting_id)

            company_block = posting.get("company")
            company_name = default_company
            if isinstance(company_block, dict) and company_block.get("name"):
                company_name = str(company_block["name"])

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=str(posting.get("name") or ""),
                    company_name=company_name,
                    apply_url=self._apply_url(slug, posting_id),
                    source_job_id=posting_id,
                    location=self._location(posting.get("location")),
                    description=None,
                    posted_date_raw=posting.get("releasedDate"),
                    company_id=company_id,
                    extra={"ref": posting.get("ref")},
                )
            )
        return jobs

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
    def _location(location: Any) -> str | None:
        """Join city / region / country into a single location string."""
        if not isinstance(location, dict):
            return None
        parts = [
            str(location.get(key)).strip()
            for key in ("city", "region", "country")
            if location.get(key)
        ]
        joined = ", ".join(p for p in parts if p)
        return joined or None

    @staticmethod
    def description_from_detail(detail: Any) -> str | None:
        """Extract a cleaned plain-text description from a detail payload.

        Joins ``jobAd.sections.jobDescription.text`` and
        ``jobAd.sections.qualifications.text``. Returns ``None`` when neither is
        present so callers can leave :attr:`RawJob.description` unset.
        """
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except ValueError:
                return None
        if not isinstance(detail, dict):
            return None
        job_ad = detail.get("jobAd")
        if not isinstance(job_ad, dict):
            return None
        sections = job_ad.get("sections")
        if not isinstance(sections, dict):
            return None

        chunks: list[str] = []
        for key in ("jobDescription", "qualifications"):
            section = sections.get(key)
            if isinstance(section, dict) and section.get("text"):
                chunks.append(str(section["text"]))
        if not chunks:
            return None
        cleaned = clean_text("\n".join(chunks))
        return cleaned or None

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch postings for ``search_terms`` and enrich with descriptions.

        Paginates the list endpoint per term by ``offset`` (using ``totalFound``
        to know when to stop, capped), dedupes by posting id across all terms,
        then fetches the detail endpoint for each unique posting (capped) to
        populate the description. A failed detail fetch leaves the posting with
        ``description=None`` rather than dropping it. ``SourceBlocked`` propagates
        from the HTTP client.
        """
        slug = self._company_slug()
        if not slug:
            return []
        terms = search_terms or self.search_terms_from_company()

        by_id: dict[str, RawJob] = {}
        for term in terms:
            for posting in self._fetch_term(slug, term):
                key = posting.source_job_id
                if not key or key in by_id:
                    continue
                by_id[key] = posting

        jobs = list(by_id.values())
        self._enrich_descriptions(slug, jobs)
        return jobs

    def _fetch_term(self, slug: str, term: str) -> list[RawJob]:
        """Page through the list endpoint for one search term."""
        collected: list[RawJob] = []
        offset = 0
        for _page in range(_MAX_PAGES_PER_TERM):
            url = self._list_url(slug, offset=offset, term=term or None)
            payload = self.http.get_json(url)
            page_jobs = self.parse(payload)
            if not page_jobs:
                break
            collected.extend(page_jobs)

            total = payload.get("totalFound") if isinstance(payload, dict) else None
            offset += _PAGE_LIMIT
            if isinstance(total, int) and offset >= total:
                break
        return collected

    def _enrich_descriptions(self, slug: str, jobs: list[RawJob]) -> None:
        """Populate ``description`` from the detail endpoint, resiliently."""
        for job in jobs[:_MAX_DETAIL_FETCHES]:
            posting_id = job.source_job_id
            if not posting_id:
                continue
            try:
                detail = self.http.get_json(self._detail_url(slug, posting_id))
            except SourceBlocked:
                # A hard block applies to the whole source: let it propagate.
                raise
            except Exception:
                # Any other detail-fetch failure is non-fatal: keep the posting
                # with description left as None.
                continue
            job.description = self.description_from_detail(detail)
