"""Lever Postings API source adapter.

Lever exposes a public, unauthenticated JSON API for a company's open postings.
A single call returns a JSON *list* (not a wrapping object)::

    GET https://api.lever.co/v0/postings/{company_slug}?mode=json

Each element has the shape::

    {"id", "text"(title), "hostedUrl",
     "categories": {"location", "team", "commitment"},
     "createdAt"(epoch ms), "descriptionPlain"}

``createdAt`` is an epoch-milliseconds integer which is converted to an ISO date
string (``YYYY-MM-DD``).

As with every adapter, :meth:`LeverAdapter.parse` is pure (no network) so it can
be exercised against a fixture; :meth:`LeverAdapter.fetch` issues the single list
call and parses it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from job_monitor.models import RawJob, Source
from job_monitor.sources.base import BaseAdapter

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.http import PoliteClient

_API_ROOT = "https://api.lever.co/v0/postings"


class LeverAdapter(BaseAdapter):
    """Adapter for Lever-hosted public job postings."""

    name: ClassVar[str] = "lever"
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

    def _list_url(self, slug: str) -> str:
        return f"{_API_ROOT}/{slug}?mode=json"

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                     #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Turn a Lever postings list into ``RawJob`` records. PURE — no network.

        ``payload`` may be the already-decoded list or a JSON string. ``base_url``
        is accepted for interface parity; Lever provides absolute apply URLs.
        """
        postings = self._coerce_payload(payload)
        company_name = self._company_name()
        company_id = self._company_id()

        jobs: list[RawJob] = []
        for posting in postings:
            if not isinstance(posting, dict):
                continue
            apply_url = posting.get("hostedUrl")
            if not apply_url:
                continue

            posting_id = posting.get("id")
            description = posting.get("descriptionPlain")

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=str(posting.get("text") or ""),
                    company_name=company_name,
                    apply_url=str(apply_url),
                    source_job_id=str(posting_id) if posting_id is not None else None,
                    location=self._location(posting.get("categories")),
                    description=str(description) if description else None,
                    posted_date_raw=self._iso_date(posting.get("createdAt")),
                    company_id=company_id,
                )
            )
        return jobs

    @staticmethod
    def _coerce_payload(payload: Any) -> list[Any]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                return []
        if not isinstance(payload, list):
            return []
        return payload

    @staticmethod
    def _location(categories: Any) -> str | None:
        if isinstance(categories, dict) and categories.get("location"):
            return str(categories["location"])
        return None

    @staticmethod
    def _iso_date(created_at: Any) -> str | None:
        """Convert a ``createdAt`` epoch-millis value into an ISO date string."""
        if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
            return None
        try:
            dt = datetime.fromtimestamp(created_at / 1000.0, tz=UTC)
        except (ValueError, OverflowError, OSError):
            return None
        return dt.date().isoformat()

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch every posting for the company in a single call, then parse.

        ``search_terms`` is accepted for interface parity; the Lever postings API
        returns all open postings in one request. The endpoint returns a JSON
        list, so we read it as text and decode it (``get_json`` expects an object).
        ``SourceBlocked`` propagates from the HTTP client.
        """
        slug = self._company_slug()
        if not slug:
            return []
        payload = self.http.get_text(self._list_url(slug))
        return self.parse(payload)
