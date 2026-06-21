"""Greenhouse Job Board API source adapter.

Greenhouse exposes a public, unauthenticated JSON API for a board's open
postings. A single call returns every job (with full content when
``content=true``)::

    GET https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true

The response shape::

    {"jobs": [{"id", "title", "absolute_url",
               "location": {"name"}, "updated_at",
               "content"}, ...]}

``content`` is the HTML-escaped job description (entities like ``&lt;p&gt;``),
so it is HTML-unescaped and then run through :func:`clean_text`.

As with every adapter, :meth:`GreenhouseAdapter.parse` is pure (no network) so it
can be exercised against a fixture; :meth:`GreenhouseAdapter.fetch` issues the
single list call and parses it.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any, ClassVar

from job_monitor.models import RawJob, Source
from job_monitor.normalize import clean_text
from job_monitor.sources.base import BaseAdapter

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.http import PoliteClient

_API_ROOT = "https://boards-api.greenhouse.io/v1/boards"


class GreenhouseAdapter(BaseAdapter):
    """Adapter for Greenhouse-hosted public job boards."""

    name: ClassVar[str] = "greenhouse"
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

    def _board_token(self) -> str:
        return str(self._search().get("board_token", "") or "")

    def _company_name(self) -> str:
        return getattr(self.company, "name", "") or ""

    def _company_id(self) -> str | None:
        return getattr(self.company, "company_id", None)

    def _list_url(self, board_token: str) -> str:
        return f"{_API_ROOT}/{board_token}/jobs?content=true"

    # ------------------------------------------------------------------ #
    # Parsing (pure)                                                     #
    # ------------------------------------------------------------------ #
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Turn a board ``jobs`` payload into ``RawJob`` records. PURE — no network.

        ``payload`` may be the already-decoded dict or a JSON string. ``base_url``
        is accepted for interface parity; Greenhouse provides absolute apply URLs.
        """
        data = self._coerce_payload(payload)
        postings = data.get("jobs") or []
        company_name = self._company_name()
        company_id = self._company_id()

        jobs: list[RawJob] = []
        for posting in postings:
            if not isinstance(posting, dict):
                continue
            apply_url = posting.get("absolute_url")
            if not apply_url:
                continue

            posting_id = posting.get("id")
            description = self._description(posting.get("content"))

            jobs.append(
                RawJob(
                    source=Source.OFFICIAL_ATS,
                    title=str(posting.get("title") or ""),
                    company_name=company_name,
                    apply_url=str(apply_url),
                    source_job_id=str(posting_id) if posting_id is not None else None,
                    location=self._location(posting.get("location")),
                    description=description,
                    posted_date_raw=posting.get("updated_at"),
                    company_id=company_id,
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
        if isinstance(location, dict) and location.get("name"):
            return str(location["name"])
        return None

    @staticmethod
    def _description(content: Any) -> str | None:
        """Unescape the HTML-escaped ``content`` and clean it to plain text."""
        if not content:
            return None
        cleaned = clean_text(html.unescape(str(content)))
        return cleaned or None

    # ------------------------------------------------------------------ #
    # Fetching (network)                                                 #
    # ------------------------------------------------------------------ #
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch every posting on the board in a single call, then parse.

        ``search_terms`` is accepted for interface parity; the Greenhouse board
        API returns all open postings in one request and is not filtered
        server-side here. ``SourceBlocked`` propagates from the HTTP client.
        """
        board_token = self._board_token()
        if not board_token:
            return []
        payload = self.http.get_json(self._list_url(board_token))
        return self.parse(payload)
