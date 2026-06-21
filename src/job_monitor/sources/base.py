"""Source-adapter base class.

An adapter knows how to fetch postings from one ATS platform / job board and
turn the raw payload into loosely-shaped :class:`~job_monitor.models.RawJob`
records. Two responsibilities are split deliberately:

* :meth:`BaseAdapter.fetch` does the network I/O (and is what live runs call);
* :meth:`BaseAdapter.parse` is pure — it takes a payload and returns
  ``RawJob`` records with no I/O — so fixture-driven tests can exercise the
  parsing logic deterministically and offline.

:meth:`BaseAdapter.healthcheck` is concrete and never raises: it runs a small
live ``fetch`` and classifies the outcome for the ``validate`` CLI command.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from job_monitor.models import RawJob, Source, SourceBlocked, SourceHealth

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.http import PoliteClient

# Used when a company has no configured search terms (spec default domain).
DEFAULT_SEARCH_TERMS: list[str] = ["quality manager", "food safety"]

# Keys, in priority order, under which a company may store its search terms.
_SEARCH_KEYS = ("search_terms", "keywords", "q")


class BaseAdapter(ABC):
    """Base class every source adapter subclasses.

    Subclasses set the :attr:`name` / :attr:`source` class attributes and
    implement :meth:`fetch` and :meth:`parse`.
    """

    name: ClassVar[str]
    source: ClassVar[Source]

    def __init__(
        self,
        *,
        http: PoliteClient,
        company: CompanyConfig | None,
        settings: Settings,
    ) -> None:
        self.http = http
        self.company = company
        self.settings = settings

    # ------------------------------------------------------------------ #
    # Abstract interface                                                 #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch postings for ``search_terms`` (performs network I/O)."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        """Turn a raw ``payload`` into ``RawJob`` records. PURE — no network."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared behaviour                                                   #
    # ------------------------------------------------------------------ #
    def search_terms_from_company(self) -> list[str]:
        """Return the company's configured search terms, or a sensible default.

        Reads ``company.search`` looking for ``search_terms`` / ``keywords`` /
        ``q`` (in that order). Values may be a single string or a list of
        strings. Falls back to :data:`DEFAULT_SEARCH_TERMS`.
        """
        search = getattr(self.company, "search", None) or {}
        for key in _SEARCH_KEYS:
            value = search.get(key)
            if not value:
                continue
            if isinstance(value, str):
                return [value]
            terms = [str(v) for v in value if str(v).strip()]
            if terms:
                return terms
        return list(DEFAULT_SEARCH_TERMS)

    def healthcheck(self) -> SourceHealth:
        """Run a small live ``fetch`` and classify the outcome. Never raises.

        Statuses: ``ok`` (returned jobs), ``empty`` (returned none),
        ``blocked`` (:class:`SourceBlocked` raised), ``error`` (any other
        exception, with its message captured).
        """
        terms = self.search_terms_from_company()
        start = time.monotonic()
        try:
            jobs = self.fetch(terms)
        except SourceBlocked as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            return SourceHealth(
                name=self.name,
                source=self.source,
                ok=False,
                status="blocked",
                job_count=0,
                latency_ms=latency_ms,
                error=str(exc) or "blocked",
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            return SourceHealth(
                name=self.name,
                source=self.source,
                ok=False,
                status="error",
                job_count=0,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

        latency_ms = (time.monotonic() - start) * 1000.0
        count = len(jobs)
        if count > 0:
            return SourceHealth(
                name=self.name,
                source=self.source,
                ok=True,
                status="ok",
                job_count=count,
                latency_ms=latency_ms,
            )
        return SourceHealth(
            name=self.name,
            source=self.source,
            ok=False,
            status="empty",
            job_count=0,
            latency_ms=latency_ms,
        )
