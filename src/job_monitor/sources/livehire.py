"""LiveHire public careers search."""

from __future__ import annotations

from typing import Any

from job_monitor.models import RawJob, Source
from job_monitor.sources.base import BaseAdapter

_BASE_URL = "https://www.livehire.com"


class LiveHireAdapter(BaseAdapter):
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        if self.company is None:
            return []
        tenant = self.company.search["tenant"]
        base = (base_url or _BASE_URL).rstrip("/")
        jobs = []
        for record in payload["jobs"]:
            if record.get("internalOnly") or record.get("isEoi"):
                continue
            code = record.get("urlCode")
            advert = record.get("advertUrlCode")
            title = record.get("roleName")
            if not code or not advert or not title:
                continue
            slug = record.get("seoSlug") or ""
            jobs.append(RawJob(
                source=Source.OFFICIAL_ATS,
                source_job_id=code,
                company_id=self.company.company_id,
                company_name=self.company.name,
                title=title,
                location=record.get("physicalLocation") or record.get("countryName"),
                description=record.get("summary"),
                posted_date_raw=record.get("postedFrom"),
                apply_url=f"{base}/careers/{tenant}/job/{code}/{advert}/{slug}",
            ))
        return jobs

    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        """Fetch the default public results page."""
        if self.company is None:
            return []
        tenant = self.company.search["tenant"]
        payload = self.http.get_json(
            f"{_BASE_URL}/careers-api/search/{tenant}", method="POST", json_body={},
        )
        return self.parse(payload)
