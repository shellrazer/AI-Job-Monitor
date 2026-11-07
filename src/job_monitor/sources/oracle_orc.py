"""Oracle Recruiting Cloud public career-site listings."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from job_monitor.models import RawJob, Source
from job_monitor.sources.base import BaseAdapter


class OracleOrcAdapter(BaseAdapter):
    def parse(self, payload: Any, *, base_url: str | None = None) -> list[RawJob]:
        if self.company is None:
            return []
        search = self.company.search
        base = (base_url or search["base_url"]).rstrip("/")
        site = search["site"]
        jobs = []
        for item in payload["items"]:
            for record in item["requisitionList"]:
                if record.get("PrimaryLocationCountry") != "AU":
                    continue
                job_id = str(record["Id"])
                jobs.append(RawJob(
                    source=Source.OFFICIAL_ATS,
                    source_job_id=job_id,
                    company_id=self.company.company_id,
                    company_name=self.company.name,
                    title=record["Title"],
                    location=record.get("PrimaryLocation") or "Australia",
                    description=record.get("ShortDescriptionStr"),
                    posted_date_raw=record.get("PostedDate"),
                    apply_url=f"{base}/hcmUI/CandidateExperience/en/sites/{site}/job/{job_id}",
                ))
        return jobs

    def fetch(self, search_terms: list[str]) -> list[RawJob]:
        if self.company is None:
            return []
        search = self.company.search
        base = search["base_url"].rstrip("/")
        site = search["site"]
        limit = 25
        jobs: dict[str, RawJob] = {}
        for offset in range(0, limit * int(search.get("max_pages", 10)), limit):
            query = urlencode({
                "onlyData": "true",
                "expand": "requisitionList",
                "finder": f"findReqs;siteNumber={site},limit={limit},offset={offset}",
            })
            endpoint = f"{base}/hcmRestApi/resources/latest/recruitingCEJobRequisitions?{query}"
            payload = self.http.get_json(endpoint, method="GET")
            items = payload["items"]
            if not items:
                break
            result = items[0]
            records = result["requisitionList"]
            for job in self.parse(payload):
                jobs[job.apply_url] = job
            if not records or offset + limit >= int(result["TotalJobsCount"]):
                break
        return list(jobs.values())
