"""Capture live responses into tests/fixtures/ for offline adapter parse tests.

Run: `uv run python scripts/capture_fixtures.py`

Network-dependent and best-effort: ATS endpoints/tenant slugs change and SEEK/Jora
are Cloudflare-protected. Re-run when a site changes. It prints a per-source report
and only overwrites a fixture when it actually got data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def _report(name: str, ok: bool, detail: str) -> None:
    mark = "OK  " if ok else "MISS"
    print(f"[{mark}] {name}: {detail}")


def capture_workday(name: str, tenant: str, host: str, site: str, search: str) -> None:
    import httpx

    url = f"https://{tenant}.{host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
    body = {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": search}
    headers = {"Accept": "application/json", "Content-Type": "application/json",
               "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        r = httpx.post(url, json=body, headers=headers, timeout=30, follow_redirects=True)
        if r.status_code == 200 and "jobPostings" in r.text:
            data = r.json()
            (FIXTURES / f"workday_{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            _report(f"workday_{name}", True, f"{len(data.get('jobPostings', []))} postings, total={data.get('total')}")
        else:
            _report(f"workday_{name}", False, f"HTTP {r.status_code}, len={len(r.text)}")
    except Exception as exc:
        _report(f"workday_{name}", False, f"{type(exc).__name__}: {exc}")


def capture_impersonate(name: str, url: str) -> None:
    try:
        from curl_cffi import requests as cffi

        r = cffi.get(url, impersonate="chrome", timeout=30)
        has_ldjson = "JobPosting" in r.text or "application/ld+json" in r.text
        if r.status_code == 200 and len(r.text) > 2000:
            (FIXTURES / f"{name}.html").write_text(r.text, encoding="utf-8")
            _report(name, True, f"HTTP 200, {len(r.text)} bytes, ld+json JobPosting={has_ldjson}")
        else:
            _report(name, False, f"HTTP {r.status_code}, len={len(r.text)} (likely Cloudflare)")
    except Exception as exc:
        _report(name, False, f"{type(exc).__name__}: {exc}")


def capture_text(name: str, url: str) -> None:
    import httpx

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        r = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        if r.status_code == 200 and len(r.text) > 1000:
            (FIXTURES / f"{name}.html").write_text(r.text, encoding="utf-8")
            _report(name, True, f"HTTP 200, {len(r.text)} bytes")
        else:
            _report(name, False, f"HTTP {r.status_code}, len={len(r.text)}")
    except Exception as exc:
        _report(name, False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print(f"Capturing fixtures into {FIXTURES}\n")
    capture_workday("bega", "begacheese", "wd3", "Bega_Careers", "quality")
    capture_workday("saputo", "saputo", "wd5", "Saputo_External_Careers", "quality")
    capture_impersonate(
        "seek_search",
        "https://www.seek.com.au/Quality-Manager-jobs/in-All-Sydney-NSW",
    )
    capture_impersonate("jora_search", "https://au.jora.com/j?q=quality+manager&l=Sydney+NSW")
    capture_text("graincorp_careers", "https://jobs.graincorp.com.au/")
    print("\nDone. Captured fixtures (if any) are in tests/fixtures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
