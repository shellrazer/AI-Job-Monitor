"""Source adapters: one module per ATS platform / job board.

The adapter registry maps a config ``adapter`` name to a :class:`BaseAdapter`
subclass. ``build_adapter`` is the single entry point the pipeline uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from job_monitor.config import CompanyConfig, Settings
    from job_monitor.sources.base import BaseAdapter
    from job_monitor.sources.http import PoliteClient


def get_adapter_registry() -> dict[str, type[BaseAdapter]]:
    """Return the adapter-name -> adapter-class registry.

    Imported lazily so that importing :mod:`job_monitor.sources` does not pull
    in optional heavy backends (e.g. Playwright) until an adapter is built.
    """
    from job_monitor.sources.generic_html import GenericHtmlAdapter
    from job_monitor.sources.jora import JoraAdapter
    from job_monitor.sources.seek import SeekAdapter
    from job_monitor.sources.successfactors import SuccessFactorsAdapter
    from job_monitor.sources.workday import WorkdayAdapter

    return {
        "workday": WorkdayAdapter,
        "successfactors": SuccessFactorsAdapter,
        "seek": SeekAdapter,
        "jora": JoraAdapter,
        "generic_html": GenericHtmlAdapter,
    }


def build_adapter(
    company: CompanyConfig | None,
    settings: Settings,
    http: PoliteClient,
    *,
    adapter_name: str | None = None,
) -> BaseAdapter:
    """Construct the adapter bound to ``company`` (or an explicit ``adapter_name``)."""
    name = adapter_name or (company.adapter if company else None)
    if name is None:
        raise ValueError("build_adapter requires a company with an adapter or an explicit adapter_name")
    registry = get_adapter_registry()
    try:
        cls = registry[name]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unknown adapter '{name}'. Known: {sorted(registry)}") from exc
    return cls(http=http, company=company, settings=settings)
