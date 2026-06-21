"""Logging setup and per-run statistics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure rich logging once and return the package logger."""
    global _CONFIGURED
    logger = logging.getLogger("job_monitor")
    if not _CONFIGURED:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONFIGURED = True
    return logger


@dataclass
class SourceStat:
    """Per-source fetch outcome for a run."""

    company_id: str
    name: str
    status: str = "ok"  # ok | blocked | empty | error
    fetched: int = 0
    error: str | None = None


@dataclass
class RunStats:
    """Aggregate counters for a single pipeline run."""

    run_id: str
    sources: list[SourceStat] = field(default_factory=list)
    raw_fetched: int = 0
    canonical: int = 0
    duplicates: int = 0
    persisted: int = 0
    alerts: int = 0

    def add_source(self, stat: SourceStat) -> None:
        self.sources.append(stat)
        self.raw_fetched += stat.fetched

    def summary(self) -> str:
        blocked = sum(1 for s in self.sources if s.status == "blocked")
        errored = sum(1 for s in self.sources if s.status == "error")
        ok = sum(1 for s in self.sources if s.status == "ok")
        return (
            f"run {self.run_id}: {self.raw_fetched} fetched from {len(self.sources)} sources "
            f"({ok} ok, {blocked} blocked, {errored} error); "
            f"{self.canonical} unique, {self.duplicates} dupes, {self.persisted} persisted, "
            f"{self.alerts} alerts"
        )
