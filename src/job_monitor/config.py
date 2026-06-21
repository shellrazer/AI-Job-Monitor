"""Load and validate the YAML config files into typed pydantic objects.

Single source of runtime truth. Resolves ``${ENV_VAR}`` references against the
environment and expands ``~`` in paths. Fails fast with clear errors.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` in strings with the environment value.

    Missing variables resolve to an empty string (callers validate required
    fields downstream so the failure message is specific).
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _interpolate_env(data)


def expand_path(p: str | Path) -> Path:
    """Expand ``~`` and environment variables in a path-like value."""
    return Path(os.path.expandvars(str(p))).expanduser()


# --------------------------------------------------------------------------- #
# companies.yaml                                                              #
# --------------------------------------------------------------------------- #
class CompanyConfig(BaseModel):
    company_id: str
    name: str
    sector: str = ""
    priority_tier: str = "P3"
    careers_url: str = ""
    ats_platform: str = ""
    adapter: str
    active: bool = True
    search: dict[str, Any] = Field(default_factory=dict)


class CompaniesConfig(BaseModel):
    companies: list[CompanyConfig] = Field(default_factory=list)

    def active(self) -> list[CompanyConfig]:
        return [c for c in self.companies if c.active]


# --------------------------------------------------------------------------- #
# keywords.yaml                                                               #
# --------------------------------------------------------------------------- #
class KeywordsConfig(BaseModel):
    core: list[str] = Field(default_factory=list)
    semantic: list[str] = Field(default_factory=list)
    industry: list[str] = Field(default_factory=list)
    exclusion: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# scoring.yaml                                                                #
# --------------------------------------------------------------------------- #
class ScoringWeights(BaseModel):
    seniority: float = 0.30
    semantic: float = 0.25
    industry: float = 0.20
    company: float = 0.15
    location_salary: float = 0.10


class ScoringConfig(BaseModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    promote_senior_specialist: bool = True
    # Editable scoring tables (spec §10). Stored as plain dicts so they can be
    # retuned in YAML without code changes. The scorer reads these.
    seniority_points: dict[str, int] = Field(default_factory=dict)
    industry_points: dict[str, int] = Field(default_factory=dict)
    responsibility_points: dict[str, int] = Field(default_factory=dict)
    company_points: dict[str, int] = Field(default_factory=dict)
    location_points: dict[str, int] = Field(default_factory=dict)
    salary_bands: list[list[float]] = Field(default_factory=list)
    tier_bands: list[list[Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# settings.yaml                                                               #
# --------------------------------------------------------------------------- #
class EmailSettings(BaseModel):
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender: str = ""
    app_password: str = ""
    recipients: list[str] = Field(default_factory=list)


class EmbeddingSettings(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_dir: str = "~/.job_monitor/models"
    dim: int = 384


class HttpSettings(BaseModel):
    user_agent: str = "JobMonitor/0.1 (+personal use)"
    per_host_rate_limit_rps: float = 0.5
    timeout_seconds: float = 20.0
    max_retries: int = 4
    backoff_base_seconds: float = 1.5
    cache_dir: str = "~/.job_monitor/http_cache"
    cache_ttl_seconds: int = 3600
    respect_robots: bool = True
    impersonate: str = "chrome"  # curl_cffi browser fingerprint for SEEK/Jora


class ReportSettings(BaseModel):
    output_dir: str = "~/.job_monitor/reports"
    filename_template: str = "report_{date}.html"
    open_after_run: bool = False
    min_tier: str = "C"  # lowest tier to include in report/email
    email_new_only: bool = True  # daily digest = only roles newly discovered this run


class Settings(BaseModel):
    db_path: str = "~/.job_monitor/jobs.sqlite"
    report: ReportSettings = Field(default_factory=ReportSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    locations_include: list[str] = Field(default_factory=list)
    run_hour_local: int = 7

    @property
    def db_file(self) -> Path:
        return expand_path(self.db_path)


# --------------------------------------------------------------------------- #
# profile.yaml                                                                #
# --------------------------------------------------------------------------- #
class ProfileConfig(BaseModel):
    full_text: str
    years_experience: float = 0.0
    current_titles: list[str] = Field(default_factory=list)
    target_titles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    salary_floor: int | None = None
    work_rights: str = ""

    @field_validator("full_text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("profile.full_text must not be empty (it is embedded each run)")
        return v


# --------------------------------------------------------------------------- #
# Aggregate                                                                   #
# --------------------------------------------------------------------------- #
class AppConfig(BaseModel):
    settings: Settings
    companies: CompaniesConfig
    keywords: KeywordsConfig
    scoring: ScoringConfig
    profile: ProfileConfig

    def validate_email_ready(self) -> None:
        """Raise if email is enabled but credentials are missing."""
        if self.settings.email.enabled and (
            not self.settings.email.sender or not self.settings.email.app_password
        ):
            raise ValueError(
                "Email is enabled but JOB_MONITOR_GMAIL_USER / "
                "JOB_MONITOR_GMAIL_APP_PASSWORD are not set in the environment."
            )


def load_config(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> AppConfig:
    """Load every config file from ``config_dir`` into a validated :class:`AppConfig`."""
    cdir = Path(config_dir)
    return AppConfig(
        settings=Settings(**_load_yaml(cdir / "settings.yaml")),
        companies=CompaniesConfig(**_load_yaml(cdir / "companies.yaml")),
        keywords=KeywordsConfig(**_load_yaml(cdir / "keywords.yaml")),
        scoring=ScoringConfig(**_load_yaml(cdir / "scoring.yaml")),
        profile=ProfileConfig(**_load_yaml(cdir / "profile.yaml")),
    )
