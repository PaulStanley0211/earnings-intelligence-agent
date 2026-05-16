"""Centralised application configuration.

Loaded from environment variables (or a ``.env`` file in development) via
``pydantic-settings``. Required keys fail startup fast - never silently default.
The set of required keys is the canonical list in ``CLAUDE.md`` plus a few
runtime knobs needed by the bootstrap modules.

The ``Settings`` object is cached behind :func:`get_settings` so the rest of the
codebase has a single source of truth and we do not re-parse the environment on
every import.
"""

from __future__ import annotations

import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# SEC requires a contact email in the User-Agent header. Format: "<name> <email>".
_EDGAR_UA_PATTERN = re.compile(r"^[^<>]{2,}\s+\S+@\S+\.\S+$")


class Environment(StrEnum):
    """Deployment environment selector."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    TEST = "test"


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings sourced from the process environment.

    Missing required values raise on construction so the app cannot start in a
    half-configured state. See :file:`.env.example` for the full key list.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM ----
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key for Claude.")
    max_daily_llm_cost_usd: float = Field(
        ..., gt=0, description="Hard daily LLM spend cap. The client fails closed past this."
    )
    llm_cache_dir: Path = Field(
        default=Path("./llm_cache"),
        description="On-disk LLM response cache (production cache, not test cassettes).",
    )

    # ---- Market data ----
    finnhub_api_key: SecretStr = Field(..., description="Finnhub API key for analyst consensus.")

    # ---- Embeddings (Phase 3) ----
    openai_api_key: SecretStr = Field(
        ..., description="OpenAI API key for the embeddings client (Phase 3)."
    )
    embeddings_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embeddings model used by the language differ.",
    )

    # ---- Datastores ----
    database_url: str = Field(..., description="SQLAlchemy URL for the primary Postgres database.")
    redis_url: str = Field(..., description="Redis connection URL used by RQ and caches.")

    # ---- EDGAR ----
    edgar_user_agent: str = Field(
        ...,
        description="SEC EDGAR User-Agent. Format: '<name> <email>'.",
    )
    edgar_poll_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="How often the watcher polls EDGAR for new filings.",
    )

    # ---- Runtime ----
    log_level: LogLevel = Field(default="INFO", description="Loguru log level.")
    environment: Environment = Field(
        default=Environment.DEV, description="Runtime environment selector."
    )

    # ---- Optional delivery ----
    smtp_host: str | None = None
    smtp_user: str | None = None
    smtp_pass: SecretStr | None = None
    slack_webhook_url: SecretStr | None = None

    # ---- Optional API gating ----
    api_keys: str | None = Field(
        default=None,
        description="Comma-separated allowlist of API keys for /api/* endpoints.",
    )

    @field_validator("edgar_user_agent")
    @classmethod
    def _validate_edgar_user_agent(cls, value: str) -> str:
        """Reject EDGAR User-Agent strings that do not include a contact email.

        SEC will throttle or block requests with a missing or malformed
        identifier, so we surface the error at startup rather than at first poll.
        """
        if not _EDGAR_UA_PATTERN.match(value.strip()):
            raise ValueError(
                "EDGAR_USER_AGENT must be of the form '<name> <email>' "
                "with a real contact email; see CLAUDE.md."
            )
        return value.strip()

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        """Require a Postgres URL - the rest of the stack assumes it."""
        if not value.startswith(("postgresql://", "postgresql+psycopg://", "postgresql+asyncpg://")):
            raise ValueError("DATABASE_URL must be a Postgres URL.")
        return value

    @property
    def api_key_set(self) -> frozenset[str]:
        """Parse :attr:`api_keys` into a hashable lookup set."""
        if not self.api_keys:
            return frozenset()
        return frozenset(key.strip() for key in self.api_keys.split(",") if key.strip())

    @property
    def is_production(self) -> bool:
        """Return True when running in the production environment."""
        return self.environment is Environment.PROD


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance (cached).

    Tests should use :func:`reset_settings_cache` to drop and reload after
    mutating the environment.
    """
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Drop the cached :class:`Settings` so the next call re-reads the env.

    Intended for tests; production code should never need this.
    """
    get_settings.cache_clear()
