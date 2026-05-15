"""Shared pytest fixtures.

The most important job here is to give every test a deterministic environment.
Settings are sourced from the process environment via pydantic-settings, so we
install a known-good set of values before the application is imported.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# These env vars must be set before `app.config` is imported anywhere.
_DEFAULT_TEST_ENV: dict[str, str] = {
    "ANTHROPIC_API_KEY": "sk-ant-test-placeholder-key-not-real-do-not-call",
    "FINNHUB_API_KEY": "finnhub-test-placeholder",
    "DATABASE_URL": "postgresql+psycopg://earnings:earnings@localhost:5432/earnings_test",
    "REDIS_URL": "redis://localhost:6379/1",
    "EDGAR_USER_AGENT": "Test Suite tests@example.com",
    "MAX_DAILY_LLM_COST_USD": "1.00",
    "LOG_LEVEL": "WARNING",
    "ENVIRONMENT": "test",
    "LLM_CACHE_DIR": "./llm_cache_test",
}

for key, value in _DEFAULT_TEST_ENV.items():
    os.environ.setdefault(key, value)


@pytest.fixture()
def fresh_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Drop the cached :class:`Settings` so the next call re-reads the env.

    Tests that mutate the environment must depend on this fixture or call
    :func:`app.config.reset_settings_cache` directly.
    """
    from app.config import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture()
def cassette_dir(tmp_path: Path) -> Path:
    """Return a per-test cassette directory under pytest's tmp_path."""
    target = tmp_path / "cassettes"
    target.mkdir(parents=True, exist_ok=True)
    return target
