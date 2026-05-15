"""Settings validation tests.

Required keys must fail fast at construction. The EDGAR User-Agent must
contain a contact email (SEC policy). Tests mutate the environment through
``monkeypatch`` and refresh the cached :class:`Settings` via the
``fresh_settings`` fixture so the next call re-reads the env.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Environment, Settings, get_settings, reset_settings_cache


def test_settings_load_with_valid_env(fresh_settings: None) -> None:
    settings = get_settings()
    assert settings.environment is Environment.TEST
    assert settings.edgar_user_agent.endswith("@example.com")
    assert settings.max_daily_llm_cost_usd > 0


def test_missing_anthropic_key_raises(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings_cache()
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_invalid_edgar_user_agent_rejected(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDGAR_USER_AGENT", "missing-email-here")
    reset_settings_cache()
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_invalid_database_url_rejected(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "mysql://localhost/earnings")
    reset_settings_cache()
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_api_keys_parsed_into_frozenset(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_KEYS", "k1, k2 ,k3,")
    reset_settings_cache()
    keys = Settings().api_key_set  # type: ignore[call-arg]
    assert keys == frozenset({"k1", "k2", "k3"})
