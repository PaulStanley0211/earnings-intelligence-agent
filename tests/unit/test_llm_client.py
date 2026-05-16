"""Tests for the cassette-aware LLM client.

The Anthropic SDK is replaced by a stub that returns a deterministic response,
so these tests never touch the real API. Coverage:

- Recording produces a cassette JSON file at the SHA-keyed path.
- Replay reads the cassette without invoking the stub.
- A test-mode call with no cassette and ``REC`` unset raises :class:`CassetteMiss`.
- The daily cost cap fails closed when a projected call would exceed it.
- The secret-scrubbing log filter strips API keys from log records.
- The async :meth:`LLMClient.acomplete` reads/writes the Postgres-backed
  daily spend through the injected repository.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.llm.client import (
    CassetteMiss,
    CostCapExceeded,
    LLMClient,
    _DailyCostTracker,
    _hash_call,
)
from app.observability.logging import _scrub


class _StubMessage:
    def __init__(self, text: str) -> None:
        self.content = [MagicMock(type="text", text=text)]
        self.usage = MagicMock(input_tokens=100, output_tokens=50)


def _stub_anthropic(text: str = "hello") -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _StubMessage(text)
    return client


@pytest.fixture()
def llm_client(
    fresh_settings: None,
    monkeypatch: pytest.MonkeyPatch,
    cassette_dir: Path,
) -> LLMClient:
    """A client wired to a temp cassette dir and a stub Anthropic SDK."""
    monkeypatch.delenv("REC", raising=False)
    return LLMClient(
        cassette_dir=cassette_dir,
        anthropic_client=_stub_anthropic(),
    )


def test_first_call_writes_a_cassette(
    llm_client: LLMClient,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In test mode, recording requires the REC=1 escape hatch.
    monkeypatch.setenv("REC", "1")
    response = llm_client.complete(
        prompt_version="test/v1",
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-6",
    )
    assert response.text == "hello"
    files = list(cassette_dir.glob("*.json"))
    assert len(files) == 1
    payload: dict[str, Any] = json.loads(files[0].read_text())
    assert payload["text"] == "hello"
    assert payload["prompt_version"] == "test/v1"


def test_replay_skips_the_api(
    cassette_dir: Path,
    fresh_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _stub_anthropic(text="cassette-text")
    # Record with REC=1, then replay with REC unset to prove the SDK is bypassed.
    monkeypatch.setenv("REC", "1")
    first = LLMClient(cassette_dir=cassette_dir, anthropic_client=stub)
    first.complete(
        prompt_version="test/v1",
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-6",
    )

    monkeypatch.delenv("REC", raising=False)
    second_stub = _stub_anthropic(text="would-be-fresh")
    second = LLMClient(cassette_dir=cassette_dir, anthropic_client=second_stub)
    replayed = second.complete(
        prompt_version="test/v1",
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-6",
    )
    assert replayed.cached is True
    assert replayed.text == "cassette-text"
    second_stub.messages.create.assert_not_called()


def test_missing_cassette_in_test_mode_raises(
    cassette_dir: Path,
    fresh_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REC", raising=False)
    client = LLMClient(cassette_dir=cassette_dir, anthropic_client=_stub_anthropic())
    with pytest.raises(CassetteMiss):
        client.complete(
            prompt_version="never-seen",
            messages=[{"role": "user", "content": "no cassette here"}],
            model="claude-sonnet-4-6",
        )


def test_cost_cap_fails_closed() -> None:
    tracker = _DailyCostTracker(cap_usd=0.01)
    tracker.record(0.009)
    with pytest.raises(CostCapExceeded):
        tracker.check_and_reserve(0.005)


def test_secret_scrubbing_redacts_anthropic_keys() -> None:
    scrubbed = _scrub("the key is sk-ant-abcdefghijklmnopqrstuvwxyz0123456789 in this line")
    assert "sk-ant-" not in scrubbed
    assert "[REDACTED]" in scrubbed


def test_secret_scrubbing_redacts_bearer_tokens() -> None:
    scrubbed = _scrub("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature")
    assert "eyJhbGciOiJIUzI1NiJ9" not in scrubbed


def test_cassette_key_is_stable() -> None:
    key1 = _hash_call(
        prompt_version="v1",
        messages=[{"role": "user", "content": "x"}],
        model="claude-opus-4-7",
        temperature=0.0,
        max_tokens=10,
        system=None,
    )
    key2 = _hash_call(
        prompt_version="v1",
        messages=[{"role": "user", "content": "x"}],
        model="claude-opus-4-7",
        temperature=0.0,
        max_tokens=10,
        system=None,
    )
    assert key1 == key2
    assert len(key1) == 64


class _StubSpendRepository:
    def __init__(self, *, initial: Decimal = Decimal("0")) -> None:
        self._spent = initial
        self.adds: list[Decimal] = []

    async def get_daily_spend(self, day: date) -> Decimal:
        return self._spent

    async def add_daily_spend(self, *, day: date, amount_usd: Decimal) -> Decimal:
        self.adds.append(amount_usd)
        self._spent = self._spent + amount_usd
        return self._spent


async def test_acomplete_records_spend_to_repository(
    cassette_dir: Path,
    fresh_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REC", "1")
    repo = _StubSpendRepository()
    client = LLMClient(cassette_dir=cassette_dir, anthropic_client=_stub_anthropic())
    response = await client.acomplete(
        prompt_version="acomplete-test/v1",
        messages=[{"role": "user", "content": "hi"}],
        repository=repo,
        model="claude-sonnet-4-6",
    )
    assert response.cost_usd > 0
    assert len(repo.adds) == 1
    assert repo.adds[0] > Decimal("0")


async def test_acomplete_fails_closed_when_db_spend_exceeds_cap(
    cassette_dir: Path,
    fresh_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REC", "1")
    # MAX_DAILY_LLM_COST_USD defaults to 1.0 in test env.
    repo = _StubSpendRepository(initial=Decimal("0.99"))
    client = LLMClient(cassette_dir=cassette_dir, anthropic_client=_stub_anthropic())
    with pytest.raises(CostCapExceeded):
        await client.acomplete(
            prompt_version="cap-test/v1",
            messages=[{"role": "user", "content": "hi"}],
            repository=repo,
            model="claude-opus-4-7",
            max_tokens=512,
        )
    assert repo.adds == [], "no spend should be recorded when the cap is exceeded"
