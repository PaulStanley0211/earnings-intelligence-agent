"""Unit tests for the OpenAI embeddings client wrapper."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from app.tools.embeddings import (
    DailyCostCapExceeded,
    EmbeddingsClient,
    _hash_embed_call,
)


class _RepoStub:
    def __init__(self, spent: Decimal = Decimal("0")) -> None:
        self.spent = spent
        self.added: list[Decimal] = []

    async def get_daily_spend(self, _day):  # type: ignore[no-untyped-def]
        return self.spent

    async def add_daily_spend(self, *, day, amount_usd):  # type: ignore[no-untyped-def]
        self.added.append(amount_usd)
        self.spent += amount_usd
        return self.spent


def _stub_openai(vectors: list[list[float]]) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    client.embeddings.create.return_value = response
    return client


def test_aembed_replays_cassette_without_calling_openai(tmp_path: Path):
    key = _hash_embed_call(model="text-embedding-3-small", texts=["alpha", "beta"])
    cassette = tmp_path / f"{key}.json"
    cassette.write_text(
        json.dumps(
            {
                "model": "text-embedding-3-small",
                "vectors": [[0.1] * 1536, [0.2] * 1536],
                "input_tokens": 4,
                "cost_usd": 0.0,
            }
        )
    )
    openai = MagicMock()
    repo = _RepoStub()
    client = EmbeddingsClient(
        api_key=SecretStr("sk-test"),
        repository_factory=lambda: repo,
        cassette_dir=tmp_path,
        openai_client=openai,
        max_daily_cost_usd=10.0,
    )
    vectors = asyncio.run(client.aembed(["alpha", "beta"]))
    assert vectors[0][0] == pytest.approx(0.1)
    assert vectors[1][0] == pytest.approx(0.2)
    openai.embeddings.create.assert_not_called()


def test_aembed_writes_cassette_on_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REC", "1")
    openai = _stub_openai([[0.3] * 1536])
    repo = _RepoStub()
    client = EmbeddingsClient(
        api_key=SecretStr("sk-test"),
        repository_factory=lambda: repo,
        cassette_dir=tmp_path,
        openai_client=openai,
        max_daily_cost_usd=10.0,
    )
    asyncio.run(client.aembed(["gamma"]))
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["model"] == "text-embedding-3-small"
    assert len(payload["vectors"]) == 1


def test_aembed_returns_empty_on_empty_input(tmp_path: Path):
    client = EmbeddingsClient(
        api_key=SecretStr("sk-test"),
        repository_factory=lambda: _RepoStub(),
        cassette_dir=tmp_path,
        openai_client=MagicMock(),
        max_daily_cost_usd=10.0,
    )
    assert asyncio.run(client.aembed([])) == []


def test_aembed_raises_when_projected_cost_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("REC", "1")
    repo = _RepoStub(spent=Decimal("9.99"))
    client = EmbeddingsClient(
        api_key=SecretStr("sk-test"),
        repository_factory=lambda: repo,
        cassette_dir=tmp_path,
        openai_client=MagicMock(),
        max_daily_cost_usd=10.0,
    )
    with pytest.raises(DailyCostCapExceeded):
        asyncio.run(
            client.aembed(["x" * 100_000 for _ in range(100)])  # large projected cost
        )
