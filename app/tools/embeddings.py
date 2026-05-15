"""OpenAI embeddings client with cassette replay and daily-cost guard.

The differ and the backfill script go through this one wrapper so:

* Tests run offline by default. Vectors are SHA-keyed by ``(model, sorted_texts)``
  and cassettes live under ``tests/fixtures/cassettes/embeddings/``. Re-record
  with ``REC=1``.
* Daily spend is gated on the shared ``daily_llm_spend`` Postgres table so
  embeddings and Claude calls compete for the same cap configured by
  ``MAX_DAILY_LLM_COST_USD``.
* Failure modes are explicit: an OpenAI rate-limit or network error is
  retried, a 4xx surfaces immediately, and a cap-exceeded projection raises
  :class:`DailyCostCapExceeded` before any API call is issued.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Protocol

import httpx
from openai import APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import SecretStr
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

_USD_PER_1K_TOKENS: Final[dict[str, float]] = {
    "text-embedding-3-small": 0.02 / 1000.0,
    "text-embedding-3-large": 0.13 / 1000.0,
}

_DEFAULT_BATCH_SIZE: Final[int] = 100
_DEFAULT_MAX_ATTEMPTS: Final[int] = 3


class DailyCostCapExceeded(RuntimeError):
    """Raised when an embedding call would push today's spend past the cap."""


class CassetteMiss(RuntimeError):
    """Raised when a test asked for replay but no cassette exists for the key."""


class _SupportsDailySpend(Protocol):
    """The repository shape the cost guard requires."""

    async def get_daily_spend(self, day: date) -> Decimal:
        """Return total USD spent on the given UTC day."""
        ...

    async def add_daily_spend(
        self, *, day: date, amount_usd: Decimal
    ) -> Decimal:
        """Append ``amount_usd`` to the running total for ``day``."""
        ...


def _hash_embed_call(*, model: str, texts: Sequence[str]) -> str:
    """Return a stable SHA-256 cassette key for an embedding call."""
    payload = json.dumps(
        {"model": model, "texts": sorted(texts)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingsClient:
    """Wraps the OpenAI embeddings API with cassette replay and cost guard.

    A single client instance is constructed per process and shared by the
    differ node and the backfill script. ``repository_factory`` produces a
    fresh :class:`Repository` per call so we can run inside an existing
    SQLAlchemy session in tests, or build a one-shot session in scripts.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        repository_factory: Callable[[], _SupportsDailySpend],
        model: str = "text-embedding-3-small",
        cassette_dir: Path | None = None,
        openai_client: Any = None,
        max_daily_cost_usd: float = 10.0,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        """Wire dependencies. ``openai_client`` may be a real or mock client."""
        self._api_key = api_key
        self._repository_factory = repository_factory
        self._model = model
        self._cassette_dir = cassette_dir or Path(
            "tests/fixtures/cassettes/embeddings"
        )
        self._cassette_dir.mkdir(parents=True, exist_ok=True)
        self._client = openai_client or AsyncOpenAI(
            api_key=api_key.get_secret_value()
        )
        self._max_daily_cost_usd = max_daily_cost_usd
        self._batch_size = batch_size
        self._max_attempts = max_attempts

    @property
    def model(self) -> str:
        """Return the embedding model name (e.g. ``text-embedding-3-small``)."""
        return self._model

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every entry of ``texts`` in order; returns the matching vectors.

        Cassette replay short-circuits the entire call when a cassette exists
        for the SHA-keyed input. ``REC=1`` forces a live API call and rewrites
        the cassette.
        """
        if not texts:
            return []
        key = _hash_embed_call(model=self._model, texts=texts)
        cassette = self._load_cassette(key)
        recording = os.environ.get("REC") == "1"
        if cassette is not None and not recording:
            return list(cassette["vectors"])

        await self._gate_on_daily_cost(texts)

        vectors = await self._call_with_retry(list(texts))

        cost_usd = self._estimate_cost(texts)
        await self._record_spend(cost_usd)

        if recording or cassette is None:
            self._save_cassette(
                key,
                {
                    "model": self._model,
                    "vectors": vectors,
                    "input_tokens": self._estimate_tokens(texts),
                    "cost_usd": cost_usd,
                },
            )

        _logger.bind(
            model=self._model,
            input_count=len(texts),
            cost_usd=cost_usd,
            trace_id=current_trace_id(),
        ).info("embeddings_call")
        return vectors

    def _estimate_tokens(self, texts: Sequence[str]) -> int:
        """Cheap token-count estimate (4 chars per token) for the cost guard."""
        total_chars = sum(len(t) for t in texts)
        return max(1, total_chars // 4)

    def _estimate_cost(self, texts: Sequence[str]) -> float:
        """Estimated USD cost for embedding ``texts`` at the current model."""
        per_token = _USD_PER_1K_TOKENS.get(
            self._model, _USD_PER_1K_TOKENS["text-embedding-3-large"]
        )
        return self._estimate_tokens(texts) * per_token

    async def _gate_on_daily_cost(self, texts: Sequence[str]) -> None:
        """Raise :class:`DailyCostCapExceeded` when projection would breach."""
        projected = self._estimate_cost(texts)
        repo = self._repository_factory()
        today = datetime.now(UTC).date()
        already_spent = float(await repo.get_daily_spend(today))
        if already_spent + projected > self._max_daily_cost_usd:
            raise DailyCostCapExceeded(
                f"Embedding call projected to cost ${projected:.4f} "
                f"on top of ${already_spent:.4f} already spent today "
                f"would exceed cap ${self._max_daily_cost_usd:.2f}."
            )

    async def _record_spend(self, cost_usd: float) -> None:
        """Commit actual spend to the shared daily-spend table."""
        repo = self._repository_factory()
        await repo.add_daily_spend(
            day=datetime.now(UTC).date(),
            amount_usd=Decimal(f"{cost_usd:.6f}"),
        )

    async def _call_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI with batching and tenacity-driven retry."""
        out: list[list[float]] = []
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=30.0),
            retry=retry_if_exception_type(
                (RateLimitError, APITimeoutError, httpx.RequestError)
            ),
            reraise=True,
        )
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = None
            async for attempt in retrying:
                with attempt:
                    raw = self._client.embeddings.create(
                        model=self._model, input=batch
                    )
                    response = await raw if asyncio.iscoroutine(raw) else raw
            if response is None:
                raise RuntimeError("unreachable: tenacity reraises on exhaustion")
            out.extend(list(item.embedding) for item in response.data)
        return out

    def _cassette_path(self, key: str) -> Path:
        """Return the filesystem path for the cassette identified by ``key``."""
        return self._cassette_dir / f"{key}.json"

    def _load_cassette(self, key: str) -> dict[str, Any] | None:
        """Load and return the cassette JSON for ``key``, or ``None`` if absent."""
        path = self._cassette_path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise CassetteMiss(f"Cassette at {path} is not a JSON object")
        return data

    def _save_cassette(self, key: str, payload: dict[str, Any]) -> None:
        """Serialise ``payload`` to the cassette file for ``key``."""
        path = self._cassette_path(key)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
