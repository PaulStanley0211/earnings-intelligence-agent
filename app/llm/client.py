"""The one and only Anthropic client.

Every LLM call in the codebase routes through :class:`LLMClient`. Importing
``anthropic`` anywhere else is a project-rule violation - this is the contract
that lets us add tracing, a cost cap, prompt-version recording, and a
cassette-based replay layer in exactly one place.

The cassette layer keys responses by SHA-256 over
``(prompt_version, messages, model, temperature, max_tokens)``. Tests run
against cassettes by default and so are fully offline and deterministic. Set
the environment variable ``REC=1`` to re-record cassettes from a live API
during a test run.

The cost guard tracks daily spend in two places: an in-process counter for
sync :meth:`LLMClient.complete` calls (test fixtures, low-stakes one-shots),
and the Postgres-backed ``daily_llm_spend`` table read through
:class:`~app.memory.repository.Repository` for async
:meth:`LLMClient.acomplete` calls from agent nodes. The async path is what
production uses - it survives restarts and is consistent across the web,
worker, and watcher processes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

from anthropic import Anthropic
from anthropic.types import MessageParam

from app.config import Settings, get_settings
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

# Indicative per-1k-token pricing for cost accounting. These are estimates used
# only by the in-process cost guard; the official invoice is the source of
# truth. Pricing in USD per 1,000 tokens.
_PRICING_USD_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    # model -> (input_price, output_price)
    "claude-opus-4-7": (15.0 / 1000.0, 75.0 / 1000.0),
    "claude-opus-4-6": (15.0 / 1000.0, 75.0 / 1000.0),
    "claude-sonnet-4-6": (3.0 / 1000.0, 15.0 / 1000.0),
    "claude-haiku-4-5-20251001": (1.0 / 1000.0, 5.0 / 1000.0),
}


def _model_pricing(model: str) -> tuple[float, float]:
    """Return per-1k-token (input, output) pricing for ``model``.

    Falls back to Opus pricing when the model is unknown so the cost guard
    never under-charges.
    """
    return _PRICING_USD_PER_1K_TOKENS.get(model, _PRICING_USD_PER_1K_TOKENS["claude-opus-4-7"])


# Models that have deprecated the ``temperature`` parameter at the Anthropic
# API. Passing ``temperature`` to these models raises a 400 BadRequestError. We
# still allow the prompt frontmatter to declare ``temperature: 0.0`` for
# documentation and cassette-key stability, but we omit it from the API kwargs
# for these specific models.
_NO_TEMPERATURE_MODELS: frozenset[str] = frozenset(
    {
        "claude-opus-4-7",
        "claude-opus-4-7[1m]",
    }
)


def _supports_temperature(model: str) -> bool:
    """Return ``False`` for models that have deprecated the temperature parameter."""
    return model not in _NO_TEMPERATURE_MODELS


class CostCapExceeded(RuntimeError):
    """Raised when a call would push today's spend past the configured cap."""


class CassetteMiss(RuntimeError):
    """Raised when a test asked for replay but no cassette exists for the key."""


@dataclass(frozen=True)
class LLMResponse:
    """Normalised response surfaced to callers.

    Hides the underlying SDK's response shape so we can swap providers or
    versions without rippling changes.
    """

    text: str
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    cassette_key: str

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict suitable for cassette storage."""
        return {
            "text": self.text,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


class _DailyCostTracker:
    """Thread-safe per-day spend counter enforcing the daily cap.

    Resets at UTC midnight. The state lives in-process; a Postgres-backed
    counter follows in Phase 1 so the cap survives restarts and is shared
    across the web, worker, and watcher processes.
    """

    def __init__(self, cap_usd: float) -> None:
        """Construct a tracker enforcing ``cap_usd`` per UTC day."""
        self._cap_usd = cap_usd
        self._lock = threading.Lock()
        self._date: date = datetime.now(UTC).date()
        self._spent_usd: float = 0.0

    def _roll_day_if_needed(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._date:
            self._date = today
            self._spent_usd = 0.0

    def check_and_reserve(self, projected_cost_usd: float) -> None:
        """Raise :class:`CostCapExceeded` if adding ``projected_cost_usd`` exceeds the cap."""
        with self._lock:
            self._roll_day_if_needed()
            if self._spent_usd + projected_cost_usd > self._cap_usd:
                raise CostCapExceeded(
                    f"LLM daily cap of ${self._cap_usd:.2f} would be exceeded "
                    f"by a call costing ${projected_cost_usd:.4f} on top of "
                    f"${self._spent_usd:.4f} already spent today."
                )

    def record(self, actual_cost_usd: float) -> None:
        """Commit an actual spend reading after a call has completed."""
        with self._lock:
            self._roll_day_if_needed()
            self._spent_usd += actual_cost_usd

    @property
    def spent_today(self) -> float:
        """Return the running USD total spent today."""
        with self._lock:
            self._roll_day_if_needed()
            return self._spent_usd


class _SupportsDailySpend(Protocol):
    """Subset of :class:`~app.memory.repository.Repository` the LLM cost cap needs.

    The async path uses Postgres so the cap is shared across processes; the
    Protocol lets tests inject a stub without spinning up a real database.
    """

    async def get_daily_spend(self, day: date) -> Decimal: ...
    async def add_daily_spend(self, *, day: date, amount_usd: Decimal) -> Decimal: ...


def _hash_call(
    *,
    prompt_version: str,
    messages: Iterable[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    system: str | None,
) -> str:
    """Return a stable SHA-256 cassette key for the call inputs."""
    payload = json.dumps(
        {
            "prompt_version": prompt_version,
            "messages": list(messages),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMClient:
    """Cassette-aware Anthropic client with cost guard and tracing hooks.

    Instances are normally obtained from :func:`get_llm_client`. The class is
    constructible directly for tests that need to inject a custom cassette
    directory or a stub Anthropic client.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cassette_dir: Path | None = None,
        anthropic_client: Anthropic | None = None,
        cost_tracker: _DailyCostTracker | None = None,
    ) -> None:
        """Wire dependencies; defaults read from :func:`get_settings`."""
        self._settings = settings or get_settings()
        self._cassette_dir = cassette_dir or Path("tests/fixtures/cassettes")
        self._cassette_dir.mkdir(parents=True, exist_ok=True)
        self._client = anthropic_client or Anthropic(
            api_key=self._settings.anthropic_api_key.get_secret_value()
        )
        self._cost_tracker = cost_tracker or _DailyCostTracker(
            cap_usd=self._settings.max_daily_llm_cost_usd
        )

    # ---- public API ----

    def complete(
        self,
        *,
        prompt_version: str,
        messages: list[dict[str, Any]],
        model: str = "claude-opus-4-7",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        """Run a chat-style completion through the cassette layer and cost guard.

        ``prompt_version`` must reference the versioned template under
        :file:`prompts/` so every recorded response is traceable to the prompt
        that produced it.
        """
        key = _hash_call(
            prompt_version=prompt_version,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

        cassette = self._load_cassette(key)
        recording = os.environ.get("REC") == "1"
        if cassette is not None and not recording:
            return self._response_from_cassette(cassette, key)

        if self._settings.environment.value == "test" and not recording:
            raise CassetteMiss(
                f"No cassette for key {key} under {self._cassette_dir}. "
                "Run with REC=1 to record."
            )

        # Pre-flight cost reservation. We do not know exact tokens until after
        # the call, so reserve a conservative estimate based on max_tokens.
        in_price, out_price = _model_pricing(model)
        worst_case_cost = (max_tokens / 1000.0) * out_price
        self._cost_tracker.check_and_reserve(worst_case_cost)

        # The Anthropic SDK types ``messages`` as a list of TypedDicts; our
        # internal call sites use plain dicts, which the SDK accepts verbatim.
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system or "",
            "messages": cast("list[MessageParam]", messages),
        }
        if _supports_temperature(model):
            create_kwargs["temperature"] = temperature
        api_response = self._client.messages.create(**create_kwargs)
        text = _extract_text(api_response)
        input_tokens = int(api_response.usage.input_tokens)
        output_tokens = int(api_response.usage.output_tokens)
        cost_usd = (
            (input_tokens / 1000.0) * in_price
            + (output_tokens / 1000.0) * out_price
        )
        self._cost_tracker.record(cost_usd)

        response = LLMResponse(
            text=text,
            model=model,
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cached=False,
            cassette_key=key,
        )

        if recording or self._settings.environment.value == "test":
            self._save_cassette(key, response)

        _logger.bind(
            prompt_version=prompt_version,
            model=model,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            trace_id=current_trace_id(),
        ).info("llm_call")
        return response

    @property
    def spent_today_usd(self) -> float:
        """Return USD spent today by this process across all calls."""
        return self._cost_tracker.spent_today

    async def acomplete(
        self,
        *,
        prompt_version: str,
        messages: list[dict[str, Any]],
        repository: _SupportsDailySpend,
        model: str = "claude-opus-4-7",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        """Async completion that gates on the Postgres-backed daily spend.

        The pre-flight check reads ``daily_llm_spend`` rather than the
        in-process counter so the cap is shared across web/worker/watcher
        processes and survives restarts. The actual Anthropic call runs in a
        worker thread because the SDK is sync; the in-process counter is
        kept consistent so legacy :meth:`complete` callers still see a
        unified ``spent_today`` reading.
        """
        key = _hash_call(
            prompt_version=prompt_version,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

        cassette = self._load_cassette(key)
        recording = os.environ.get("REC") == "1"
        if cassette is not None and not recording:
            return self._response_from_cassette(cassette, key)

        if self._settings.environment.value == "test" and not recording:
            raise CassetteMiss(
                f"No cassette for key {key} under {self._cassette_dir}. "
                "Run with REC=1 to record."
            )

        in_price, out_price = _model_pricing(model)
        worst_case_cost = (max_tokens / 1000.0) * out_price
        today = datetime.now(UTC).date()
        already_spent = float(await repository.get_daily_spend(today))
        if already_spent + worst_case_cost > self._settings.max_daily_llm_cost_usd:
            raise CostCapExceeded(
                f"LLM daily cap of ${self._settings.max_daily_llm_cost_usd:.2f} "
                f"would be exceeded by a call costing up to "
                f"${worst_case_cost:.4f} on top of ${already_spent:.4f} "
                "already spent today."
            )

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system or "",
            "messages": cast("list[MessageParam]", messages),
        }
        if _supports_temperature(model):
            create_kwargs["temperature"] = temperature
        # Wrap in a lambda so mypy can resolve the overload of
        # ``messages.create`` against our dynamically built kwargs.
        api_response = await asyncio.to_thread(
            lambda: self._client.messages.create(**create_kwargs)
        )
        text = _extract_text(api_response)
        input_tokens = int(api_response.usage.input_tokens)
        output_tokens = int(api_response.usage.output_tokens)
        cost_usd = (
            (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price
        )
        await repository.add_daily_spend(
            day=today, amount_usd=Decimal(f"{cost_usd:.6f}")
        )
        # Keep the in-process counter consistent for observability surfaces
        # that still consult ``spent_today_usd``.
        self._cost_tracker.record(cost_usd)

        response = LLMResponse(
            text=text,
            model=model,
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cached=False,
            cassette_key=key,
        )
        if recording or self._settings.environment.value == "test":
            self._save_cassette(key, response)
        _logger.bind(
            prompt_version=prompt_version,
            model=model,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            trace_id=current_trace_id(),
        ).info("llm_call_async")
        return response

    # ---- cassette I/O ----

    def _cassette_path(self, key: str) -> Path:
        return self._cassette_dir / f"{key}.json"

    def _load_cassette(self, key: str) -> dict[str, Any] | None:
        path = self._cassette_path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise CassetteMiss(f"Cassette at {path} is malformed (not a JSON object).")
        return data

    def _save_cassette(self, key: str, response: LLMResponse) -> None:
        path = self._cassette_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(response.as_payload(), fh, indent=2, sort_keys=True)

    def _response_from_cassette(self, payload: dict[str, Any], key: str) -> LLMResponse:
        return LLMResponse(
            text=str(payload["text"]),
            model=str(payload["model"]),
            prompt_version=str(payload["prompt_version"]),
            input_tokens=int(payload["input_tokens"]),
            output_tokens=int(payload["output_tokens"]),
            cost_usd=float(payload["cost_usd"]),
            cached=True,
            cassette_key=key,
        )


def _extract_text(api_response: Any) -> str:
    """Concatenate the text from each text block of an Anthropic response.

    The SDK returns a list of content blocks of different types; agent code
    only consumes ``text`` blocks for now. Tool-use blocks land in Phase 5+
    when specialists call out to deterministic helpers.
    """
    parts: list[str] = []
    for block in api_response.content or []:
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "".join(parts)


_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the process-wide :class:`LLMClient` (lazy singleton)."""
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton


def reset_llm_client() -> None:
    """Drop the cached client so the next call rebuilds it.

    Tests use this when they need to swap settings or inject a stub Anthropic
    client; production code should never need it.
    """
    global _singleton
    _singleton = None
