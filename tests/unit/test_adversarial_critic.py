"""Phase 5c gate: critic catches >=27/30 adversarial notes.

The deterministic critic (``critique_draft``) handles numeric and quote errors.
The LLM critic (``llm_critique``) is invoked only for the 5 ``contradicted_direction``
cases that the deterministic critic accepts, to check for semantic direction-flip errors.

Cassette policy
---------------
The 5 ``contradicted_direction`` cases require real Opus calls to catch the
direction-flip. Cassettes live under ``tests/fixtures/cassettes/adversarial_critic/``.
Record with ``REC=1 pytest tests/unit/test_adversarial_critic.py``. Subsequent
runs replay from cassettes and make no API calls.

When recording (``REC=1``), the LLM client reads the real ``ANTHROPIC_API_KEY``
from the project ``.env`` file so the test conftest placeholder does not block
the live API call.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import Anthropic

from app.agents.critic import critique_draft
from app.agents.llm_critic import llm_critique
from app.llm.client import LLMClient, _DailyCostTracker
from app.models.state import AgentState, CriticVerdict

ADV_DIR = (
    Path(__file__).parent.parent / "fixtures" / "adversarial_notes" / "perturbed"
)
CASSETTE_DIR = (
    Path(__file__).parent.parent / "fixtures" / "cassettes" / "adversarial_critic"
)

# Absolute path to the project root .env file with the real API key.
_DOTENV = Path(__file__).parent.parent.parent / ".env"

# A generous daily cap so recording 5 Opus cassettes does not trip the guard.
_RECORDING_CAP_USD = 25.0


def _all_adversarial() -> list[dict[str, Any]]:
    """Load and return all 30 perturbed adversarial note JSONs, sorted by path."""
    return [json.loads(p.read_text()) for p in sorted(ADV_DIR.glob("*.json"))]


def _state_from_snapshot(note_markdown: str, snapshot: dict[str, Any]) -> AgentState:
    """Rebuild an :class:`AgentState` from a stored snapshot + the perturbed note."""
    return AgentState.model_validate({**snapshot, "draft_note": note_markdown})


def _read_dotenv_key(key: str) -> str | None:
    """Read a single key from the project .env file without importing dotenv.

    Returns ``None`` when the file is absent or the key is not present.
    Strips surrounding whitespace and quotes from the value.
    """
    if not _DOTENV.exists():
        return None
    for line in _DOTENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        lhs, _, rhs = line.partition("=")
        if lhs.strip() == key:
            return rhs.strip().strip('"').strip("'")
    return None


def _make_llm_client() -> LLMClient:
    """Construct an :class:`LLMClient` wired to the adversarial-critic cassette dir.

    When recording (``REC=1``), the real ``ANTHROPIC_API_KEY`` from the project
    ``.env`` is injected directly so the test-conftest placeholder does not block
    the live API call. Replay runs use whatever key is in the environment (it is
    never sent to Anthropic during cassette replay).

    Uses a generous in-process cost tracker so recording 5 Opus calls does not
    trip the default $1.00 test cap.
    """
    tracker = _DailyCostTracker(cap_usd=_RECORDING_CAP_USD)
    recording = os.environ.get("REC") == "1"
    if recording:
        real_key = _read_dotenv_key("ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
        anthropic_client = Anthropic(api_key=real_key)
        return LLMClient(
            cassette_dir=CASSETTE_DIR,
            anthropic_client=anthropic_client,
            cost_tracker=tracker,
        )
    return LLMClient(
        cassette_dir=CASSETTE_DIR,
        cost_tracker=tracker,
    )


def _make_repository_stub() -> MagicMock:
    """Return a stub satisfying the ``_SupportsDailySpend`` protocol.

    ``get_daily_spend`` always reports $0.00 spent so the cost guard never
    blocks a recording run. ``add_daily_spend`` is a no-op.
    """
    repo = MagicMock()
    repo.get_daily_spend = AsyncMock(return_value=Decimal("0.00"))
    repo.add_daily_spend = AsyncMock(return_value=Decimal("0.00"))
    return repo


@pytest.mark.asyncio
async def test_adversarial_critic_catches_at_least_27_of_30() -> None:
    """Phase 5c gate: deterministic + LLM critic together catch >=27/30 seeded errors."""
    cases = _all_adversarial()
    assert len(cases) == 30, f"expected 30 perturbed cases, got {len(cases)}"

    llm = _make_llm_client()
    repo = _make_repository_stub()

    caught = 0
    misses: list[str] = []

    for case in cases:
        note_markdown: str = case["note_markdown"]
        snapshot: dict[str, Any] = case["state_snapshot"]
        base_stem: str = case["base_note_stem"]
        perturbation: str = case["perturbation"]

        state = _state_from_snapshot(note_markdown, snapshot)
        det_update = critique_draft(state)
        det_errors = [
            f
            for f in det_update.changes["critic_findings"]
            if f.severity == "error"
        ]
        if det_errors:
            caught += 1
            continue

        # Deterministic critic accepted - invoke LLM critic for semantic check.
        # Build a state that satisfies the LLM critic's entry guard:
        # ``critic_verdict == ACCEPTED`` and ``final_note`` populated.
        accepted_state = state.model_copy(
            update={
                "critic_verdict": CriticVerdict.ACCEPTED,
                "final_note": note_markdown,
                "critic_attempts": 1,
            }
        )
        llm_update = await llm_critique(accepted_state, llm=llm, repository=repo)
        llm_errors = [
            f
            for f in llm_update.changes.get("critic_findings", [])
            if f.severity == "error"
        ]
        if llm_errors:
            caught += 1
        else:
            misses.append(f"{base_stem}::{perturbation}")

    assert caught >= 27, (
        f"critic caught only {caught}/30 adversarial cases; "
        f"misses: {misses}"
    )
