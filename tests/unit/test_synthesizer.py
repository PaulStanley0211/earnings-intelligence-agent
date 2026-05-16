"""Unit tests for :mod:`app.agents.synthesizer`.

The synthesiser is exercised against a stub Anthropic client so the test
runs offline. The cassette path is covered separately by
``tests/unit/test_llm_client.py``; here we focus on the prompt rendering
and the StateUpdate the node emits.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agents.citations import LanguageCitation
from app.agents.synthesizer import OWNER, _render_language_block, synthesize_note
from app.llm.client import LLMClient
from app.models.state import (
    AgentState,
    CriticFinding,
    FilingEvent,
    FilingForm,
)


class _StubRepository:
    """Implements the :class:`app.llm.client._SupportsDailySpend` protocol."""

    def __init__(self) -> None:
        self.spent: dict[date, Decimal] = {}

    async def get_daily_spend(self, day: date) -> Decimal:
        return self.spent.get(day, Decimal("0"))

    async def add_daily_spend(self, *, day: date, amount_usd: Decimal) -> Decimal:
        self.spent[day] = self.spent.get(day, Decimal("0")) + amount_usd
        return self.spent[day]


class _StubAnthropic:
    def __init__(self, text: str) -> None:
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_system: str | None = None
        self._text = text

    @property
    def messages(self) -> Any:
        outer = self

        class _MessagesNamespace:
            def create(self, **kwargs: Any) -> Any:
                outer.last_messages = list(kwargs["messages"])
                outer.last_system = kwargs.get("system")
                return MagicMock(
                    content=[MagicMock(type="text", text=outer._text)],
                    usage=MagicMock(input_tokens=100, output_tokens=50),
                )

        return _MessagesNamespace()


def _state(
    *,
    findings: list[CriticFinding] | None = None,
    attempts: int = 0,
) -> AgentState:
    return AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="0000950170-26-000050",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
            source_url="https://www.sec.gov/...",
        ),
        financials={
            "by_concept": {
                "Revenues": [
                    {
                        "value": "61858000000",
                        "unit": "USD",
                        "period_start": "2026-01-01",
                        "period_end": "2026-03-31",
                        "fiscal_year": 2026,
                        "fiscal_period": "Q3",
                    }
                ],
            }
        },
        comparisons={
            "fiscal_year": 2026,
            "fiscal_period": "Q3",
            "period_end": "2026-03-31",
            "consensus_source": "finnhub",
            "degraded": False,
            "metrics": [
                {
                    "metric": "revenue",
                    "reported_value": "61858000000",
                    "reported_unit": "USD",
                    "consensus_value": "61000000000",
                    "consensus_source": "finnhub",
                    "surprise_abs": "858000000",
                    "surprise_pct": "1.4066",
                    "direction": "beat",
                }
            ],
        },
        critic_findings=findings or [],
        critic_attempts=attempts,
    )


@pytest.fixture()
def llm_for_synth(
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LLMClient, _StubAnthropic]:
    monkeypatch.setenv("REC", "1")  # let the stub respond and write a cassette
    stub = _StubAnthropic(text="## Headline\nRevenue $61.9 billion [F1].\n")
    client = LLMClient(cassette_dir=cassette_dir, anthropic_client=stub)  # type: ignore[arg-type]
    return client, stub


async def test_synthesize_emits_draft_and_increments_cost(
    llm_for_synth: tuple[LLMClient, _StubAnthropic],
) -> None:
    llm, stub = llm_for_synth
    repo = _StubRepository()
    update = await synthesize_note(_state(), llm=llm, repository=repo)
    assert update.owner == OWNER
    assert "draft_note" in update.changes
    assert update.changes["draft_note"].startswith("## Headline")
    assert update.changes["cost_usd"] > 0
    assert stub.last_messages is not None
    [user_msg] = stub.last_messages
    assert user_msg["role"] == "user"
    assert "<source>" in user_msg["content"]
    assert "[F1] Revenues" in user_msg["content"]
    assert "[C1] revenue" in user_msg["content"]
    # Daily spend was committed to the repository.
    assert any(amount > 0 for amount in repo.spent.values())


async def test_synthesize_includes_critic_feedback_on_retry(
    llm_for_synth: tuple[LLMClient, _StubAnthropic],
) -> None:
    llm, stub = llm_for_synth
    state = _state(
        findings=[
            CriticFinding(layer="numbers", severity="error", message="bad [F99]"),
        ],
        attempts=1,
    )
    await synthesize_note(state, llm=llm, repository=_StubRepository())
    assert stub.last_messages is not None
    body = stub.last_messages[0]["content"]
    assert "Previous critic findings" in body
    assert "bad [F99]" in body


def test_synthesizer_renders_language_diffs_block_when_present() -> None:
    """_render_language_block emits an [L#] entry for each citation."""
    citations = [
        LanguageCitation(
            identifier="L1",
            section="mda",
            change_type="modified",
            text=(
                "Operating expenses rose substantially as we accelerated"
                " AI infrastructure investment."
            ),
            severity="major",
        ),
        LanguageCitation(
            identifier="L2",
            section="risk_factors",
            change_type="added",
            text="A new geopolitical risk could affect international sales.",
            severity="major",
        ),
    ]
    rendered = _render_language_block(citations)
    assert "[L1]" in rendered
    assert "operating expenses rose substantially" in rendered.lower()
    assert "[L2]" in rendered


def test_synthesizer_renders_no_language_changes_message_when_empty() -> None:
    """_render_language_block returns a human-readable fallback for empty input."""
    assert "no language changes" in _render_language_block([]).lower()
