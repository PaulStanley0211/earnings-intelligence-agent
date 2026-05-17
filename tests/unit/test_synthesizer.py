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

from app.agents.citations import CommitmentCitation, LanguageCitation, QACitation
from app.agents.synthesizer import (
    _PROMPT_FULL,
    OWNER,
    _render_commitments_block,
    _render_language_block,
    _render_qa_pairs_block,
    _select_prompt,
    synthesize_note,
)
from app.llm.client import LLMClient
from app.models.state import (
    AgentState,
    AnswerClass,
    CommitmentExtracted,
    CriticFinding,
    FilingEvent,
    FilingForm,
    QAPairPayload,
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


def _qa_pair(ordinal: int, question: str, answer: str) -> QAPairPayload:
    return QAPairPayload(
        ordinal=ordinal,
        analyst_name="Brent Thill",
        question_text=question,
        answer_text=answer,
        answer_class=AnswerClass.DIRECT,
        sha256_text="a" * 64,
    )


def test_select_prompt_prefers_full_when_transcript_present() -> None:
    """_select_prompt returns the full prompt as soon as QA or commitments exist."""
    state = _state()
    assert _select_prompt(state) == "synthesizer/numbers_v1"

    state_with_lang = _state()
    state_with_lang = state_with_lang.model_copy(
        update={"language_diffs": [{"section": "mda", "diffs": []}]}
    )
    assert _select_prompt(state_with_lang) == "synthesizer/numbers_with_language_v1"

    state_with_qa = _state().model_copy(
        update={
            "qa_pairs": [_qa_pair(1, "q", "a")],
        }
    )
    assert _select_prompt(state_with_qa) == _PROMPT_FULL

    state_with_commitment = _state().model_copy(
        update={
            "commitments": [
                CommitmentExtracted(
                    commitment_text="c",
                    target_period="Q3 2026",
                    source_quote="verbatim quote that anchors the commitment",
                )
            ]
        }
    )
    assert _select_prompt(state_with_commitment) == _PROMPT_FULL


def test_synthesizer_renders_qa_pairs_block() -> None:
    """_render_qa_pairs_block emits a Q#/A# pair per citation."""
    citations = [
        QACitation(
            identifier="Q1",
            ordinal=1,
            analyst_name="Brent Thill",
            question_text="How should we think about Azure margins next quarter?",
            answer_text="We expect margins to remain stable around 47 percent.",
            answer_class="direct",
        ),
        QACitation(
            identifier="Q2",
            ordinal=2,
            analyst_name=None,
            question_text="FX headwinds?",
            answer_text="Two points next quarter.",
            answer_class="partial",
        ),
    ]
    rendered = _render_qa_pairs_block(citations)
    assert "Q1 (analyst: Brent Thill):" in rendered
    assert "A1 [direct]:" in rendered
    assert "Q2 (analyst: unknown):" in rendered
    assert "A2 [partial]:" in rendered


def test_synthesizer_renders_commitments_block() -> None:
    """_render_commitments_block emits one K# line per commitment."""
    citations = [
        CommitmentCitation(
            identifier="K1",
            commitment_text="Azure margin expansion of 100 bps next quarter.",
            target_period="Q3 2026",
            source_quote=(
                "we expect Azure margin expansion of 100 basis points next quarter"
            ),
        ),
        CommitmentCitation(
            identifier="K2",
            commitment_text="Operating margin will reach 45 percent for the full year.",
            target_period=None,
            source_quote="operating margin will reach 45 percent for the full year",
        ),
    ]
    rendered = _render_commitments_block(citations)
    assert "K1 (target: Q3 2026):" in rendered
    assert "K2 (target: not specified):" in rendered
    assert '(source: "operating margin will reach' in rendered


def test_render_blocks_emit_fallback_when_empty() -> None:
    """Empty inputs produce a human-readable placeholder string."""
    assert "no analyst" in _render_qa_pairs_block([]).lower()
    assert "no management" in _render_commitments_block([]).lower()


async def test_synthesizer_picks_full_v1_when_transcript_data_present(
    llm_for_synth: tuple[LLMClient, _StubAnthropic],
) -> None:
    """When qa_pairs or commitments exist the full prompt is selected and rendered."""
    llm, stub = llm_for_synth
    base = _state()
    state = base.model_copy(
        update={
            "qa_pairs": [
                _qa_pair(
                    1,
                    "How should we think about Azure margins next quarter?",
                    "We expect margins to remain stable around 47 percent.",
                )
            ],
            "commitments": [
                CommitmentExtracted(
                    commitment_text=(
                        "Azure margin expansion of 100 basis points next quarter."
                    ),
                    target_period="Q3 2026",
                    source_quote=(
                        "we expect Azure margin expansion of 100 basis points next quarter"
                    ),
                )
            ],
        }
    )
    await synthesize_note(state, llm=llm, repository=_StubRepository())
    assert stub.last_messages is not None
    body = stub.last_messages[0]["content"]
    assert 'source type="qa_pairs"' in body
    assert 'source type="commitments"' in body
    assert "Q1 (analyst: Brent Thill):" in body
    assert "K1 (target: Q3 2026):" in body
