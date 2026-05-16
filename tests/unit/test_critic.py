"""Unit tests for :mod:`app.agents.critic`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.agents.critic import _MAX_CRITIC_ATTEMPTS, OWNER, critique_draft
from app.models.state import (
    AgentState,
    CriticFinding,
    CriticVerdict,
    FilingEvent,
    FilingForm,
)


def _state(
    *,
    draft: str | None,
    financials: dict[str, Any] | None = None,
    comparisons: dict[str, Any] | None = None,
    attempts: int = 0,
    findings: list[CriticFinding] | None = None,
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
        financials=financials,
        comparisons=comparisons,
        draft_note=draft,
        critic_attempts=attempts,
        critic_findings=findings or [],
    )


def _financials() -> dict[str, Any]:
    return {
        "by_concept": {
            "EarningsPerShareDiluted": [
                {
                    "value": "1.32",
                    "unit": "USD/shares",
                    "period_start": "2026-01-01",
                    "period_end": "2026-03-31",
                    "fiscal_year": 2026,
                    "fiscal_period": "Q3",
                }
            ],
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
    }


def _comparisons() -> dict[str, Any]:
    return {
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
    }


def test_accepts_well_cited_note() -> None:
    # Citation index (built from the same helpers):
    #   F1 = EarningsPerShareDiluted, F2 = Revenues
    #   C1 = revenue comparison
    draft = (
        "## Headline\n"
        "Microsoft reported diluted EPS of $1.32 [F1] on revenue of "
        "$61.9 billion [F2].\n\n"
        "## Numbers\n"
        "- Diluted EPS: $1.32 [F1]\n"
        "- Revenue: $61.9 billion [F2]\n\n"
        "## Versus consensus\n"
        "- Revenue beat consensus of $61,000,000,000 [C1] by 1.4066% [C1]\n"
    )
    state = _state(draft=draft, financials=_financials(), comparisons=_comparisons())
    update = critique_draft(state)
    assert update.owner == OWNER
    assert update.changes["critic_verdict"] is CriticVerdict.ACCEPTED
    assert update.changes["final_note"] == draft
    assert update.changes["critic_findings"] == []
    assert update.changes["critic_attempts"] == 1


def test_rejects_when_a_number_is_uncited() -> None:
    draft = (
        "## Headline\n"
        "Microsoft reported $1.32 [F1] and revenue grew 12% from prior year.\n"
    )
    state = _state(draft=draft, financials=_financials(), comparisons=_comparisons())
    update = critique_draft(state)
    assert update.changes["critic_verdict"] is CriticVerdict.REJECTED
    findings = update.changes["critic_findings"]
    assert any("12%" in f.message for f in findings)


def test_rejects_when_cited_value_mismatches() -> None:
    draft = (
        "## Headline\n"
        "Diluted EPS came in at $2.50 [F1].\n"
    )
    state = _state(draft=draft, financials=_financials(), comparisons=_comparisons())
    update = critique_draft(state)
    assert update.changes["critic_verdict"] is CriticVerdict.REJECTED
    findings = update.changes["critic_findings"]
    assert any("F1" in f.message and "tolerance" in f.message for f in findings)


def test_rejects_when_citation_id_is_unknown() -> None:
    draft = "## Headline\nRevenue $61.9 billion [F99].\n"
    state = _state(draft=draft, financials=_financials(), comparisons=_comparisons())
    update = critique_draft(state)
    assert update.changes["critic_verdict"] is CriticVerdict.REJECTED
    findings = update.changes["critic_findings"]
    assert any("F99" in f.message for f in findings)


def test_third_rejection_emits_loop_exceeded() -> None:
    state = _state(
        draft="$99 million in pure invention.",  # uncited "$99 million"
        financials=_financials(),
        comparisons=_comparisons(),
        attempts=_MAX_CRITIC_ATTEMPTS - 1,
    )
    update = critique_draft(state)
    assert update.changes["critic_attempts"] == _MAX_CRITIC_ATTEMPTS
    assert update.changes["critic_verdict"] is CriticVerdict.LOOP_EXCEEDED


def test_missing_draft_note_is_an_error() -> None:
    state = _state(draft=None, financials=_financials(), comparisons=_comparisons())
    update = critique_draft(state)
    assert update.changes["critic_verdict"] is CriticVerdict.REJECTED
    findings = update.changes["critic_findings"]
    assert findings[0].message.startswith("critic invoked with no draft")


def test_accepts_percentage_surprise_against_comparison() -> None:
    # The comparator stored surprise_pct=1.4066 for the revenue beat. The
    # synthesiser is allowed to round to two places.
    draft = "Revenue beat consensus by 1.41% [C1]."
    state = _state(draft=draft, financials=_financials(), comparisons=_comparisons())
    update = critique_draft(state)
    assert update.changes["critic_verdict"] is CriticVerdict.ACCEPTED


@pytest.mark.parametrize(
    "draft",
    [
        "Filed: 2026-04-25",  # bare integers are not flagged
        "Section 1: numbers",
    ],
)
def test_bare_integers_are_not_flagged_as_uncited(draft: str) -> None:
    state = _state(draft=draft, financials=_financials(), comparisons=_comparisons())
    update = critique_draft(state)
    assert update.changes["critic_verdict"] is CriticVerdict.ACCEPTED


def test_critic_accepts_valid_language_citation() -> None:
    state = AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="x",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime.now(UTC),
            source_url="https://www.sec.gov/x",
        ),
        language_diffs=[
            {
                "section": "mda",
                "diffs": [
                    {
                        "change_type": "modified",
                        "current_text": (
                            "Operating expenses rose substantially as we "
                            "accelerated AI infrastructure investment."
                        ),
                        "prior_text": "Operating expenses rose modestly.",
                        "similarity": "0.7421",
                        "severity": "major",
                    },
                ],
            }
        ],
        draft_note=(
            "## Headline\n"
            "MSFT updated guidance.\n"
            "## Language changes\n"
            "- Operating expenses rose substantially as we accelerated AI "
            "infrastructure investment [L1].\n"
        ),
    )
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert not any(f.severity == "error" and "L1" in f.message for f in findings)


def test_critic_rejects_l_citation_that_does_not_match_indexed_text() -> None:
    state = AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="x",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime.now(UTC),
            source_url="https://www.sec.gov/x",
        ),
        language_diffs=[
            {
                "section": "mda",
                "diffs": [
                    {
                        "change_type": "added",
                        "text": "A new geopolitical risk could affect international sales.",
                        "severity": "major",
                    },
                ],
            }
        ],
        draft_note=(
            "## Language changes\n"
            "- We are pivoting to a subscription-only business model [L1].\n"
        ),
    )
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert any(f.severity == "error" and "L1" in f.message for f in findings)


def test_critic_rejects_l_citation_with_no_matching_index() -> None:
    state = AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="x",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime.now(UTC),
            source_url="https://www.sec.gov/x",
        ),
        draft_note=(
            "## Language changes\n"
            "- A made-up quote [L7].\n"
        ),
    )
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert any(f.severity == "error" and "L7" in f.message for f in findings)
