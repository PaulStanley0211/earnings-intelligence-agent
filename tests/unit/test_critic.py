"""Unit tests for :mod:`app.agents.critic`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.agents.critic import _MAX_CRITIC_ATTEMPTS, OWNER, critique_draft
from app.models.state import (
    AgentState,
    AnswerClass,
    CommitmentExtracted,
    CriticFinding,
    CriticVerdict,
    FilingEvent,
    FilingForm,
    QAPairPayload,
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


def _transcript_state(
    *,
    draft: str,
    qa_pairs: list[QAPairPayload] | None = None,
    commitments: list[CommitmentExtracted] | None = None,
) -> AgentState:
    return AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="0000950170-26-000050",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.TRANSCRIPT,
            filed_at=datetime.now(UTC),
            source_url="https://ir.example.com/transcript.pdf",
        ),
        qa_pairs=qa_pairs or [],
        commitments=commitments or [],
        draft_note=draft,
    )


def test_critic_accepts_qa_citation_within_tolerance() -> None:
    """A quoted Q&A phrase that resolves into the indexed answer is accepted."""
    qa = QAPairPayload(
        ordinal=1,
        analyst_name="Brent Thill",
        question_text="How should we think about Azure margins next quarter?",
        answer_text="We expect margins to remain stable around 47 percent.",
        answer_class=AnswerClass.DIRECT,
        sha256_text="a" * 64,
    )
    draft = (
        "## Q&A signals\n"
        "- We expect margins to remain stable around 47 percent [Q1].\n"
    )
    state = _transcript_state(draft=draft, qa_pairs=[qa])
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert not any(f.severity == "error" and "Q1" in f.message for f in findings)


def test_critic_rejects_qa_citation_with_low_similarity() -> None:
    """A quoted phrase that does not appear in the cited Q&A is rejected."""
    qa = QAPairPayload(
        ordinal=1,
        analyst_name="Brent Thill",
        question_text="How are Azure margins trending?",
        answer_text="Hard to forecast precisely; we will update next quarter.",
        answer_class=AnswerClass.DEFLECTED,
        sha256_text="a" * 64,
    )
    draft = (
        "## Q&A signals\n"
        "- We are pivoting away from cloud entirely and refocusing on PCs [Q1].\n"
    )
    state = _transcript_state(draft=draft, qa_pairs=[qa])
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert any(f.severity == "error" and "Q1" in f.message for f in findings)


def test_critic_accepts_commitment_citation() -> None:
    """A draft phrase matching a commitment's source_quote is accepted."""
    commitment = CommitmentExtracted(
        commitment_text="Azure margin expansion of 100 basis points next quarter.",
        target_period="Q3 2026",
        source_quote="we expect Azure margin expansion of 100 basis points next quarter",
    )
    draft = (
        "## Commitments\n"
        "- we expect Azure margin expansion of 100 basis points next quarter [K1].\n"
    )
    state = _transcript_state(draft=draft, commitments=[commitment])
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert not any(f.severity == "error" and "K1" in f.message for f in findings)


def test_critic_rejects_unresolved_qa_citation() -> None:
    """A [Q#] that points past the end of the qa_pairs list is rejected."""
    qa = QAPairPayload(
        ordinal=1,
        analyst_name=None,
        question_text="q",
        answer_text="a",
        answer_class=AnswerClass.DIRECT,
        sha256_text="a" * 64,
    )
    draft = "## Q&A signals\n- Mystery phrase [Q99].\n"
    state = _transcript_state(draft=draft, qa_pairs=[qa])
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert any(f.severity == "error" and "Q99" in f.message for f in findings)


def test_critic_rejects_unresolved_commitment_citation() -> None:
    """A [K#] without a backing commitment is rejected."""
    draft = "## Commitments\n- A made-up promise [K3].\n"
    state = _transcript_state(draft=draft)
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert any(f.severity == "error" and "K3" in f.message for f in findings)


def test_critic_resolves_unknown_p_citation() -> None:
    """When the synthesizer cites [P0] but peer_context is empty, the critic
    must flag the citation as unknown."""
    from app.models.state import FilingEventSource

    state = AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 1, 1, tzinfo=UTC),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
        draft_note='Peer says "growth strong" [P0].',
        peer_context=[],
    )
    update = critique_draft(state)
    assert any(
        f.severity == "error" and "P0" in f.message
        for f in update.changes["critic_findings"]
    )


def test_critic_resolves_known_p_citation() -> None:
    """A [P#] citation that resolves against peer_context with matching text is accepted."""
    from app.models.state import FilingEventSource, PeerContextEntry

    state = AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 1, 1, tzinfo=UTC),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
        # Draft cites the peer text verbatim so the 90% similarity check passes.
        draft_note="Cloud pricing pressure intensified during the quarter [P0].",
        peer_context=[
            PeerContextEntry(
                peer_ticker="GOOGL",
                kind="language_diff",
                text="Cloud pricing pressure intensified during the quarter.",
                source_filing_accession="x-1",
                severity="major",
            )
        ],
    )
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    # No error finding referencing P0 should remain.
    assert all(f.severity != "error" or "P0" not in f.message for f in findings)


def test_language_match_uses_quoted_substring_when_line_has_quotes() -> None:
    """Editorial framing around a quoted phrase must not fail the match."""
    from app.agents.critic import _language_match

    quoted_line = 'Sarah Lee asked "what is the cloud margin outlook for Q3"'
    indexed = "What is the cloud margin outlook for Q3? Is it on the high end?"

    assert _language_match(quoted_line, indexed) is True


def test_language_match_falls_back_to_full_line_without_quotes() -> None:
    from app.agents.critic import _language_match

    line = "cloud margin outlook for Q3"
    indexed = "What is the cloud margin outlook for Q3? Is it on the high end?"

    assert _language_match(line, indexed) is True


def test_language_match_rejects_wrong_quoted_substring() -> None:
    from app.agents.critic import _language_match

    quoted_line = 'Analyst said "earnings will collapse to zero"'
    indexed = "We anticipate solid margin expansion."

    assert _language_match(quoted_line, indexed) is False


def test_numbers_in_language_cited_lines_are_not_flagged_as_uncited() -> None:
    """Numbers inside a [K#]-cited quote must not be flagged as uncited figures.

    The synthesizer quotes commitment source text verbatim, e.g.
    '"revenue guidance of 2.55 to 2.60 billion dollars" [K2]'.
    Those numbers are numeric evidence, not independent financial claims;
    ``_language_cited_line_spans`` marks the whole line as covered so
    ``_find_uncited`` does not raise an error on them.
    """
    from app.models.state import CommitmentExtracted

    commitment = CommitmentExtracted(
        commitment_text="Revenue guidance of 2.55 to 2.60 billion.",
        target_period="FY2026",
        source_quote=(
            "reiterating our previously communicated full-year fiscal 2026 "
            "revenue guidance of 2.55 to 2.60 billion dollars"
        ),
    )
    draft = (
        "## Commitments\n"
        '- Management said "reiterating our previously communicated full-year '
        "fiscal 2026 revenue guidance of 2.55 to 2.60 billion dollars\" [K1].\n"
    )
    state = _transcript_state(draft=draft, commitments=[commitment])
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    # The 2.55 / 2.60 billion values must not be flagged as uncited numbers.
    assert not any("2.60" in f.message or "2.55" in f.message for f in findings), (
        f"Numbers inside a [K#]-cited line must not be flagged; got: {findings}"
    )
