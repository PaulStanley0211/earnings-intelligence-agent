"""Tests for the :class:`AgentState` contract and :class:`StateUpdate` ownership."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models.state import (
    AgentState,
    AnswerClass,
    CommitmentExtracted,
    CommitmentStatus,
    CommitmentStatusUpdate,
    CriticVerdict,
    FilingEvent,
    FilingForm,
    QAPairPayload,
    StateUpdate,
)


def _build_state() -> AgentState:
    return AgentState(
        trace_id="trace-1",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="0000950170-25-000001",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime.now(UTC),
            source_url="https://www.sec.gov/some/index.htm",
        ),
    )


def test_minimal_agent_state_constructs() -> None:
    state = _build_state()
    assert state.cost_usd == 0.0
    assert state.plan == []
    assert state.critic_findings == []
    assert state.final_note is None


def test_planner_update_applies_changes() -> None:
    state = _build_state()
    update = StateUpdate(
        owner="planner",
        changes={"plan": ["financial_extractor", "comparator"], "cost_usd": 0.21},
    )
    new_state = update.apply(state)
    assert new_state.plan == ["financial_extractor", "comparator"]
    assert new_state.cost_usd == 0.21
    # Original state must not be mutated.
    assert state.plan == []


def test_cost_is_accumulated_not_overwritten() -> None:
    state = _build_state().model_copy(update={"cost_usd": 0.5})
    update = StateUpdate(owner="synthesizer", changes={"cost_usd": 0.3, "draft_note": "x"})
    new_state = update.apply(state)
    assert new_state.cost_usd == pytest.approx(0.8)
    assert new_state.draft_note == "x"


def test_update_rejects_field_outside_owners_allowlist() -> None:
    with pytest.raises(ValidationError):
        StateUpdate(owner="planner", changes={"draft_note": "not yours"})


def test_update_rejects_unknown_owner() -> None:
    with pytest.raises(ValidationError):
        StateUpdate(owner="not_a_real_node", changes={"cost_usd": 0.0})


def test_critic_can_set_final_note_and_verdict() -> None:
    state = _build_state().model_copy(update={"draft_note": "draft"})
    update = StateUpdate(
        owner="critic",
        changes={
            "final_note": "final",
            "critic_verdict": CriticVerdict.ACCEPTED,
            "critic_attempts": 1,
        },
    )
    new_state = update.apply(state)
    assert new_state.final_note == "final"
    assert new_state.critic_verdict is CriticVerdict.ACCEPTED
    assert new_state.critic_attempts == 1


def test_filing_event_is_immutable() -> None:
    event = _build_state().filing_event
    with pytest.raises(ValidationError):
        event.ticker = "NVDA"  # type: ignore[misc]


def test_language_differ_owns_language_diffs() -> None:
    update = StateUpdate(
        owner="language_differ",
        changes={"language_diffs": [{"section": "mda", "diffs": []}]},
    )
    assert update.changes["language_diffs"][0]["section"] == "mda"


def test_language_differ_cannot_mutate_comparisons() -> None:
    with pytest.raises(ValidationError):
        StateUpdate(owner="language_differ", changes={"comparisons": {}})


def test_filing_event_defaults_to_watcher_source() -> None:
    """Existing code that builds FilingEvent without ``source`` keeps working."""
    from datetime import UTC, datetime

    from app.models.state import FilingEvent, FilingEventSource, FilingForm

    event = FilingEvent(
        accession_number="0001193125-26-027198",
        cik="0000789019",
        ticker="MSFT",
        form=FilingForm.FORM_8K,
        filed_at=datetime(2026, 1, 28, tzinfo=UTC),
        source_url="https://example.com",
    )
    assert event.source is FilingEventSource.WATCHER


def test_filing_event_accepts_upload_source() -> None:
    from datetime import UTC, datetime

    from app.models.state import FilingEvent, FilingEventSource, FilingForm

    event = FilingEvent(
        accession_number="upload-001",
        cik="0000789019",
        ticker="MSFT",
        form=FilingForm.FORM_8K,
        filed_at=datetime(2026, 1, 28, tzinfo=UTC),
        source_url="https://example.com",
        source=FilingEventSource.UPLOAD,
    )
    assert event.source is FilingEventSource.UPLOAD


# ---- Phase 4B: transcript analyzer state fields ----


def _build_qa_pair(ordinal: int = 1) -> QAPairPayload:
    return QAPairPayload(
        ordinal=ordinal,
        analyst_name="Analyst A",
        question_text="q",
        answer_text="a",
        answer_class=AnswerClass.DIRECT,
        sha256_text="0" * 64,
    )


def test_phase4b_fields_default_empty() -> None:
    """A freshly built AgentState has empty qa_pairs / commitments / commitment_updates."""
    state = _build_state()
    assert state.qa_pairs == []
    assert state.commitments == []
    assert state.commitment_updates == []


def test_qa_pairs_applied_by_transcript_analyzer() -> None:
    state = _build_state()
    update = StateUpdate(
        owner="transcript_analyzer",
        changes={"qa_pairs": [_build_qa_pair()]},
    )
    new_state = update.apply(state)
    assert len(new_state.qa_pairs) == 1
    assert new_state.qa_pairs[0].ordinal == 1
    assert new_state.qa_pairs[0].answer_class is AnswerClass.DIRECT


def test_commitments_applied_by_transcript_analyzer() -> None:
    state = _build_state()
    commitment = CommitmentExtracted(
        commitment_text="We will launch X in Q3 2026.",
        target_period="Q3 2026",
        source_quote="we will launch X in Q3 2026",
    )
    update = StateUpdate(
        owner="transcript_analyzer",
        changes={"commitments": [commitment]},
    )
    new_state = update.apply(state)
    assert len(new_state.commitments) == 1
    assert new_state.commitments[0].target_period == "Q3 2026"


def test_commitment_updates_applied_by_transcript_analyzer() -> None:
    state = _build_state()
    update_payload = CommitmentStatusUpdate(
        commitment_id=42,
        new_status=CommitmentStatus.MET,
        reason="Management confirmed launch.",
    )
    update = StateUpdate(
        owner="transcript_analyzer",
        changes={"commitment_updates": [update_payload]},
    )
    new_state = update.apply(state)
    assert len(new_state.commitment_updates) == 1
    assert new_state.commitment_updates[0].commitment_id == 42
    assert new_state.commitment_updates[0].new_status is CommitmentStatus.MET


@pytest.mark.parametrize(
    ("owner", "field", "payload"),
    [
        ("comparator", "qa_pairs", [_build_qa_pair()]),
        (
            "synthesizer",
            "commitments",
            [
                CommitmentExtracted(
                    commitment_text="t", target_period=None, source_quote="q"
                )
            ],
        ),
        (
            "language_differ",
            "commitment_updates",
            [
                CommitmentStatusUpdate(
                    commitment_id=1,
                    new_status=CommitmentStatus.STILL_OPEN,
                    reason="r",
                )
            ],
        ),
    ],
)
def test_phase4b_fields_rejected_for_other_owners(
    owner: str,
    field: str,
    payload: object,
) -> None:
    with pytest.raises(ValidationError):
        StateUpdate(owner=owner, changes={field: payload})


def test_phase4b_payloads_are_frozen() -> None:
    """In-state payload models are immutable."""
    pair = _build_qa_pair()
    commitment = CommitmentExtracted(
        commitment_text="t", target_period=None, source_quote="q"
    )
    status_update = CommitmentStatusUpdate(
        commitment_id=1, new_status=CommitmentStatus.OPEN, reason="r"
    )
    with pytest.raises(ValidationError):
        pair.ordinal = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        commitment.target_period = "x"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        status_update.commitment_id = 99  # type: ignore[misc]
