"""Tests for the :class:`AgentState` contract and :class:`StateUpdate` ownership."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models.state import (
    AgentState,
    CriticVerdict,
    FilingEvent,
    FilingForm,
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
