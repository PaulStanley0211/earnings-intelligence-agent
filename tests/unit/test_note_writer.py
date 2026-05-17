"""Unit tests for the note_writer agent node."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.note_writer import OWNER, write_note
from app.models.state import (
    AgentState,
    CriticVerdict,
    FilingEvent,
    FilingEventSource,
    FilingForm,
)


def _state(*, verdict: CriticVerdict, final_note: str | None) -> AgentState:
    return AgentState(
        trace_id="t-1",
        started_at=datetime(2025, 4, 15, tzinfo=UTC),
        filing_event=FilingEvent(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 4, 15, tzinfo=UTC),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
        draft_note=final_note,
        critic_verdict=verdict,
        critic_attempts=1,
        final_note=final_note,
    )


@pytest.mark.asyncio
async def test_writes_note_when_critic_accepted() -> None:
    state = _state(verdict=CriticVerdict.ACCEPTED, final_note="# Body\n\nText [F1].")
    repo = MagicMock()
    repo.insert_note = AsyncMock(return_value=99)

    update = await write_note(
        state,
        repository=repo,
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
    )

    assert update.owner == OWNER
    assert update.changes == {"persisted_note_id": 99}
    repo.insert_note.assert_awaited_once()


@pytest.mark.asyncio
async def test_skips_when_loop_exceeded() -> None:
    state = _state(verdict=CriticVerdict.LOOP_EXCEEDED, final_note=None)
    repo = MagicMock()
    repo.insert_note = AsyncMock()

    update = await write_note(
        state,
        repository=repo,
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
    )

    assert update.changes == {}
    repo.insert_note.assert_not_awaited()


@pytest.mark.asyncio
async def test_swallows_db_error_logs_and_continues() -> None:
    state = _state(verdict=CriticVerdict.ACCEPTED, final_note="x [F1]")
    repo = MagicMock()
    repo.insert_note = AsyncMock(side_effect=RuntimeError("boom"))

    update = await write_note(
        state,
        repository=repo,
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
    )

    # Note persistence failure must NOT block the response; persisted_note_id
    # stays None.
    assert update.changes == {"persisted_note_id": None}
