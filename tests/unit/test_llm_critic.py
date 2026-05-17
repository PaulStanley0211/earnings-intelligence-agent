"""Unit tests for the LLM critic node."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.llm_critic import OWNER, llm_critique
from app.models.state import (
    AgentState,
    CriticFinding,
    CriticVerdict,
    FilingEvent,
    FilingEventSource,
    FilingForm,
)


def _accepted_state(note: str) -> AgentState:
    return AgentState(
        trace_id="t",
        started_at=datetime(2025, 1, 1, tzinfo=UTC),
        filing_event=FilingEvent(
            accession_number="acc-1",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 1, 1, tzinfo=UTC),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
        draft_note=note,
        final_note=note,
        critic_verdict=CriticVerdict.ACCEPTED,
        critic_attempts=1,
    )


@pytest.mark.asyncio
async def test_accepts_clean_note() -> None:
    llm = MagicMock()
    llm.acomplete = AsyncMock(return_value='{"findings": []}')
    repo = MagicMock()

    update = await llm_critique(
        _accepted_state("# Clean Note\n\nRevenue rose $1B [F1]."),
        llm=llm,
        repository=repo,
    )

    assert update.owner == OWNER
    assert update.changes["critic_verdict"] is CriticVerdict.ACCEPTED


@pytest.mark.asyncio
async def test_rejects_when_findings_present() -> None:
    llm = MagicMock()
    finding_json = (
        '{"findings": [{"layer":"semantic","severity":"error",'
        '"claim":"X","evidence":"Y","recommended_fix":"Z"}]}'
    )
    llm.acomplete = AsyncMock(return_value=finding_json)
    repo = MagicMock()

    update = await llm_critique(
        _accepted_state("# Note\n\nText."), llm=llm, repository=repo
    )
    assert update.changes["critic_verdict"] is CriticVerdict.REJECTED
    assert len(update.changes["critic_findings"]) >= 1
    finding = update.changes["critic_findings"][-1]
    assert isinstance(finding, CriticFinding)
    assert finding.layer == "semantic"


@pytest.mark.asyncio
async def test_malformed_json_retries_once_then_rejects() -> None:
    llm = MagicMock()
    llm.acomplete = AsyncMock(side_effect=["not json", "still not json"])
    repo = MagicMock()

    update = await llm_critique(
        _accepted_state("# Note"), llm=llm, repository=repo
    )
    assert llm.acomplete.await_count == 2
    assert update.changes["critic_verdict"] is CriticVerdict.REJECTED
    assert any(
        "unparseable" in f.message for f in update.changes["critic_findings"]
    )


@pytest.mark.asyncio
async def test_skips_when_det_critic_rejected_or_loop_exceeded() -> None:
    state = _accepted_state("# Note")
    state = state.model_copy(update={"critic_verdict": CriticVerdict.REJECTED})
    llm = MagicMock()
    llm.acomplete = AsyncMock()

    update = await llm_critique(state, llm=llm, repository=MagicMock())
    assert update.changes == {}
