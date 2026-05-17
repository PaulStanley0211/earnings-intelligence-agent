"""Unit tests for the peer_reader agent node."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.peer_reader import OWNER, read_peers
from app.memory.schemas import (
    PeerCommitmentSignal,
    PeerLanguageDiffSignal,
    PeerSignals,
)
from app.models.state import (
    AgentState,
    FilingEvent,
    FilingEventSource,
    FilingForm,
    PeerContextEntry,
)


def _state(ticker: str = "MSFT") -> AgentState:
    return AgentState(
        trace_id="t-1",
        started_at=datetime(2025, 4, 15, tzinfo=UTC),
        filing_event=FilingEvent(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker=ticker,
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 4, 15, tzinfo=UTC),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
    )


@pytest.mark.asyncio
async def test_no_peers_yields_empty_context() -> None:
    repo = MagicMock()
    repo.list_peers = AsyncMock(return_value=[])

    update = await read_peers(_state(), repository=repo)

    assert update.owner == OWNER
    assert update.changes == {"peer_context": []}


@pytest.mark.asyncio
async def test_one_peer_returns_combined_signals() -> None:
    repo = MagicMock()
    repo.list_peers = AsyncMock(return_value=["GOOGL"])
    repo.get_recent_peer_signals = AsyncMock(
        return_value=PeerSignals(
            language_diffs=[
                PeerLanguageDiffSignal(
                    text="Cloud pricing pressure intensified.",
                    severity="major",
                    source_filing_accession="0000123-25-000002",
                ),
            ],
            commitments=[
                PeerCommitmentSignal(
                    text="We expect cloud margins to expand next quarter.",
                    source_filing_accession="0000123-25-000003",
                ),
            ],
        )
    )

    update = await read_peers(_state(), repository=repo)
    entries = update.changes["peer_context"]
    assert len(entries) == 2
    assert {e.kind for e in entries} == {"language_diff", "commitment"}
    assert all(isinstance(e, PeerContextEntry) for e in entries)


@pytest.mark.asyncio
async def test_db_error_degrades_to_empty_context() -> None:
    repo = MagicMock()
    repo.list_peers = AsyncMock(side_effect=RuntimeError("db down"))

    update = await read_peers(_state(), repository=repo)
    assert update.changes == {"peer_context": []}
