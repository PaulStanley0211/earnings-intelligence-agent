"""Unit tests for peer repository methods.

These tests require a live Postgres instance (the same DATABASE_URL used by
the integration suite). Schema is created and torn down per-test via
``Base.metadata`` so alembic migrations are not required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.db import build_engine
from app.memory.models import Base
from app.memory.repository import Repository
from app.memory.schemas import NewFiling, PeerCreate
from app.models.state import FilingForm

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture()
async def session() -> AsyncIterator[AsyncSession]:
    """Build an engine, recreate the schema, yield a clean session."""
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as live:
        yield live
    await engine.dispose()


def _new_filing(
    accession_number: str,
    ticker: str,
    form: FilingForm = FilingForm.FORM_10Q,
    filed_at: datetime | None = None,
    status: str = "processed",
) -> NewFiling:
    return NewFiling(
        accession_number=accession_number,
        cik="0000999",
        ticker=ticker,
        form=form,
        filed_at=filed_at or datetime(2025, 4, 15, tzinfo=UTC),
        source_url=f"https://www.sec.gov/Archives/edgar/data/0000999/{accession_number}.htm",
    )


@pytest.mark.asyncio
async def test_upsert_peer_inserts_then_no_ops_on_duplicate(
    session: AsyncSession,
) -> None:
    """upsert_peer is idempotent: a second call with the same pair leaves one row."""
    repo = Repository(session)
    await repo.upsert_peer(PeerCreate(ticker="MSFT", peer_ticker="GOOGL"))
    await session.commit()
    await repo.upsert_peer(PeerCreate(ticker="MSFT", peer_ticker="GOOGL"))
    await session.commit()
    peers = await repo.list_peers(ticker="MSFT")
    assert peers == ["GOOGL"]


@pytest.mark.asyncio
async def test_list_peers_returns_empty_for_unknown_ticker(
    session: AsyncSession,
) -> None:
    """list_peers returns [] when no mapping exists for a ticker."""
    repo = Repository(session)
    assert await repo.list_peers(ticker="UNKNOWN") == []


@pytest.mark.asyncio
async def test_get_recent_peer_signals_empty_when_cold_start(
    session: AsyncSession,
) -> None:
    """get_recent_peer_signals returns empty lists when no data exists for a peer."""
    repo = Repository(session)
    sig = await repo.get_recent_peer_signals(peer_ticker="GOOGL")
    assert sig.language_diffs == []
    assert sig.commitments == []


@pytest.mark.asyncio
async def test_get_recent_peer_signals_skips_stale_filings(
    session: AsyncSession,
) -> None:
    """Filings older than max_age_days are excluded from peer signals."""
    repo = Repository(session)
    stale_filed_at = datetime.now(UTC) - timedelta(days=400)
    filing = _new_filing(
        accession_number="0000999-24-000001",
        ticker="GOOGL",
        filed_at=stale_filed_at,
    )
    await repo.record_filing(filing=filing)
    await session.commit()

    # Mark as processed so it would qualify if not for age.
    await repo.mark_filing_processed("0000999-24-000001")
    await session.commit()

    sig = await repo.get_recent_peer_signals(peer_ticker="GOOGL", max_age_days=180)
    assert sig.language_diffs == []
    assert sig.commitments == []
