"""Unit tests for Note repository methods.

These tests require a live Postgres instance (the same DATABASE_URL used by
the integration suite). Schema is created and torn down per-test via
``Base.metadata`` so alembic migrations are not required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.db import build_engine
from app.memory.models import Base
from app.memory.repository import Repository
from app.memory.schemas import NewFiling, NoteCreate
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


def _new_filing(accession_number: str = "0000123-25-000001") -> NewFiling:
    return NewFiling(
        accession_number=accession_number,
        cik="0000123",
        ticker="MSFT",
        form=FilingForm.FORM_10Q,
        filed_at=datetime(2025, 4, 15, tzinfo=UTC),
        source_url="https://www.sec.gov/Archives/edgar/data/0000123/index.htm",
    )


@pytest.mark.asyncio
async def test_insert_note_persists_and_returns_id(session: AsyncSession) -> None:
    """insert_note returns an id; a second call with the same accession is a no-op."""
    repo = Repository(session)
    await repo.record_filing(filing=_new_filing())
    await session.commit()

    note = NoteCreate(
        filing_accession="0000123-25-000001",
        ticker="MSFT",
        markdown_body="Body",
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
        critic_attempts=2,
    )
    note_id = await repo.insert_note(note)
    await session.commit()
    assert note_id is not None

    again_id = await repo.insert_note(note)
    await session.commit()
    assert again_id == note_id, "second insert should return existing id"


@pytest.mark.asyncio
async def test_get_latest_note_returns_most_recent(session: AsyncSession) -> None:
    """get_latest_note returns the most-recently-inserted note for the ticker."""
    repo = Repository(session)
    await repo.record_filing(filing=_new_filing())
    await session.commit()

    await repo.insert_note(
        NoteCreate(
            filing_accession="0000123-25-000001",
            ticker="MSFT",
            markdown_body="Older",
            prompt_template_name="synthesizer/full_v1",
            prompt_template_sha="a" * 64,
            critic_attempts=1,
        )
    )
    await session.commit()

    latest = await repo.get_latest_note(ticker="MSFT")
    assert latest is not None
    assert latest.markdown_body == "Older"
    assert latest.ticker == "MSFT"


@pytest.mark.asyncio
async def test_get_latest_note_returns_none_for_unknown_ticker(
    session: AsyncSession,
) -> None:
    """get_latest_note returns None when no note exists for the ticker."""
    repo = Repository(session)
    result = await repo.get_latest_note(ticker="NOPE")
    assert result is None
