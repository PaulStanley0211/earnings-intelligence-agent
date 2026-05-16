"""Integration test for the language backfill CLI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.memory.db import build_engine
from app.memory.models import Base, FilingSection
from app.memory.repository import Repository
from app.scripts.backfill_language import run_backfill
from app.tools.edgar import RecentFiling, SubmissionsResponse

pytestmark = pytest.mark.integration


class _Edgar:
    async def get_submissions(self, *, cik: str) -> SubmissionsResponse:
        # Months offset from April 2026: i=1 -> March, i=2 -> February, i=3 -> January
        filings = [
            RecentFiling(
                accession_number=f"0000950170-26-{i:06d}",
                form="10-Q",
                filing_date=date(2026, 4 - i, 25),
                report_date=date(2026, 4 - i, 1),
                primary_document=f"msft-q{i}.htm",
            )
            for i in range(1, 4)
        ]
        return SubmissionsResponse(
            cik=cik.zfill(10),
            entity_name="Microsoft Corp",
            tickers=["MSFT"],
            recent_filings=filings,
        )

    async def get_filing_document(
        self, *, cik: str, accession_number: str, primary_document: str
    ) -> str:
        return (
            "<html><body>"
            "<p>Item 2. Management's Discussion and Analysis</p>"
            f"<p>Revenue grew during the quarter ending {primary_document}.</p>"
            "<p>Item 3. Other</p>"
            "</body></html>"
        )


class _Embeddings:
    @property
    def model(self) -> str:
        return "openai/text-embedding-3-small"

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        # Return 1536-dim vectors. Each text gets a unique first component.
        return [[float(i) / 1000.0] + [0.0] * 1535 for i in range(len(texts))]


@pytest_asyncio.fixture()
async def session_factory_with_msft() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        from sqlalchemy import text

        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        await Repository(session).upsert_watchlist_entry(
            ticker="MSFT", cik="0000789019", company_name="Microsoft Corp"
        )
        await session.commit()
    yield factory
    await engine.dispose()


async def test_run_backfill_inserts_sections_for_each_filing(
    session_factory_with_msft: async_sessionmaker[AsyncSession],
) -> None:
    from sqlalchemy import select

    summary = await run_backfill(
        tickers=["MSFT"],
        quarters=3,
        edgar=_Edgar(),
        embeddings=_Embeddings(),
        session_factory=session_factory_with_msft,
    )
    assert summary["filings_parsed"] == 3
    async with session_factory_with_msft() as session:
        rows = (await session.execute(select(FilingSection))).scalars().all()
    # 3 filings * 1 substantive paragraph each (post-filter).
    assert len(rows) == 3


async def test_run_backfill_is_idempotent(
    session_factory_with_msft: async_sessionmaker[AsyncSession],
) -> None:
    await run_backfill(
        tickers=["MSFT"],
        quarters=3,
        edgar=_Edgar(),
        embeddings=_Embeddings(),
        session_factory=session_factory_with_msft,
    )
    summary = await run_backfill(
        tickers=["MSFT"],
        quarters=3,
        edgar=_Edgar(),
        embeddings=_Embeddings(),
        session_factory=session_factory_with_msft,
    )
    # Filings already exist; no new paragraphs inserted on the second run.
    assert summary["paragraphs_inserted"] == 0
