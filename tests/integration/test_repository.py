"""Integration tests for the memory repository.

These tests require a live Postgres at ``DATABASE_URL`` (the docker-compose
stack or the CI services job both provide one). The schema is created and torn
down per-test from ``Base.metadata`` so the tests do not depend on alembic
having run first - alembic itself is covered by ``test_migrations.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.db import build_engine
from app.memory.models import Base, FilingSection, LanguageDiff
from app.memory.repository import Repository
from app.memory.schemas import (
    FilingStatus,
    NewComparison,
    NewConsensusEstimate,
    NewFiling,
    NewFinancialFact,
    NewPollLog,
    PollStatus,
)
from app.models.state import FilingForm

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture()
async def session() -> AsyncIterator[AsyncSession]:
    """Build an engine, recreate the schema, yield a clean session.

    The pgvector extension must be enabled before ``create_all`` so the
    ``vector(1536)`` column type resolves. The extension is idempotent and
    persists for the lifetime of the database, so enabling it here does not
    pollute other test runs.
    """
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = AsyncSession
    async with AsyncSession(engine, expire_on_commit=False) as live:
        yield live
    await engine.dispose()
    _ = factory  # silence unused-name linter on the alias above


def _new_filing(
    *,
    accession_number: str = "0000950170-25-000001",
    ticker: str = "MSFT",
    cik: str = "0000789019",
    form: FilingForm = FilingForm.FORM_10Q,
) -> NewFiling:
    return NewFiling(
        accession_number=accession_number,
        cik=cik,
        ticker=ticker,
        form=form,
        filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
        source_url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}-index.htm",
        report_period_end=date(2026, 3, 31),
    )


async def test_record_filing_persists_and_is_idempotent(session: AsyncSession) -> None:
    repo = Repository(session)
    first = await repo.record_filing(filing=_new_filing())
    await session.commit()
    assert first is not None
    assert first.accession_number == "0000950170-25-000001"
    assert first.status is FilingStatus.DETECTED

    duplicate = await repo.record_filing(filing=_new_filing())
    await session.commit()
    assert duplicate is None, "second insert of the same accession must be a no-op"


async def test_known_accession_numbers_filters_by_cik(session: AsyncSession) -> None:
    repo = Repository(session)
    await repo.record_filing(filing=_new_filing(accession_number="0000111111-25-000001"))
    await repo.record_filing(
        filing=_new_filing(
            accession_number="0000222222-25-000001", ticker="NVDA", cik="0001045810"
        )
    )
    await session.commit()
    msft = await repo.known_accession_numbers(cik="0000789019")
    nvda = await repo.known_accession_numbers(cik="0001045810")
    assert msft == {"0000111111-25-000001"}
    assert nvda == {"0000222222-25-000001"}


async def test_mark_filing_processed_sets_timestamp_and_status(session: AsyncSession) -> None:
    repo = Repository(session)
    await repo.record_filing(filing=_new_filing())
    await session.commit()
    await repo.mark_filing_processed("0000950170-25-000001")
    await session.commit()
    stored = await repo.get_filing("0000950170-25-000001")
    assert stored is not None
    assert stored.status is FilingStatus.PROCESSED
    assert stored.processed_at is not None


async def test_insert_financial_facts_is_idempotent_on_unique_key(
    session: AsyncSession,
) -> None:
    repo = Repository(session)
    await repo.record_filing(filing=_new_filing())
    await session.commit()
    fact = NewFinancialFact(
        filing_accession="0000950170-25-000001",
        cik="0000789019",
        taxonomy="us-gaap",
        concept="Revenues",
        unit="USD",
        value=Decimal("61858000000"),
        period_type="duration",
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        fiscal_year=2026,
        fiscal_period="Q3",
        form="10-Q",
        filed=date(2026, 4, 25),
        frame=None,
    )
    inserted_first = await repo.insert_financial_facts("0000950170-25-000001", [fact, fact])
    await session.commit()
    assert inserted_first == 1, "the second fact is a duplicate and must be skipped"

    inserted_second = await repo.insert_financial_facts("0000950170-25-000001", [fact])
    await session.commit()
    assert inserted_second == 0


async def test_get_facts_for_filing_returns_rows(session: AsyncSession) -> None:
    repo = Repository(session)
    await repo.record_filing(filing=_new_filing())
    fact = NewFinancialFact(
        filing_accession="0000950170-25-000001",
        cik="0000789019",
        taxonomy="us-gaap",
        concept="NetIncomeLoss",
        unit="USD",
        value=Decimal("21939000000"),
        period_type="duration",
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        fiscal_year=2026,
        fiscal_period="Q3",
        form="10-Q",
        filed=date(2026, 4, 25),
        frame=None,
    )
    await repo.insert_financial_facts("0000950170-25-000001", [fact])
    await session.commit()
    rows = await repo.get_facts_for_filing("0000950170-25-000001")
    assert len(rows) == 1
    assert rows[0].concept == "NetIncomeLoss"
    assert rows[0].value == Decimal("21939000000")


async def test_record_and_read_last_successful_poll(session: AsyncSession) -> None:
    repo = Repository(session)
    assert await repo.last_successful_poll_at() is None
    await repo.record_poll(
        NewPollLog(tickers_checked=5, filings_found=0, status=PollStatus.OK)
    )
    await repo.record_poll(
        NewPollLog(
            tickers_checked=5,
            filings_found=0,
            status=PollStatus.ERROR,
            error_message="EDGAR 503",
        )
    )
    await session.commit()
    last = await repo.last_successful_poll_at()
    assert last is not None
    assert last.status is PollStatus.OK
    assert last.polled_at.tzinfo is not None, "timestamps must be timezone-aware"


async def test_mark_filing_failed_records_error(session: AsyncSession) -> None:
    repo = Repository(session)
    await repo.record_filing(filing=_new_filing())
    await session.commit()
    await repo.mark_filing_failed("0000950170-25-000001", error="boom")
    await session.commit()
    stored = await repo.get_filing("0000950170-25-000001")
    assert stored is not None
    assert stored.status is FilingStatus.FAILED
    assert stored.error_message == "boom"
    assert stored.processed_at is not None


async def test_list_filings_for_ticker_orders_newest_first(session: AsyncSession) -> None:
    repo = Repository(session)
    await repo.record_filing(
        filing=NewFiling(
            accession_number="0000950170-25-000001",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2026, 1, 25, tzinfo=UTC),
            source_url="https://www.sec.gov/x",
        )
    )
    await repo.record_filing(
        filing=NewFiling(
            accession_number="0000950170-25-000002",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_8K,
            filed_at=datetime(2026, 4, 25, tzinfo=UTC),
            source_url="https://www.sec.gov/y",
        )
    )
    await session.commit()
    rows = await repo.list_filings_for_ticker("MSFT")
    assert [r.accession_number for r in rows] == [
        "0000950170-25-000002",
        "0000950170-25-000001",
    ]


async def test_daily_spend_accumulates_atomically(session: AsyncSession) -> None:
    repo = Repository(session)
    today = date(2026, 5, 15)
    assert await repo.get_daily_spend(today) == Decimal("0")

    first = await repo.add_daily_spend(day=today, amount_usd=Decimal("0.5"))
    second = await repo.add_daily_spend(day=today, amount_usd=Decimal("0.25"))
    await session.commit()
    assert first == Decimal("0.5")
    assert second == Decimal("0.75")
    assert await repo.get_daily_spend(today) == Decimal("0.75")


async def test_list_active_watchlist(session: AsyncSession) -> None:
    repo = Repository(session)
    await repo.upsert_watchlist_entry(ticker="MSFT", cik="0000789019", company_name="Microsoft")
    await repo.upsert_watchlist_entry(
        ticker="NVDA", cik="0001045810", company_name="NVIDIA Corp", active=False
    )
    await session.commit()
    active = await repo.list_active_watchlist()
    tickers = sorted(entry.ticker for entry in active)
    assert tickers == ["MSFT"]


# ---- Phase 2: consensus estimates ----


async def test_upsert_consensus_estimate_refreshes_value(session: AsyncSession) -> None:
    repo = Repository(session)
    first = await repo.upsert_consensus_estimate(
        NewConsensusEstimate(
            ticker="MSFT",
            fiscal_year=2026,
            fiscal_period="Q3",
            metric="eps_diluted",
            value=Decimal("1.30"),
            analyst_count=18,
            source="finnhub",
        )
    )
    await session.commit()
    assert first.value == Decimal("1.30")

    second = await repo.upsert_consensus_estimate(
        NewConsensusEstimate(
            ticker="MSFT",
            fiscal_year=2026,
            fiscal_period="Q3",
            metric="eps_diluted",
            value=Decimal("1.32"),
            analyst_count=20,
            source="finnhub",
        )
    )
    await session.commit()
    # Same primary key (auto-id may differ but the unique key is fixed): the
    # value moves to the refreshed reading.
    assert second.value == Decimal("1.32")
    assert second.analyst_count == 20

    rows = await repo.get_consensus_estimates(
        ticker="MSFT", fiscal_year=2026, fiscal_period="Q3"
    )
    assert {(r.metric, r.source) for r in rows} == {("eps_diluted", "finnhub")}


async def test_get_consensus_estimates_filters_by_period(session: AsyncSession) -> None:
    repo = Repository(session)
    await repo.upsert_consensus_estimate(
        NewConsensusEstimate(
            ticker="MSFT",
            fiscal_year=2026,
            fiscal_period="Q3",
            metric="revenue",
            value=Decimal("61000000000"),
            source="finnhub",
        )
    )
    await repo.upsert_consensus_estimate(
        NewConsensusEstimate(
            ticker="MSFT",
            fiscal_year=2026,
            fiscal_period="Q2",
            metric="revenue",
            value=Decimal("65000000000"),
            source="finnhub",
        )
    )
    await session.commit()
    q3 = await repo.get_consensus_estimates(
        ticker="MSFT", fiscal_year=2026, fiscal_period="Q3"
    )
    assert len(q3) == 1
    assert q3[0].value == Decimal("61000000000")


# ---- Phase 2: comparisons ----


async def test_insert_comparison_upserts_on_filing_and_metric(
    session: AsyncSession,
) -> None:
    repo = Repository(session)
    await repo.record_filing(filing=_new_filing())
    await session.commit()
    first = await repo.insert_comparison(
        NewComparison(
            filing_accession="0000950170-25-000001",
            metric="revenue",
            reported_value=Decimal("61858000000"),
            reported_unit="USD",
            consensus_value=Decimal("61000000000"),
            consensus_source="finnhub",
            surprise_abs=Decimal("858000000"),
            surprise_pct=Decimal("1.4066"),
            direction="beat",
        )
    )
    await session.commit()
    assert first.direction == "beat"

    # Re-running the comparator updates the surprise.
    second = await repo.insert_comparison(
        NewComparison(
            filing_accession="0000950170-25-000001",
            metric="revenue",
            reported_value=Decimal("61858000000"),
            reported_unit="USD",
            consensus_value=Decimal("62000000000"),
            consensus_source="finnhub",
            surprise_abs=Decimal("-142000000"),
            surprise_pct=Decimal("-0.2290"),
            direction="in_line",
        )
    )
    await session.commit()
    assert second.direction == "in_line"
    rows = await repo.list_comparisons_for_filing("0000950170-25-000001")
    assert len(rows) == 1
    assert rows[0].consensus_value == Decimal("62000000000")


# ---- Phase 3: filing_sections and language_diffs ----


async def test_filing_section_model_roundtrips(session: AsyncSession) -> None:
    """FilingSection rows persist and can be fetched back by section_kind."""
    repo = Repository(session)
    await repo.record_filing(
        filing=NewFiling(
            accession_number="0000000000-26-000001",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
            source_url="https://www.sec.gov/x",
        )
    )
    await session.commit()

    section = FilingSection(
        filing_accession="0000000000-26-000001",
        cik="0000789019",
        ticker="MSFT",
        section_kind="mda",
        paragraph_index=0,
        text="The company saw strong demand.",
        text_sha="a" * 64,
        embedding=None,
        embedding_model=None,
    )
    session.add(section)
    await session.commit()

    rows = (await session.execute(select(FilingSection))).scalars().all()
    assert len(rows) == 1
    assert rows[0].section_kind == "mda"


async def test_language_diff_model_roundtrips(session: AsyncSession) -> None:
    """LanguageDiff rows persist and can be fetched back by change_type."""
    repo = Repository(session)
    await repo.record_filing(
        filing=NewFiling(
            accession_number="0000000000-26-000002",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
            source_url="https://www.sec.gov/x",
        )
    )
    await session.commit()

    diff = LanguageDiff(
        filing_accession="0000000000-26-000002",
        prior_filing_accession=None,
        section_kind="mda",
        change_type="added",
        current_section_id=None,
        prior_section_id=None,
        similarity=None,
        severity="major",
    )
    session.add(diff)
    await session.commit()

    rows = (await session.execute(select(LanguageDiff))).scalars().all()
    assert len(rows) == 1
    assert rows[0].change_type == "added"
