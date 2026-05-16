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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.memory.db import build_engine
from app.memory.models import Base, FilingSection, LanguageDiff
from app.memory.repository import Repository
from app.memory.schemas import (
    ChangeType,
    FilingStatus,
    NewComparison,
    NewConsensusEstimate,
    NewFiling,
    NewFilingSection,
    NewFinancialFact,
    NewLanguageDiff,
    NewPollLog,
    PollStatus,
    SectionKind,
    Severity,
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


# ---- Phase 3: repository methods for filing_sections ----


@pytest_asyncio.fixture()
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Build an engine, recreate the schema, yield a sessionmaker.

    Unlike the ``session`` fixture, this yields a factory so individual tests
    can open and close multiple sessions to simulate separate transactions.
    The pgvector extension is enabled idempotently before ``create_all``.
    """
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


async def test_insert_filing_sections_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    accession = "0000000000-26-000010"
    async with session_factory() as session:
        await Repository(session).record_filing(
            filing=NewFiling(
                accession_number=accession,
                cik="0000789019",
                ticker="MSFT",
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
                source_url="https://www.sec.gov/x",
            )
        )
        rows = [
            NewFilingSection(
                filing_accession=accession,
                cik="0000789019",
                ticker="MSFT",
                section_kind=SectionKind.MDA,
                paragraph_index=i,
                text=f"Paragraph {i}.",
                text_sha=f"{i:064d}",
                embedding=None,
                embedding_model=None,
            )
            for i in range(3)
        ]
        first = await Repository(session).insert_filing_sections(rows)
        await session.commit()

    async with session_factory() as session:
        second = await Repository(session).insert_filing_sections(rows)
        await session.commit()

    assert first == 3
    assert second == 0


async def test_update_section_embeddings_sets_vector_and_model(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    accession = "0000000000-26-000011"
    async with session_factory() as session:
        await Repository(session).record_filing(
            filing=NewFiling(
                accession_number=accession,
                cik="0000789019",
                ticker="MSFT",
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
                source_url="https://www.sec.gov/x",
            )
        )
        ids: list[int] = []
        for i in range(2):
            section = FilingSection(
                filing_accession=accession,
                cik="0000789019",
                ticker="MSFT",
                section_kind="mda",
                paragraph_index=i,
                text=f"p{i}",
                text_sha=f"{i:064d}",
                embedding=None,
                embedding_model=None,
            )
            session.add(section)
            await session.flush()
            ids.append(section.id)
        await Repository(session).update_section_embeddings(
            updates=[
                (ids[0], [0.0] * 1536, "openai/text-embedding-3-small"),
                (ids[1], [0.5] * 1536, "openai/text-embedding-3-small"),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        rows = (
            await session.execute(select(FilingSection).order_by(FilingSection.id))
        ).scalars().all()
        assert all(r.embedding is not None for r in rows)
        assert rows[0].embedding_model == "openai/text-embedding-3-small"


async def test_get_prior_quarter_sections_returns_most_recent_filing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        repo = Repository(session)
        for accession, filed in [
            ("0000000000-26-000020", datetime(2026, 1, 25, tzinfo=UTC)),
            ("0000000000-26-000021", datetime(2026, 4, 25, tzinfo=UTC)),
        ]:
            await repo.record_filing(
                filing=NewFiling(
                    accession_number=accession,
                    cik="0000789019",
                    ticker="MSFT",
                    form=FilingForm.FORM_10Q,
                    filed_at=filed,
                    source_url="https://www.sec.gov/x",
                )
            )
            await repo.insert_filing_sections(
                [
                    NewFilingSection(
                        filing_accession=accession,
                        cik="0000789019",
                        ticker="MSFT",
                        section_kind=SectionKind.MDA,
                        paragraph_index=0,
                        text=f"Filed at {filed.date().isoformat()}.",
                        text_sha=accession.ljust(64, "0"),
                        embedding=None,
                        embedding_model=None,
                    )
                ]
            )
        await session.commit()

    async with session_factory() as session:
        rows = await Repository(session).get_prior_quarter_sections(
            ticker="MSFT",
            section_kind=SectionKind.MDA,
            before=date(2026, 4, 25),
        )
        assert len(rows) == 1
        assert rows[0].filing_accession == "0000000000-26-000020"
        assert rows[0].text == "Filed at 2026-01-25."


# ---- Phase 3: repository methods for language_diffs ----


async def test_insert_language_diffs_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    accession = "0000000000-26-000030"
    async with session_factory() as session:
        await Repository(session).record_filing(
            filing=NewFiling(
                accession_number=accession,
                cik="0000789019",
                ticker="MSFT",
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
                source_url="https://www.sec.gov/x",
            )
        )
        rows = [
            NewLanguageDiff(
                filing_accession=accession,
                section_kind=SectionKind.MDA,
                change_type=ChangeType.ADDED,
                severity=Severity.MAJOR,
            ),
            NewLanguageDiff(
                filing_accession=accession,
                section_kind=SectionKind.MDA,
                change_type=ChangeType.REMOVED,
                severity=Severity.MINOR,
            ),
        ]
        first = await Repository(session).insert_language_diffs(rows)
        await session.commit()

    async with session_factory() as session:
        second = await Repository(session).insert_language_diffs(rows)
        await session.commit()

    assert first == 2
    assert second == 0


async def test_list_language_diffs_for_filing_returns_inserted_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    accession = "0000000000-26-000031"
    async with session_factory() as session:
        await Repository(session).record_filing(
            filing=NewFiling(
                accession_number=accession,
                cik="0000789019",
                ticker="MSFT",
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
                source_url="https://www.sec.gov/x",
            )
        )
        await Repository(session).insert_language_diffs(
            [
                NewLanguageDiff(
                    filing_accession=accession,
                    section_kind=SectionKind.MDA,
                    change_type=ChangeType.ADDED,
                    severity=Severity.MAJOR,
                )
            ]
        )
        await session.commit()

    async with session_factory() as session:
        rows = await Repository(session).list_language_diffs_for_filing(accession)
        assert len(rows) == 1
        assert rows[0].change_type == ChangeType.ADDED
        assert rows[0].severity == Severity.MAJOR
        assert rows[0].filing_accession == accession


# ---- Phase 4A: uploaded_documents ----


async def test_add_and_fetch_uploaded_document(session: AsyncSession) -> None:
    """An uploaded document round-trips through the repository methods."""
    from app.memory.schemas import NewUploadedDocument
    repo = Repository(session)

    new = NewUploadedDocument(
        upload_id="upload-test-001",
        ticker="MSFT",
        filing_type="8-K",
        original_filename="msft-8k-q2.pdf",
        content_sha256="a" * 64,
        parsed_text="Microsoft reported revenue of $XX billion.",
        parsed_char_count=42,
        page_count=14,
    )
    stored = await repo.add_uploaded_document(new)
    await session.commit()
    assert stored.upload_id == "upload-test-001"
    assert stored.ticker == "MSFT"

    by_sha = await repo.get_uploaded_document_by_sha256("a" * 64)
    assert by_sha is not None
    assert by_sha.original_filename == "msft-8k-q2.pdf"

    by_id = await repo.get_uploaded_document("upload-test-001")
    assert by_id is not None
    assert by_id.parsed_char_count == 42


async def test_uploaded_document_sha256_unique(session: AsyncSession) -> None:
    """Re-uploading the same content (same sha256) violates the unique index."""
    import sqlalchemy.exc

    from app.memory.schemas import NewUploadedDocument
    repo = Repository(session)

    base = NewUploadedDocument(
        upload_id="upload-a",
        ticker="MSFT",
        filing_type="8-K",
        original_filename="a.pdf",
        content_sha256="b" * 64,
        parsed_text="hi",
        parsed_char_count=2,
        page_count=1,
    )
    await repo.add_uploaded_document(base)
    await session.commit()

    duplicate = base.model_copy(update={"upload_id": "upload-b"})
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        await repo.add_uploaded_document(duplicate)


# ---- Phase 4B: qa_pairs and commitments ----


async def _record_msft_filing(
    session: AsyncSession,
    *,
    accession: str = "0000789019-26-000001",
    filed_at: datetime | None = None,
) -> None:
    """Record a synthetic MSFT 10-Q filing so the FK target exists."""
    await Repository(session).record_filing(
        filing=NewFiling(
            accession_number=accession,
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=filed_at or datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
            source_url=f"https://www.sec.gov/Archives/edgar/data/789019/{accession}-index.htm",
        )
    )


async def _record_nvda_filing(
    session: AsyncSession,
    *,
    accession: str = "0001045810-26-000001",
) -> None:
    """Record a synthetic NVDA 10-Q filing so the FK target exists."""
    await Repository(session).record_filing(
        filing=NewFiling(
            accession_number=accession,
            cik="0001045810",
            ticker="NVDA",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2026, 5, 21, 20, 5, tzinfo=UTC),
            source_url=(
                f"https://www.sec.gov/Archives/edgar/data/1045810/{accession}-index.htm"
            ),
        )
    )


async def test_add_qa_pairs_idempotent_on_filing_accession_ordinal(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Re-inserting the same (filing_accession, ordinal) tuples is a no-op."""
    from app.memory.schemas import AnswerClass, NewQAPair

    accession = "0000789019-26-000001"
    async with session_factory() as session:
        await _record_msft_filing(session, accession=accession)
        first_batch = [
            NewQAPair(
                filing_accession=accession,
                ordinal=i,
                analyst_name=f"Analyst {i}",
                question_text=f"Question {i}?",
                answer_text=f"Answer {i}.",
                answer_class=AnswerClass.DIRECT,
                sha256_text=f"{i:064d}",
            )
            for i in range(1, 4)
        ]
        first = await Repository(session).add_qa_pairs(
            filing_accession=accession, pairs=first_batch
        )
        await session.commit()
        assert len(first) == 3

    async with session_factory() as session:
        second_batch = [
            *first_batch,
            NewQAPair(
                filing_accession=accession,
                ordinal=4,
                analyst_name="Analyst 4",
                question_text="Question 4?",
                answer_text="Answer 4.",
                answer_class=AnswerClass.PARTIAL,
                sha256_text=f"{4:064d}",
            ),
        ]
        second = await Repository(session).add_qa_pairs(
            filing_accession=accession, pairs=second_batch
        )
        await session.commit()
        # The follow-up SELECT must surface all four ordinals once each.
        assert {p.ordinal for p in second} == {1, 2, 3, 4}

    async with session_factory() as session:
        listed = await Repository(session).list_qa_pairs_for_filing(accession)
        assert [p.ordinal for p in listed] == [1, 2, 3, 4]
        assert listed[3].answer_class == AnswerClass.PARTIAL


async def test_add_commitments_idempotent_on_source_quote(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A repeat batch with the same source_quote values must not duplicate rows."""
    from app.memory.schemas import NewCommitment

    accession = "0000789019-26-000002"
    async with session_factory() as session:
        await _record_msft_filing(session, accession=accession)
        first_batch = [
            NewCommitment(
                filing_accession=accession,
                ticker="MSFT",
                commitment_text="Expect double-digit revenue growth in Q3.",
                target_period="Q3 FY26",
                source_quote="We expect double-digit revenue growth in Q3.",
            ),
            NewCommitment(
                filing_accession=accession,
                ticker="MSFT",
                commitment_text="Operating margin to expand 100 bps.",
                target_period="FY26",
                source_quote="Operating margin should expand by about 100 basis points.",
            ),
        ]
        first = await Repository(session).add_commitments(
            filing_accession=accession,
            ticker="MSFT",
            commitments=first_batch,
        )
        await session.commit()
        assert len(first) == 2

    async with session_factory() as session:
        second_batch = [
            *first_batch,
            NewCommitment(
                filing_accession=accession,
                ticker="MSFT",
                commitment_text="Capex to exceed $80B for FY26.",
                target_period="FY26",
                source_quote="We anticipate capex to exceed $80 billion this fiscal year.",
            ),
        ]
        second = await Repository(session).add_commitments(
            filing_accession=accession,
            ticker="MSFT",
            commitments=second_batch,
        )
        await session.commit()
        assert len(second) == 3, "all three rows present, none duplicated"
        # The set of returned source_quotes covers the full input batch.
        assert {c.source_quote for c in second} == {
            "We expect double-digit revenue growth in Q3.",
            "Operating margin should expand by about 100 basis points.",
            "We anticipate capex to exceed $80 billion this fiscal year.",
        }

    # Verify table state directly: exactly three rows for this filing, no duplicates.
    async with session_factory() as session:
        from app.memory.models import Commitment

        rows = (
            await session.execute(
                select(Commitment).where(Commitment.filing_accession == accession)
            )
        ).scalars().all()
        assert len(rows) == 3


async def test_get_open_commitments_returns_only_open_and_filters_by_ticker(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``get_open_commitments`` excludes closed rows AND rows for other tickers."""
    from app.memory.schemas import CommitmentStatus, NewCommitment

    msft_accession = "0000789019-26-000003"
    nvda_accession = "0001045810-26-000003"

    async with session_factory() as session:
        await _record_msft_filing(session, accession=msft_accession)
        await _record_nvda_filing(session, accession=nvda_accession)
        msft_rows = await Repository(session).add_commitments(
            filing_accession=msft_accession,
            ticker="MSFT",
            commitments=[
                NewCommitment(
                    filing_accession=msft_accession,
                    ticker="MSFT",
                    commitment_text="Open MSFT commitment 1.",
                    target_period="Q3",
                    source_quote="quote-msft-open-1",
                ),
                NewCommitment(
                    filing_accession=msft_accession,
                    ticker="MSFT",
                    commitment_text="MSFT commitment that will be marked met.",
                    target_period="Q3",
                    source_quote="quote-msft-met",
                ),
            ],
        )
        await Repository(session).add_commitments(
            filing_accession=nvda_accession,
            ticker="NVDA",
            commitments=[
                NewCommitment(
                    filing_accession=nvda_accession,
                    ticker="NVDA",
                    commitment_text="Open NVDA commitment.",
                    target_period="Q2",
                    source_quote="quote-nvda-open",
                ),
            ],
        )
        # Resolve one MSFT commitment so it is no longer open.
        to_close = next(c for c in msft_rows if c.source_quote == "quote-msft-met")
        await Repository(session).update_commitment_status(
            commitment_id=to_close.id,
            status=CommitmentStatus.MET,
            resolved_filing_accession=msft_accession,
            resolved_reason="reported result matched the guidance band",
        )
        await session.commit()

    async with session_factory() as session:
        open_msft = await Repository(session).get_open_commitments("MSFT")
        assert len(open_msft) == 1
        assert open_msft[0].source_quote == "quote-msft-open-1"
        assert open_msft[0].status == CommitmentStatus.OPEN
        # No NVDA rows must leak through the MSFT filter.
        assert all(row.ticker == "MSFT" for row in open_msft)


async def test_update_commitment_status_atomically_updates_resolved_fields_and_updated_at(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``update_commitment_status`` rewrites all mutable fields plus ``updated_at``."""
    import asyncio

    from app.memory.schemas import CommitmentStatus, NewCommitment

    open_accession = "0000789019-26-000004"
    resolving_accession = "0000789019-26-000005"

    async with session_factory() as session:
        await _record_msft_filing(session, accession=open_accession)
        await _record_msft_filing(
            session,
            accession=resolving_accession,
            filed_at=datetime(2026, 7, 25, 20, 5, tzinfo=UTC),
        )
        rows = await Repository(session).add_commitments(
            filing_accession=open_accession,
            ticker="MSFT",
            commitments=[
                NewCommitment(
                    filing_accession=open_accession,
                    ticker="MSFT",
                    commitment_text="Guide raise next quarter.",
                    target_period="Q3 FY26",
                    source_quote="quote-update-target",
                ),
            ],
        )
        await session.commit()
        target = rows[0]
        original_updated_at = target.updated_at
        assert target.status == CommitmentStatus.OPEN

    # Give Postgres' clock at least one tick before the UPDATE so the
    # ``updated_at`` comparison cannot collapse to equality.
    await asyncio.sleep(0.05)

    async with session_factory() as session:
        await Repository(session).update_commitment_status(
            commitment_id=target.id,
            status=CommitmentStatus.MET,
            resolved_filing_accession=resolving_accession,
            resolved_reason="Q3 met guidance",
        )
        await session.commit()

    async with session_factory() as session:
        from app.memory.models import Commitment

        refreshed = await session.get(Commitment, target.id)
        assert refreshed is not None
        assert refreshed.status == CommitmentStatus.MET.value
        assert refreshed.resolved_filing_accession == resolving_accession
        assert refreshed.resolved_reason == "Q3 met guidance"
        assert refreshed.updated_at > original_updated_at
