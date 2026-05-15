"""Repository over the Phase 1 memory schema.

Every query and write that touches the database lives here so agent code
never needs to import SQLAlchemy. Methods accept and return the Pydantic
DTOs from :mod:`app.memory.schemas` to keep callers decoupled from the
session lifecycle.

The repository deliberately does **not** commit transactions - the caller
owns the transaction boundary so multiple repository calls can land in one
atomic unit of work. Tests close the session under an ``async with`` block,
which rolls back any uncommitted state.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import desc, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.models import (
    Comparison,
    ConsensusEstimate,
    DailyLLMSpend,
    EdgarPollLog,
    Filing,
    FinancialFact,
    WatchlistEntry,
)
from app.memory.schemas import (
    ComparisonRecord,
    ConsensusEstimateRecord,
    FilingRecord,
    FilingStatus,
    FinancialFactRecord,
    NewComparison,
    NewConsensusEstimate,
    NewFiling,
    NewFinancialFact,
    NewPollLog,
    PollLogRecord,
    PollStatus,
    WatchlistRecord,
)


class Repository:
    """Async repository - one instance per :class:`AsyncSession`."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to ``session``; the caller owns the transaction."""
        self._session = session

    # ---- watchlist ----

    async def upsert_watchlist_entry(
        self,
        *,
        ticker: str,
        cik: str,
        company_name: str,
        active: bool = True,
    ) -> WatchlistRecord:
        """Insert or update a watchlist entry, keyed by ticker."""
        insert_stmt = pg_insert(WatchlistEntry).values(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            active=active,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[WatchlistEntry.ticker],
            set_={
                "cik": insert_stmt.excluded.cik,
                "company_name": insert_stmt.excluded.company_name,
                "active": insert_stmt.excluded.active,
            },
        ).returning(WatchlistEntry)
        result = await self._session.execute(upsert_stmt)
        row = result.scalar_one()
        return WatchlistRecord.model_validate(row)

    async def list_active_watchlist(self) -> Sequence[WatchlistRecord]:
        """Return every active watchlist entry, ticker-sorted."""
        stmt = (
            select(WatchlistEntry)
            .where(WatchlistEntry.active.is_(True))
            .order_by(WatchlistEntry.ticker)
        )
        result = await self._session.execute(stmt)
        return [WatchlistRecord.model_validate(row) for row in result.scalars().all()]

    # ---- filings ----

    async def record_filing(self, *, filing: NewFiling) -> FilingRecord | None:
        """Insert a filing if its accession is new; return ``None`` otherwise.

        Idempotent by design - the EDGAR watcher restarts often, and replaying
        a poll cycle must never duplicate rows.
        """
        stmt = (
            pg_insert(Filing)
            .values(
                accession_number=filing.accession_number,
                cik=filing.cik,
                ticker=filing.ticker,
                form=filing.form.value,
                filed_at=filing.filed_at,
                source_url=filing.source_url,
                report_period_end=filing.report_period_end,
                status=FilingStatus.DETECTED.value,
            )
            .on_conflict_do_nothing(index_elements=[Filing.accession_number])
            .returning(Filing)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return FilingRecord.model_validate(row) if row is not None else None

    async def get_filing(self, accession_number: str) -> FilingRecord | None:
        """Return a single :class:`FilingRecord` by accession, or ``None``."""
        stmt = select(Filing).where(Filing.accession_number == accession_number)
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return FilingRecord.model_validate(row) if row is not None else None

    async def known_accession_numbers(self, cik: str) -> set[str]:
        """Return the set of accession numbers already recorded for ``cik``.

        The watcher uses this to skip filings it has already processed.
        """
        stmt = select(Filing.accession_number).where(Filing.cik == cik)
        result = await self._session.execute(stmt)
        return set(result.scalars().all())

    async def list_filings_for_ticker(
        self, ticker: str, *, limit: int = 20
    ) -> Sequence[FilingRecord]:
        """Return the most-recent filings for ``ticker``, newest first."""
        stmt = (
            select(Filing)
            .where(Filing.ticker == ticker)
            .order_by(desc(Filing.filed_at))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [FilingRecord.model_validate(row) for row in result.scalars().all()]

    async def mark_filing_processed(self, accession_number: str) -> None:
        """Stamp ``accession_number`` as fully processed."""
        filing = await self._session.get(Filing, accession_number)
        if filing is None:
            return
        filing.status = FilingStatus.PROCESSED.value
        filing.processed_at = datetime.now(UTC)
        filing.error_message = None

    async def mark_filing_failed(self, accession_number: str, *, error: str) -> None:
        """Record a terminal error for ``accession_number``."""
        filing = await self._session.get(Filing, accession_number)
        if filing is None:
            return
        filing.status = FilingStatus.FAILED.value
        filing.error_message = error[:2000]
        filing.processed_at = datetime.now(UTC)

    # ---- financial facts ----

    async def insert_financial_facts(
        self,
        accession_number: str,
        facts: Iterable[NewFinancialFact],
    ) -> int:
        """Insert facts attached to ``accession_number``, skipping duplicates.

        Returns the number of rows actually inserted. The unique constraint
        on ``(filing_accession, concept, period_end, period_start, unit)``
        deduplicates within a single call as well as across calls.
        """
        payload = [
            {
                "filing_accession": accession_number,
                "cik": fact.cik,
                "taxonomy": fact.taxonomy,
                "concept": fact.concept,
                "unit": fact.unit,
                "value": fact.value,
                "period_type": fact.period_type,
                "period_start": fact.period_start,
                "period_end": fact.period_end,
                "fiscal_year": fact.fiscal_year,
                "fiscal_period": fact.fiscal_period,
                "form": fact.form,
                "filed": fact.filed,
                "frame": fact.frame,
            }
            for fact in facts
        ]
        if not payload:
            return 0
        stmt = (
            pg_insert(FinancialFact)
            .values(payload)
            .on_conflict_do_nothing(
                constraint="uq_financial_facts_filing_concept_period_unit",
            )
            .returning(FinancialFact.id)
        )
        result = await self._session.execute(stmt)
        return len(result.scalars().all())

    async def get_facts_for_filing(
        self, accession_number: str
    ) -> Sequence[FinancialFactRecord]:
        """Return every fact attached to ``accession_number``, concept-sorted."""
        stmt = (
            select(FinancialFact)
            .where(FinancialFact.filing_accession == accession_number)
            .order_by(FinancialFact.concept, FinancialFact.period_end)
        )
        result = await self._session.execute(stmt)
        return [FinancialFactRecord.model_validate(row) for row in result.scalars().all()]

    # ---- poll log ----

    async def record_poll(self, log: NewPollLog) -> PollLogRecord:
        """Persist one EDGAR poll cycle outcome."""
        row = EdgarPollLog(
            tickers_checked=log.tickers_checked,
            filings_found=log.filings_found,
            status=log.status.value,
            error_message=log.error_message,
        )
        self._session.add(row)
        await self._session.flush()
        return PollLogRecord.model_validate(row)

    async def last_successful_poll_at(self) -> PollLogRecord | None:
        """Return the most-recent ``ok`` poll, or ``None``."""
        stmt = (
            select(EdgarPollLog)
            .where(EdgarPollLog.status == PollStatus.OK.value)
            .order_by(desc(EdgarPollLog.polled_at))
            .limit(1)
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return PollLogRecord.model_validate(row) if row is not None else None

    # ---- daily LLM spend ----

    async def add_daily_spend(self, *, day: date, amount_usd: Decimal) -> Decimal:
        """Atomically add ``amount_usd`` to the day's spend and return the new total.

        Uses ``INSERT ... ON CONFLICT DO UPDATE`` so the counter survives
        concurrent calls from the web, worker, and watcher processes.
        """
        insert_stmt = pg_insert(DailyLLMSpend).values(day=day, spent_usd=amount_usd)
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[DailyLLMSpend.day],
            set_={
                "spent_usd": DailyLLMSpend.spent_usd + insert_stmt.excluded.spent_usd,
                "updated_at": datetime.now(UTC),
            },
        ).returning(DailyLLMSpend.spent_usd)
        result = await self._session.execute(upsert_stmt)
        return Decimal(result.scalar_one())

    async def get_daily_spend(self, day: date) -> Decimal:
        """Return today's recorded LLM spend (``0`` if no row yet)."""
        stmt = select(DailyLLMSpend.spent_usd).where(DailyLLMSpend.day == day)
        value = (await self._session.execute(stmt)).scalar_one_or_none()
        return Decimal(value) if value is not None else Decimal("0")

    # ---- consensus estimates ----

    async def upsert_consensus_estimate(
        self, estimate: NewConsensusEstimate
    ) -> ConsensusEstimateRecord:
        """Insert or refresh an analyst consensus row.

        Conflicts on ``(ticker, fiscal_year, fiscal_period, metric, source)``
        update the value, analyst count, and ``fetched_at`` timestamp so the
        most-recent fetch wins. Per-source rows coexist so the comparator can
        prefer Finnhub when both are present.
        """
        insert_stmt = pg_insert(ConsensusEstimate).values(
            ticker=estimate.ticker,
            fiscal_year=estimate.fiscal_year,
            fiscal_period=estimate.fiscal_period,
            metric=estimate.metric,
            value=estimate.value,
            analyst_count=estimate.analyst_count,
            source=estimate.source,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            constraint="uq_consensus_estimates_ticker_period_metric_source",
            set_={
                "value": insert_stmt.excluded.value,
                "analyst_count": insert_stmt.excluded.analyst_count,
                "fetched_at": datetime.now(UTC),
            },
        ).returning(ConsensusEstimate)
        result = await self._session.execute(upsert_stmt)
        return ConsensusEstimateRecord.model_validate(result.scalar_one())

    async def get_consensus_estimates(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
    ) -> Sequence[ConsensusEstimateRecord]:
        """Return every consensus row matching the (ticker, period) coordinate.

        Ordered by ``source`` so callers can prefer the first row when ranking
        by source priority (Finnhub before yfinance).
        """
        stmt = (
            select(ConsensusEstimate)
            .where(ConsensusEstimate.ticker == ticker)
            .where(ConsensusEstimate.fiscal_year == fiscal_year)
            .where(ConsensusEstimate.fiscal_period == fiscal_period)
            .order_by(ConsensusEstimate.source)
        )
        result = await self._session.execute(stmt)
        return [ConsensusEstimateRecord.model_validate(row) for row in result.scalars().all()]

    # ---- comparisons ----

    async def insert_comparison(self, comparison: NewComparison) -> ComparisonRecord:
        """Insert one comparison row; upserts on (filing_accession, metric).

        Re-running the comparator for the same filing overwrites the previous
        values so a critic-triggered retry sees the fresh numbers.
        """
        insert_stmt = pg_insert(Comparison).values(
            filing_accession=comparison.filing_accession,
            metric=comparison.metric,
            reported_value=comparison.reported_value,
            reported_unit=comparison.reported_unit,
            consensus_value=comparison.consensus_value,
            consensus_source=comparison.consensus_source,
            surprise_abs=comparison.surprise_abs,
            surprise_pct=comparison.surprise_pct,
            direction=comparison.direction,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            constraint="uq_comparisons_filing_metric",
            set_={
                "reported_value": insert_stmt.excluded.reported_value,
                "reported_unit": insert_stmt.excluded.reported_unit,
                "consensus_value": insert_stmt.excluded.consensus_value,
                "consensus_source": insert_stmt.excluded.consensus_source,
                "surprise_abs": insert_stmt.excluded.surprise_abs,
                "surprise_pct": insert_stmt.excluded.surprise_pct,
                "direction": insert_stmt.excluded.direction,
            },
        ).returning(Comparison)
        result = await self._session.execute(upsert_stmt)
        return ComparisonRecord.model_validate(result.scalar_one())

    async def list_comparisons_for_filing(
        self, accession_number: str
    ) -> Sequence[ComparisonRecord]:
        """Return every comparison row for ``accession_number``, metric-sorted."""
        stmt = (
            select(Comparison)
            .where(Comparison.filing_accession == accession_number)
            .order_by(Comparison.metric)
        )
        result = await self._session.execute(stmt)
        return [ComparisonRecord.model_validate(row) for row in result.scalars().all()]
