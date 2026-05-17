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
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.models import (
    Commitment,
    Comparison,
    ConsensusEstimate,
    DailyLLMSpend,
    EdgarPollLog,
    Filing,
    FilingSection,
    FinancialFact,
    LanguageDiff,
    Note,
    Peer,
    QAPair,
    UploadedDocument,
    WatchlistEntry,
)
from app.memory.schemas import (
    CommitmentRecord,
    CommitmentStatus,
    ComparisonRecord,
    ConsensusEstimateRecord,
    FilingRecord,
    FilingSectionRecord,
    FilingStatus,
    FinancialFactRecord,
    LanguageDiffRecord,
    NewCommitment,
    NewComparison,
    NewConsensusEstimate,
    NewFiling,
    NewFilingSection,
    NewFinancialFact,
    NewLanguageDiff,
    NewPollLog,
    NewQAPair,
    NewUploadedDocument,
    NoteCreate,
    NoteRead,
    PeerCommitmentSignal,
    PeerCreate,
    PeerLanguageDiffSignal,
    PeerSignals,
    PollLogRecord,
    PollStatus,
    QAPairRecord,
    SectionKind,
    UploadedDocumentRecord,
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

    async def get_watchlist_entry_by_ticker(
        self, ticker: str
    ) -> WatchlistRecord | None:
        """Return the watchlist entry whose ticker matches (case-insensitive), or None."""
        stmt = select(WatchlistEntry).where(WatchlistEntry.ticker == ticker.upper())
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return WatchlistRecord.model_validate(row) if row is not None else None

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

    # ---- filing sections ----

    async def insert_filing_sections(
        self,
        rows: Iterable[NewFilingSection],
    ) -> int:
        """Insert filing-section paragraphs, skipping duplicates.

        Conflicts on ``(filing_accession, section_kind, paragraph_index)`` are
        silently ignored so the differ can re-run safely without creating
        phantom rows.  Returns the number of rows actually inserted.
        """
        payload = [
            {
                "filing_accession": row.filing_accession,
                "cik": row.cik,
                "ticker": row.ticker,
                "section_kind": row.section_kind.value,
                "paragraph_index": row.paragraph_index,
                "text": row.text,
                "text_sha": row.text_sha,
                "embedding": row.embedding,
                "embedding_model": row.embedding_model,
            }
            for row in rows
        ]
        if not payload:
            return 0
        stmt = (
            pg_insert(FilingSection)
            .values(payload)
            .on_conflict_do_nothing(
                constraint="uq_filing_sections_filing_section_paragraph",
            )
            .returning(FilingSection.id)
        )
        result = await self._session.execute(stmt)
        return len(result.scalars().all())

    async def update_section_embeddings(
        self,
        *,
        updates: Sequence[tuple[int, list[float], str]],
    ) -> int:
        """Set the ``embedding`` and ``embedding_model`` columns for rows by id.

        Used by the differ after a successful batched embeddings call to
        back-fill vectors onto previously-inserted rows.  Returns the number of
        rows actually updated.
        """
        if not updates:
            return 0
        count = 0
        for row_id, vector, model in updates:
            section = await self._session.get(FilingSection, row_id)
            if section is None:
                continue
            section.embedding = vector
            section.embedding_model = model
            count += 1
        return count

    async def get_prior_quarter_sections(
        self,
        *,
        ticker: str,
        section_kind: SectionKind,
        before: date,
    ) -> Sequence[FilingSectionRecord]:
        """Return paragraphs from the most-recent filing strictly before ``before``.

        The differ uses this to find the baseline section to align the current
        filing against.  The ``before`` bound is exclusive: a filing whose
        ``filed_at`` equals ``before`` midnight UTC is NOT included.  Returns
        ``[]`` when no prior filing with that section exists.
        """
        cutoff = datetime.combine(before, datetime.min.time(), tzinfo=UTC)
        anchor_stmt = (
            select(Filing.accession_number)
            .join(FilingSection, FilingSection.filing_accession == Filing.accession_number)
            .where(Filing.ticker == ticker)
            .where(FilingSection.section_kind == section_kind.value)
            .where(Filing.filed_at < cutoff)
            .order_by(desc(Filing.filed_at))
            .limit(1)
        )
        accession = (await self._session.execute(anchor_stmt)).scalar_one_or_none()
        if accession is None:
            return []
        stmt = (
            select(FilingSection)
            .where(FilingSection.filing_accession == accession)
            .where(FilingSection.section_kind == section_kind.value)
            .order_by(FilingSection.paragraph_index)
        )
        result = await self._session.execute(stmt)
        return [FilingSectionRecord.model_validate(row) for row in result.scalars().all()]

    async def get_filing_sections(
        self, *, accession_number: str, section_kind: SectionKind
    ) -> Sequence[FilingSectionRecord]:
        """Return paragraphs for one filing's section, ordered by paragraph_index."""
        stmt = (
            select(FilingSection)
            .where(FilingSection.filing_accession == accession_number)
            .where(FilingSection.section_kind == section_kind.value)
            .order_by(FilingSection.paragraph_index)
        )
        result = await self._session.execute(stmt)
        return [
            FilingSectionRecord.model_validate(row) for row in result.scalars().all()
        ]

    # ---- language diffs ----

    async def insert_language_diffs(
        self,
        rows: Iterable[NewLanguageDiff],
    ) -> int:
        """Insert language-diff rows, skipping duplicates.

        Conflicts on the unique constraint over
        ``(filing_accession, section_kind, change_type, current_section_id, prior_section_id)``
        are ignored so re-running the differ for a filing is safe.
        Returns the number of rows actually inserted.
        """
        payload = [
            {
                "filing_accession": row.filing_accession,
                "prior_filing_accession": row.prior_filing_accession,
                "section_kind": row.section_kind.value,
                "change_type": row.change_type.value,
                "current_section_id": row.current_section_id,
                "prior_section_id": row.prior_section_id,
                "similarity": row.similarity,
                "severity": row.severity.value,
            }
            for row in rows
        ]
        if not payload:
            return 0
        stmt = (
            pg_insert(LanguageDiff)
            .values(payload)
            .on_conflict_do_nothing(
                constraint="uq_language_diffs_filing_section_change_pair",
            )
            .returning(LanguageDiff.id)
        )
        result = await self._session.execute(stmt)
        return len(result.scalars().all())

    async def list_language_diffs_for_filing(
        self, accession_number: str
    ) -> Sequence[LanguageDiffRecord]:
        """Return every language-diff row attached to ``accession_number``.

        Results are ordered by insertion order (``id`` ascending) so callers
        see diffs in the sequence the differ produced them.
        """
        stmt = (
            select(LanguageDiff)
            .where(LanguageDiff.filing_accession == accession_number)
            .order_by(LanguageDiff.id)
        )
        result = await self._session.execute(stmt)
        return [
            LanguageDiffRecord.model_validate(row) for row in result.scalars().all()
        ]

    # ---- uploaded documents ----

    async def add_uploaded_document(
        self, new: NewUploadedDocument
    ) -> UploadedDocumentRecord:
        """Insert a new ``uploaded_documents`` row and return its detached record.

        Callers commit. Re-inserting the same ``content_sha256`` raises
        ``sqlalchemy.exc.IntegrityError`` on flush.
        """
        row = UploadedDocument(
            upload_id=new.upload_id,
            ticker=new.ticker,
            filing_type=new.filing_type,
            original_filename=new.original_filename,
            content_sha256=new.content_sha256,
            parsed_text=new.parsed_text,
            parsed_char_count=new.parsed_char_count,
            page_count=new.page_count,
        )
        self._session.add(row)
        await self._session.flush()
        return UploadedDocumentRecord.model_validate(row)

    async def get_uploaded_document_by_sha256(
        self, content_sha256: str
    ) -> UploadedDocumentRecord | None:
        """Return the document with the given content hash, or ``None``."""
        stmt = select(UploadedDocument).where(
            UploadedDocument.content_sha256 == content_sha256
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return UploadedDocumentRecord.model_validate(row) if row is not None else None

    async def get_uploaded_document(
        self, upload_id: str
    ) -> UploadedDocumentRecord | None:
        """Return the document with the given ``upload_id``, or ``None``."""
        stmt = select(UploadedDocument).where(UploadedDocument.upload_id == upload_id)
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return UploadedDocumentRecord.model_validate(row) if row is not None else None

    # ---- qa pairs ----

    async def add_qa_pairs(
        self,
        *,
        filing_accession: str,
        pairs: Sequence[NewQAPair],
    ) -> Sequence[QAPairRecord]:
        """Bulk-insert Q&A pairs; idempotent on ``(filing_accession, ordinal)``.

        Uses ``ON CONFLICT DO NOTHING`` on the unique constraint so re-running
        the transcript analyzer is safe. After the insert, a follow-up
        ``SELECT`` returns the persisted DTOs for every supplied ordinal -
        callers receive a consistent view regardless of whether a given row
        was newly inserted or already present.
        """
        if not pairs:
            return []
        payload = [
            {
                "filing_accession": filing_accession,
                "ordinal": p.ordinal,
                "analyst_name": p.analyst_name,
                "question_text": p.question_text,
                "answer_text": p.answer_text,
                "answer_class": p.answer_class.value,
                "sha256_text": p.sha256_text,
            }
            for p in pairs
        ]
        insert_stmt = (
            pg_insert(QAPair)
            .values(payload)
            .on_conflict_do_nothing(
                constraint="uq_qa_pairs_filing_accession_ordinal",
            )
        )
        await self._session.execute(insert_stmt)
        ordinals = [p.ordinal for p in pairs]
        select_stmt = (
            select(QAPair)
            .where(QAPair.filing_accession == filing_accession)
            .where(QAPair.ordinal.in_(ordinals))
            .order_by(QAPair.ordinal)
        )
        result = await self._session.execute(select_stmt)
        return [QAPairRecord.model_validate(row) for row in result.scalars().all()]

    async def list_qa_pairs_for_filing(
        self, filing_accession: str
    ) -> Sequence[QAPairRecord]:
        """Return Q&A pairs for ``filing_accession`` in ascending ordinal order."""
        stmt = (
            select(QAPair)
            .where(QAPair.filing_accession == filing_accession)
            .order_by(QAPair.ordinal)
        )
        result = await self._session.execute(stmt)
        return [QAPairRecord.model_validate(row) for row in result.scalars().all()]

    # ---- commitments ----

    async def add_commitments(
        self,
        *,
        filing_accession: str,
        ticker: str,
        commitments: Sequence[NewCommitment],
    ) -> Sequence[CommitmentRecord]:
        """Bulk-insert commitments; idempotent on ``(filing_accession, source_quote)``.

        The schema has no UNIQUE constraint covering this pair (the source
        quote is free-form text), so idempotency is enforced in Python: the
        method SELECTs existing rows for the supplied ``source_quote`` values
        and only inserts the missing ones. A follow-up SELECT returns DTOs
        for every supplied ``source_quote`` so the caller gets one row per
        input regardless of whether it was new or already present.

        Callers must commit. The caller is responsible for ensuring the
        ``source_quote`` strings are stable verbatim transcript spans - any
        whitespace or punctuation drift will defeat the dedupe.
        """
        if not commitments:
            return []
        quotes = [c.source_quote for c in commitments]
        existing_stmt = select(Commitment.source_quote).where(
            Commitment.filing_accession == filing_accession,
            Commitment.source_quote.in_(quotes),
        )
        existing_result = await self._session.execute(existing_stmt)
        existing_quotes = set(existing_result.scalars().all())

        to_insert = [c for c in commitments if c.source_quote not in existing_quotes]
        if to_insert:
            payload = [
                {
                    "filing_accession": filing_accession,
                    "ticker": ticker,
                    "commitment_text": c.commitment_text,
                    "target_period": c.target_period,
                    "source_quote": c.source_quote,
                }
                for c in to_insert
            ]
            await self._session.execute(pg_insert(Commitment).values(payload))

        select_stmt = (
            select(Commitment)
            .where(Commitment.filing_accession == filing_accession)
            .where(Commitment.source_quote.in_(quotes))
            .order_by(Commitment.id)
        )
        result = await self._session.execute(select_stmt)
        return [CommitmentRecord.model_validate(row) for row in result.scalars().all()]

    async def get_open_commitments(self, ticker: str) -> Sequence[CommitmentRecord]:
        """Return ``status='open'`` commitments for ``ticker``, oldest first.

        The cross-quarter reconciliation pass uses this to find prior
        guidance that still needs a verdict.
        """
        stmt = (
            select(Commitment)
            .where(Commitment.ticker == ticker)
            .where(Commitment.status == CommitmentStatus.OPEN.value)
            .order_by(Commitment.created_at)
        )
        result = await self._session.execute(stmt)
        return [CommitmentRecord.model_validate(row) for row in result.scalars().all()]

    async def update_commitment_status(
        self,
        *,
        commitment_id: int,
        status: CommitmentStatus,
        resolved_filing_accession: str | None,
        resolved_reason: str | None,
    ) -> None:
        """Atomically rewrite a commitment's four mutable fields plus ``updated_at``.

        This is the only place commitment rows are mutated. ``updated_at`` is
        set to ``now()`` in the same UPDATE statement because the schema has
        no trigger; relying on the DB clock keeps the timestamp consistent
        with the row's ``created_at`` (also DB-driven).
        """
        stmt = (
            update(Commitment)
            .where(Commitment.id == commitment_id)
            .values(
                status=status.value,
                resolved_filing_accession=resolved_filing_accession,
                resolved_reason=resolved_reason,
                updated_at=func.now(),
            )
        )
        await self._session.execute(stmt)

    # ---- notes ----

    async def insert_note(self, note: NoteCreate) -> int:
        """Idempotent note insert keyed by ``filing_accession``.

        Returns the id of the newly-inserted row, or the id of the existing
        row when a note for the same filing already exists. This preserves
        the append-only memory rule: re-running the synthesizer for the same
        filing does not overwrite the accepted note.
        """
        stmt = (
            pg_insert(Note)
            .values(
                filing_accession=note.filing_accession,
                ticker=note.ticker,
                markdown_body=note.markdown_body,
                prompt_template_name=note.prompt_template_name,
                prompt_template_sha=note.prompt_template_sha,
                critic_attempts=note.critic_attempts,
            )
            .on_conflict_do_nothing(index_elements=[Note.filing_accession])
            .returning(Note.id)
        )
        result = await self._session.execute(stmt)
        inserted_id = result.scalar_one_or_none()
        if inserted_id is not None:
            return int(inserted_id)
        existing = await self._session.execute(
            select(Note.id).where(Note.filing_accession == note.filing_accession)
        )
        return int(existing.scalar_one())

    async def get_latest_note(self, *, ticker: str) -> NoteRead | None:
        """Return the most-recently-created note for ``ticker``, or ``None``.

        Used by the peer-critic node to surface the prior accepted note for
        cross-quarter comparison.
        """
        stmt = (
            select(Note)
            .where(Note.ticker == ticker)
            .order_by(Note.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return NoteRead(
            id=row.id,
            filing_accession=row.filing_accession,
            ticker=row.ticker,
            markdown_body=row.markdown_body,
            prompt_template_name=row.prompt_template_name,
            prompt_template_sha=row.prompt_template_sha,
            critic_attempts=row.critic_attempts,
            created_at=row.created_at,
        )

    # ---- peers ----

    async def upsert_peer(self, peer: PeerCreate) -> None:
        """Idempotent peer insert; no-op on duplicate ``(ticker, peer_ticker)``."""
        stmt = (
            pg_insert(Peer)
            .values(
                ticker=peer.ticker,
                peer_ticker=peer.peer_ticker,
                source=peer.source,
            )
            .on_conflict_do_nothing(index_elements=[Peer.ticker, Peer.peer_ticker])
        )
        await self._session.execute(stmt)

    async def list_peers(self, *, ticker: str) -> list[str]:
        """Return peer tickers for ``ticker``, sorted alphabetically."""
        stmt = (
            select(Peer.peer_ticker)
            .where(Peer.ticker == ticker)
            .order_by(Peer.peer_ticker)
        )
        result = await self._session.execute(stmt)
        return [str(r) for r in result.scalars().all()]

    async def get_recent_peer_signals(
        self,
        *,
        peer_ticker: str,
        max_age_days: int = 180,
    ) -> PeerSignals:
        """Return the peer's most-recent language diffs and open commitments.

        ``language_diffs``: from the peer's most recent processed 10-K or
        10-Q within ``max_age_days``, filtered to ``severity='major'``.
        ``commitments``: from the peer's most recent processed TRANSCRIPT
        within ``max_age_days``, filtered to ``status='open'``.
        Returns an empty :class:`PeerSignals` for cold-start or stale peers.
        """
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)

        language_filing_result = await self._session.execute(
            select(Filing.accession_number)
            .where(
                Filing.ticker == peer_ticker,
                Filing.form.in_(("10-K", "10-Q")),
                Filing.filed_at >= cutoff,
                Filing.status == "processed",
            )
            .order_by(desc(Filing.filed_at))
            .limit(1)
        )
        accession = language_filing_result.scalar_one_or_none()

        language_diffs: list[PeerLanguageDiffSignal] = []
        if accession is not None:
            diff_rows = await self._session.execute(
                select(LanguageDiff)
                .where(
                    LanguageDiff.filing_accession == accession,
                    LanguageDiff.severity == "major",
                )
            )
            for row in diff_rows.scalars().all():
                text_body = await self._language_diff_text(row)
                language_diffs.append(
                    PeerLanguageDiffSignal(
                        text=text_body,
                        severity=row.severity,
                        source_filing_accession=accession,
                    )
                )

        transcript_filing_result = await self._session.execute(
            select(Filing.accession_number)
            .where(
                Filing.ticker == peer_ticker,
                Filing.form == "TRANSCRIPT",
                Filing.filed_at >= cutoff,
                Filing.status == "processed",
            )
            .order_by(desc(Filing.filed_at))
            .limit(1)
        )
        t_accession = transcript_filing_result.scalar_one_or_none()

        commitments: list[PeerCommitmentSignal] = []
        if t_accession is not None:
            commitment_rows = await self._session.execute(
                select(Commitment)
                .where(
                    Commitment.filing_accession == t_accession,
                    Commitment.status == "open",
                )
            )
            for c in commitment_rows.scalars().all():
                commitments.append(
                    PeerCommitmentSignal(
                        text=c.commitment_text,
                        source_filing_accession=t_accession,
                    )
                )

        return PeerSignals(language_diffs=language_diffs, commitments=commitments)

    async def _language_diff_text(self, row: LanguageDiff) -> str:
        """Fetch the current-section text for a language diff row.

        Returns an empty string when ``current_section_id`` is ``None`` or
        the referenced section has been deleted.
        """
        if row.current_section_id is None:
            return ""
        sec = await self._session.execute(
            select(FilingSection.text).where(FilingSection.id == row.current_section_id)
        )
        return str(sec.scalar_one_or_none() or "")
