"""SQLAlchemy ORM declarations for the memory layer.

The memory layer owns every interaction with Postgres. Agent code talks to
the :class:`~app.memory.repository.Repository` and never imports these models
directly - this keeps the project-rule "no raw SQL in agent code" verifiable
by inspecting imports.

Phase 1 introduced five tables:

- ``filings``: one row per detected SEC filing on the watchlist.
- ``financial_facts``: numbers extracted from a filing's XBRL companyfacts.
- ``watchlist``: tickers the watcher polls.
- ``edgar_poll_log``: one row per poll cycle for the ``/health`` last-poll check.
- ``daily_llm_spend``: Postgres-backed daily LLM cost cap.

Phase 2 adds two:

- ``consensus_estimates``: analyst consensus values (mean) per metric/period
  fetched from Finnhub with a yfinance fallback.
- ``comparisons``: the comparator's per-metric reported-vs-consensus row,
  one per filing and metric.

Phase 3 adds two:

- ``filing_sections``: one row per paragraph of a parsed MD&A or Risk Factors
  section, with an optional pgvector embedding for similarity search.
- ``language_diffs``: material changes (added / removed / modified) between a
  current and prior quarter's section, as produced by the language-differ node.

The module is approaching the project's 300-line guideline as of Phase 3.
Future additions should consider splitting along a clean responsibility
boundary (e.g., language-vs-numbers tables) rather than continuing to
grow this single module.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for every ORM model in the memory layer."""


class Filing(Base):
    """An SEC filing the watcher has detected.

    Status is a state machine: ``detected -> processing -> processed`` on the
    happy path, with ``failed`` as the terminal error state. The ``status``
    column is also the only mutable column on this row - the rest are append-
    only per the project rule on memory.
    """

    __tablename__ = "filings"

    # Widened from 32 to 64 in Phase 4B (migration 0007) so upload-derived
    # accessions (e.g. ``upload-{uuid4().hex}`` is 39 chars) fit alongside
    # the 18-char real SEC accession numbers.
    accession_number: Mapped[str] = mapped_column(String(64), primary_key=True)
    cik: Mapped[str] = mapped_column(String(10), nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    # Widened from 8 to 16 in Phase 4B (migration 0006) so the literal
    # "TRANSCRIPT" fits alongside the SEC-form abbreviations.
    form: Mapped[str] = mapped_column(String(16), nullable=False)
    filed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    primary_document: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="detected")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    facts: Mapped[list[FinancialFact]] = relationship(
        back_populates="filing", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint(
            "form IN ('10-K', '10-Q', '8-K', 'TRANSCRIPT')",
            name="filings_form_supported",
        ),
        CheckConstraint(
            "status IN ('detected', 'processing', 'processed', 'failed')",
            name="filings_status_valid",
        ),
        Index("ix_filings_cik_filed_at", "cik", "filed_at"),
        Index("ix_filings_ticker_filed_at", "ticker", "filed_at"),
    )


class FinancialFact(Base):
    """A single XBRL fact attached to a filing.

    One row per (concept, period, unit) so a filing typically owns dozens of
    rows. Deduped at insert time by a composite unique constraint - the
    repository uses ``ON CONFLICT DO NOTHING`` for idempotent loads.
    """

    __tablename__ = "financial_facts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filing_accession: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="CASCADE"),
        nullable=False,
    )
    cik: Mapped[str] = mapped_column(String(10), nullable=False)
    taxonomy: Mapped[str] = mapped_column(String(32), nullable=False)
    concept: Mapped[str] = mapped_column(String(128), nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(28, 6), nullable=False)
    period_type: Mapped[str] = mapped_column(String(16), nullable=False)
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fiscal_period: Mapped[str | None] = mapped_column(String(8), nullable=True)
    form: Mapped[str | None] = mapped_column(String(8), nullable=True)
    filed: Mapped[date | None] = mapped_column(Date, nullable=True)
    frame: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    filing: Mapped[Filing] = relationship(back_populates="facts")

    __table_args__ = (
        UniqueConstraint(
            "filing_accession",
            "concept",
            "period_end",
            "period_start",
            "unit",
            name="uq_financial_facts_filing_concept_period_unit",
        ),
        CheckConstraint(
            "period_type IN ('instant', 'duration')",
            name="financial_facts_period_type_valid",
        ),
        Index("ix_financial_facts_cik_concept", "cik", "concept"),
    )


class WatchlistEntry(Base):
    """A ticker the EDGAR watcher polls.

    The repository upserts by ticker, so updating company metadata or toggling
    the ``active`` flag is a single call. Phase 1 seeds five tickers; phase 6
    gives the dashboard a UI to manage them.
    """

    __tablename__ = "watchlist"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    cik: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    company_name: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class EdgarPollLog(Base):
    """One row per EDGAR poll cycle.

    The ``/health`` endpoint reads the most recent ``ok`` row to assert the
    watcher polled within the SLA window (5 minutes by default).
    """

    __tablename__ = "edgar_poll_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    polled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    tickers_checked: Mapped[int] = mapped_column(Integer, nullable=False)
    filings_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(8), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("status IN ('ok', 'error')", name="edgar_poll_log_status_valid"),
        Index("ix_edgar_poll_log_polled_at", "polled_at"),
    )


class DailyLLMSpend(Base):
    """Postgres-backed counter for the daily LLM cost cap.

    Phase 0 enforced the cap in-process. Phase 1 lifts the counter into the
    database so the web, worker, and watcher processes share a single budget
    and the cap survives restarts.
    """

    __tablename__ = "daily_llm_spend"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    spent_usd: Mapped[Decimal] = mapped_column(
        Numeric(18, 6), nullable=False, default=Decimal("0")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ConsensusEstimate(Base):
    """Analyst consensus value for one (ticker, period, metric).

    The comparator pairs each row with the matching :class:`FinancialFact` to
    compute the per-metric surprise. Rows are append-only; the most-recent
    row per ``(ticker, fiscal_year, fiscal_period, metric, source)`` wins via
    the ``fetched_at`` ordering.
    """

    __tablename__ = "consensus_estimates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_period: Mapped[str] = mapped_column(String(8), nullable=False)
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(28, 6), nullable=False)
    analyst_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "fiscal_year",
            "fiscal_period",
            "metric",
            "source",
            name="uq_consensus_estimates_ticker_period_metric_source",
        ),
        CheckConstraint(
            "source IN ('finnhub', 'yfinance')",
            name="consensus_estimates_source_valid",
        ),
        Index("ix_consensus_estimates_ticker_period", "ticker", "fiscal_year", "fiscal_period"),
    )


class Comparison(Base):
    """Reported vs consensus comparison for one metric on one filing.

    Surprise is signed: positive means the reported value beat consensus on a
    higher-is-better metric. The comparator owns the sign convention - see
    :mod:`app.agents.comparator`.
    """

    __tablename__ = "comparisons"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filing_accession: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="CASCADE"),
        nullable=False,
    )
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    reported_value: Mapped[Decimal] = mapped_column(Numeric(28, 6), nullable=False)
    reported_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    consensus_value: Mapped[Decimal | None] = mapped_column(Numeric(28, 6), nullable=True)
    consensus_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    surprise_abs: Mapped[Decimal | None] = mapped_column(Numeric(28, 6), nullable=True)
    surprise_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    direction: Mapped[str | None] = mapped_column(String(8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "filing_accession",
            "metric",
            name="uq_comparisons_filing_metric",
        ),
        CheckConstraint(
            "direction IS NULL OR direction IN ('beat', 'miss', 'in_line')",
            name="comparisons_direction_valid",
        ),
        Index("ix_comparisons_filing_accession", "filing_accession"),
    )


class FilingSection(Base):
    """One paragraph of a parsed MD&A or Risk Factors section.

    The differ persists every paragraph of every parsed section so each
    filing seeds the next quarter's baseline, regardless of whether the
    current run's diff itself degrades.
    """

    __tablename__ = "filing_sections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filing_accession: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="CASCADE"),
        nullable=False,
    )
    cik: Mapped[str] = mapped_column(String(10), nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    section_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    paragraph_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_sha: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    # psycopg3 returns list[float]; psycopg2 would return numpy.ndarray.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "filing_accession",
            "section_kind",
            "paragraph_index",
            name="uq_filing_sections_filing_section_paragraph",
        ),
        CheckConstraint(
            "section_kind IN ('mda', 'risk_factors')",
            name="filing_sections_section_kind_valid",
        ),
        Index(
            "ix_filing_sections_ticker_section_filing",
            "ticker",
            "section_kind",
            "filing_accession",
        ),
        Index("ix_filing_sections_cik_section", "cik", "section_kind"),
    )


class LanguageDiff(Base):
    """One material change between a current and prior quarter's section.

    Unchanged paragraphs are NOT persisted - only ``added`` / ``removed`` /
    ``modified`` reach the table. Severity is computed by the agent node.
    """

    __tablename__ = "language_diffs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filing_accession: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="CASCADE"),
        nullable=False,
    )
    prior_filing_accession: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="SET NULL"),
        nullable=True,
    )
    section_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    change_type: Mapped[str] = mapped_column(String(16), nullable=False)
    current_section_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("filing_sections.id", ondelete="CASCADE"),
        nullable=True,
    )
    prior_section_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("filing_sections.id", ondelete="SET NULL"),
        nullable=True,
    )
    similarity: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    severity: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # NULLS NOT DISTINCT (Postgres 15+) makes NULL == NULL inside this
        # constraint, so re-running the differ for a cold-start filing (where
        # both section IDs are NULL) produces exactly one row, not N duplicates.
        UniqueConstraint(
            "filing_accession",
            "section_kind",
            "change_type",
            "current_section_id",
            "prior_section_id",
            name="uq_language_diffs_filing_section_change_pair",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "change_type IN ('added', 'removed', 'modified')",
            name="language_diffs_change_type_valid",
        ),
        CheckConstraint(
            "severity IN ('major', 'minor')",
            name="language_diffs_severity_valid",
        ),
        CheckConstraint(
            "section_kind IN ('mda', 'risk_factors')",
            name="language_diffs_section_kind_valid",
        ),
        Index(
            "ix_language_diffs_filing_section",
            "filing_accession",
            "section_kind",
        ),
    )


class UploadedDocument(Base):
    """Append-only record of a user-uploaded filing.

    SHA-256 of the raw bytes is unique so re-uploading the same content
    returns the existing row instead of inserting a duplicate.
    """

    __tablename__ = "uploaded_documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    upload_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    filing_type: Mapped[str] = mapped_column(String(16), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    parsed_text: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class QAPair(Base):
    """One analyst Q&A exchange extracted from a transcript filing.

    Append-only. The transcript-analyzer bulk-inserts rows under a single
    transaction with ``ON CONFLICT DO NOTHING`` on
    ``(filing_accession, ordinal)`` so re-running the agent on the same
    transcript is safe.
    """

    __tablename__ = "qa_pairs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filing_accession: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    analyst_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_class: Mapped[str] = mapped_column(String(16), nullable=False)
    sha256_text: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "filing_accession",
            "ordinal",
            name="uq_qa_pairs_filing_accession_ordinal",
        ),
        CheckConstraint(
            "answer_class IN ('direct', 'partial', 'deflected')",
            name="qa_pairs_answer_class_valid",
        ),
        Index("ix_qa_pairs_filing_accession", "filing_accession"),
    )


class Commitment(Base):
    """A forward-looking commitment a management team made on the transcript.

    The ``status``, ``resolved_filing_accession``, ``resolved_reason``, and
    ``updated_at`` columns are the ONLY mutable surface in the memory layer
    outside the small set of ``filings``/``daily_llm_spend`` updates - the
    cross-quarter reconciliation pass flips ``open`` to ``met`` / ``missed``
    / ``still_open`` as later filings arrive.
    """

    __tablename__ = "commitments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filing_accession: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="CASCADE"),
        nullable=False,
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    commitment_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_period: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_quote: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="open"
    )
    resolved_filing_accession: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="SET NULL"),
        nullable=True,
    )
    resolved_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('open', 'met', 'missed', 'still_open')",
            name="commitments_status_valid",
        ),
        Index("ix_commitments_ticker_status", "ticker", "status"),
    )


class Note(Base):
    """An accepted synthesized note for one filing.

    Append-only. One row per filing_accession; re-runs return the existing
    row via ON CONFLICT DO NOTHING. Per-event cost/latency are deliberately
    not stored here -- a per-event metrics table is deferred to Phase 7.
    """

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filing_accession: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    markdown_body: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_template_name: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_template_sha: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    critic_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_notes_ticker_created", "ticker", "created_at"),
    )


class Peer(Base):
    """A curated ``(ticker, peer_ticker)`` mapping for the peer reader.

    Append-only via upsert. The ``source`` column is forward-compatible for
    auto-discovery; currently constrained to ``'curated'``.
    """

    __tablename__ = "peers"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    peer_ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="curated")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("ticker <> peer_ticker", name="peers_no_self_reference"),
        CheckConstraint("source IN ('curated')", name="peers_source_valid"),
        Index("ix_peers_ticker", "ticker"),
    )
