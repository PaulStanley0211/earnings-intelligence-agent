"""Pydantic DTOs and enums used at the boundary of the memory layer.

The ORM models in :mod:`app.memory.models` describe what is on disk; these
DTOs are what callers hand to and receive from the
:class:`~app.memory.repository.Repository`. Keeping the boundary types
separate from the ORM rows means callers do not need an active SQLAlchemy
session attached to every result they want to inspect.

The module is approaching the project's 300-line guideline as of Phase 3.
Future additions should consider splitting along a clean responsibility
boundary (numbers-track DTOs vs language-track DTOs) rather than continuing
to grow this single module.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from app.models.state import (
    AnswerClass as AnswerClass,
)
from app.models.state import (
    CommitmentStatus as CommitmentStatus,
)
from app.models.state import (
    FilingForm,
)


class FilingStatus(StrEnum):
    """Lifecycle states for a :class:`~app.memory.models.Filing` row."""

    DETECTED = "detected"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class PollStatus(StrEnum):
    """Outcome states for an :class:`~app.memory.models.EdgarPollLog` row."""

    OK = "ok"
    ERROR = "error"


PeriodType = Literal["instant", "duration"]


class NewFiling(BaseModel):
    """Inputs to :meth:`Repository.record_filing`."""

    model_config = ConfigDict(frozen=True)

    accession_number: str
    cik: str
    ticker: str
    form: FilingForm
    filed_at: datetime
    source_url: str
    report_period_end: date | None = None


class FilingRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.Filing` row."""

    model_config = ConfigDict(from_attributes=True)

    accession_number: str
    cik: str
    ticker: str
    form: FilingForm
    filed_at: datetime
    source_url: str
    primary_document: str | None = None
    report_period_end: date | None
    status: FilingStatus
    processed_at: datetime | None
    error_message: str | None
    created_at: datetime


class NewFinancialFact(BaseModel):
    """Inputs to :meth:`Repository.insert_financial_facts`."""

    model_config = ConfigDict(frozen=True)

    filing_accession: str
    cik: str
    taxonomy: str
    concept: str
    unit: str
    value: Decimal
    period_type: PeriodType
    period_start: date | None
    period_end: date
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    form: str | None = None
    filed: date | None = None
    frame: str | None = None


class FinancialFactRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.FinancialFact` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filing_accession: str
    cik: str
    taxonomy: str
    concept: str
    unit: str
    value: Decimal
    period_type: PeriodType
    period_start: date | None
    period_end: date
    fiscal_year: int | None
    fiscal_period: str | None
    form: str | None
    filed: date | None
    frame: str | None
    created_at: datetime


class NewPollLog(BaseModel):
    """Inputs to :meth:`Repository.record_poll`."""

    model_config = ConfigDict(frozen=True)

    tickers_checked: int = Field(..., ge=0)
    filings_found: int = Field(default=0, ge=0)
    status: PollStatus
    error_message: str | None = None


class PollLogRecord(BaseModel):
    """Detached view of an :class:`~app.memory.models.EdgarPollLog` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    polled_at: datetime
    tickers_checked: int
    filings_found: int
    status: PollStatus
    error_message: str | None


class WatchlistRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.WatchlistEntry` row."""

    model_config = ConfigDict(from_attributes=True)

    ticker: str
    cik: str
    company_name: str
    active: bool
    added_at: datetime


# ---- Phase 2: consensus and comparisons ----


ConsensusSource = Literal["finnhub", "yfinance"]
ComparisonDirection = Literal["beat", "miss", "in_line"]
ComparisonMetric = Literal["revenue", "eps_diluted", "eps_basic", "net_income"]


class NewConsensusEstimate(BaseModel):
    """Inputs to :meth:`Repository.upsert_consensus_estimate`."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    fiscal_year: int
    fiscal_period: str
    metric: ComparisonMetric
    value: Decimal
    analyst_count: int | None = None
    source: ConsensusSource


class ConsensusEstimateRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.ConsensusEstimate` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    ticker: str
    fiscal_year: int
    fiscal_period: str
    metric: ComparisonMetric
    value: Decimal
    analyst_count: int | None
    source: ConsensusSource
    fetched_at: datetime


class NewComparison(BaseModel):
    """Inputs to :meth:`Repository.insert_comparison`."""

    model_config = ConfigDict(frozen=True)

    filing_accession: str
    metric: ComparisonMetric
    reported_value: Decimal
    reported_unit: str
    consensus_value: Decimal | None = None
    consensus_source: ConsensusSource | None = None
    surprise_abs: Decimal | None = None
    surprise_pct: Decimal | None = None
    direction: ComparisonDirection | None = None


class ComparisonRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.Comparison` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filing_accession: str
    metric: ComparisonMetric
    reported_value: Decimal
    reported_unit: str
    consensus_value: Decimal | None
    consensus_source: ConsensusSource | None
    surprise_abs: Decimal | None
    surprise_pct: Decimal | None
    direction: ComparisonDirection | None
    created_at: datetime


# ---- Phase 3: filing sections and language diffs ----


class SectionKind(StrEnum):
    """Kind of parsed filing section the language differ recognises."""

    MDA = "mda"
    RISK_FACTORS = "risk_factors"


class ChangeType(StrEnum):
    """Classification of a single language change."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class Severity(StrEnum):
    """Severity tier for a persisted language diff."""

    MAJOR = "major"
    MINOR = "minor"


class NewFilingSection(BaseModel):
    """Inputs to :meth:`Repository.insert_filing_sections`."""

    model_config = ConfigDict(frozen=True)

    filing_accession: str
    cik: str
    ticker: str
    section_kind: SectionKind
    paragraph_index: int = Field(..., ge=0)
    text: str
    text_sha: str = Field(..., min_length=64, max_length=64)
    embedding: list[float] | None = None
    embedding_model: str | None = None


class FilingSectionRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.FilingSection` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filing_accession: str
    cik: str
    ticker: str
    section_kind: SectionKind
    paragraph_index: int
    text: str
    text_sha: str
    embedding: list[float] | None
    embedding_model: str | None
    created_at: datetime


class NewLanguageDiff(BaseModel):
    """Inputs to :meth:`Repository.insert_language_diffs`."""

    model_config = ConfigDict(frozen=True)

    filing_accession: str
    prior_filing_accession: str | None = None
    section_kind: SectionKind
    change_type: ChangeType
    current_section_id: int | None = None
    prior_section_id: int | None = None
    similarity: Decimal | None = None
    severity: Severity


class LanguageDiffRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.LanguageDiff` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filing_accession: str
    prior_filing_accession: str | None
    section_kind: SectionKind
    change_type: ChangeType
    current_section_id: int | None
    prior_section_id: int | None
    similarity: Decimal | None
    severity: Severity
    created_at: datetime


# ---- Phase 4A: uploaded documents ----


class NewUploadedDocument(BaseModel):
    """Input shape for :meth:`Repository.add_uploaded_document`."""

    model_config = ConfigDict(frozen=True)

    upload_id: str
    ticker: str
    filing_type: str
    original_filename: str
    content_sha256: str = Field(..., min_length=64, max_length=64)
    parsed_text: str
    parsed_char_count: int
    page_count: int | None = None


class UploadedDocumentRecord(BaseModel):
    """Detached view of an :class:`~app.memory.models.UploadedDocument` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    upload_id: str
    ticker: str
    filing_type: str
    original_filename: str
    content_sha256: str
    parsed_text: str
    parsed_char_count: int
    page_count: int | None
    uploaded_at: datetime


# ---- Phase 4B: transcript Q&A pairs and management commitments ----
#
# ``AnswerClass`` and ``CommitmentStatus`` are defined in
# :mod:`app.models.state` so the in-graph ``AgentState`` payload models
# can use them without a circular import. They are re-exported at the top
# of this module for backwards compatibility with existing callers that
# import them from here.


class NewQAPair(BaseModel):
    """Inputs to :meth:`Repository.add_qa_pairs`."""

    model_config = ConfigDict(frozen=True)

    ordinal: int
    analyst_name: str | None
    question_text: str
    answer_text: str
    answer_class: AnswerClass
    sha256_text: str = Field(..., min_length=64, max_length=64)


class QAPairRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.QAPair` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filing_accession: str
    ordinal: int
    analyst_name: str | None
    question_text: str
    answer_text: str
    answer_class: AnswerClass
    sha256_text: str
    created_at: datetime


class NewCommitment(BaseModel):
    """Inputs to :meth:`Repository.add_commitments`.

    ``status`` is intentionally omitted - the DB column defaults to ``open``
    so freshly-extracted commitments enter the open queue automatically.
    """

    model_config = ConfigDict(frozen=True)

    commitment_text: str
    target_period: str | None
    source_quote: str


class CommitmentRecord(BaseModel):
    """Detached view of a :class:`~app.memory.models.Commitment` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filing_accession: str
    ticker: str
    commitment_text: str
    target_period: str | None
    source_quote: str
    status: CommitmentStatus
    resolved_filing_accession: str | None
    resolved_reason: str | None
    created_at: datetime
    updated_at: datetime


# ---- Phase 5A: persisted synthesized notes ----


class NoteCreate(BaseModel):
    """Pre-persistence note payload."""

    model_config = ConfigDict(frozen=True)

    filing_accession: str
    ticker: str
    markdown_body: str
    prompt_template_name: str
    prompt_template_sha: str = Field(..., min_length=64, max_length=64)
    critic_attempts: int = Field(..., ge=1)


class NoteRead(BaseModel):
    """Read-side note projection."""

    model_config = ConfigDict(frozen=True)

    id: int
    filing_accession: str
    ticker: str
    markdown_body: str
    prompt_template_name: str
    prompt_template_sha: str
    critic_attempts: int
    created_at: datetime


# ---- Phase 5B: peer context DTOs ----


class PeerCreate(BaseModel):
    """Pre-persistence peer mapping."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    peer_ticker: str
    source: str = "curated"

    @field_validator("peer_ticker")
    @classmethod
    def _no_self_reference(cls, v: str, info: ValidationInfo) -> str:
        ticker = info.data.get("ticker")
        if ticker is not None and v == ticker:
            raise ValueError("peer_ticker must differ from ticker")
        return v


class PeerLanguageDiffSignal(BaseModel):
    """One major language diff signal from a peer."""

    model_config = ConfigDict(frozen=True)

    text: str
    severity: str
    source_filing_accession: str


class PeerCommitmentSignal(BaseModel):
    """One open commitment signal from a peer."""

    model_config = ConfigDict(frozen=True)

    text: str
    source_filing_accession: str


class PeerSignals(BaseModel):
    """Bundle of peer signals returned by the repository."""

    model_config = ConfigDict(frozen=True)

    language_diffs: list[PeerLanguageDiffSignal] = Field(default_factory=list)
    commitments: list[PeerCommitmentSignal] = Field(default_factory=list)
