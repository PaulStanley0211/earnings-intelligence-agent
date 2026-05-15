"""Pydantic DTOs and enums used at the boundary of the memory layer.

The ORM models in :mod:`app.memory.models` describe what is on disk; these
DTOs are what callers hand to and receive from the
:class:`~app.memory.repository.Repository`. Keeping the boundary types
separate from the ORM rows means callers do not need an active SQLAlchemy
session attached to every result they want to inspect.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.state import FilingForm


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
