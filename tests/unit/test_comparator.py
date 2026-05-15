"""Unit tests for :mod:`app.agents.comparator`."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.agents.comparator import (
    OWNER,
    _compute_surprise,
    compare_against_consensus,
)
from app.memory.schemas import (
    ComparisonDirection,
    ComparisonRecord,
    NewComparison,
    NewConsensusEstimate,
)
from app.models.state import AgentState, FilingEvent, FilingForm


class _StubFetcher:
    def __init__(self, rows: list[NewConsensusEstimate]) -> None:
        self.rows = rows
        self.last_call: dict[str, object] | None = None

    async def fetch(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
        period_end: date,
    ) -> list[NewConsensusEstimate]:
        self.last_call = {
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "period_end": period_end,
        }
        return self.rows


class _StubRepository:
    def __init__(self) -> None:
        self.consensus_rows: list[NewConsensusEstimate] = []
        self.comparison_rows: list[NewComparison] = []

    async def upsert_consensus_estimate(self, estimate: NewConsensusEstimate) -> Any:
        self.consensus_rows.append(estimate)
        return estimate

    async def insert_comparison(self, comparison: NewComparison) -> ComparisonRecord:
        self.comparison_rows.append(comparison)
        return ComparisonRecord(
            id=len(self.comparison_rows),
            filing_accession=comparison.filing_accession,
            metric=comparison.metric,
            reported_value=comparison.reported_value,
            reported_unit=comparison.reported_unit,
            consensus_value=comparison.consensus_value,
            consensus_source=comparison.consensus_source,
            surprise_abs=comparison.surprise_abs,
            surprise_pct=comparison.surprise_pct,
            direction=comparison.direction,
            created_at=datetime.now(UTC),
        )


def _state_with_financials() -> AgentState:
    filing_event = FilingEvent(
        accession_number="0000950170-26-000050",
        cik="0000789019",
        ticker="MSFT",
        form=FilingForm.FORM_10Q,
        filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
        source_url="https://www.sec.gov/...",
    )
    financials: dict[str, Any] = {
        "source": "companyfacts",
        "by_concept": {
            "Revenues": [
                {
                    "value": "61858000000",
                    "unit": "USD",
                    "period_start": "2026-01-01",
                    "period_end": "2026-03-31",
                    "fiscal_year": 2026,
                    "fiscal_period": "Q3",
                }
            ],
            "EarningsPerShareDiluted": [
                {
                    "value": "1.32",
                    "unit": "USD/shares",
                    "period_start": "2026-01-01",
                    "period_end": "2026-03-31",
                    "fiscal_year": 2026,
                    "fiscal_period": "Q3",
                }
            ],
            "NetIncomeLoss": [
                {
                    "value": "21939000000",
                    "unit": "USD",
                    "period_start": "2026-01-01",
                    "period_end": "2026-03-31",
                    "fiscal_year": 2026,
                    "fiscal_period": "Q3",
                }
            ],
        },
    }
    return AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=filing_event,
        financials=financials,
    )


async def test_compare_builds_summary_for_every_reported_metric() -> None:
    fetcher = _StubFetcher(
        [
            NewConsensusEstimate(
                ticker="MSFT",
                fiscal_year=2026,
                fiscal_period="Q3",
                metric="revenue",
                value=Decimal("61000000000"),
                source="finnhub",
            ),
            NewConsensusEstimate(
                ticker="MSFT",
                fiscal_year=2026,
                fiscal_period="Q3",
                metric="eps_diluted",
                value=Decimal("1.30"),
                source="finnhub",
            ),
        ]
    )
    repo = _StubRepository()
    state = _state_with_financials()

    update = await compare_against_consensus(
        state, consensus_fetcher=fetcher, repository=repo
    )

    assert update.owner == OWNER
    summary = update.changes["comparisons"]
    assert summary["fiscal_year"] == 2026
    assert summary["fiscal_period"] == "Q3"
    assert summary["period_end"] == "2026-03-31"
    assert summary["consensus_source"] == "finnhub"
    assert summary["degraded"] is False
    metrics = {m["metric"]: m for m in summary["metrics"]}
    assert metrics["revenue"]["direction"] == "beat"
    # Revenue beat: (61.858 - 61) / 61 = ~1.41%
    assert metrics["revenue"]["consensus_value"] == "61000000000"
    # EPS beat: (1.32 - 1.30) / 1.30 = ~1.5%, > 0.5% band so 'beat'.
    assert metrics["eps_diluted"]["direction"] == "beat"
    # Net income had no consensus row.
    assert metrics["net_income"]["consensus_value"] is None
    assert metrics["net_income"]["direction"] is None
    assert len(repo.consensus_rows) == 2
    assert len(repo.comparison_rows) == 3
    assert fetcher.last_call == {
        "ticker": "MSFT",
        "fiscal_year": 2026,
        "fiscal_period": "Q3",
        "period_end": date(2026, 3, 31),
    }


async def test_compare_degraded_when_no_consensus_rows() -> None:
    fetcher = _StubFetcher([])
    repo = _StubRepository()
    state = _state_with_financials()

    update = await compare_against_consensus(
        state, consensus_fetcher=fetcher, repository=repo
    )

    summary = update.changes["comparisons"]
    assert summary["degraded"] is True
    assert summary["consensus_source"] is None
    assert all(m["consensus_value"] is None for m in summary["metrics"])


async def test_compare_short_circuits_with_empty_financials() -> None:
    fetcher = _StubFetcher([])
    repo = _StubRepository()
    state = AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="X",
            cik="0",
            ticker="X",
            form=FilingForm.FORM_10Q,
            filed_at=datetime.now(UTC),
            source_url="https://example",
        ),
    )

    update = await compare_against_consensus(
        state, consensus_fetcher=fetcher, repository=repo
    )

    summary = update.changes["comparisons"]
    assert summary == {
        "fiscal_year": None,
        "fiscal_period": None,
        "period_end": None,
        "metrics": [],
        "consensus_source": None,
        "degraded": True,
    }
    assert repo.consensus_rows == []
    assert repo.comparison_rows == []


@pytest.mark.parametrize(
    ("reported", "consensus", "expected_direction"),
    [
        (Decimal("100"), None, None),
        (Decimal("100"), Decimal("100"), "in_line"),
        (Decimal("100.1"), Decimal("100"), "in_line"),  # 0.1% within band
        (Decimal("110"), Decimal("100"), "beat"),
        (Decimal("90"), Decimal("100"), "miss"),
    ],
)
def test_compute_surprise_direction_bands(
    reported: Decimal,
    consensus: Decimal | None,
    expected_direction: ComparisonDirection | None,
) -> None:
    _, _, direction = _compute_surprise(reported=reported, consensus=consensus)
    assert direction == expected_direction
