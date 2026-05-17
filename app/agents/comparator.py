"""The comparator agent node.

Reads the financial-extractor's output, queries the consensus fetcher for
the matching ``(ticker, fiscal_year, fiscal_period)``, computes the surprise
per metric, and persists both the consensus rows and the resulting
comparison rows. The :class:`StateUpdate` it emits carries a structured
summary the synthesiser and critic consume.

The comparator is the first node that needs to align EDGAR-side concepts
(``us-gaap:Revenues``) with consensus-side metrics (``revenue``). The
mapping lives in :data:`_CONCEPT_TO_METRIC` so it is reviewable in one
place; extending Phase 2 with new metrics is a single-line change.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from typing import Any, Protocol

from app.memory.repository import Repository
from app.memory.schemas import (
    ComparisonDirection,
    ComparisonMetric,
    NewComparison,
    NewConsensusEstimate,
)
from app.models.state import AgentState, FilingForm, StateUpdate
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER = "comparator"

# Threshold (in absolute percent) below which a surprise is reported as "in
# line" rather than a beat or miss. Tight enough to flag a real divergence
# from consensus, loose enough to tolerate rounding noise in EDGAR's XBRL
# precision.
_IN_LINE_BAND_PCT: Decimal = Decimal("0.5")

# Map every us-gaap concept the financial extractor emits to the comparator's
# metric vocabulary. Multiple concepts can collapse to one metric (the SEC
# supports two revenue tags that companies use interchangeably). The lookup
# is intentionally one-way; the comparator chooses the first matching
# concept per metric in iteration order.
_CONCEPT_TO_METRIC: dict[str, ComparisonMetric] = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    "EarningsPerShareDiluted": "eps_diluted",
    "EarningsPerShareBasic": "eps_basic",
    "NetIncomeLoss": "net_income",
}


class _SupportsConsensusFetch(Protocol):
    """Minimal protocol covering what the comparator needs from the fetcher."""

    async def fetch(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
        period_end: date,
    ) -> list[NewConsensusEstimate]: ...


async def compare_against_consensus(
    state: AgentState,
    *,
    consensus_fetcher: _SupportsConsensusFetch,
    repository: Repository,
) -> StateUpdate:
    """Build the comparison rows for the current filing's reported metrics.

    The function expects :attr:`AgentState.financials` to be populated by the
    upstream financial-extractor node. When the extractor produced nothing
    usable (no quarter-aligned facts), it short-circuits and returns an empty
    comparisons summary so the synthesiser can degrade gracefully.

    Self-skips on ``TRANSCRIPT`` filings: transcripts have no reported
    numbers to compare against consensus, so the node yields an empty
    update and lets the parallel ``transcript_analyzer`` carry the payload
    for that branch of the graph.
    """
    filing = state.filing_event
    if filing.form == FilingForm.TRANSCRIPT:
        return StateUpdate(owner=OWNER, changes={})
    financials = state.financials or {}
    by_concept = financials.get("by_concept") or {}
    reported = _select_reported_values(by_concept)
    if not reported:
        _logger.bind(
            accession=filing.accession_number,
            trace_id=current_trace_id(),
        ).warning("comparator_no_reported_metrics")
        return StateUpdate(
            owner=OWNER,
            changes={
                "comparisons": {
                    "fiscal_year": None,
                    "fiscal_period": None,
                    "period_end": None,
                    "metrics": [],
                    "consensus_source": None,
                    "degraded": True,
                }
            },
        )

    fiscal_year, fiscal_period, period_end = _pick_anchor_period(reported.values())
    consensus_rows = await consensus_fetcher.fetch(
        ticker=filing.ticker,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_end=period_end,
    )
    for row in consensus_rows:
        await repository.upsert_consensus_estimate(row)
    consensus_by_metric = {row.metric: row for row in consensus_rows}

    metrics_summary: list[dict[str, Any]] = []
    for metric, reported_fact in reported.items():
        consensus = consensus_by_metric.get(metric)
        surprise_abs, surprise_pct, direction = _compute_surprise(
            reported=reported_fact["value"],
            consensus=consensus.value if consensus is not None else None,
        )
        comparison = NewComparison(
            filing_accession=filing.accession_number,
            metric=metric,
            reported_value=reported_fact["value"],
            reported_unit=reported_fact["unit"],
            consensus_value=consensus.value if consensus is not None else None,
            consensus_source=consensus.source if consensus is not None else None,
            surprise_abs=surprise_abs,
            surprise_pct=surprise_pct,
            direction=direction,
        )
        record = await repository.insert_comparison(comparison)
        metrics_summary.append(
            {
                "metric": metric,
                "concept": reported_fact["concept"],
                "reported_value": str(record.reported_value),
                "reported_unit": record.reported_unit,
                "consensus_value": (
                    str(record.consensus_value)
                    if record.consensus_value is not None
                    else None
                ),
                "consensus_source": record.consensus_source,
                "surprise_abs": (
                    str(record.surprise_abs) if record.surprise_abs is not None else None
                ),
                "surprise_pct": (
                    str(record.surprise_pct) if record.surprise_pct is not None else None
                ),
                "direction": record.direction,
            }
        )

    summary = {
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "period_end": period_end.isoformat(),
        "metrics": metrics_summary,
        "consensus_source": _summary_consensus_source(consensus_rows),
        "degraded": not consensus_rows,
    }
    _logger.bind(
        accession=filing.accession_number,
        ticker=filing.ticker,
        metric_count=len(metrics_summary),
        consensus_count=len(consensus_rows),
        trace_id=current_trace_id(),
    ).info("comparator_complete")
    return StateUpdate(owner=OWNER, changes={"comparisons": summary})


def _select_reported_values(
    by_concept: dict[str, list[dict[str, Any]]],
) -> dict[ComparisonMetric, dict[str, Any]]:
    """Collapse the extractor's concept dump into one reported value per metric.

    The extractor emits each us-gaap concept once per period. The comparator
    cares about the most-recent period (largest ``period_end``) for the
    primary fiscal_period+fiscal_year recorded on the filing's facts.
    """
    reported: dict[ComparisonMetric, dict[str, Any]] = {}
    for concept, entries in by_concept.items():
        metric = _CONCEPT_TO_METRIC.get(concept)
        if metric is None or metric in reported:
            continue
        latest = _latest_quarterly_entry(entries)
        if latest is None:
            continue
        reported[metric] = {
            "concept": concept,
            "value": Decimal(str(latest["value"])),
            "unit": str(latest["unit"]),
            "fiscal_year": int(latest.get("fiscal_year") or 0),
            "fiscal_period": str(latest.get("fiscal_period") or ""),
            "period_end": date.fromisoformat(str(latest["period_end"])),
        }
    return reported


def _latest_quarterly_entry(
    entries: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the entry with the largest ``period_end`` (None when empty)."""
    valid = [e for e in entries if e.get("period_end")]
    if not valid:
        return None
    return max(valid, key=lambda e: str(e["period_end"]))


def _pick_anchor_period(
    facts: Iterable[dict[str, Any]],
) -> tuple[int, str, date]:
    """Pick the dominant (fiscal_year, fiscal_period, period_end) for the filing.

    Falls back to the latest-period fact when fiscal labels disagree. The
    consensus fetcher matches by ``period_end`` regardless, so the fiscal
    labels are only there for downstream display.
    """
    facts_list = list(facts)
    anchor = max(facts_list, key=lambda f: f["period_end"])
    return (
        int(anchor.get("fiscal_year") or 0),
        str(anchor.get("fiscal_period") or ""),
        anchor["period_end"],
    )


def _compute_surprise(
    *,
    reported: Decimal,
    consensus: Decimal | None,
) -> tuple[Decimal | None, Decimal | None, ComparisonDirection | None]:
    """Compute absolute surprise, percent surprise, and direction.

    Direction is ``beat`` when the surprise percent exceeds the in-line band
    and the metric is higher-is-better (all Phase 2 metrics are); ``miss``
    when below the negative band; ``in_line`` when inside the band.
    """
    if consensus is None:
        return None, None, None
    abs_diff = reported - consensus
    pct: Decimal | None
    if consensus == 0:
        pct = None
    else:
        pct = (abs_diff / consensus) * Decimal("100")
        pct = pct.quantize(Decimal("0.0001"))
    direction: ComparisonDirection
    if pct is None:
        direction = "in_line" if abs_diff == 0 else ("beat" if abs_diff > 0 else "miss")
    elif pct.copy_abs() <= _IN_LINE_BAND_PCT:
        direction = "in_line"
    elif pct > 0:
        direction = "beat"
    else:
        direction = "miss"
    return abs_diff, pct, direction


def _summary_consensus_source(rows: list[NewConsensusEstimate]) -> str | None:
    """Return the single source used (or ``None`` when no rows came back)."""
    sources = {row.source for row in rows}
    if not sources:
        return None
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"
