"""Citation indexing shared by the synthesiser and the critic.

Both nodes need to agree on how a row in :attr:`AgentState.financials` or
:attr:`AgentState.comparisons` maps to a citation identifier like ``F3`` or
``C2``. Centralising the index here guarantees they cannot drift: the
synthesiser cites identifiers from :func:`build_fact_citations` and the
critic resolves identifiers against the same function on the same state.

Identifier conventions:

* ``F<n>`` - reported financial fact (one per concept/period entry from the
  extractor), numbered in concept-sorted then period-end-descending order.
* ``C<n>`` - per-metric reported-vs-consensus comparison row from the
  comparator, numbered in iteration order of
  :attr:`AgentState.comparisons` ``metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class FactCitation:
    """One numbered financial-fact entry the critic can resolve by id."""

    identifier: str
    concept: str
    value: Decimal
    unit: str
    period_end: str | None
    period_start: str | None
    fiscal_year: int | None
    fiscal_period: str | None


@dataclass(frozen=True)
class ComparisonCitation:
    """One numbered comparison entry the critic can resolve by id."""

    identifier: str
    metric: str
    reported_value: Decimal
    reported_unit: str
    consensus_value: Decimal | None
    consensus_source: str | None
    surprise_abs: Decimal | None
    surprise_pct: Decimal | None
    direction: str | None


def build_fact_citations(
    financials: dict[str, Any] | None,
) -> list[FactCitation]:
    """Numbered fact citations from the extractor's structured summary."""
    by_concept = (financials or {}).get("by_concept") or {}
    flat: list[tuple[str, dict[str, Any]]] = []
    for concept in sorted(by_concept.keys()):
        entries = by_concept.get(concept) or []
        for entry in sorted(
            entries,
            key=lambda e: str(e.get("period_end") or ""),
            reverse=True,
        ):
            flat.append((concept, entry))
    citations: list[FactCitation] = []
    for idx, (concept, entry) in enumerate(flat, start=1):
        value = _safe_decimal(entry.get("value"))
        if value is None:
            continue
        citations.append(
            FactCitation(
                identifier=f"F{idx}",
                concept=concept,
                value=value,
                unit=str(entry.get("unit") or ""),
                period_end=_str_or_none(entry.get("period_end")),
                period_start=_str_or_none(entry.get("period_start")),
                fiscal_year=_int_or_none(entry.get("fiscal_year")),
                fiscal_period=_str_or_none(entry.get("fiscal_period")),
            )
        )
    return citations


def build_comparison_citations(
    comparisons: dict[str, Any] | None,
) -> list[ComparisonCitation]:
    """Numbered comparison citations from the comparator's summary."""
    metrics = (comparisons or {}).get("metrics") or []
    citations: list[ComparisonCitation] = []
    for idx, metric in enumerate(metrics, start=1):
        reported = _safe_decimal(metric.get("reported_value"))
        if reported is None:
            continue
        citations.append(
            ComparisonCitation(
                identifier=f"C{idx}",
                metric=str(metric.get("metric") or ""),
                reported_value=reported,
                reported_unit=str(metric.get("reported_unit") or ""),
                consensus_value=_safe_decimal(metric.get("consensus_value")),
                consensus_source=_str_or_none(metric.get("consensus_source")),
                surprise_abs=_safe_decimal(metric.get("surprise_abs")),
                surprise_pct=_safe_decimal(metric.get("surprise_pct")),
                direction=_str_or_none(metric.get("direction")),
            )
        )
    return citations


def _safe_decimal(value: Any) -> Decimal | None:
    """Parse a value into :class:`Decimal`, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    """Stringify ``value`` or return ``None`` when unset."""
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_none(value: Any) -> int | None:
    """Cast ``value`` to ``int`` or return ``None`` on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
