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
* ``L<n>`` - language-diff entry from the language differ's per-section
  summaries, numbered in iteration order across sections.
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


@dataclass(frozen=True)
class LanguageCitation:
    """One numbered language-diff entry the critic can resolve by id."""

    identifier: str
    section: str
    change_type: str
    text: str
    severity: str


def build_language_citations(
    language_diffs: list[dict[str, Any]] | None,
) -> list[LanguageCitation]:
    """Numbered language citations from the differ's per-section summaries.

    Identifiers are assigned ``L1``, ``L2``, ... in iteration order across
    sections. For ``modified`` diffs the indexed text is ``current_text``
    (the new wording); for ``removed`` diffs it is ``prior_text``; for
    ``added`` diffs it is ``text``.
    """
    payloads = language_diffs or []
    citations: list[LanguageCitation] = []
    idx = 1
    for section_payload in payloads:
        section = str(section_payload.get("section") or "")
        for diff in section_payload.get("diffs") or []:
            change_type = str(diff.get("change_type") or "")
            text = _language_cite_text(change_type, diff)
            if not text:
                continue
            citations.append(
                LanguageCitation(
                    identifier=f"L{idx}",
                    section=section,
                    change_type=change_type,
                    text=text,
                    severity=str(diff.get("severity") or ""),
                )
            )
            idx += 1
    return citations


def _language_cite_text(change_type: str, diff: dict[str, Any]) -> str:
    """Pick the text the citation should resolve against."""
    if change_type == "modified":
        return str(diff.get("current_text") or "")
    if change_type == "removed":
        return str(diff.get("prior_text") or diff.get("text") or "")
    return str(diff.get("text") or "")


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
