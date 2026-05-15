"""Parse EDGAR ``companyfacts`` JSON into :class:`NewFinancialFact` rows.

The companyfacts payload mixes every fact a company has ever reported into a
single deeply-nested document::

    facts:
      us-gaap:
        Revenues:
          units:
            USD: [ { start, end, val, accn, fy, fp, form, filed, frame }, ... ]

This parser walks that structure lazily and emits one
:class:`~app.memory.schemas.NewFinancialFact` per data point. Callers usually
filter by the ``accn`` of a freshly-detected filing, which is how Phase 1
associates pre-parsed XBRL with the filing the watcher just spotted.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from app.memory.schemas import NewFinancialFact, PeriodType
from app.observability.logging import get_logger
from app.tools.edgar import CompanyFactsResponse

_logger = get_logger()

# Conservative default concept allowlist. Phase 1 only ships financials when
# the comparator and synthesiser are wired in, but the watcher already needs
# a reasonable subset so ``poll_once`` can dump something useful for review.
# The numbers track in Phase 2 will refine and extend this list.
DEFAULT_CONCEPT_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "CostOfRevenue",
        "GrossProfit",
        "OperatingIncomeLoss",
        "NetIncomeLoss",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "CashAndCashEquivalentsAtCarryingValue",
        "ResearchAndDevelopmentExpense",
    }
)


def parse_company_facts(
    response: CompanyFactsResponse,
    *,
    accession_number: str | None,
    concepts: Iterable[str] | None,
) -> list[NewFinancialFact]:
    """Flatten ``response`` into a list of :class:`NewFinancialFact`.

    ``accession_number`` filters facts whose ``accn`` field matches; pass
    ``None`` to emit every fact regardless of filing.

    ``concepts`` filters to the named concepts; pass ``None`` to emit every
    concept the companyfacts payload exposes.
    """
    allowlist = frozenset(concepts) if concepts is not None else None
    cik = response.cik
    out: list[NewFinancialFact] = []

    facts_root = response.raw.get("facts", {}) or {}
    for taxonomy, concepts_map in facts_root.items():
        if not isinstance(concepts_map, dict):
            continue
        for concept, body in concepts_map.items():
            if allowlist is not None and concept not in allowlist:
                continue
            if not isinstance(body, dict):
                continue
            units = body.get("units", {}) or {}
            for unit, entries in units.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    parsed = _build_fact(
                        entry=entry,
                        cik=cik,
                        taxonomy=str(taxonomy),
                        concept=str(concept),
                        unit=str(unit),
                        target_accession=accession_number,
                    )
                    if parsed is not None:
                        out.append(parsed)
    return out


def _build_fact(
    *,
    entry: Any,
    cik: str,
    taxonomy: str,
    concept: str,
    unit: str,
    target_accession: str | None,
) -> NewFinancialFact | None:
    """Convert one companyfacts entry into a :class:`NewFinancialFact`."""
    if not isinstance(entry, dict):
        return None
    accn = entry.get("accn")
    if target_accession is not None and accn != target_accession:
        return None
    end_raw = entry.get("end")
    val_raw = entry.get("val")
    if end_raw is None or val_raw is None or accn is None:
        return None
    try:
        end_date = date.fromisoformat(str(end_raw))
        value = _to_decimal(val_raw)
    except (ValueError, InvalidOperation):
        _logger.bind(concept=concept, accn=accn).warning("companyfacts_fact_skipped_malformed")
        return None

    start_raw = entry.get("start")
    period_start: date | None
    period_type: PeriodType
    if start_raw is None:
        period_start = None
        period_type = "instant"
    else:
        try:
            period_start = date.fromisoformat(str(start_raw))
        except ValueError:
            _logger.bind(concept=concept, accn=accn).warning(
                "companyfacts_fact_skipped_bad_start"
            )
            return None
        period_type = "duration"

    filed_raw = entry.get("filed")
    filed: date | None
    try:
        filed = date.fromisoformat(str(filed_raw)) if filed_raw else None
    except ValueError:
        filed = None

    return NewFinancialFact(
        filing_accession=str(accn),
        cik=cik,
        taxonomy=taxonomy,
        concept=concept,
        unit=unit,
        value=value,
        period_type=period_type,
        period_start=period_start,
        period_end=end_date,
        fiscal_year=_as_int(entry.get("fy")),
        fiscal_period=_as_str_or_none(entry.get("fp")),
        form=_as_str_or_none(entry.get("form")),
        filed=filed,
        frame=_as_str_or_none(entry.get("frame")),
    )


def _to_decimal(value: Any) -> Decimal:
    """Parse ``value`` into a :class:`Decimal`, tolerating EDGAR's mixed types."""
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        return Decimal(value.replace(",", "").strip())
    raise InvalidOperation(f"unsupported fact value type {type(value).__name__}")


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
