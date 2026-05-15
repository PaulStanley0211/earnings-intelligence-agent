"""Unit tests for the shared citation index used by synthesiser and critic."""

from __future__ import annotations

from decimal import Decimal

from app.agents.citations import build_comparison_citations, build_fact_citations


def test_build_fact_citations_orders_concept_then_period_desc() -> None:
    financials = {
        "by_concept": {
            "Revenues": [
                {"value": "61858000000", "unit": "USD", "period_end": "2026-03-31"},
                {"value": "65585000000", "unit": "USD", "period_end": "2025-12-31"},
            ],
            "EarningsPerShareDiluted": [
                {"value": "1.32", "unit": "USD/shares", "period_end": "2026-03-31"},
            ],
        }
    }
    citations = build_fact_citations(financials)
    assert [c.identifier for c in citations] == ["F1", "F2", "F3"]
    assert citations[0].concept == "EarningsPerShareDiluted"
    # Revenues entries are period-end-descending within their concept.
    assert citations[1].concept == "Revenues"
    assert citations[1].period_end == "2026-03-31"
    assert citations[2].period_end == "2025-12-31"


def test_build_fact_citations_skips_unparseable_values() -> None:
    citations = build_fact_citations(
        {
            "by_concept": {
                "Revenues": [
                    {"value": "not-a-number", "unit": "USD", "period_end": "2026-03-31"},
                ]
            }
        }
    )
    assert citations == []


def test_build_comparison_citations_preserves_input_order() -> None:
    comparisons = {
        "metrics": [
            {
                "metric": "revenue",
                "reported_value": "61858000000",
                "reported_unit": "USD",
                "consensus_value": "61000000000",
                "consensus_source": "finnhub",
                "surprise_abs": "858000000",
                "surprise_pct": "1.4066",
                "direction": "beat",
            },
            {
                "metric": "eps_diluted",
                "reported_value": "1.32",
                "reported_unit": "USD/shares",
                "consensus_value": None,
                "consensus_source": None,
                "surprise_abs": None,
                "surprise_pct": None,
                "direction": None,
            },
        ]
    }
    citations = build_comparison_citations(comparisons)
    assert [c.identifier for c in citations] == ["C1", "C2"]
    assert citations[0].metric == "revenue"
    assert citations[0].consensus_value == Decimal("61000000000")
    assert citations[1].consensus_value is None
