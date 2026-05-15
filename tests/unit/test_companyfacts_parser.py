"""Unit tests for the companyfacts JSON parser."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.tools.companyfacts import (
    DEFAULT_CONCEPT_ALLOWLIST,
    parse_company_facts,
)
from app.tools.edgar import CompanyFactsResponse


def _response(facts: dict[str, object]) -> CompanyFactsResponse:
    return CompanyFactsResponse(
        cik="0000789019",
        entity_name="Microsoft Corp",
        raw={"cik": 789019, "entityName": "Microsoft Corp", "facts": facts},
    )


def _us_gaap(concept: str, unit: str, entries: list[dict[str, object]]) -> dict[str, object]:
    return {"us-gaap": {concept: {"label": concept, "units": {unit: entries}}}}


def test_parser_filters_by_accession_number() -> None:
    response = _response(
        _us_gaap(
            "Revenues",
            "USD",
            [
                {
                    "start": "2026-01-01",
                    "end": "2026-03-31",
                    "val": 61858000000,
                    "accn": "0000950170-26-000050",
                    "fy": 2026,
                    "fp": "Q3",
                    "form": "10-Q",
                    "filed": "2026-04-25",
                },
                {
                    "start": "2025-10-01",
                    "end": "2025-12-31",
                    "val": 65585000000,
                    "accn": "0000950170-26-000020",
                    "fy": 2026,
                    "fp": "Q2",
                    "form": "10-Q",
                    "filed": "2026-01-30",
                },
            ],
        )
    )
    facts = parse_company_facts(
        response, accession_number="0000950170-26-000050", concepts=None
    )
    assert len(facts) == 1
    only = facts[0]
    assert only.concept == "Revenues"
    assert only.taxonomy == "us-gaap"
    assert only.unit == "USD"
    assert only.value == Decimal("61858000000")
    assert only.period_type == "duration"
    assert only.period_start == date(2026, 1, 1)
    assert only.period_end == date(2026, 3, 31)
    assert only.fiscal_year == 2026
    assert only.fiscal_period == "Q3"
    assert only.form == "10-Q"
    assert only.filed == date(2026, 4, 25)


def test_parser_handles_instant_facts() -> None:
    response = _response(
        _us_gaap(
            "Assets",
            "USD",
            [
                {
                    "end": "2026-03-31",
                    "val": 480000000000,
                    "accn": "0000950170-26-000050",
                    "fy": 2026,
                    "fp": "Q3",
                    "form": "10-Q",
                    "filed": "2026-04-25",
                }
            ],
        )
    )
    facts = parse_company_facts(
        response,
        accession_number="0000950170-26-000050",
        concepts=("Assets",),
    )
    assert len(facts) == 1
    assert facts[0].period_type == "instant"
    assert facts[0].period_start is None
    assert facts[0].period_end == date(2026, 3, 31)


def test_parser_emits_all_facts_when_accession_is_none() -> None:
    response = _response(
        _us_gaap(
            "NetIncomeLoss",
            "USD",
            [
                {
                    "start": "2026-01-01",
                    "end": "2026-03-31",
                    "val": 21939000000,
                    "accn": "A",
                    "fy": 2026,
                    "fp": "Q3",
                    "form": "10-Q",
                    "filed": "2026-04-25",
                },
                {
                    "start": "2025-10-01",
                    "end": "2025-12-31",
                    "val": 24108000000,
                    "accn": "B",
                    "fy": 2026,
                    "fp": "Q2",
                    "form": "10-Q",
                    "filed": "2026-01-30",
                },
            ],
        )
    )
    facts = parse_company_facts(response, accession_number=None, concepts=None)
    assert {fact.filing_accession for fact in facts} == {"A", "B"}


def test_parser_applies_concept_allowlist_when_provided() -> None:
    response = _response(
        {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues",
                    "units": {
                        "USD": [
                            {
                                "start": "2026-01-01",
                                "end": "2026-03-31",
                                "val": 1,
                                "accn": "X",
                                "fy": 2026,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2026-04-25",
                            }
                        ]
                    },
                },
                "DebtCurrent": {
                    "label": "DebtCurrent",
                    "units": {
                        "USD": [
                            {
                                "end": "2026-03-31",
                                "val": 2,
                                "accn": "X",
                                "fy": 2026,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2026-04-25",
                            }
                        ]
                    },
                },
            }
        }
    )
    facts = parse_company_facts(response, accession_number=None, concepts=["Revenues"])
    assert [fact.concept for fact in facts] == ["Revenues"]


def test_parser_skips_malformed_entries_but_keeps_going() -> None:
    response = _response(
        _us_gaap(
            "Revenues",
            "USD",
            [
                {"val": 1, "accn": "A"},  # missing end - skipped
                {
                    "start": "2026-01-01",
                    "end": "2026-03-31",
                    "val": 2,
                    "accn": "A",
                    "fy": 2026,
                    "fp": "Q3",
                    "form": "10-Q",
                    "filed": "2026-04-25",
                },
            ],
        )
    )
    facts = parse_company_facts(response, accession_number=None, concepts=None)
    assert [fact.value for fact in facts] == [Decimal("2")]


def test_default_concept_allowlist_covers_phase1_keys() -> None:
    assert "Revenues" in DEFAULT_CONCEPT_ALLOWLIST
    assert "NetIncomeLoss" in DEFAULT_CONCEPT_ALLOWLIST
    assert "EarningsPerShareDiluted" in DEFAULT_CONCEPT_ALLOWLIST


def test_parser_handles_string_value_with_thousands_separators() -> None:
    response = _response(
        _us_gaap(
            "Revenues",
            "USD",
            [
                {
                    "start": "2026-01-01",
                    "end": "2026-03-31",
                    "val": "61,858,000,000",
                    "accn": "A",
                    "fy": 2026,
                    "fp": "Q3",
                    "form": "10-Q",
                    "filed": "2026-04-25",
                }
            ],
        )
    )
    facts = parse_company_facts(response, accession_number=None, concepts=None)
    assert facts[0].value == Decimal("61858000000")


def test_parser_carries_cik_from_response() -> None:
    response = _response(
        _us_gaap(
            "Revenues",
            "USD",
            [
                {
                    "start": "2026-01-01",
                    "end": "2026-03-31",
                    "val": 1,
                    "accn": "A",
                    "fy": 2026,
                    "fp": "Q3",
                    "form": "10-Q",
                    "filed": "2026-04-25",
                }
            ],
        )
    )
    facts = parse_company_facts(response, accession_number=None, concepts=None)
    assert facts[0].cik == "0000789019"
