"""Unit tests for the shared citation index used by synthesiser and critic."""

from __future__ import annotations

from decimal import Decimal

from app.agents.citations import (
    build_commitment_citations,
    build_comparison_citations,
    build_fact_citations,
    build_language_citations,
    build_qa_citations,
)
from app.models.state import AnswerClass, CommitmentExtracted, QAPairPayload


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


def test_build_language_citations_indexes_added_modified_and_removed() -> None:
    payload = [
        {
            "section": "mda",
            "diffs": [
                {"change_type": "added", "text": "New paragraph one.", "severity": "major"},
                {
                    "change_type": "modified",
                    "current_text": "Updated paragraph.",
                    "prior_text": "Old paragraph.",
                    "similarity": "0.7421",
                    "severity": "major",
                },
                {"change_type": "removed", "text": "Removed paragraph.", "severity": "minor"},
            ],
        },
    ]
    citations = build_language_citations(payload)
    assert [c.identifier for c in citations] == ["L1", "L2", "L3"]
    assert citations[0].text == "New paragraph one."
    assert citations[1].text == "Updated paragraph."
    assert citations[2].text == "Removed paragraph."


def test_build_language_citations_empty_for_missing_payload() -> None:
    assert build_language_citations(None) == []
    assert build_language_citations([]) == []


def _qa(ordinal: int, question: str, answer: str) -> QAPairPayload:
    return QAPairPayload(
        ordinal=ordinal,
        analyst_name="Brent Thill",
        question_text=question,
        answer_text=answer,
        answer_class=AnswerClass.DIRECT,
        sha256_text="a" * 64,
    )


def test_resolve_qa_citation() -> None:
    """build_qa_citations indexes Q&A pairs in order, source_text concatenates Q+A."""
    pairs = [
        _qa(1, "Azure outlook?", "We expect Azure growth to remain strong."),
        _qa(2, "FX headwinds?", "Two points of headwind next quarter."),
        _qa(3, "Capex?", "Capex will exceed $50 billion this fiscal year."),
    ]
    citations = build_qa_citations(pairs)
    assert [c.identifier for c in citations] == ["Q1", "Q2", "Q3"]
    q3 = citations[2]
    assert q3.question_text == "Capex?"
    assert "$50 billion" in q3.source_text
    # source_text contains both the question and the answer.
    assert "Capex?" in q3.source_text


def test_qa_citation_index_out_of_range() -> None:
    """Indexing past the end of the list yields no Q99 entry."""
    citations = build_qa_citations([_qa(1, "q", "a"), _qa(2, "q", "a"), _qa(3, "q", "a")])
    index = {c.identifier: c for c in citations}
    assert "Q99" not in index
    assert index.get("Q99") is None


def test_qa_citations_accept_plain_dicts() -> None:
    """build_qa_citations is dict-tolerant for lightweight test fixtures."""
    citations = build_qa_citations(
        [
            {
                "ordinal": 1,
                "analyst_name": None,
                "question_text": "What is guidance?",
                "answer_text": "We expect modest growth.",
                "answer_class": "direct",
            }
        ]
    )
    assert len(citations) == 1
    assert citations[0].identifier == "Q1"
    assert citations[0].analyst_name is None


def _commitment(text: str, quote: str, target: str | None = "Q3 2026") -> CommitmentExtracted:
    return CommitmentExtracted(
        commitment_text=text,
        target_period=target,
        source_quote=quote,
    )


def test_resolve_commitment_citation() -> None:
    """build_commitment_citations exposes source_quote as the anchor text."""
    commitments = [
        _commitment(
            "Azure margin expansion of 100 basis points next quarter.",
            "we expect Azure margin expansion of 100 basis points next quarter",
        ),
        _commitment(
            "Operating margin will reach 45 percent for the full year.",
            "operating margin will reach 45 percent for the full year",
            target="FY2026",
        ),
    ]
    citations = build_commitment_citations(commitments)
    assert [c.identifier for c in citations] == ["K1", "K2"]
    assert citations[0].source_text == (
        "we expect Azure margin expansion of 100 basis points next quarter"
    )
    assert citations[1].target_period == "FY2026"


def test_commitment_citation_index_out_of_range() -> None:
    """Indexing past the end of the list yields no K99 entry."""
    citations = build_commitment_citations([_commitment("only one", "verbatim quote one")])
    index = {c.identifier: c for c in citations}
    assert "K99" not in index
    assert index.get("K99") is None


def test_commitment_citations_skip_entries_without_source_quote() -> None:
    """Entries with no source_quote cannot be anchored and are skipped."""
    citations = build_commitment_citations(
        [
            {
                "commitment_text": "Vague forward statement.",
                "target_period": "FY2026",
                "source_quote": "",
            }
        ]
    )
    assert citations == []


def test_commitment_citations_empty_for_missing_payload() -> None:
    assert build_commitment_citations(None) == []
    assert build_commitment_citations([]) == []


def test_qa_citations_empty_for_missing_payload() -> None:
    assert build_qa_citations(None) == []
    assert build_qa_citations([]) == []
