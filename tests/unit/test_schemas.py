"""DTO sanity checks for the Phase 3 memory schema additions."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.memory.schemas import (
    ChangeType,
    FilingSectionRecord,
    LanguageDiffRecord,
    NewFilingSection,
    NewLanguageDiff,
    SectionKind,
    Severity,
)


def test_new_filing_section_is_frozen():
    row = NewFilingSection(
        filing_accession="0000000000-26-000001",
        cik="0000789019",
        ticker="MSFT",
        section_kind=SectionKind.MDA,
        paragraph_index=0,
        text="The company saw strong demand.",
        text_sha="a" * 64,
        embedding=None,
        embedding_model=None,
    )
    with pytest.raises(ValidationError):
        row.text = "mutated"


def test_new_language_diff_defaults_optional_fields():
    row = NewLanguageDiff(
        filing_accession="0000000000-26-000001",
        section_kind=SectionKind.MDA,
        change_type=ChangeType.ADDED,
        severity=Severity.MAJOR,
    )
    assert row.prior_filing_accession is None
    assert row.current_section_id is None
    assert row.prior_section_id is None
    assert row.similarity is None


def test_filing_section_record_from_attributes():
    class _Stub:
        id = 1
        filing_accession = "0000000000-26-000001"
        cik = "0000789019"
        ticker = "MSFT"
        section_kind = "mda"
        paragraph_index = 0
        text = "Demand was strong."
        text_sha = "a" * 64
        embedding = [0.1] * 1536
        embedding_model = "openai/text-embedding-3-small"
        created_at = datetime(2026, 5, 15, tzinfo=UTC)

    record = FilingSectionRecord.model_validate(_Stub())
    assert record.section_kind == SectionKind.MDA
    assert record.embedding is not None and len(record.embedding) == 1536


def test_language_diff_record_serialises_similarity():
    class _Stub:
        id = 1
        filing_accession = "0000000000-26-000001"
        prior_filing_accession = "0000000000-26-000000"
        section_kind = "mda"
        change_type = "modified"
        current_section_id = 10
        prior_section_id = 5
        similarity = Decimal("0.8400")
        severity = "minor"
        created_at = datetime(2026, 5, 15, tzinfo=UTC)

    record = LanguageDiffRecord.model_validate(_Stub())
    assert record.change_type == ChangeType.MODIFIED
    assert record.severity == Severity.MINOR
    assert record.similarity == Decimal("0.8400")
