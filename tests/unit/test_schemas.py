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


# ---- Phase 5B: peer context DTOs ----


def test_peer_create_validates_no_self_reference() -> None:
    import pytest

    from app.memory.schemas import PeerCreate

    with pytest.raises(ValueError):
        PeerCreate(ticker="MSFT", peer_ticker="MSFT")


def test_peer_signals_defaults_to_empty() -> None:
    from app.memory.schemas import PeerSignals

    sig = PeerSignals(language_diffs=[], commitments=[])
    assert sig.language_diffs == []
    assert sig.commitments == []


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


def test_note_create_round_trips_through_pydantic() -> None:
    """NoteCreate is frozen, validates sha length, and survives a JSON round-trip."""
    from app.memory.schemas import NoteCreate

    note = NoteCreate(
        filing_accession="0000123-25-000001",
        ticker="MSFT",
        markdown_body="# Microsoft Q3 FY25\n\nRevenue rose [F1].",
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
        critic_attempts=1,
    )
    assert note.filing_accession == "0000123-25-000001"
    assert note.ticker == "MSFT"
    assert len(note.prompt_template_sha) == 64

    # Round-trip via model_dump_json -> validate
    rebuilt = NoteCreate.model_validate_json(note.model_dump_json())
    assert rebuilt == note
