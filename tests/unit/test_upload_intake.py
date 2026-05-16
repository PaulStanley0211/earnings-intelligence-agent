"""Unit tests for the upload_intake agent node."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.agents.upload_intake import intake_upload
from app.memory.schemas import (
    NewUploadedDocument,
    UploadedDocumentRecord,
    WatchlistRecord,
)
from app.tools.documents import ParsedDocument


class _FakeRepository:
    """Tracks every insert; mimics the production repository surface."""

    def __init__(self) -> None:
        self.saved: list[UploadedDocumentRecord] = []
        self._next_id = 1

    async def add_uploaded_document(
        self, new: NewUploadedDocument
    ) -> UploadedDocumentRecord:
        record = UploadedDocumentRecord(
            id=self._next_id,
            upload_id=new.upload_id,
            ticker=new.ticker,
            filing_type=new.filing_type,
            original_filename=new.original_filename,
            content_sha256=new.content_sha256,
            parsed_text=new.parsed_text,
            parsed_char_count=new.parsed_char_count,
            page_count=new.page_count,
            uploaded_at=datetime.now(UTC),
        )
        self._next_id += 1
        self.saved.append(record)
        return record

    async def get_uploaded_document_by_sha256(
        self, content_sha256: str
    ) -> UploadedDocumentRecord | None:
        for record in self.saved:
            if record.content_sha256 == content_sha256:
                return record
        return None

    async def get_watchlist_entry_by_ticker(
        self, ticker: str
    ) -> WatchlistRecord | None:
        if ticker.upper() != "MSFT":
            return None
        return WatchlistRecord(
            ticker="MSFT",
            cik="0000789019",
            company_name="Microsoft Corp",
            active=True,
            added_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


@pytest.mark.asyncio
async def test_intake_creates_filing_event_with_upload_source() -> None:
    from app.models.state import FilingEventSource

    repo = _FakeRepository()
    parsed = ParsedDocument(
        text="Microsoft reported revenue of $X billion.",
        char_count=42,
        page_count=14,
        content_sha256="c" * 64,
    )
    event = await intake_upload(
        ticker="msft",
        filing_type="8-K",
        original_filename="msft-8k.pdf",
        parsed=parsed,
        repository=repo,
    )
    assert event.source is FilingEventSource.UPLOAD
    assert event.ticker == "MSFT"
    assert event.form.value == "8-K"
    # The upload row is persisted exactly once.
    assert len(repo.saved) == 1
    assert repo.saved[0].content_sha256 == "c" * 64
    # accession_number is the synthetic upload-{upload_id} form.
    assert event.accession_number.startswith("upload-")


@pytest.mark.asyncio
async def test_intake_idempotent_on_duplicate_sha256() -> None:
    """Re-uploading the same content returns the existing row, no duplicate insert."""
    repo = _FakeRepository()
    parsed = ParsedDocument(
        text="hi", char_count=2, page_count=1, content_sha256="d" * 64
    )
    e1 = await intake_upload(
        ticker="MSFT",
        filing_type="8-K",
        original_filename="x.pdf",
        parsed=parsed,
        repository=repo,
    )
    e2 = await intake_upload(
        ticker="MSFT",
        filing_type="8-K",
        original_filename="x.pdf",
        parsed=parsed,
        repository=repo,
    )
    assert len(repo.saved) == 1
    assert e1.accession_number == e2.accession_number


@pytest.mark.asyncio
async def test_intake_rejects_unknown_ticker() -> None:
    """A ticker that isn't in the watchlist should fail fast."""
    repo = _FakeRepository()
    parsed = ParsedDocument(
        text="hi", char_count=2, page_count=1, content_sha256="e" * 64
    )
    with pytest.raises(ValueError, match="watchlist"):
        await intake_upload(
            ticker="ZZZZ",
            filing_type="8-K",
            original_filename="x.pdf",
            parsed=parsed,
            repository=repo,
        )


@pytest.mark.asyncio
async def test_intake_rejects_unsupported_filing_type() -> None:
    """An unknown filing_type string must fail before any DB writes."""
    repo = _FakeRepository()
    parsed = ParsedDocument(
        text="hi", char_count=2, page_count=1, content_sha256="f" * 64
    )
    with pytest.raises(ValueError, match="Unsupported filing_type"):
        await intake_upload(
            ticker="MSFT",
            filing_type="DEF14A",
            original_filename="x.pdf",
            parsed=parsed,
            repository=repo,
        )
    assert len(repo.saved) == 0
