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


@pytest.mark.asyncio
async def test_intake_recovers_from_concurrent_duplicate_insert() -> None:
    """When a concurrent caller wins the insert race, we recover idempotently."""
    from sqlalchemy.exc import IntegrityError

    class _RacyRepository:
        """Repository where ``add_uploaded_document`` always raises IntegrityError
        because a sibling caller already committed the same content.
        """

        def __init__(self, winner: UploadedDocumentRecord) -> None:
            self._winner = winner
            self.add_calls = 0

        async def add_uploaded_document(
            self, new: NewUploadedDocument
        ) -> UploadedDocumentRecord:
            self.add_calls += 1
            raise IntegrityError("uq violation", {}, None)  # type: ignore[arg-type]

        async def get_uploaded_document_by_sha256(
            self, content_sha256: str
        ) -> UploadedDocumentRecord | None:
            # First call (before the race): row not visible yet -> None.
            # Subsequent calls (after the race): winner is visible.
            if self.add_calls == 0:
                return None
            return self._winner

        async def get_watchlist_entry_by_ticker(
            self, ticker: str
        ) -> WatchlistRecord | None:
            return WatchlistRecord(
                ticker="MSFT",
                cik="0000789019",
                company_name="Microsoft Corp",
                active=True,
                added_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    winner = UploadedDocumentRecord(
        id=1,
        upload_id="winner-id",
        ticker="MSFT",
        filing_type="8-K",
        original_filename="x.pdf",
        content_sha256="g" * 64,
        parsed_text="hi",
        parsed_char_count=2,
        page_count=1,
        uploaded_at=datetime.now(UTC),
    )

    parsed = ParsedDocument(
        text="hi", char_count=2, page_count=1, content_sha256="g" * 64
    )
    repo = _RacyRepository(winner=winner)

    event = await intake_upload(
        ticker="MSFT",
        filing_type="8-K",
        original_filename="x.pdf",
        parsed=parsed,
        repository=repo,
    )
    # We tried exactly once to insert, then recovered.
    assert repo.add_calls == 1
    # The returned event reuses the winner's upload_id.
    assert event.accession_number == "upload-winner-id"


@pytest.mark.asyncio
async def test_intake_rejects_rebind_to_different_ticker_or_form() -> None:
    """A second upload of the same bytes under a different ticker/form must fail."""
    repo = _FakeRepository()
    parsed = ParsedDocument(
        text="hi", char_count=2, page_count=1, content_sha256="h" * 64
    )
    # First upload -- MSFT / 8-K.
    await intake_upload(
        ticker="MSFT",
        filing_type="8-K",
        original_filename="x.pdf",
        parsed=parsed,
        repository=repo,
    )
    assert len(repo.saved) == 1
    # Same bytes, different filing_type -- must raise, must NOT insert a second row.
    with pytest.raises(ValueError, match="previously uploaded"):
        await intake_upload(
            ticker="MSFT",
            filing_type="10-Q",
            original_filename="x.pdf",
            parsed=parsed,
            repository=repo,
        )
    assert len(repo.saved) == 1
