"""Upload intake agent node.

Persists an uploaded document and produces the canonical
:class:`~app.models.state.FilingEvent` that drives the downstream graph.
The intake is idempotent on the SHA-256 of the raw bytes: re-uploading the
same content -- including under concurrent requests racing into the same
``add_uploaded_document`` -- returns the existing row's ``upload_id``
rather than producing a duplicate.

The graph downstream is identical to the watcher-driven path; only the
``FilingEvent.source`` discriminator differs.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from app.memory.schemas import (
    FilingRecord,
    NewFiling,
    NewUploadedDocument,
    UploadedDocumentRecord,
    WatchlistRecord,
)
from app.models.state import FilingEvent, FilingEventSource, FilingForm
from app.tools.documents import ParsedDocument


class _SupportsUploadStorage(Protocol):
    """Minimum repository surface the intake node depends on."""

    async def add_uploaded_document(
        self, new: NewUploadedDocument
    ) -> UploadedDocumentRecord: ...

    async def get_uploaded_document_by_sha256(
        self, content_sha256: str
    ) -> UploadedDocumentRecord | None: ...

    async def get_watchlist_entry_by_ticker(
        self, ticker: str
    ) -> WatchlistRecord | None: ...

    async def record_filing(self, *, filing: NewFiling) -> FilingRecord | None: ...


def _filing_form(filing_type: str) -> FilingForm:
    """Map the user-supplied filing-type string to the enum the graph uses.

    Accepts every member of :class:`FilingForm`, including ``TRANSCRIPT``
    (Phase 4B), so user-uploaded earnings-call transcripts route through
    the transcript-analyzer track in the compiled graph.
    """
    try:
        return FilingForm(filing_type)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported filing_type {filing_type!r}; expected one of "
            f"{[m.value for m in FilingForm]}."
        ) from exc


def _reject_if_rebind(
    existing: UploadedDocumentRecord, ticker_upper: str, form_value: str
) -> None:
    """Raise if the existing row's ticker/form disagree with the new call.

    Without this guard, a second upload of the same bytes under a different
    ticker or filing_type would silently reuse the original row's upload_id,
    producing a FilingEvent whose ``ticker`` / ``form`` disagree with the
    persisted audit row.
    """
    if existing.ticker != ticker_upper or existing.filing_type != form_value:
        raise ValueError(
            f"Content was previously uploaded for "
            f"{existing.ticker!r}/{existing.filing_type!r}; re-uploading "
            f"the same bytes as {ticker_upper!r}/{form_value!r} would "
            "create an inconsistent audit row. Use a different file."
        )


def _default_clock() -> datetime:
    """Production clock used when callers omit ``clock=``.

    Wall-clock ``datetime.now(UTC)`` is non-deterministic across runs, which
    means ``filed_at`` reaches the synthesizer prompt with a fresh timestamp
    on every invocation. That moves the SHA-keyed cassette key and breaks
    replay. Production paths still want a real timestamp, so the default
    stays here -- tests inject a frozen clock to keep cassette keys stable.
    """
    return datetime.now(UTC)


async def intake_upload(
    *,
    ticker: str,
    filing_type: str,
    original_filename: str,
    parsed: ParsedDocument,
    repository: _SupportsUploadStorage,
    clock: Callable[[], datetime] | None = None,
) -> FilingEvent:
    """Persist (or recover) the uploaded document and return its FilingEvent.

    The returned event always carries ``source=FilingEventSource.UPLOAD`` so
    downstream tracing distinguishes user-driven runs from watcher-driven ones.

    ``clock`` is an injection seam for tests. Production callers omit it and
    receive :func:`datetime.now` (UTC). Tests pass a frozen clock so the
    synthesizer's ``{filed_at}`` substitution lands at a deterministic value
    and cassette SHA keys stay stable across runs.
    """
    now = clock if clock is not None else _default_clock
    ticker_upper = ticker.upper()
    form = _filing_form(filing_type)
    entry = await repository.get_watchlist_entry_by_ticker(ticker_upper)
    if entry is None:
        raise ValueError(
            f"Ticker {ticker_upper!r} not on watchlist; add it before uploading."
        )

    existing = await repository.get_uploaded_document_by_sha256(parsed.content_sha256)
    if existing is not None:
        _reject_if_rebind(existing, ticker_upper, form.value)
        upload_id = existing.upload_id
    else:
        candidate_upload_id = uuid4().hex
        try:
            await repository.add_uploaded_document(
                NewUploadedDocument(
                    upload_id=candidate_upload_id,
                    ticker=ticker_upper,
                    filing_type=form.value,
                    original_filename=original_filename,
                    content_sha256=parsed.content_sha256,
                    parsed_text=parsed.text,
                    parsed_char_count=parsed.char_count,
                    page_count=parsed.page_count,
                )
            )
            upload_id = candidate_upload_id
        except IntegrityError:
            winner = await repository.get_uploaded_document_by_sha256(
                parsed.content_sha256
            )
            if winner is None:
                raise
            _reject_if_rebind(winner, ticker_upper, form.value)
            upload_id = winner.upload_id

    accession_number = f"upload-{upload_id}"
    filed_at = now()
    source_url = f"upload://{upload_id}"

    # Mirror the upload onto the ``filings`` table. Downstream agents
    # (transcript_analyzer, comparator, language_differ) persist rows
    # whose ``filing_accession`` columns FK to ``filings.accession_number``;
    # without a matching parent row those inserts hit a FK violation. The
    # repository method is idempotent on ``accession_number``
    # (ON CONFLICT DO NOTHING), so a retry or a duplicate-content path
    # that reuses an existing ``upload_id`` is safe.
    await repository.record_filing(
        filing=NewFiling(
            accession_number=accession_number,
            cik=entry.cik,
            ticker=ticker_upper,
            form=form,
            filed_at=filed_at,
            # No SEC URL exists for user uploads. Reuse the ``upload://``
            # scheme so audit tooling can distinguish upload-derived rows
            # from watcher-derived rows by source_url prefix alone.
            source_url=source_url,
            # Unknown for user uploads: a transcript/PDF carries no
            # canonical "report period end". The financial-extractor and
            # comparator infer the period from XBRL or the transcript
            # itself, not from this column.
            report_period_end=None,
        )
    )

    event = FilingEvent(
        accession_number=accession_number,
        cik=entry.cik,
        ticker=ticker_upper,
        form=form,
        filed_at=filed_at,
        source_url=source_url,
        source=FilingEventSource.UPLOAD,
    )
    # Belt-and-suspenders: upstream Task 6 defaults source to WATCHER, so
    # a future caller forgetting the kw-arg would silently misroute. Assert
    # here so the intake path can never produce a non-UPLOAD event.
    assert event.source is FilingEventSource.UPLOAD
    return event
