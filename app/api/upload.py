"""``POST /api/upload``: ingest a user-supplied filing, run the graph.

The validation order is deliberate: content type first (saves bandwidth on
unsupported types), then size (cheap header-driven check), then parse, then
intake, then graph invocation. A 413 short-circuits before we decode bytes
into a PDF/text parser, a 415 short-circuits before we even read the body.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Final, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.upload_intake import intake_upload
from app.api.dependencies import get_compiled_graph, get_session
from app.config import get_settings
from app.memory.repository import Repository
from app.models.state import AgentState
from app.tools.documents import (
    DocumentParseError,
    ParsedDocument,
    parse_pdf,
    parse_plain_text,
)

router = APIRouter(prefix="/api", tags=["upload"])

_ACCEPTED_PDF_TYPES: Final[set[str]] = {"application/pdf"}
_ACCEPTED_TEXT_TYPES: Final[set[str]] = {"text/plain"}


class AnalysisPayload(BaseModel):
    """The structured analysis distilled from the upload."""

    financials: dict[str, Any] | None
    comparisons: dict[str, Any] | None
    language_diffs: list[dict[str, Any]]
    draft_note: str | None
    final_note: str | None
    critic_verdict: str | None


class UploadResponse(BaseModel):
    """Response shape for ``POST /api/upload``."""

    upload_id: str
    trace_id: str
    status: Literal["completed", "failed"]
    analysis: AnalysisPayload


def _reject_unsupported_type(content_type: str | None) -> None:
    """Raise 415 when the upload's declared content type is not accepted."""
    if (
        content_type not in _ACCEPTED_PDF_TYPES
        and content_type not in _ACCEPTED_TEXT_TYPES
    ):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type {content_type!r}. "
                "Accepted: application/pdf, text/plain."
            ),
        )


def _parse_or_422(content_type: str | None, raw: bytes) -> ParsedDocument:
    """Parse the upload according to its declared content type.

    Raises :class:`HTTPException` 422 for parser-detected problems (e.g.
    scanned PDFs, empty text). The 415 branch is handled separately so it
    short-circuits before the file body is even read.
    """
    # ``_reject_unsupported_type`` runs first, so the only remaining
    # option after the PDF branch is plain-text.
    parser = (
        parse_pdf if content_type in _ACCEPTED_PDF_TYPES else parse_plain_text
    )
    try:
        return parser(raw)
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/upload", response_model=UploadResponse)
async def post_upload(
    ticker: Annotated[str, Form()],
    filing_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    session: Annotated[AsyncSession, Depends(get_session)],
    graph: Annotated[
        CompiledStateGraph[Any, Any, Any, Any], Depends(get_compiled_graph)
    ],
) -> UploadResponse:
    """Run the full Phase 1-3 pipeline against the uploaded document.

    Returns the structured analysis (financials, comparisons, language
    diffs, draft + final note, critic verdict) once the graph finishes.
    """
    _reject_unsupported_type(file.content_type)

    settings = get_settings()
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload exceeds MAX_UPLOAD_BYTES "
                f"({settings.max_upload_bytes} bytes)."
            ),
        )

    parsed = _parse_or_422(file.content_type, raw)

    repository = Repository(session)
    try:
        filing_event = await intake_upload(
            ticker=ticker,
            filing_type=filing_type,
            original_filename=file.filename or "upload.bin",
            parsed=parsed,
            repository=repository,
        )
    except ValueError as exc:
        # Unknown ticker or unsupported filing_type. ``get_session`` rolls
        # back automatically when this propagates.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    trace_id = uuid4().hex
    initial_state = AgentState(
        trace_id=trace_id,
        started_at=datetime.now(UTC),
        filing_event=filing_event,
    )
    final = await graph.ainvoke(initial_state)
    final_state = (
        final if isinstance(final, AgentState) else AgentState.model_validate(final)
    )

    upload_id = filing_event.accession_number.removeprefix("upload-")
    verdict = (
        final_state.critic_verdict.value
        if final_state.critic_verdict is not None
        else None
    )
    return UploadResponse(
        upload_id=upload_id,
        trace_id=trace_id,
        status="completed",
        analysis=AnalysisPayload(
            financials=final_state.financials,
            comparisons=final_state.comparisons,
            language_diffs=final_state.language_diffs,
            draft_note=final_state.draft_note,
            final_note=final_state.final_note,
            critic_verdict=verdict,
        ),
    )
