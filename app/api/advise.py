"""``POST /api/advise``: given a ticker, return the upload checklist."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.document_advisor import UnknownTickerError, advise
from app.api.dependencies import get_edgar_client, get_session
from app.memory.repository import Repository
from app.tools.advisor import AdvisorOutput
from app.tools.edgar import EdgarClient

router = APIRouter(prefix="/api", tags=["advise"])


class AdviseRequest(BaseModel):
    """Request body for ``POST /api/advise``."""

    ticker: str


class AdviseFilingResponse(BaseModel):
    """One row of the response checklist."""

    filing_type: str
    accession_number: str
    filed_at: str
    edgar_index_url: str
    primary_document: str | None


class AdviseResponse(BaseModel):
    """Response body for ``POST /api/advise``."""

    ticker: str
    cik: str
    suggested: list[AdviseFilingResponse]
    transcript_hint: str


def _to_response(output: AdvisorOutput) -> AdviseResponse:
    """Render an :class:`AdvisorOutput` as the JSON-serialisable response."""
    return AdviseResponse(
        ticker=output.ticker,
        cik=output.cik,
        suggested=[
            AdviseFilingResponse(
                filing_type=f.filing_type,
                accession_number=f.accession_number,
                filed_at=f.filed_at.isoformat(),
                edgar_index_url=f.edgar_index_url,
                primary_document=f.primary_document,
            )
            for f in output.suggested
        ],
        transcript_hint=output.transcript_hint,
    )


@router.post("/advise", response_model=AdviseResponse)
async def post_advise(
    body: AdviseRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    edgar: Annotated[EdgarClient, Depends(get_edgar_client)],
) -> AdviseResponse:
    """Return the upload checklist for ``body.ticker``."""
    try:
        output = await advise(
            ticker=body.ticker, repository=Repository(session), edgar=edgar
        )
    except UnknownTickerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(output)
