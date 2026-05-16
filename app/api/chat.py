"""``POST /api/chat``: stubbed in Phase 4A.

The citation-enforced chat agent lands in Phase 6. We register the route
now so the upload UI can be wired against a stable URL.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """Request body for ``POST /api/chat`` (Phase 6 will accept this shape)."""

    trace_id: str
    message: str


@router.post("/chat")
async def post_chat(_body: ChatRequest) -> None:
    """Return HTTP 501 until Phase 6 ships the citation-enforced chat agent."""
    raise HTTPException(
        status_code=501,
        detail=(
            "/api/chat is reserved for the Phase 6 citation-enforced chat agent. "
            "Phase 4A ships only the route shape."
        ),
    )
