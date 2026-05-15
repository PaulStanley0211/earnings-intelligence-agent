"""``GET /health`` endpoint.

Phase 0 stub: returns a fixed payload so ``docker compose up`` succeeds before
the database and Redis are wired in. Phase 1 will replace it with real checks
(Postgres ping, Redis ping, last EDGAR poll within 5 minutes) per PLAN.md
section 8 ("Deployment").
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__

router = APIRouter(tags=["health"])

_STATUS_OK: Final[str] = "ok"


class HealthResponse(BaseModel):
    """Shape of the ``/health`` payload."""

    status: str
    version: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return liveness for orchestrators and the Docker healthcheck.

    Phase 1 will extend this to a readiness check covering Postgres, Redis,
    and the last successful EDGAR poll, per PLAN.md section 8.
    """
    return HealthResponse(
        status=_STATUS_OK,
        version=__version__,
        timestamp=datetime.now(UTC).isoformat(),
    )
