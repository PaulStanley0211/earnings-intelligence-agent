"""``GET /health`` endpoint.

The endpoint runs three readiness checks:

1. Postgres - issues ``SELECT 1`` against the live engine.
2. Redis - pings the configured server.
3. EDGAR watcher freshness - reads the most-recent ``ok`` row from
   ``edgar_poll_log`` and compares its age to the configured threshold.

A failing database check returns HTTP 503 because the API cannot serve
useful traffic without Postgres. A degraded Redis or stale watcher returns
HTTP 200 with ``status: degraded`` so Docker does not flap the API
container while the operator investigates.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Final, Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.memory.db import get_engine
from app.memory.redis_client import get_redis
from app.memory.repository import Repository
from app.observability.logging import get_logger

router = APIRouter(tags=["health"])

_logger = get_logger()

_STATUS_OK: Final[str] = "ok"
_STATUS_DEGRADED: Final[str] = "degraded"
_STATUS_ERROR: Final[str] = "error"

# Watcher must have polled within this window to be considered fresh. Five
# minutes matches PLAN.md section 8 ("last successful EDGAR poll within 5 min").
_POLL_FRESHNESS_THRESHOLD: Final[timedelta] = timedelta(minutes=5)


CheckStatus = Literal["ok", "stale", "unknown", "error", "not_applicable"]


class HealthChecks(BaseModel):
    """Per-subsystem readiness verdicts."""

    database: CheckStatus
    redis: CheckStatus
    edgar_watcher: CheckStatus


class HealthDetails(BaseModel):
    """Optional diagnostics alongside the verdicts."""

    last_poll_age_seconds: float | None = None
    database_error: str | None = None
    redis_error: str | None = None


class HealthResponse(BaseModel):
    """Shape of the ``/health`` payload."""

    status: str
    version: str
    timestamp: str
    checks: HealthChecks
    details: HealthDetails


@router.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    """Return overall readiness plus per-subsystem detail."""
    details = HealthDetails()
    db_status = await _check_database(details)
    redis_status = await _check_redis(details)
    watcher_status = await _check_watcher(details)

    overall: str
    if db_status == "error":
        overall = _STATUS_ERROR
        response.status_code = 503
    elif (
        redis_status != "ok"
        or watcher_status in {"stale", "unknown", "error"}
    ):
        overall = _STATUS_DEGRADED
    else:
        overall = _STATUS_OK

    return HealthResponse(
        status=overall,
        version=__version__,
        timestamp=datetime.now(UTC).isoformat(),
        checks=HealthChecks(
            database=db_status,
            redis=redis_status,
            edgar_watcher=watcher_status,
        ),
        details=details,
    )


async def _check_database(details: HealthDetails) -> CheckStatus:
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        details.database_error = str(exc)[:200]
        _logger.bind(error=str(exc)).warning("health_database_check_failed")
        return "error"


async def _check_redis(details: HealthDetails) -> CheckStatus:
    try:
        pong = await get_redis().ping()
        return "ok" if pong else "error"
    except Exception as exc:
        details.redis_error = str(exc)[:200]
        _logger.bind(error=str(exc)).warning("health_redis_check_failed")
        return "error"


async def _check_watcher(details: HealthDetails) -> CheckStatus:
    from app.config import get_settings

    if not get_settings().watcher_mode_enabled:
        return "not_applicable"
    try:
        engine = get_engine()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            last = await Repository(session).last_successful_poll_at()
        if last is None:
            return "unknown"
        age = datetime.now(UTC) - last.polled_at
        details.last_poll_age_seconds = age.total_seconds()
        return "ok" if age <= _POLL_FRESHNESS_THRESHOLD else "stale"
    except Exception as exc:
        details.database_error = (details.database_error or "") + f"; watcher: {exc!s}"
        _logger.bind(error=str(exc)).warning("health_watcher_check_failed")
        return "error"
