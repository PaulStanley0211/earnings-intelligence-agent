"""Integration tests for the upgraded ``/health`` endpoint.

The endpoint exercises three checks - Postgres reachability, Redis ping, and
the freshness of the most recent EDGAR poll. The first two are real (the
docker-compose stack supplies both); the poll-log row is seeded by the test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import create_app
from app.memory.db import build_engine, dispose_engine, get_engine
from app.memory.models import Base, EdgarPollLog

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture()
async def fresh_schema() -> AsyncIterator[None]:
    """Reset the schema and process-wide engine before each /health test."""
    await dispose_engine()
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    # Force the next get_engine() call to build a clean pool from the same DSN.
    await dispose_engine()
    _ = get_engine()
    yield
    await dispose_engine()


async def _seed_poll_log(*, polled_at: datetime, status: str) -> None:
    engine = get_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = EdgarPollLog(
            polled_at=polled_at,
            tickers_checked=5,
            filings_found=0,
            status=status,
        )
        session.add(row)
        await session.commit()


async def test_health_returns_ok_when_recent_poll_exists(fresh_schema: None) -> None:
    await _seed_poll_log(polled_at=datetime.now(UTC), status="ok")
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["edgar_watcher"] == "ok"


async def test_health_reports_stale_poll_as_degraded(fresh_schema: None) -> None:
    await _seed_poll_log(
        polled_at=datetime.now(UTC) - timedelta(minutes=10),
        status="ok",
    )
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    # The watcher being stale is degraded, not down - keep 200 so Docker
    # does not restart the API container in lock-step with the watcher.
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["edgar_watcher"] == "stale"


async def test_health_marks_watcher_unknown_when_never_polled(fresh_schema: None) -> None:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["checks"]["edgar_watcher"] == "unknown"
    assert body["status"] == "degraded"


async def test_health_database_check_executes_a_real_query(fresh_schema: None) -> None:
    # If the DB check is wired correctly, dropping the table the endpoint
    # touches must not break /health - the endpoint should only run SELECT 1.
    engine = get_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await session.execute(text("DROP TABLE filings CASCADE"))
        await session.commit()
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["checks"]["database"] == "ok"
