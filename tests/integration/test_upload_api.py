"""End-to-end tests for /api/advise (and later /api/upload, /api/chat)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_edgar_client
from app.main import create_app
from app.memory.db import build_engine, dispose_engine, get_engine
from app.memory.models import Base
from app.memory.repository import Repository
from app.tools.edgar import RecentFiling, SubmissionsResponse

pytestmark = pytest.mark.integration


class _FakeEdgar:
    """Stub EDGAR client matching :class:`EdgarClient`'s submission contract."""

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse:
        """Return a fixed pair of recent filings (one 8-K, one 10-Q)."""
        return SubmissionsResponse(
            cik=cik,
            entity_name="Microsoft Corp",
            tickers=["MSFT"],
            sic_description=None,
            recent_filings=[
                RecentFiling(
                    accession_number="0001193125-26-191457",
                    form="8-K",
                    filing_date=date(2026, 4, 29),
                    report_date=date(2026, 4, 29),
                    primary_document="msft-20260429.htm",
                ),
                RecentFiling(
                    accession_number="0001193125-26-027207",
                    form="10-Q",
                    filing_date=date(2026, 1, 28),
                    report_date=date(2025, 12, 31),
                    primary_document="msft-20260128.htm",
                ),
            ],
        )


@pytest_asyncio.fixture()
async def fresh_schema() -> AsyncIterator[None]:
    """Reset the schema and process-wide engine before each test."""
    await dispose_engine()
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    await dispose_engine()
    _ = get_engine()
    yield
    await dispose_engine()


@pytest_asyncio.fixture()
async def seed_watchlist_msft(fresh_schema: None) -> AsyncIterator[None]:
    """Insert MSFT into the watchlist so the advisor can resolve it."""
    engine = get_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await Repository(session).upsert_watchlist_entry(
            ticker="MSFT",
            cik="0000789019",
            company_name="Microsoft Corp",
            active=True,
        )
        await session.commit()
    yield


async def _stub_edgar() -> AsyncIterator[_FakeEdgar]:
    """Async generator yielding a fake EDGAR client."""
    yield _FakeEdgar()


@pytest_asyncio.fixture()
async def app_with_fake_edgar() -> AsyncIterator[AsyncClient]:
    """Build a FastAPI app with the EDGAR dependency overridden to the fake."""
    app = create_app()
    app.dependency_overrides[get_edgar_client] = _stub_edgar
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def test_advise_for_msft_returns_checklist(
    seed_watchlist_msft: None, app_with_fake_edgar: AsyncClient
) -> None:
    """A seeded ticker yields a 200 with at least the 8-K in the checklist."""
    response = await app_with_fake_edgar.post("/api/advise", json={"ticker": "MSFT"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert any(s["filing_type"] == "8-K" for s in payload["suggested"])
    assert "transcript" in payload["transcript_hint"].lower()


async def test_advise_unknown_ticker_404(
    fresh_schema: None, app_with_fake_edgar: AsyncClient
) -> None:
    """An unseeded ticker yields a 404 mentioning 'watchlist'."""
    response = await app_with_fake_edgar.post(
        "/api/advise", json={"ticker": "ZZZZZZ"}
    )
    assert response.status_code == 404
    assert "watchlist" in response.json()["detail"].lower()
