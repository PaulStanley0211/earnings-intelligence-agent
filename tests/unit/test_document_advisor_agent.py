"""Unit tests for the document_advisor agent node."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from app.agents.document_advisor import UnknownTickerError
from app.agents.document_advisor import advise as advise_node
from app.memory.schemas import WatchlistRecord
from app.tools.edgar import RecentFiling, SubmissionsResponse


class _FakeEdgar:
    """Stub for ``EdgarClient.get_submissions``."""

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse:
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
                    primary_document="msft.htm",
                )
            ],
        )


class _FakeRepository:
    """Stand-in for Repository: looks up CIK by ticker."""

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
async def test_advise_for_known_ticker() -> None:
    output = await advise_node(
        ticker="MSFT", repository=_FakeRepository(), edgar=_FakeEdgar()
    )
    assert output.ticker == "MSFT"
    assert len(output.suggested) == 1
    assert output.suggested[0].filing_type == "8-K"


@pytest.mark.asyncio
async def test_advise_for_unknown_ticker_raises() -> None:
    """Tickers not in the watchlist need to be added before advising."""
    with pytest.raises(UnknownTickerError):
        await advise_node(
            ticker="ZZZZ", repository=_FakeRepository(), edgar=_FakeEdgar()
        )
