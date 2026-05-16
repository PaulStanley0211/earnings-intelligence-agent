"""Unit tests for the document advisor tool."""
from __future__ import annotations

from datetime import date

import pytest

from app.tools.advisor import (
    AdvisorOutput,
    advise_for_ticker,
)
from app.tools.edgar import RecentFiling, SubmissionsResponse


class _FakeEdgar:
    """Stub EDGAR client returning a canned recent-filings list."""

    def __init__(self) -> None:
        self._filings = SubmissionsResponse(
            cik="0000789019",
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
                RecentFiling(
                    accession_number="0001193125-26-027198",
                    form="8-K",
                    filing_date=date(2026, 1, 28),
                    report_date=date(2026, 1, 28),
                    primary_document="msft-20260128b.htm",
                ),
                RecentFiling(
                    accession_number="0000950170-25-100235",
                    form="10-K",
                    filing_date=date(2025, 8, 15),
                    report_date=date(2025, 6, 30),
                    primary_document="msft-20250630.htm",
                ),
            ],
        )

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse:
        return self._filings


@pytest.mark.asyncio
async def test_advise_returns_latest_per_type() -> None:
    output = await advise_for_ticker(
        ticker="MSFT", cik="0000789019", edgar=_FakeEdgar()
    )
    assert isinstance(output, AdvisorOutput)
    forms = [f.filing_type for f in output.suggested]
    assert "8-K" in forms
    assert "10-Q" in forms
    assert "10-K" in forms
    # Latest 8-K must be the Apr 29 one (newer of the two 8-Ks).
    eight_k = next(f for f in output.suggested if f.filing_type == "8-K")
    assert eight_k.accession_number == "0001193125-26-191457"
    # Every suggestion exposes the canonical EDGAR archive URL.
    for filing in output.suggested:
        assert filing.edgar_index_url.startswith(
            "https://www.sec.gov/Archives/edgar/data/789019/"
        )
    # Transcript hint is plain text, not a fetched URL.
    assert "transcript" in output.transcript_hint.lower()


@pytest.mark.asyncio
async def test_advise_orders_8k_before_10q_before_10k() -> None:
    """``suggested`` reflects upload priority for an earnings analysis."""
    output = await advise_for_ticker(
        ticker="MSFT", cik="0000789019", edgar=_FakeEdgar()
    )
    assert [f.filing_type for f in output.suggested] == ["8-K", "10-Q", "10-K"]
