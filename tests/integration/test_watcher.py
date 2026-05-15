"""Integration tests for the EDGAR watcher.

Drives :func:`app.agents.watcher.poll_once` against a stub EDGAR client and
the real Postgres test database. The stub mirrors the shape of
:class:`~app.tools.edgar.EdgarClient` so the test exercises the same call
sites the production client implements.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.watcher import PollResult, poll_once
from app.memory.db import build_engine
from app.memory.models import Base
from app.memory.repository import Repository
from app.memory.schemas import FilingStatus
from app.tools.edgar import CompanyFactsResponse, RecentFiling, SubmissionsResponse

pytestmark = pytest.mark.integration


class StubEdgarClient:
    """Test double matching the methods :func:`poll_once` calls."""

    def __init__(
        self,
        submissions: dict[str, SubmissionsResponse],
        facts: dict[str, CompanyFactsResponse],
    ) -> None:
        self._submissions = submissions
        self._facts = facts
        self.submissions_calls: list[str] = []
        self.facts_calls: list[str] = []

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse:
        padded = cik.zfill(10)
        self.submissions_calls.append(padded)
        return self._submissions[padded]

    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse:
        padded = cik.zfill(10)
        self.facts_calls.append(padded)
        return self._facts[padded]


@pytest_asyncio.fixture()
async def session() -> AsyncIterator[AsyncSession]:
    """Clean schema per-test so we can assert on insert counts."""
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as live:
        yield live
    await engine.dispose()


async def _seed_watchlist(session: AsyncSession) -> None:
    repo = Repository(session)
    await repo.upsert_watchlist_entry(
        ticker="MSFT", cik="0000789019", company_name="Microsoft Corp"
    )
    await session.commit()


def _submissions(filings: list[RecentFiling]) -> SubmissionsResponse:
    return SubmissionsResponse(
        cik="0000789019",
        entity_name="Microsoft Corp",
        tickers=["MSFT"],
        recent_filings=filings,
    )


def _company_facts(accession: str, value: int = 61858000000) -> CompanyFactsResponse:
    return CompanyFactsResponse(
        cik="0000789019",
        entity_name="Microsoft Corp",
        raw={
            "cik": 789019,
            "entityName": "Microsoft Corp",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenues",
                        "units": {
                            "USD": [
                                {
                                    "start": "2026-01-01",
                                    "end": "2026-03-31",
                                    "val": value,
                                    "accn": accession,
                                    "fy": 2026,
                                    "fp": "Q3",
                                    "form": "10-Q",
                                    "filed": "2026-04-25",
                                }
                            ]
                        },
                    }
                }
            },
        },
    )


async def test_poll_once_detects_and_persists_new_filing(session: AsyncSession) -> None:
    await _seed_watchlist(session)
    accession = "0000950170-26-000050"
    edgar = StubEdgarClient(
        submissions={
            "0000789019": _submissions(
                [
                    RecentFiling(
                        accession_number=accession,
                        form="10-Q",
                        filing_date=date(2026, 4, 25),
                        report_date=date(2026, 3, 31),
                        primary_document="msft.htm",
                    )
                ]
            )
        },
        facts={"0000789019": _company_facts(accession)},
    )
    result = await poll_once(edgar=edgar, session=session)
    assert isinstance(result, PollResult)
    assert result.tickers_checked == 1
    assert result.filings_found == 1

    repo = Repository(session)
    stored = await repo.get_filing(accession)
    assert stored is not None
    assert stored.status is FilingStatus.PROCESSED

    facts = await repo.get_facts_for_filing(accession)
    assert {fact.concept for fact in facts} == {"Revenues"}


async def test_poll_once_is_idempotent_across_runs(session: AsyncSession) -> None:
    await _seed_watchlist(session)
    accession = "0000950170-26-000050"
    edgar = StubEdgarClient(
        submissions={
            "0000789019": _submissions(
                [
                    RecentFiling(
                        accession_number=accession,
                        form="10-Q",
                        filing_date=date(2026, 4, 25),
                        report_date=date(2026, 3, 31),
                        primary_document="msft.htm",
                    )
                ]
            )
        },
        facts={"0000789019": _company_facts(accession)},
    )

    first = await poll_once(edgar=edgar, session=session)
    second = await poll_once(edgar=edgar, session=session)
    assert first.filings_found == 1
    assert second.filings_found == 0, "second pass must skip the already-known filing"
    # The second pass must not fetch companyfacts at all - that would be wasted spend.
    assert len(edgar.facts_calls) == 1


async def test_poll_once_skips_unsupported_forms(session: AsyncSession) -> None:
    await _seed_watchlist(session)
    edgar = StubEdgarClient(
        submissions={
            "0000789019": _submissions(
                [
                    RecentFiling(
                        accession_number="0000950170-26-DEF14",
                        form="DEF 14A",
                        filing_date=date(2026, 4, 25),
                        report_date=None,
                        primary_document="proxy.htm",
                    )
                ]
            )
        },
        facts={},
    )
    result = await poll_once(edgar=edgar, session=session)
    assert result.filings_found == 0


async def test_poll_once_records_a_poll_log_entry(session: AsyncSession) -> None:
    await _seed_watchlist(session)
    edgar = StubEdgarClient(
        submissions={"0000789019": _submissions([])},
        facts={},
    )
    await poll_once(edgar=edgar, session=session)
    repo = Repository(session)
    last = await repo.last_successful_poll_at()
    assert last is not None
    assert last.tickers_checked == 1
    assert last.filings_found == 0
