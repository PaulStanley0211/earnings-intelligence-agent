"""Integration test for the Phase 1 LangGraph skeleton.

Compiles the graph, fires it once with a synthetic filing event against a
stub EDGAR client, and verifies the financial-extractor node ran and wrote
its summary into ``AgentState.financials``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.graph import build_graph
from app.memory.db import build_engine
from app.memory.models import Base
from app.memory.repository import Repository
from app.memory.schemas import NewFiling
from app.models.state import AgentState, FilingEvent, FilingForm
from app.tools.edgar import CompanyFactsResponse

pytestmark = pytest.mark.integration


class StubEdgar:
    """Test double that returns one revenue fact for the requested CIK."""

    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse:
        return CompanyFactsResponse(
            cik=cik.zfill(10),
            entity_name="Microsoft Corp",
            raw={
                "cik": int(cik),
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
                                        "val": 61858000000,
                                        "accn": "0000950170-26-000050",
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


@pytest_asyncio.fixture()
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    # Seed the filing row so the financial_extractor has a parent to insert facts under.
    accession = "0000950170-26-000050"
    async with factory() as session:
        await Repository(session).record_filing(
            filing=NewFiling(
                accession_number=accession,
                cik="0000789019",
                ticker="MSFT",
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
                source_url="https://www.sec.gov/...",
            )
        )
        await session.commit()
    yield factory
    await engine.dispose()


async def test_compiled_graph_runs_financial_extractor(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    graph = build_graph(edgar=StubEdgar(), session_factory=session_factory)
    initial = AgentState(
        trace_id="trace-test",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="0000950170-26-000050",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
            source_url="https://www.sec.gov/...",
        ),
    )
    final = await graph.ainvoke(initial)
    financials = final["financials"] if isinstance(final, dict) else final.financials
    assert financials is not None
    assert financials["source"] == "companyfacts"
    assert financials["parsed_count"] == 1
    assert "Revenues" in financials["concepts"]
