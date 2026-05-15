"""Integration test for the Phase 2 LangGraph numbers track.

Compiles the graph, fires it once with a synthetic filing event against
stub clients (EDGAR, consensus, Anthropic), and verifies the full chain -
extractor -> comparator -> synthesizer -> critic - executes end-to-end
and lands an accepted note in ``AgentState.final_note``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.graph import build_graph
from app.llm.client import LLMClient
from app.memory.db import build_engine
from app.memory.models import Base
from app.memory.repository import Repository
from app.memory.schemas import NewConsensusEstimate, NewFiling
from app.models.state import AgentState, FilingEvent, FilingForm
from app.tools.edgar import CompanyFactsResponse

pytestmark = pytest.mark.integration


class StubEdgar:
    """Test double returning one revenue fact for the requested CIK."""

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


class StubConsensus:
    """Returns a single revenue consensus row matching the requested period."""

    async def fetch(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
        period_end: date,
    ) -> list[NewConsensusEstimate]:
        return [
            NewConsensusEstimate(
                ticker=ticker,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                metric="revenue",
                value=Decimal("61000000000"),
                source="finnhub",
            )
        ]


def _stub_anthropic(text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock(
        content=[MagicMock(type="text", text=text)],
        usage=MagicMock(input_tokens=100, output_tokens=80),
    )
    client.messages.create.return_value = msg
    return client


@pytest_asyncio.fixture()
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
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


async def test_numbers_track_graph_accepts_well_cited_draft(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REC", "1")
    # The synthesiser stub returns a draft that cites every figure - so the
    # critic accepts on the first pass.
    accepted_note = (
        "## Headline\n"
        "MSFT reported revenue of $61.9 billion [F1].\n\n"
        "## Numbers\n"
        "- Revenue: $61.9 billion [F1]\n\n"
        "## Versus consensus\n"
        "- Revenue beat consensus by 1.41% [C1]\n"
    )
    llm = LLMClient(
        cassette_dir=cassette_dir,
        anthropic_client=_stub_anthropic(accepted_note),
    )
    graph = build_graph(
        edgar=StubEdgar(),
        consensus_fetcher=StubConsensus(),
        llm=llm,
        session_factory=session_factory,
    )
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
    if not isinstance(final, dict):
        final = final.__dict__
    assert final["financials"]["source"] == "companyfacts"
    assert final["comparisons"]["consensus_source"] == "finnhub"
    metrics = final["comparisons"]["metrics"]
    assert any(m["metric"] == "revenue" for m in metrics)
    # The synthesiser strips trailing whitespace before publishing the draft.
    assert final["final_note"] == accepted_note.rstrip()
    findings: list[Any] = final["critic_findings"]
    assert findings == []
    assert final["critic_attempts"] == 1
    assert final["cost_usd"] > 0
