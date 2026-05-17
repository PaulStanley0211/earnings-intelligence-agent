"""Integration test: accepted critic -> notes row written via note_writer.

Mirrors the inline graph-invocation pattern from
:mod:`tests.integration.test_graph` — no speculative fixtures are used.
The speculative ``invoke_graph_for_filing`` / ``test_session_factory``
fixtures from the plan do not exist; instead each test constructs a
``FilingEvent``, invokes ``build_graph(...).ainvoke(...)``, and asserts
on the final state. The ``session_factory`` fixture matches the one in
``test_graph.py`` exactly (fresh schema, pre-seeded 10-Q filing).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
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

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEdgar:
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

    async def get_filing_document(
        self,
        *,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> str:
        return (
            "<html><body>"
            "<p>Item 2. Management's Discussion and Analysis</p>"
            "<p>Revenue grew supported by enterprise demand for cloud platform.</p>"
            "<p>Item 3. Other</p>"
            "</body></html>"
        )


class _StubConsensus:
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


class _StubEmbeddings:
    """Deterministic 1536-dim vectors; satisfies ``_SupportsEmbed``."""

    @property
    def model(self) -> str:
        return "openai/text-embedding-3-small"

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [float(ord(t[0]) % 7) / 7.0 + 0.001] + [0.0] * 1535
            for t in texts
        ]


_LLM_CRITIC_JSON = '{"findings": []}'
"""Canned LLM critic response: no semantic findings, note accepted."""

_LLM_CRITIC_MARKER = '<source name="draft_note">'
"""Substring present in every rendered llm_v1 prompt body, absent from all others."""


def _stub_anthropic(text_: str) -> MagicMock:
    """Return a MagicMock Anthropic client that routes by call content.

    LLM critic calls are detected via the ``<source name="draft_note">`` marker
    in the rendered prompt body and receive a clean ``{"findings": []}`` response.
    All other calls receive ``text_``.
    """
    client = MagicMock()

    def _create(**kwargs: Any) -> MagicMock:
        messages: list[dict[str, str]] = kwargs.get("messages", [])
        user_text = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        response_text = _LLM_CRITIC_JSON if _LLM_CRITIC_MARKER in user_text else text_
        return MagicMock(
            content=[MagicMock(type="text", text=response_text)],
            usage=MagicMock(input_tokens=100, output_tokens=80),
        )

    client.messages.create.side_effect = _create
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ACCESSION = "0000950170-26-000050"
_CIK = "0000789019"
_TICKER = "MSFT"
_FILED_AT = datetime(2026, 4, 25, 20, 5, tzinfo=UTC)

_ACCEPTED_NOTE = (
    "## Headline\n"
    "MSFT reported revenue of $61.9 billion [F1].\n\n"
    "## Numbers\n"
    "- Revenue: $61.9 billion [F1]\n\n"
    "## Versus consensus\n"
    "- Revenue beat consensus by 1.41% [C1]\n"
)


@pytest_asyncio.fixture()
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Fresh schema + pre-seeded 10-Q filing; matches the pattern in test_graph.py."""
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        await Repository(session).record_filing(
            filing=NewFiling(
                accession_number=_ACCESSION,
                cik=_CIK,
                ticker=_TICKER,
                form=FilingForm.FORM_10Q,
                filed_at=_FILED_AT,
                source_url="https://www.sec.gov/...",
            )
        )
        await session.commit()
    yield factory
    await engine.dispose()


def _make_graph(
    cassette_dir: Path,
    session_factory: async_sessionmaker[AsyncSession],
    note_text: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Assemble a graph with stubbed clients and the given canned LLM response."""
    monkeypatch.setenv("REC", "1")
    llm = LLMClient(
        cassette_dir=cassette_dir,
        anthropic_client=_stub_anthropic(note_text),
    )
    return build_graph(
        edgar=_StubEdgar(),
        consensus_fetcher=_StubConsensus(),
        embeddings=_StubEmbeddings(),
        llm=llm,
        session_factory=session_factory,
    )


def _make_initial_state() -> AgentState:
    """Build the initial AgentState for the pre-seeded 10-Q filing."""
    return AgentState(
        trace_id="trace-notes-persistence",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number=_ACCESSION,
            cik=_CIK,
            ticker=_TICKER,
            form=FilingForm.FORM_10Q,
            filed_at=_FILED_AT,
            source_url="https://www.sec.gov/...",
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_accepted_note_persists_one_row(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the critic accepts, the note_writer must insert one row into ``notes``.

    Asserts that:
    - ``critic_verdict == "accepted"``
    - ``persisted_note_id`` is not None (the note_writer ran and wrote the row)
    - The row in the DB has the correct markdown body and critic_attempts count
    """
    graph = _make_graph(cassette_dir, session_factory, _ACCEPTED_NOTE, monkeypatch=monkeypatch)
    final = await graph.ainvoke(_make_initial_state())
    if not isinstance(final, dict):
        final = final.__dict__

    assert final["critic_verdict"] == "accepted"
    assert final["persisted_note_id"] is not None, (
        "persisted_note_id should be set when note_writer runs after accepted critic"
    )

    async with session_factory() as session:
        repo = Repository(session)
        latest = await repo.get_latest_note(ticker=_TICKER)

    assert latest is not None, "notes table should have a row for MSFT"
    assert latest.markdown_body == _ACCEPTED_NOTE.rstrip()
    assert latest.critic_attempts == final["critic_attempts"]


async def test_rerun_returns_same_note_id(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second graph run for the same filing must return the same ``persisted_note_id``.

    The repository's ``insert_note`` is idempotent (ON CONFLICT DO NOTHING), so
    re-invoking the full pipeline for an already-processed filing must not
    duplicate the notes row.
    """
    graph = _make_graph(cassette_dir, session_factory, _ACCEPTED_NOTE, monkeypatch=monkeypatch)

    final1 = await graph.ainvoke(_make_initial_state())
    if not isinstance(final1, dict):
        final1 = final1.__dict__

    final2 = await graph.ainvoke(_make_initial_state())
    if not isinstance(final2, dict):
        final2 = final2.__dict__

    assert final1["persisted_note_id"] is not None
    assert final1["persisted_note_id"] == final2["persisted_note_id"], (
        "idempotent re-run must return the same note id, not insert a duplicate"
    )
