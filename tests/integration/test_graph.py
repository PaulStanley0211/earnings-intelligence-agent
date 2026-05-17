"""Integration tests for the LangGraph pipeline.

Phase 2 test: compiles the graph, fires it once with a synthetic filing event
against stub clients (EDGAR, consensus, Anthropic), and verifies the full chain -
extractor -> comparator -> synthesizer -> critic - executes end-to-end and
lands an accepted note in ``AgentState.final_note``.

Phase 3 test: additionally exercises the parallel language_differ branch and
asserts that ``language_diffs`` is populated in the final state.

Phase 4B tests: exercise the topology change that adds ``transcript_analyzer``
as a third parallel branch. One test drives a ``TRANSCRIPT`` filing and
asserts the transcript analyzer ran while the financial trio self-skipped;
another drives an ``8-K`` filing and asserts the reverse -- the financial
track ran while the transcript analyzer self-skipped.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.graph import build_graph
from app.llm.client import LLMClient
from app.memory.db import build_engine
from app.memory.models import Base, Filing
from app.memory.repository import Repository
from app.memory.schemas import (
    NewConsensusEstimate,
    NewFiling,
    NewFilingSection,
    NewUploadedDocument,
    SectionKind,
)
from app.models.state import AgentState, FilingEvent, FilingEventSource, FilingForm
from app.tools.edgar import CompanyFactsResponse

pytestmark = pytest.mark.integration


class StubEdgar:
    """Test double returning one revenue fact for the requested CIK.

    Also satisfies ``_SupportsFilingDocument`` with a minimal HTML stub so the
    language differ can parse at least one section.
    """

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
            "<p>Operating expenses rose modestly as we expanded R&amp;D headcount.</p>"
            "<p>Item 3. Other</p>"
            "</body></html>"
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


class _StubEmbeddings:
    """Deterministic vectors; matches Vector(1536) schema via list multiplier."""

    @property
    def model(self) -> str:
        return "openai/text-embedding-3-small"

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        # 1536-dim vectors; each text gets a unique pattern based on first char.
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
        embeddings=_StubEmbeddings(),
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


async def test_phase3_graph_runs_language_differ_in_parallel(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fan-out from financial_extractor to both comparator and language_differ.

    Seeds a prior filing section so the differ has a real baseline, then
    asserts that ``language_diffs`` is populated with at least one MDA entry.
    """
    monkeypatch.setenv("REC", "1")
    accession_prior = "0000950170-26-000040"
    async with session_factory() as session:
        await Repository(session).record_filing(
            filing=NewFiling(
                accession_number=accession_prior,
                cik="0000789019",
                ticker="MSFT",
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 1, 25, tzinfo=UTC),
                source_url="https://www.sec.gov/x",
            )
        )
        await Repository(session).insert_filing_sections(
            [
                NewFilingSection(
                    filing_accession=accession_prior,
                    cik="0000789019",
                    ticker="MSFT",
                    section_kind=SectionKind.MDA,
                    paragraph_index=0,
                    text="Revenue grew supported by enterprise demand for cloud platform.",
                    text_sha="a" * 64,
                    embedding=[float(ord("R") % 7) / 7.0 + 0.001] + [0.0] * 1535,
                    embedding_model="openai/text-embedding-3-small",
                )
            ]
        )
        # Stamp primary_document on the current filing so the differ can fetch it.
        await session.execute(
            sa_update(Filing)
            .where(Filing.accession_number == "0000950170-26-000050")
            .values(primary_document="msft-20260331.htm")
        )
        await session.commit()

    accepted_note = (
        "## Headline\n"
        "MSFT reported revenue of $61.9 billion [F1].\n"
        "## Numbers\n- Revenue: $61.9 billion [F1]\n"
        "## Versus consensus\n- Revenue beat consensus by 1.41% [C1]\n"
    )
    llm = LLMClient(
        cassette_dir=cassette_dir,
        anthropic_client=_stub_anthropic(accepted_note),
    )
    graph = build_graph(
        edgar=StubEdgar(),
        consensus_fetcher=StubConsensus(),
        embeddings=_StubEmbeddings(),
        llm=llm,
        session_factory=session_factory,
    )
    initial = AgentState(
        trace_id="trace-phase3",
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
    payload = final["language_diffs"]
    assert isinstance(payload, list)
    assert any(s["section"] == "mda" for s in payload)


def _sequenced_anthropic(texts: list[str]) -> MagicMock:
    """Return a MagicMock that dequeues one canned text per ``messages.create``.

    LLM critic calls are detected via the ``<source name="draft_note">`` marker
    and short-circuit to a clean ``{"findings": []}`` response without consuming
    from the queue.
    """
    client = MagicMock()
    queue = list(texts)

    def _create(**kwargs: Any) -> Any:
        messages: list[dict[str, str]] = kwargs.get("messages", [])
        user_text = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        if _LLM_CRITIC_MARKER in user_text:
            return MagicMock(
                content=[MagicMock(type="text", text=_LLM_CRITIC_JSON)],
                usage=MagicMock(input_tokens=100, output_tokens=80),
            )
        if not queue:
            raise AssertionError("no more canned LLM responses configured")
        return MagicMock(
            content=[MagicMock(type="text", text=queue.pop(0))],
            usage=MagicMock(input_tokens=100, output_tokens=80),
        )

    client.messages.create.side_effect = _create
    return client


async def test_graph_runs_transcript_analyzer_on_transcript_filing(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TRANSCRIPT filing runs the transcript analyzer and self-skips the others.

    Asserts the transcript analyzer populated ``qa_pairs`` and
    ``commitments`` while ``financials``, ``comparisons``, and
    ``language_diffs`` all remain empty because their owning nodes
    self-skipped on the unsupported form.
    """
    monkeypatch.setenv("REC", "1")
    accession = "upload-tttest1"
    upload_id = "tttest1"
    async with session_factory() as session:
        repo = Repository(session)
        await repo.upsert_watchlist_entry(
            ticker="MSFT", cik="0000789019", company_name="Microsoft Corp"
        )
        await repo.add_uploaded_document(
            NewUploadedDocument(
                upload_id=upload_id,
                ticker="MSFT",
                filing_type="TRANSCRIPT",
                original_filename="msft-q3-transcript.txt",
                content_sha256="a" * 64,
                parsed_text=(
                    "Analyst: Can you discuss the cloud platform growth drivers?\n"
                    "Management: We expect to expand operating margin next quarter."
                ),
                parsed_char_count=120,
                page_count=None,
            )
        )
        await repo.record_filing(
            filing=NewFiling(
                accession_number=accession,
                cik="0000789019",
                ticker="MSFT",
                form=FilingForm.TRANSCRIPT,
                filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
                source_url=f"upload://{upload_id}",
            )
        )
        await session.commit()

    # Two canned LLM responses: the transcript analyzer's extract pass
    # (one Q&A, one commitment, no survivors so no reconcile call), then
    # the synthesizer's draft. The critic accepts because the note has
    # no numbers and quotes no transcript text -- so the citation index
    # has nothing to validate.
    extract_json = json.dumps(
        {
            "qa_pairs": [
                {
                    "ordinal": 1,
                    "analyst_name": "Analyst One",
                    "question_text": "Can you discuss the cloud platform growth drivers?",
                    "answer_text": "Demand for our cloud platform remains strong.",
                    "answer_class": "direct",
                }
            ],
            "commitments": [
                {
                    "commitment_text": "Expand operating margin next quarter.",
                    "target_period": "Q4 2026",
                    "source_quote": "We expect to expand operating margin next quarter.",
                }
            ],
        }
    )
    synth_note = (
        "## Headline\n"
        "Management addressed cloud platform demand and committed to margin expansion.\n"
    )
    llm = LLMClient(
        cassette_dir=cassette_dir,
        anthropic_client=_sequenced_anthropic([extract_json, synth_note]),
    )
    graph = build_graph(
        edgar=StubEdgar(),
        consensus_fetcher=StubConsensus(),
        embeddings=_StubEmbeddings(),
        llm=llm,
        session_factory=session_factory,
    )
    initial = AgentState(
        trace_id="trace-transcript",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number=accession,
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.TRANSCRIPT,
            filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
            source_url=f"upload://{upload_id}",
            source=FilingEventSource.UPLOAD,
        ),
    )
    final = await graph.ainvoke(initial)
    if not isinstance(final, dict):
        final = final.__dict__
    qa_pairs = final["qa_pairs"]
    commitments = final["commitments"]
    assert len(qa_pairs) == 1
    assert qa_pairs[0].question_text.startswith("Can you discuss")
    assert len(commitments) == 1
    assert commitments[0].target_period == "Q4 2026"
    # Financial trio self-skipped: their owned fields stay at their
    # AgentState defaults (None for financials/comparisons, [] for diffs).
    # LangGraph only surfaces fields a node actually wrote, so checking
    # for absence-or-default covers both the dict and AgentState shapes.
    assert final.get("financials") is None
    assert final.get("comparisons") is None
    assert final.get("language_diffs", []) == []
    assert final["final_note"] == synth_note.rstrip()


async def test_graph_skips_transcript_analyzer_on_non_transcript_filing(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 10-Q filing skips the transcript analyzer and runs the financial track.

    Asserts ``qa_pairs`` / ``commitments`` / ``commitment_updates`` remain
    empty because the transcript analyzer self-skipped on the non-transcript
    form, while ``financials`` and ``comparisons`` carry the extracted XBRL
    data. Re-uses the ``session_factory`` fixture's pre-seeded 10-Q filing
    so the ``StubEdgar`` companyfacts (whose ``accn`` is hard-coded to that
    accession) actually match.
    """
    monkeypatch.setenv("REC", "1")
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
        anthropic_client=_sequenced_anthropic([accepted_note]),
    )
    graph = build_graph(
        edgar=StubEdgar(),
        consensus_fetcher=StubConsensus(),
        embeddings=_StubEmbeddings(),
        llm=llm,
        session_factory=session_factory,
    )
    initial = AgentState(
        trace_id="trace-10q-skip",
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
    # Transcript analyzer self-skipped on a non-TRANSCRIPT form.
    assert final["qa_pairs"] == []
    assert final["commitments"] == []
    assert final["commitment_updates"] == []
    # Financial track ran end-to-end on the 10-Q.
    assert final["financials"]["source"] == "companyfacts"
    assert final["comparisons"]["consensus_source"] == "finnhub"
    assert final["final_note"] == accepted_note.rstrip()
