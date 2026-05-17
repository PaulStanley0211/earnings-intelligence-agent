"""Phase 5b gate: peer_reader e2e emits [P#] citations.

Workflow
--------
1. Seed the database with a MSFT -> GOOGL peer mapping.
2. Seed a processed GOOGL TRANSCRIPT filing with one open commitment whose
   ``commitment_text`` is a known string the stub synthesizer will quote.
3. Invoke the full Phase 5b graph for a MSFT 10-Q (uses the same MSFT
   fixture as ``test_notes_persistence.py``).
4. Assert:
   - The final note contains at least one ``[P\\d+]`` citation.
   - The critic accepted the note (verdict == "accepted").

LLM strategy
------------
A controlled stub Anthropic client handles the single synthesizer call and
returns a draft note that quotes the GOOGL peer commitment text verbatim so
the deterministic critic can resolve the ``[P0]`` reference within the
standard 90% character-similarity tolerance. No real API calls are made.

The test patches ``REC=1`` to bypass cassette lookup in
:class:`~app.llm.client.LLMClient`.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import select as sa_select
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.graph import build_graph
from app.llm.client import LLMClient
from app.memory.db import build_engine
from app.memory.models import Base, FilingSection
from app.memory.repository import Repository
from app.memory.schemas import (
    ChangeType,
    NewCommitment,
    NewConsensusEstimate,
    NewFiling,
    NewFilingSection,
    NewLanguageDiff,
    PeerCreate,
    SectionKind,
    Severity,
)
from app.models.state import AgentState, FilingEvent, FilingForm
from app.tools.edgar import CompanyFactsResponse

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MSFT filing identifiers (primary ticker under test)
_MSFT_ACCESSION = "0000950170-26-000050"
_MSFT_CIK = "0000789019"
_MSFT_TICKER = "MSFT"
_MSFT_FILED_AT = datetime(2026, 4, 25, 20, 5, tzinfo=UTC)

# GOOGL peer identifiers
_GOOGL_ACCESSION = "0001652044-26-000100"
_GOOGL_TRANSCRIPT_ACCESSION = "0001652044-26-000200"
_GOOGL_CIK = "0001652044"
_GOOGL_TICKER = "GOOGL"
_GOOGL_FILED_AT = datetime(2026, 4, 1, 18, 0, tzinfo=UTC)
_GOOGL_TRANSCRIPT_FILED_AT = datetime(2026, 4, 2, 18, 0, tzinfo=UTC)

# Verbatim peer commitment text the stub synthesizer will quote
_PEER_COMMITMENT_TEXT = "Cloud pricing pressure intensified across all regions"

# The draft note the stub returns; [P0] quotes the verbatim peer commitment text
# so the critic's substring match succeeds on the first attempt.
_DRAFT_NOTE = (
    "## Headline\n"
    "MSFT reported solid results for Q3 2026.\n\n"
    "## Peer context\n"
    f"- {_PEER_COMMITMENT_TEXT} [P0]\n"
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEdgar:
    """Minimal EDGAR double returning one revenue fact for MSFT."""

    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse:
        """Return a minimal CompanyFactsResponse with one revenue figure."""
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
                                        "accn": _MSFT_ACCESSION,
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
        """Return a trivial HTML document; language_differ runs but finds nothing."""
        return (
            "<html><body>"
            "<p>Item 2. Management's Discussion and Analysis</p>"
            "<p>Revenue grew driven by cloud demand.</p>"
            "</body></html>"
        )


class _StubConsensus:
    """Returns one revenue consensus row for MSFT; skips for other tickers."""

    async def fetch(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
        period_end: date,
    ) -> list[NewConsensusEstimate]:
        """Return a single revenue consensus estimate for MSFT."""
        if ticker != _MSFT_TICKER:
            return []
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
        """Return the stub embeddings model identifier."""
        return "openai/text-embedding-3-small"

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return deterministic unit vectors for each input text."""
        return [[0.001] + [0.0] * 1535 for _ in texts]


_LLM_CRITIC_JSON = '{"findings": []}'
"""Canned LLM critic response: no semantic findings, note accepted."""

_LLM_CRITIC_MARKER = '<source name="draft_note">'
"""Substring present in every rendered llm_v1 prompt body, absent from all others."""


def _make_stub_anthropic() -> MagicMock:
    """Return a MagicMock Anthropic client that routes by call content.

    The draft note quotes the GOOGL peer commitment text verbatim so the
    deterministic critic resolves ``[P0]`` via a substring match.
    LLM critic calls are detected via the ``<source name="draft_note">`` marker
    and receive a clean ``{"findings": []}`` response.
    """
    client = MagicMock()

    def _create(**kwargs: Any) -> MagicMock:
        messages: list[dict[str, str]] = kwargs.get("messages", [])
        user_text = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        response_text = _LLM_CRITIC_JSON if _LLM_CRITIC_MARKER in user_text else _DRAFT_NOTE
        return MagicMock(
            content=[MagicMock(type="text", text=response_text)],
            usage=MagicMock(input_tokens=120, output_tokens=60),
        )

    client.messages.create.side_effect = _create
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Fresh schema seeded with MSFT 10-Q, GOOGL peer rows, and GOOGL signals.

    Schema lifecycle: drop-all -> create-all on entry; engine disposed on exit.
    """
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with factory() as session:
        repo = Repository(session)

        # MSFT 10-Q filing (the primary filing under analysis)
        await repo.record_filing(
            filing=NewFiling(
                accession_number=_MSFT_ACCESSION,
                cik=_MSFT_CIK,
                ticker=_MSFT_TICKER,
                form=FilingForm.FORM_10Q,
                filed_at=_MSFT_FILED_AT,
                source_url="https://www.sec.gov/msft-10q",
            )
        )

        # GOOGL processed 10-Q filing (peer language-diff source)
        await repo.record_filing(
            filing=NewFiling(
                accession_number=_GOOGL_ACCESSION,
                cik=_GOOGL_CIK,
                ticker=_GOOGL_TICKER,
                form=FilingForm.FORM_10Q,
                filed_at=_GOOGL_FILED_AT,
                source_url="https://www.sec.gov/googl-10q",
            )
        )
        # Mark the GOOGL 10-Q as processed so peer_reader finds it
        await repo.mark_filing_processed(_GOOGL_ACCESSION)

        # GOOGL processed TRANSCRIPT filing (peer commitment source)
        await repo.record_filing(
            filing=NewFiling(
                accession_number=_GOOGL_TRANSCRIPT_ACCESSION,
                cik=_GOOGL_CIK,
                ticker=_GOOGL_TICKER,
                form=FilingForm.TRANSCRIPT,
                filed_at=_GOOGL_TRANSCRIPT_FILED_AT,
                source_url="https://www.sec.gov/googl-transcript",
            )
        )
        # Mark the GOOGL TRANSCRIPT as processed so peer_reader finds it
        await repo.mark_filing_processed(_GOOGL_TRANSCRIPT_ACCESSION)

        # Seed a FilingSection for the GOOGL 10-Q so the language_diff
        # current_section_id FK resolves to a real row and the diff text
        # is retrievable by _language_diff_text.
        section_sha = "a" * 64
        await repo.insert_filing_sections(
            [
                NewFilingSection(
                    filing_accession=_GOOGL_ACCESSION,
                    cik=_GOOGL_CIK,
                    ticker=_GOOGL_TICKER,
                    section_kind=SectionKind.MDA,
                    paragraph_index=0,
                    text=_PEER_COMMITMENT_TEXT,
                    text_sha=section_sha,
                )
            ]
        )
        # Retrieve the inserted section id so we can reference it in the diff
        section_result = await session.execute(
            sa_select(FilingSection.id)
            .where(FilingSection.filing_accession == _GOOGL_ACCESSION)
        )
        section_id = section_result.scalar_one()

        # Seed a major language diff for the GOOGL 10-Q pointing at the section
        await repo.insert_language_diffs(
            [
                NewLanguageDiff(
                    filing_accession=_GOOGL_ACCESSION,
                    prior_filing_accession=None,
                    section_kind=SectionKind.MDA,
                    change_type=ChangeType.ADDED,
                    current_section_id=section_id,
                    prior_section_id=None,
                    similarity=None,
                    severity=Severity.MAJOR,
                )
            ]
        )

        # Seed an open GOOGL commitment in the TRANSCRIPT filing
        await repo.add_commitments(
            filing_accession=_GOOGL_TRANSCRIPT_ACCESSION,
            ticker=_GOOGL_TICKER,
            commitments=[
                NewCommitment(
                    commitment_text=_PEER_COMMITMENT_TEXT,
                    target_period="Q4 2026",
                    source_quote=_PEER_COMMITMENT_TEXT,
                )
            ],
        )

        # Seed the MSFT -> GOOGL peer mapping
        await repo.upsert_peer(
            PeerCreate(ticker=_MSFT_TICKER, peer_ticker=_GOOGL_TICKER)
        )

        await session.commit()

    yield factory
    await engine.dispose()


def _make_initial_state() -> AgentState:
    """Build the initial AgentState for the pre-seeded MSFT 10-Q filing."""
    return AgentState(
        trace_id="trace-peer-reader-e2e",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number=_MSFT_ACCESSION,
            cik=_MSFT_CIK,
            ticker=_MSFT_TICKER,
            form=FilingForm.FORM_10Q,
            filed_at=_MSFT_FILED_AT,
            source_url="https://www.sec.gov/msft-10q",
        ),
    )


# ---------------------------------------------------------------------------
# Gate test
# ---------------------------------------------------------------------------

_PEER_CITATION_RE = re.compile(r"\[P\d+\]")


async def test_peer_reader_e2e_emits_p_citation(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 5b gate: the synthesized note must contain >= 1 [P#] citation.

    Assertions:
    - ``critic_verdict == "accepted"`` - the critic resolved [P0] without error.
    - The ``draft_note`` contains at least one ``[P\\d+]`` token.
    - ``peer_context`` on the final state is non-empty (peer_reader ran).
    """
    monkeypatch.setenv("REC", "1")
    stub_anthropic = _make_stub_anthropic()
    llm = LLMClient(cassette_dir=cassette_dir, anthropic_client=stub_anthropic)
    graph = build_graph(
        edgar=_StubEdgar(),
        consensus_fetcher=_StubConsensus(),
        embeddings=_StubEmbeddings(),
        llm=llm,
        session_factory=session_factory,
    )

    final = await graph.ainvoke(_make_initial_state())
    if not isinstance(final, dict):
        final = final.__dict__

    # --- Gate: peer_reader populated peer_context ---------------------------
    peer_context = final.get("peer_context") or []
    assert len(peer_context) >= 1, (
        "peer_reader must populate peer_context with >= 1 entry when a MSFT->GOOGL "
        "peer mapping and GOOGL signals exist in the database."
    )

    # --- Gate: draft note contains >= 1 [P#] citation ----------------------
    draft_note = final.get("draft_note") or ""
    assert _PEER_CITATION_RE.search(draft_note), (
        f"The synthesized note must contain at least one [P#] citation. "
        f"Got draft_note:\n{draft_note!r}"
    )

    # --- Gate: critic accepted the note ------------------------------------
    verdict = final.get("critic_verdict")
    assert str(verdict) == "accepted", (
        f"Critic must accept the note when [P0] resolves correctly. "
        f"Got verdict={verdict!r}. "
        f"Critic findings: {final.get('critic_findings')}"
    )
