"""Multi-quarter synthetic E2E gate for Phase 5a.

Runs the full LangGraph pipeline twice - first for the NIMBUS Q2 2026
transcript, then for Q3 2026 - using the same labelled fixtures that the
cross-quarter reconciliation test uses. The two runs share a single fresh
database so the Q2 commitments are visible to the Q3 reconcile pass.

Phase 5a gate assertions (all three must pass):

1. The ``notes`` table contains exactly two rows for ticker ``NIMBUS``
   after both pipeline runs complete (one per quarter, idempotent on
   re-runs of the same filing).
2. At least one Q2 commitment row has ``status`` other than ``open``
   after the Q3 run (i.e., cross-quarter reconciliation actually closed
   something).
3. No orphan FK rows: every ``qa_pairs.filing_accession`` and
   ``commitments.filing_accession`` in the DB resolves to a real
   ``filings.accession_number``.

LLM strategy
-------------
The transcript_analyzer calls (extract + reconcile) are driven by a
controlled stub :class:`_NimbusStub` that detects each call type from the
message content and returns appropriate JSON. The synthesizer call uses the
same stub and returns a simple note text that the deterministic critic
accepts on the first attempt (no financial figures means no citation check
needed). ``REC=1`` is patched in so the :class:`~app.llm.client.LLMClient`
routes every call through the stub rather than checking cassettes.

Graph wiring
------------
The test builds the real compiled :func:`~app.graph.build_graph` with a
fresh database, stub EDGAR/consensus/embeddings clients (identical to
:mod:`tests.integration.test_notes_persistence`), and the stub LLM. The
Q2 run is invoked first and committed; the Q3 run follows, which causes the
transcript_analyzer's reconcile pass to see and close Q2 commitments.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.graph import build_graph
from app.llm.client import LLMClient
from app.memory.db import build_engine
from app.memory.models import Base, Commitment, Note, QAPair
from app.memory.repository import Repository
from app.memory.schemas import (
    CommitmentStatus,
    NewFiling,
    NewUploadedDocument,
)
from app.models.state import AgentState, FilingEvent, FilingEventSource, FilingForm
from app.tools.edgar import CompanyFactsResponse

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_REAL_DIR: Path = (
    Path(__file__).resolve().parents[1] / "fixtures" / "transcripts" / "real"
)

_Q2_PATH: Path = _REAL_DIR / "transcript_nimbus_q2_2026.txt"
_Q3_PATH: Path = _REAL_DIR / "transcript_nimbus_q3_2026.txt"

# Synthetic identifiers matching the labelled-fixture convention.
_TICKER: str = "NIMBUS"
_CIK: str = "0009999999"
_COMPANY_NAME: str = "Nimbus Observability"

_Q2_UPLOAD_ID: str = "mqnimbusq2"
_Q3_UPLOAD_ID: str = "mqnimbusq3"
_Q2_ACCESSION: str = f"upload-{_Q2_UPLOAD_ID}"
_Q3_ACCESSION: str = f"upload-{_Q3_UPLOAD_ID}"

_FILED_AT_Q2: datetime = datetime(2026, 2, 28, 20, 0, tzinfo=UTC)
_FILED_AT_Q3: datetime = datetime(2026, 5, 16, 20, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Canned LLM responses used by the stub
# ---------------------------------------------------------------------------

# A single Q&A pair + one commitment extracted from the Q2 transcript.
# Kept deliberately small so the reconcile call has exactly one prior
# commitment (id=1) to close, making the DB assertion fully deterministic.
_Q2_EXTRACT_JSON: str = json.dumps(
    {
        "qa_pairs": [
            {
                "ordinal": 1,
                "analyst_name": "Aaron Mitchell",
                "question_text": (
                    "Daniel, can you give us more color on the Cirrus Analytics "
                    "integration plan?"
                ),
                "answer_text": (
                    "we expect native Cirrus query capabilities to be generally "
                    "available inside the Stratus console by the end of fiscal Q4 2026."
                ),
                "answer_class": "direct",
            }
        ],
        "commitments": [
            {
                "commitment_text": (
                    "Close the Cirrus Analytics acquisition by the end of fiscal Q3 2026."
                ),
                "target_period": "Q3 2026",
                "source_quote": (
                    "we expect to close the previously announced Cirrus Analytics "
                    "acquisition by the end of fiscal Q3 2026, which will materially "
                    "expand our footprint in real-time log analytics."
                ),
            }
        ],
    }
)

# A different Q&A pair + new commitment for the Q3 run.
_Q3_EXTRACT_JSON: str = json.dumps(
    {
        "qa_pairs": [
            {
                "ordinal": 1,
                "analyst_name": "Theo Bennett",
                "question_text": (
                    "Daniel, on the Stratus Copilot product, are you still on track?"
                ),
                "answer_text": (
                    "We are not going to hit the end of fiscal Q4 2026 GA target for "
                    "Stratus Copilot that we committed to last quarter."
                ),
                "answer_class": "direct",
            }
        ],
        "commitments": [
            {
                "commitment_text": (
                    "Stratus Copilot will reach general availability in fiscal Q2 2027."
                ),
                "target_period": "Q2 2027",
                "source_quote": (
                    "We now expect Stratus Copilot to reach general availability in "
                    "fiscal Q2 2027 rather than the end of fiscal Q4 2026 timing we "
                    "previously shared."
                ),
            }
        ],
    }
)

# Reconcile verdict: close the single Q2 commitment as 'met'.
# commitment_id=1 is the first row inserted by the Q2 extract pass in a fresh DB.
_Q3_RECONCILE_JSON: str = json.dumps(
    {
        "verdicts": [
            {
                "commitment_id": 1,
                "new_status": "met",
                "reason": (
                    "Cho confirmed closing the Cirrus Analytics acquisition on "
                    "schedule in the final week of fiscal Q3."
                ),
            }
        ]
    }
)

# A minimal note that contains no financial figures and no uncited numbers
# so the deterministic critic accepts it on the first attempt.
_SYNTH_NOTE: str = (
    "## Nimbus Earnings Note\n\n"
    "Management discussed the Cirrus Analytics acquisition and Stratus Copilot "
    "during the earnings call. The company confirmed key milestones and provided "
    "guidance for the coming quarters. No specific financial figures are cited "
    "in this note."
)

# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


_LLM_CRITIC_JSON = '{"findings": []}'
"""Canned LLM critic response: no semantic findings, note accepted."""

_LLM_CRITIC_MARKER = '<source name="draft_note">'
"""Substring present in every rendered llm_v1 prompt body, absent from all others."""


def _make_nimbus_stub() -> MagicMock:
    """Return a MagicMock Anthropic client that routes by call content.

    Four call types are handled (checked in priority order):

    * **LLM critic**: the rendered prompt contains
      ``<source name="draft_note">`` (from ``llm_v1.md``).
      Returns :data:`_LLM_CRITIC_JSON` (clean, no findings).
    * **Reconcile**: the rendered prompt contains
      ``<source type="prior_commitments">`` (from ``reconcile_v1.md``).
      Returns :data:`_Q3_RECONCILE_JSON`.
    * **Extract Q3**: the rendered prompt contains Q3-specific phrasing
      that appears only in the Q3 transcript but not the Q2 one.
      Returns :data:`_Q3_EXTRACT_JSON`.
    * **Extract Q2**: all other Sonnet calls with a long user message
      (transcript) but no reconcile marker.  Returns :data:`_Q2_EXTRACT_JSON`.
    * **Synthesizer (Opus)**: everything else (synthesis prompts use the
      Opus model). Returns :data:`_SYNTH_NOTE`.
    """
    client = MagicMock()

    def _create(**kwargs: Any) -> MagicMock:
        messages: list[dict[str, str]] = kwargs.get("messages", [])
        user_text = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        model: str = str(kwargs.get("model", ""))

        # LLM critic call: detected by the draft_note source block marker.
        if _LLM_CRITIC_MARKER in user_text:
            response_text = _LLM_CRITIC_JSON
        # Reconcile call: the rendered prompt contains the prior_commitments block
        elif "<source type=" in user_text and "prior_commitments" in user_text:
            response_text = _Q3_RECONCILE_JSON
        elif "claude-sonnet" in model or "sonnet" in model.lower():
            # Extract pass: distinguish Q2 from Q3 by presence of Q3-specific phrase
            if "Stratus Copilot to reach general availability in fiscal Q2 2027" in user_text:
                response_text = _Q3_EXTRACT_JSON
            else:
                response_text = _Q2_EXTRACT_JSON
        else:
            # Synthesizer (Opus) or any other caller
            response_text = _SYNTH_NOTE

        return MagicMock(
            content=[MagicMock(type="text", text=response_text)],
            usage=MagicMock(input_tokens=100, output_tokens=80),
        )

    client.messages.create.side_effect = _create
    return client


class _StubEdgar:
    """Minimal EDGAR double - the financial_extractor and language_differ
    self-skip for TRANSCRIPT filings so these methods are never called."""

    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse:
        return CompanyFactsResponse(
            cik=cik.zfill(10),
            entity_name=_COMPANY_NAME,
            raw={
                "cik": int(cik),
                "entityName": _COMPANY_NAME,
                "facts": {"us-gaap": {}},
            },
        )

    async def get_filing_document(
        self,
        *,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> str:
        return "<html><body><p>Item 2. MD&A</p></body></html>"


class _StubConsensus:
    """Minimal consensus double - comparator self-skips for TRANSCRIPT filings."""

    async def fetch(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
        period_end: date,
    ) -> list[Any]:
        return []


class _StubEmbeddings:
    """Minimal embeddings double - language_differ self-skips for TRANSCRIPT."""

    @property
    def model(self) -> str:
        return "openai/text-embedding-3-small"

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.001] + [0.0] * 1535 for _ in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Build a per-test async session factory bound to a clean schema.

    Drops and recreates the full schema so each test starts from an empty
    database with auto-increment IDs resetting to 1.
    """
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


async def _seed_quarter(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    upload_id: str,
    accession: str,
    transcript_path: Path,
    filed_at: datetime,
    content_sha256: str,
) -> None:
    """Insert the uploaded_document + filing rows the transcript_analyzer needs.

    Both rows must be committed before the graph is invoked so the
    transcript_analyzer's separately-opened session can read them.
    """
    transcript_text = transcript_path.read_text(encoding="utf-8")
    async with session_factory() as session:
        repo = Repository(session)
        await repo.upsert_watchlist_entry(
            ticker=_TICKER, cik=_CIK, company_name=_COMPANY_NAME
        )
        await repo.add_uploaded_document(
            NewUploadedDocument(
                upload_id=upload_id,
                ticker=_TICKER,
                filing_type=FilingForm.TRANSCRIPT.value,
                original_filename=transcript_path.name,
                content_sha256=content_sha256,
                parsed_text=transcript_text,
                parsed_char_count=len(transcript_text),
                page_count=None,
            )
        )
        await repo.record_filing(
            filing=NewFiling(
                accession_number=accession,
                cik=_CIK,
                ticker=_TICKER,
                form=FilingForm.TRANSCRIPT,
                filed_at=filed_at,
                source_url=f"upload://{upload_id}",
            )
        )
        await session.commit()


def _make_initial_state(
    *,
    accession: str,
    upload_id: str,
    filed_at: datetime,
    trace_id: str,
) -> AgentState:
    """Build the initial :class:`AgentState` for an upload-driven TRANSCRIPT."""
    return AgentState(
        trace_id=trace_id,
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number=accession,
            cik=_CIK,
            ticker=_TICKER,
            form=FilingForm.TRANSCRIPT,
            filed_at=filed_at,
            source_url=f"upload://{upload_id}",
            source=FilingEventSource.UPLOAD,
        ),
    )


# ---------------------------------------------------------------------------
# Phase 5a gate test
# ---------------------------------------------------------------------------


async def test_multi_quarter_run_persists_notes_and_closes_commitments(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 5a gate: multi-quarter run persists notes and closes commitments.

    Workflow:
    1. Seed Q2: insert uploaded_document + filing rows, then invoke the
       full graph for the Q2 transcript. The transcript_analyzer extract
       pass persists Q2 qa_pairs + 1 commitment (open). The synthesizer
       writes a note; the note_writer persists it to the notes table.
    2. Seed Q3: insert rows for the Q3 transcript, then invoke the full
       graph. The transcript_analyzer reconcile pass closes the Q2
       commitment as 'met' and persists Q3 qa_pairs + new Q3 commitments.
       The synthesizer writes a second note.

    Assertions:
    - notes table has exactly 2 rows for NIMBUS (one per quarter).
    - At least 1 Q2 commitment row has status != 'open' (reconciliation
      actually closed something).
    - No orphan FK rows: every qa_pairs and commitments row has a
      filing_accession that exists in the filings table.
    """
    # Patch REC=1 so the LLM client bypasses cassette lookup and always
    # calls the stub anthropic client.
    monkeypatch.setenv("REC", "1")
    stub_anthropic = _make_nimbus_stub()
    llm = LLMClient(
        cassette_dir=cassette_dir,
        anthropic_client=stub_anthropic,
    )
    graph = build_graph(
        edgar=_StubEdgar(),
        consensus_fetcher=_StubConsensus(),
        embeddings=_StubEmbeddings(),
        llm=llm,
        session_factory=session_factory,
    )

    # ---- Q2 run ----
    await _seed_quarter(
        session_factory=session_factory,
        upload_id=_Q2_UPLOAD_ID,
        accession=_Q2_ACCESSION,
        transcript_path=_Q2_PATH,
        filed_at=_FILED_AT_Q2,
        content_sha256="a" * 64,
    )
    q2_initial = _make_initial_state(
        accession=_Q2_ACCESSION,
        upload_id=_Q2_UPLOAD_ID,
        filed_at=_FILED_AT_Q2,
        trace_id="trace-mq-nimbus-q2",
    )
    q2_final = await graph.ainvoke(q2_initial)
    if not isinstance(q2_final, dict):
        q2_final = q2_final.__dict__

    # Sanity: the Q2 run produced a note (critic accepted or loop_exceeded)
    assert q2_final.get("draft_note") is not None, (
        "Q2 graph run produced no draft note; synthesizer may have failed"
    )

    # ---- Q3 run ----
    await _seed_quarter(
        session_factory=session_factory,
        upload_id=_Q3_UPLOAD_ID,
        accession=_Q3_ACCESSION,
        transcript_path=_Q3_PATH,
        filed_at=_FILED_AT_Q3,
        content_sha256="b" * 64,
    )
    q3_initial = _make_initial_state(
        accession=_Q3_ACCESSION,
        upload_id=_Q3_UPLOAD_ID,
        filed_at=_FILED_AT_Q3,
        trace_id="trace-mq-nimbus-q3",
    )
    q3_final = await graph.ainvoke(q3_initial)
    if not isinstance(q3_final, dict):
        q3_final = q3_final.__dict__

    assert q3_final.get("draft_note") is not None, (
        "Q3 graph run produced no draft note; synthesizer may have failed"
    )

    # ---- Gate assertion 1: notes table has 2 rows for NIMBUS ----
    async with session_factory() as session:
        note_count_result = await session.execute(
            select(func.count()).select_from(Note).where(Note.ticker == _TICKER)
        )
        note_count = note_count_result.scalar_one()

    assert note_count == 2, (
        f"Expected exactly 2 notes for ticker {_TICKER!r}, got {note_count}. "
        "The note_writer must persist one note per accepted critic run."
    )

    # ---- Gate assertion 2: at least 1 Q2 commitment was closed ----
    async with session_factory() as session:
        all_commitments_result = await session.execute(
            select(Commitment).order_by(Commitment.id)
        )
        all_commitments = list(all_commitments_result.scalars().all())

    q2_commitments = [
        c for c in all_commitments if c.filing_accession == _Q2_ACCESSION
    ]
    assert q2_commitments, (
        "No Q2 commitments found in the database after the Q2 run. "
        "The transcript_analyzer extract pass must persist commitments."
    )

    closed_q2 = [
        c
        for c in q2_commitments
        if c.status in {CommitmentStatus.MET.value, CommitmentStatus.MISSED.value}
    ]
    assert len(closed_q2) >= 1, (
        "Spec gate requires >= 1 Q2 commitment to be closed (met/missed) after "
        f"the Q3 run. Found {len(q2_commitments)} Q2 commitment(s) with statuses: "
        f"{[c.status for c in q2_commitments]}. "
        "The Q3 reconcile pass must close at least one prior commitment."
    )

    # ---- Gate assertion 3: no orphan FK rows ----
    async with session_factory() as session:
        # Collect all accession numbers that exist in the filings table.
        filings_result = await session.execute(
            text("SELECT accession_number FROM filings")
        )
        known_accessions: set[str] = {row[0] for row in filings_result.fetchall()}

        # Check qa_pairs
        qa_result = await session.execute(
            select(QAPair.filing_accession).distinct()
        )
        qa_accessions: set[str] = set(qa_result.scalars().all())

        # Check commitments
        commitment_result = await session.execute(
            select(Commitment.filing_accession).distinct()
        )
        commitment_accessions: set[str] = set(commitment_result.scalars().all())

    orphan_qa = qa_accessions - known_accessions
    orphan_commitments = commitment_accessions - known_accessions

    assert not orphan_qa, (
        f"Orphan qa_pairs rows: filing_accession values {orphan_qa!r} do not "
        "exist in the filings table."
    )
    assert not orphan_commitments, (
        f"Orphan commitments rows: filing_accession values {orphan_commitments!r} "
        "do not exist in the filings table."
    )
