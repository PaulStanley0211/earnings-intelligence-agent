"""Unit tests for :mod:`app.agents.transcript_analyzer`.

The node is exercised against a stub Anthropic client wrapped in a real
:class:`LLMClient`, mirroring ``tests/unit/test_synthesizer.py``. A stub
repository captures persistence calls so the test stays offline and
makes no network requests.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agents.transcript_analyzer import (
    OWNER,
    _has_keyword_overlap,
    _reconcile_prefilter,
    _upload_id_from_accession,
    transcript_analyzer,
)
from app.llm.client import LLMClient
from app.memory.schemas import (
    CommitmentRecord,
    CommitmentStatus,
    NewCommitment,
    NewQAPair,
    UploadedDocumentRecord,
)
from app.models.state import (
    AgentState,
    AnswerClass,
    FilingEvent,
    FilingEventSource,
    FilingForm,
)

# ---- Stubs -----------------------------------------------------------------


class _StubRepository:
    """Test double covering :class:`_SupportsTranscriptStorage`.

    Tracks every write so assertions can verify the agent persists what
    its returned :class:`StateUpdate` advertises. Implements the
    :class:`app.llm.client._SupportsDailySpend` protocol too so the
    same instance can be handed to :meth:`LLMClient.acomplete`.
    """

    def __init__(
        self,
        *,
        uploaded_text: str | None = "transcript text",
        prior_open: list[CommitmentRecord] | None = None,
    ) -> None:
        self._uploaded_text = uploaded_text
        self._prior_open = prior_open or []
        self.spent: dict[date, Decimal] = {}
        self.qa_inserts: list[tuple[str, list[NewQAPair]]] = []
        self.commitment_inserts: list[tuple[str, str, list[NewCommitment]]] = []
        self.status_updates: list[dict[str, Any]] = []

    async def get_uploaded_document(
        self, upload_id: str
    ) -> UploadedDocumentRecord | None:
        if self._uploaded_text is None:
            return None
        return UploadedDocumentRecord(
            id=1,
            upload_id=upload_id,
            ticker="MSFT",
            filing_type="TRANSCRIPT",
            original_filename="transcript.txt",
            content_sha256="0" * 64,
            parsed_text=self._uploaded_text,
            parsed_char_count=len(self._uploaded_text),
            page_count=None,
            uploaded_at=datetime.now(UTC),
        )

    async def get_open_commitments(self, ticker: str) -> list[CommitmentRecord]:
        return [c for c in self._prior_open if c.ticker == ticker]

    async def add_qa_pairs(
        self, *, filing_accession: str, pairs: list[NewQAPair]
    ) -> list[object]:
        self.qa_inserts.append((filing_accession, list(pairs)))
        return []

    async def add_commitments(
        self,
        *,
        filing_accession: str,
        ticker: str,
        commitments: list[NewCommitment],
    ) -> list[object]:
        self.commitment_inserts.append((filing_accession, ticker, list(commitments)))
        return []

    async def update_commitment_status(
        self,
        *,
        commitment_id: int,
        status: CommitmentStatus,
        resolved_filing_accession: str | None,
        resolved_reason: str | None,
    ) -> None:
        self.status_updates.append(
            {
                "commitment_id": commitment_id,
                "status": status,
                "resolved_filing_accession": resolved_filing_accession,
                "resolved_reason": resolved_reason,
            }
        )

    async def get_daily_spend(self, day: date) -> Decimal:
        return self.spent.get(day, Decimal("0"))

    async def add_daily_spend(self, *, day: date, amount_usd: Decimal) -> Decimal:
        self.spent[day] = self.spent.get(day, Decimal("0")) + amount_usd
        return self.spent[day]


class _SequencedAnthropic:
    """Anthropic stub that returns a queued sequence of text bodies.

    Each ``messages.create`` call dequeues the next text. Tests assert
    against ``self.calls`` which captures every kwargs payload.
    """

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[dict[str, Any]] = []

    @property
    def messages(self) -> Any:
        outer = self

        class _MessagesNamespace:
            def create(self, **kwargs: Any) -> Any:
                if not outer._texts:
                    raise AssertionError("no more canned responses configured")
                text = outer._texts.pop(0)
                outer.calls.append(kwargs)
                return MagicMock(
                    content=[MagicMock(type="text", text=text)],
                    usage=MagicMock(input_tokens=100, output_tokens=50),
                )

        return _MessagesNamespace()


# ---- Helpers ---------------------------------------------------------------


def _build_state(form: FilingForm = FilingForm.TRANSCRIPT) -> AgentState:
    return AgentState(
        trace_id="t-trace",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="upload-abc123",
            cik="0000789019",
            ticker="MSFT",
            form=form,
            filed_at=datetime(2026, 4, 25, 20, 5, tzinfo=UTC),
            source_url="upload://abc123",
            source=FilingEventSource.UPLOAD,
        ),
    )


def _prior_commitment(
    *,
    commitment_id: int,
    text: str,
    target: str | None,
    ticker: str = "MSFT",
) -> CommitmentRecord:
    now = datetime.now(UTC)
    return CommitmentRecord(
        id=commitment_id,
        filing_accession="upload-prior-001",
        ticker=ticker,
        commitment_text=text,
        target_period=target,
        source_quote=text,
        status=CommitmentStatus.OPEN,
        resolved_filing_accession=None,
        resolved_reason=None,
        created_at=now,
        updated_at=now,
    )


def _extract_json(qa: list[dict[str, Any]], commitments: list[dict[str, Any]]) -> str:
    return json.dumps({"qa_pairs": qa, "commitments": commitments})


def _reconcile_json(verdicts: list[dict[str, Any]]) -> str:
    return json.dumps({"verdicts": verdicts})


@pytest.fixture()
def llm_with_texts(
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):  # type: ignore[no-untyped-def]
    """Factory that yields an ``LLMClient`` returning queued canned bodies."""
    monkeypatch.setenv("REC", "1")  # bypass cassette miss, write fresh

    def _build(texts: list[str]) -> tuple[LLMClient, _SequencedAnthropic]:
        stub = _SequencedAnthropic(texts)
        client = LLMClient(
            cassette_dir=cassette_dir,
            anthropic_client=stub,  # type: ignore[arg-type]
        )
        return client, stub

    return _build


# ---- Tests -----------------------------------------------------------------


async def test_transcript_analyzer_skips_when_form_is_not_transcript(
    llm_with_texts: Any,
) -> None:
    """Non-transcript filings yield an empty update without any LLM call."""
    state = _build_state(form=FilingForm.FORM_10Q)
    llm, stub = llm_with_texts([])  # no canned responses -- must not be called
    repo = _StubRepository()
    update = await transcript_analyzer(state, llm=llm, repository=repo)
    assert update.owner == OWNER
    assert update.changes == {}
    assert stub.calls == []
    assert repo.qa_inserts == []
    assert repo.commitment_inserts == []
    assert repo.status_updates == []


async def test_transcript_analyzer_extracts_qa_pairs_and_commitments(
    llm_with_texts: Any,
) -> None:
    """A clean extract response populates StateUpdate.qa_pairs + commitments."""
    extract_body = _extract_json(
        qa=[
            {
                "ordinal": 1,
                "analyst_name": "Brent Thill",
                "question_text": "Azure growth?",
                "answer_text": "Azure grew 31 percent.",
                "answer_class": "direct",
            }
        ],
        commitments=[
            {
                "commitment_text": "Operating margin will expand 100 bps next quarter.",
                "target_period": "Q4 2026",
                "source_quote": "we expect 100 bps of margin expansion",
            }
        ],
    )
    llm, stub = llm_with_texts([extract_body])
    repo = _StubRepository(
        uploaded_text="Azure grew 31 percent. we expect 100 bps of margin expansion."
    )
    update = await transcript_analyzer(_build_state(), llm=llm, repository=repo)
    assert update.owner == OWNER
    assert len(update.changes["qa_pairs"]) == 1
    pair = update.changes["qa_pairs"][0]
    assert pair.ordinal == 1
    assert pair.answer_class is AnswerClass.DIRECT
    assert pair.sha256_text and len(pair.sha256_text) == 64
    assert len(update.changes["commitments"]) == 1
    commitment = update.changes["commitments"][0]
    assert commitment.target_period == "Q4 2026"
    # No prior open commitments -> no reconcile call, empty verdicts.
    assert update.changes["commitment_updates"] == []
    # Only one LLM call (extract); reconcile was skipped because no prior open.
    assert len(stub.calls) == 1
    # Persistence happened.
    assert len(repo.qa_inserts) == 1
    assert repo.qa_inserts[0][0] == "upload-abc123"
    assert len(repo.commitment_inserts) == 1
    assert repo.commitment_inserts[0][1] == "MSFT"


async def test_transcript_analyzer_reconciles_prior_open_commitments(
    llm_with_texts: Any,
) -> None:
    """A prior open commitment with keyword overlap closes via reconcile."""
    prior = _prior_commitment(
        commitment_id=42,
        text="Azure margin expansion of 100 basis points next quarter.",
        target="Q3 2026",
    )
    transcript = (
        "Operator: Azure margin expanded by 110 basis points this quarter, "
        "exceeding our prior guidance."
    )
    extract_body = _extract_json(qa=[], commitments=[])
    reconcile_body = _reconcile_json(
        [
            {
                "commitment_id": 42,
                "new_status": "met",
                "reason": "Azure margin expanded by 110 bps.",
            }
        ]
    )
    llm, stub = llm_with_texts([extract_body, reconcile_body])
    repo = _StubRepository(uploaded_text=transcript, prior_open=[prior])
    update = await transcript_analyzer(_build_state(), llm=llm, repository=repo)
    verdicts = update.changes["commitment_updates"]
    assert len(verdicts) == 1
    assert verdicts[0].commitment_id == 42
    assert verdicts[0].new_status is CommitmentStatus.MET
    # Both extract + reconcile calls fired.
    assert len(stub.calls) == 2
    # Status update persisted with the filing accession as the resolver.
    assert len(repo.status_updates) == 1
    persisted = repo.status_updates[0]
    assert persisted["commitment_id"] == 42
    assert persisted["status"] is CommitmentStatus.MET
    assert persisted["resolved_filing_accession"] == "upload-abc123"
    assert persisted["resolved_reason"].startswith("Azure margin")


async def test_transcript_analyzer_returns_degraded_empty_on_malformed_extract_json(
    llm_with_texts: Any,
) -> None:
    """Two malformed extract responses -> empty update, warning logged."""
    llm, stub = llm_with_texts(["not json", "still not json"])
    repo = _StubRepository(uploaded_text="some transcript")
    update = await transcript_analyzer(_build_state(), llm=llm, repository=repo)
    assert update.owner == OWNER
    assert update.changes == {}
    # Retried once -> two LLM calls.
    assert len(stub.calls) == 2
    # No persistence happened.
    assert repo.qa_inserts == []
    assert repo.commitment_inserts == []
    assert repo.status_updates == []


async def test_transcript_analyzer_skips_reconcile_when_no_prior_open_commitments(
    llm_with_texts: Any,
) -> None:
    """Empty prior-open list -> single extract call, no reconcile call."""
    extract_body = _extract_json(
        qa=[
            {
                "ordinal": 1,
                "analyst_name": None,
                "question_text": "How is FX impact?",
                "answer_text": "FX was a 1 point headwind.",
                "answer_class": "direct",
            }
        ],
        commitments=[],
    )
    llm, stub = llm_with_texts([extract_body])  # only one canned response
    repo = _StubRepository(uploaded_text="transcript body")
    update = await transcript_analyzer(_build_state(), llm=llm, repository=repo)
    assert update.changes["commitment_updates"] == []
    assert len(stub.calls) == 1
    assert repo.status_updates == []


def test_reconcile_prefilter_keyword_overlap() -> None:
    """A 5+ char token from commitment_text appearing in the transcript wins."""
    matched = _prior_commitment(
        commitment_id=1,
        text="Azure margin expansion target",
        target=None,
    )
    unmatched = _prior_commitment(
        commitment_id=2,
        text="Foo bar baz quux",
        target=None,
    )
    transcript = "Operator: We saw Azure growth and margin compression."
    survivors = _reconcile_prefilter([matched, unmatched], transcript)
    assert [c.id for c in survivors] == [1]


def test_reconcile_prefilter_period_overlap() -> None:
    """target_period match survives even with zero keyword overlap."""
    candidate = _prior_commitment(
        commitment_id=7,
        text="Zorblax frobnicate vampyric quantum",
        target="Q3 2026",
    )
    transcript = "Looking ahead to Q3 2026 we expect headwinds to ease."
    survivors = _reconcile_prefilter([candidate], transcript)
    assert [c.id for c in survivors] == [7]


def test_reconcile_prefilter_drops_short_tokens() -> None:
    """Tokens under 5 chars do not trigger a survive verdict."""
    candidate = _prior_commitment(
        commitment_id=9,
        text="we do",
        target=None,
    )
    transcript = "we do believe..."
    assert _reconcile_prefilter([candidate], transcript) == []


def test_has_keyword_overlap_is_case_insensitive() -> None:
    assert _has_keyword_overlap("Azure Margin", "the azure quarter")


def test_upload_id_from_accession_strips_prefix() -> None:
    assert _upload_id_from_accession("upload-abc123") == "abc123"


def test_upload_id_from_accession_returns_none_for_non_upload() -> None:
    assert _upload_id_from_accession("0000950170-26-000050") is None
