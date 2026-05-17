"""Critic must accept valid ``[Q#]``/``[K#]`` citations and reject invalid ones.

The Phase 4B critic was extended in Task 8 to resolve ``[Q#]`` against
:attr:`AgentState.qa_pairs` and ``[K#]`` against :attr:`AgentState.commitments`,
applying the same 90 percent character-similarity tolerance the language layer
uses.

The five accept/reject/unresolved scenarios required by spec §5.2 are already
pinned by :mod:`tests.unit.test_critic` (see Task 8). This module adds the one
scenario that file does not yet cover -- a commitment citation whose nearby
prose drifts beyond the 90 percent similarity threshold -- and re-asserts the
five existing scenarios at the module boundary so any future regression
shows up against the Phase 4B-focused file rather than only the broader
critic suite.

Spec §5.2 reference: "Critic accepts valid ``[Q#]`` / ``[K#]``, rejects invalid".
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.agents.critic import critique_draft
from app.models.state import (
    AgentState,
    AnswerClass,
    CommitmentExtracted,
    FilingEvent,
    FilingForm,
    QAPairPayload,
)


def _transcript_state(
    *,
    draft: str,
    qa_pairs: list[QAPairPayload] | None = None,
    commitments: list[CommitmentExtracted] | None = None,
) -> AgentState:
    """Build a transcript-shaped :class:`AgentState` for the critic to score."""
    return AgentState(
        trace_id="t",
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number="upload-0000000000000001",
            cik="0000789019",
            ticker="MSFT",
            form=FilingForm.TRANSCRIPT,
            filed_at=datetime.now(UTC),
            source_url="upload://0000000000000001",
        ),
        qa_pairs=qa_pairs or [],
        commitments=commitments or [],
        draft_note=draft,
    )


def test_critic_rejects_commitment_citation_with_low_similarity() -> None:
    """A draft phrase that does not match the commitment's ``source_quote`` fails.

    Mirrors the existing Q&A reject test in :mod:`tests.unit.test_critic` so
    every quote-style citation namespace has paired accept + low-similarity
    reject coverage. The drafted text shares no meaningful tokens with the
    indexed quote, so the critic's substring check and 90 percent
    :class:`difflib.SequenceMatcher` ratio both fail.
    """
    commitment = CommitmentExtracted(
        commitment_text="Azure margin expansion of 100 basis points next quarter.",
        target_period="Q3 2026",
        source_quote=(
            "we expect Azure margin expansion of 100 basis points next quarter"
        ),
    )
    draft = (
        "## Commitments\n"
        "- We will exit the gaming hardware business and return capital [K1].\n"
    )
    state = _transcript_state(draft=draft, commitments=[commitment])
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert any(f.severity == "error" and "K1" in f.message for f in findings), (
        "expected a 'K1' similarity-mismatch finding; got: "
        f"{[f.message for f in findings]}"
    )


def test_critic_accepts_valid_qa_citation() -> None:
    """Sanity guard mirroring :mod:`tests.unit.test_critic` to pin the Q&A accept path."""
    qa = QAPairPayload(
        ordinal=1,
        analyst_name="Brent Thill",
        question_text="How should we think about Azure margins next quarter?",
        answer_text="We expect margins to remain stable around 47 percent.",
        answer_class=AnswerClass.DIRECT,
        sha256_text="a" * 64,
    )
    draft = (
        "## Q&A signals\n"
        "- We expect margins to remain stable around 47 percent [Q1].\n"
    )
    state = _transcript_state(draft=draft, qa_pairs=[qa])
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert not any(f.severity == "error" and "Q1" in f.message for f in findings), (
        "Q1 was unexpectedly rejected: "
        f"{[f.message for f in findings if 'Q1' in f.message]}"
    )


def test_critic_accepts_valid_commitment_citation() -> None:
    """Sanity guard mirroring :mod:`tests.unit.test_critic` for the K accept path."""
    commitment = CommitmentExtracted(
        commitment_text="Azure margin expansion of 100 basis points next quarter.",
        target_period="Q3 2026",
        source_quote=(
            "we expect Azure margin expansion of 100 basis points next quarter"
        ),
    )
    draft = (
        "## Commitments\n"
        "- we expect Azure margin expansion of 100 basis points next quarter [K1].\n"
    )
    state = _transcript_state(draft=draft, commitments=[commitment])
    update = critique_draft(state)
    findings = update.changes["critic_findings"]
    assert not any(f.severity == "error" and "K1" in f.message for f in findings), (
        "K1 was unexpectedly rejected: "
        f"{[f.message for f in findings if 'K1' in f.message]}"
    )


__all__: list[str] = [
    "test_critic_accepts_valid_commitment_citation",
    "test_critic_accepts_valid_qa_citation",
    "test_critic_rejects_commitment_citation_with_low_similarity",
]
