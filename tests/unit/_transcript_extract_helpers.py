"""Shared helpers for the transcript_analyzer F1 + recall gates.

The two gate test modules (``test_transcript_analyzer_f1.py`` and
``test_commitment_extraction.py``) both need to drive the extract-only
pass of :mod:`app.agents.transcript_analyzer` against the labelled
transcript fixtures and parse the raw JSON output. This module owns that
shared driver plus the small bipartite-matching helpers so each gate
module stays focused on its own metric.

The driver routes every call through :class:`app.llm.client.LLMClient`
with cassette-based replay, matching the existing pattern used by the
synthesizer/critic/language-differ gates. Cassettes live under
``tests/fixtures/cassettes/transcript_analyzer_f1/`` and are committed
to the repo so the gates are fully deterministic at PR time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Final

from app.llm.client import LLMClient
from app.llm.prompts import load_prompt
from app.models.state import AnswerClass

_FIXTURES_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[1] / "fixtures" / "transcripts"
)
"""Root of the labelled transcript corpus (``synthetic/`` + ``real/``)."""

CASSETTE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cassettes" / "transcript_analyzer_f1"
)
"""Cassette directory dedicated to the F1 + recall gates."""

RECONCILIATION_CASSETTE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cassettes"
    / "transcript_reconciliation"
)
"""Cassette directory dedicated to the cross-quarter reconciliation gate.

The Task 11 integration test (``tests/integration/test_commitment_reconciliation.py``)
records both the extract and reconcile Sonnet calls for each NIMBUS transcript
under this directory. Keeping these cassettes separate from the F1 gate's
extract-only cassettes avoids accidental cross-contamination if either prompt
template's body SHA ever drifts."""

EXTRACT_PROMPT_NAME: Final[str] = "transcript_analyzer/extract_v1"
"""Mirrors :data:`app.agents.transcript_analyzer.EXTRACT_PROMPT_NAME`."""

_EXTRACT_MAX_TOKENS: Final[int] = 4096
"""Mirrors :data:`app.agents.transcript_analyzer._EXTRACT_MAX_TOKENS`."""

QUESTION_SIMILARITY_THRESHOLD: Final[float] = 0.90
"""Fraction-of-chars overlap required to call two questions the same pair."""

QUOTE_SIMILARITY_THRESHOLD: Final[float] = 0.90
"""Fraction-of-chars overlap required to match a commitment source quote."""


@dataclass(frozen=True)
class LabelledTranscript:
    """A single labelled transcript fixture loaded into memory."""

    name: str
    transcript_path: Path
    labels_path: Path
    transcript_text: str
    qa_pairs: list[dict[str, Any]]
    commitments: list[dict[str, Any]]


class _ZeroSpendRepository:
    """Minimal stub of :class:`app.llm.client._SupportsDailySpend`.

    The F1 gate runs offline against cassette replay so daily-spend
    accounting is irrelevant; this stub always reports zero spend and
    discards new entries. Production code never sees this class - it
    only satisfies the LLM client's protocol so cassette replay works
    without a real Postgres connection.
    """

    async def get_daily_spend(self, day: date) -> Decimal:
        """Return zero so the cost-cap pre-flight never trips during replay."""
        return Decimal("0")

    async def add_daily_spend(self, *, day: date, amount_usd: Decimal) -> Decimal:
        """Pretend to record spend; never consulted during replay."""
        return amount_usd


def iter_labelled_transcripts() -> Iterator[LabelledTranscript]:
    """Yield every (transcript, labels) pair across ``synthetic/`` and ``real/``.

    Sorted by file name so the iteration order is deterministic across
    runs and platforms. The Q3 NIMBUS labels file's optional
    ``reconciliation_targets`` block is intentionally ignored here - it
    belongs to a future reconciliation gate.
    """
    for labels_path in sorted(_FIXTURES_ROOT.rglob("*.labels.json")):
        transcript_path = labels_path.with_suffix("").with_suffix(".txt")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        yield LabelledTranscript(
            name=transcript_path.stem,
            transcript_path=transcript_path,
            labels_path=labels_path,
            transcript_text=transcript_path.read_text(encoding="utf-8"),
            qa_pairs=list(labels.get("qa_pairs", [])),
            commitments=list(labels.get("commitments", [])),
        )


def build_llm_client() -> LLMClient:
    """Construct an :class:`LLMClient` pointed at the gate's cassette dir.

    Test mode (``ENVIRONMENT=test``) makes the client raise
    :class:`app.llm.client.CassetteMiss` on any unrecorded call - the
    spec requires that, so missing cassettes surface as a hard failure
    rather than silently triggering a network call.
    """
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    return LLMClient(cassette_dir=CASSETTE_DIR)


def build_reconciliation_llm_client() -> LLMClient:
    """Construct an :class:`LLMClient` for the reconciliation integration test.

    Points at :data:`RECONCILIATION_CASSETTE_DIR` so the extract + reconcile
    cassettes recorded for the cross-quarter test stay isolated from the F1
    gate's extract-only cassettes. Test mode (``ENVIRONMENT=test``) makes
    the client raise :class:`app.llm.client.CassetteMiss` on any unrecorded
    call so missing cassettes surface as a hard failure rather than
    silently triggering a network call.
    """
    RECONCILIATION_CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    return LLMClient(cassette_dir=RECONCILIATION_CASSETTE_DIR)


async def run_extract(
    transcript: LabelledTranscript,
    *,
    llm: LLMClient,
) -> dict[str, list[dict[str, Any]]]:
    """Invoke the extract prompt on ``transcript`` and return parsed JSON.

    Mirrors :func:`app.agents.transcript_analyzer._call_extract` but
    without the retry loop or pydantic validation - we want to measure
    the raw model output against the labels, not the agent's degraded
    fallbacks. Returns a dict with ``qa_pairs`` and ``commitments`` keys.
    """
    template = load_prompt(EXTRACT_PROMPT_NAME)
    user_content = template.render(transcript_text=transcript.transcript_text)
    response = await llm.acomplete(
        prompt_version=f"{template.prompt_version}#{template.body_sha[:8]}",
        messages=[{"role": "user", "content": user_content}],
        repository=_ZeroSpendRepository(),
        model=template.model,
        temperature=template.temperature,
        max_tokens=_EXTRACT_MAX_TOKENS,
    )
    payload = json.loads(response.text)
    if not isinstance(payload, dict):
        raise AssertionError(
            f"extract response for {transcript.name} was not a JSON object"
        )
    qa = payload.get("qa_pairs", [])
    commitments = payload.get("commitments", [])
    if not isinstance(qa, list) or not isinstance(commitments, list):
        raise AssertionError(
            f"extract response for {transcript.name} has wrong shape: {payload!r}"
        )
    return {"qa_pairs": qa, "commitments": commitments}


def char_similarity(a: str, b: str) -> float:
    """Return ``SequenceMatcher`` ratio after lowercase + whitespace trim.

    Strips leading/trailing whitespace and lowercases both sides so
    cosmetic differences do not artificially depress the score.
    """
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


@dataclass(frozen=True)
class MatchedPair:
    """A labelled Q&A pair paired with its best-matching extracted pair."""

    label: dict[str, Any]
    extracted: dict[str, Any]
    similarity: float


def match_qa_pairs(
    extracted: Iterable[dict[str, Any]],
    labelled: Iterable[dict[str, Any]],
    threshold: float = QUESTION_SIMILARITY_THRESHOLD,
) -> tuple[list[MatchedPair], list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy bipartite match by question_text similarity.

    Returns ``(matched, unmatched_labels, unmatched_extracted)``. Each
    extracted pair can match at most one label, and matches are made in
    order of descending similarity so the strongest candidates lock in
    first. Threshold defaults to 90 percent char similarity per spec.
    """
    extracted_list = list(extracted)
    label_list = list(labelled)
    candidates: list[tuple[float, int, int]] = []
    for li, label in enumerate(label_list):
        label_q = str(label.get("question_text", ""))
        for ei, ext in enumerate(extracted_list):
            ext_q = str(ext.get("question_text", ""))
            sim = char_similarity(label_q, ext_q)
            if sim >= threshold:
                candidates.append((sim, li, ei))
    candidates.sort(reverse=True)

    used_labels: set[int] = set()
    used_extracted: set[int] = set()
    matched: list[MatchedPair] = []
    for sim, li, ei in candidates:
        if li in used_labels or ei in used_extracted:
            continue
        used_labels.add(li)
        used_extracted.add(ei)
        matched.append(
            MatchedPair(
                label=label_list[li],
                extracted=extracted_list[ei],
                similarity=sim,
            )
        )
    unmatched_labels = [
        label for li, label in enumerate(label_list) if li not in used_labels
    ]
    unmatched_extracted = [
        ext for ei, ext in enumerate(extracted_list) if ei not in used_extracted
    ]
    return matched, unmatched_labels, unmatched_extracted


def known_classes() -> tuple[str, ...]:
    """Return the answer-class label set the gate enforces."""
    return tuple(c.value for c in AnswerClass)
