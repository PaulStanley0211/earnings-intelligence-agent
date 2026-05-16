"""Commitment-extraction recall gate for ``transcript_analyzer``.

Implements the §5.2 commitment-recall gate from the Phase 4B design
spec: the extract pass must surface >= 80 percent of the labelled
forward-looking commitments across the labelled transcript corpus.

Matching rule: a labelled commitment counts as found when any
extracted commitment's ``source_quote`` matches the labelled
``source_quote`` at >= 90 percent ``SequenceMatcher`` character
similarity. The source quote is the right field to match on because it
is the verbatim transcript anchor the spec promises - ``commitment_text``
is an LLM paraphrase and would let cosmetic rewording silently degrade
the gate.

The spec only mandates recall. Precision is harder to score (the LLM
may legitimately surface additional commitments the labellers did not
mark) so this module emits a precision figure as an informational
metric without gating on it.

LLM calls are replayed from committed cassettes under
``tests/fixtures/cassettes/transcript_analyzer_f1/``; the recall gate
shares the same cassette pool as the F1 gate because both consume the
exact same extract response per transcript.
"""

from __future__ import annotations

from tests.unit._transcript_extract_helpers import (
    QUOTE_SIMILARITY_THRESHOLD,
    build_llm_client,
    char_similarity,
    iter_labelled_transcripts,
    run_extract,
)

_MIN_RECALL: float = 0.80
"""Minimum micro-aggregate commitment recall across all transcripts."""


async def test_commitment_recall_meets_gate() -> None:
    """Aggregate commitment recall across all labelled transcripts >= 0.80.

    For each labelled commitment we check whether at least one extracted
    commitment carries a 90 percent character-similar ``source_quote``.
    Recall is computed once over the global match counts so longer
    transcripts do not dominate the metric. Precision is logged
    informationally - the spec does not gate on it.
    """
    llm = build_llm_client()

    matched_labels = 0
    total_labels = 0
    matched_extracted_ids: set[tuple[str, int]] = set()
    total_extracted = 0
    misses: list[str] = []

    for transcript in iter_labelled_transcripts():
        result = await run_extract(transcript, llm=llm)
        extracted_commitments = result["commitments"]
        total_extracted += len(extracted_commitments)
        for label in transcript.commitments:
            total_labels += 1
            label_quote = str(label.get("source_quote", ""))
            best_index: int | None = None
            best_score = 0.0
            for index, ext in enumerate(extracted_commitments):
                ext_quote = str(ext.get("source_quote", ""))
                score = char_similarity(label_quote, ext_quote)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_score >= QUOTE_SIMILARITY_THRESHOLD and best_index is not None:
                matched_labels += 1
                matched_extracted_ids.add((transcript.name, best_index))
            else:
                misses.append(
                    f"{transcript.name}: best similarity {best_score:.2f} for "
                    f"{label_quote[:60]!r}"
                )

    assert total_labels > 0, "no labelled commitments found"
    recall = matched_labels / total_labels
    informational_precision = (
        len(matched_extracted_ids) / total_extracted if total_extracted > 0 else 0.0
    )
    assert recall >= _MIN_RECALL, (
        f"commitment recall {recall:.3f} ({matched_labels}/{total_labels}) "
        f"below gate {_MIN_RECALL}. Misses: {misses}. "
        f"Informational precision {informational_precision:.3f} "
        f"({len(matched_extracted_ids)}/{total_extracted})."
    )


__all__: list[str] = ["test_commitment_recall_meets_gate"]
