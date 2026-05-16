"""Q&A F1 + per-class precision/recall gates for ``transcript_analyzer``.

Implements the §5.2 quality gates from the Phase 4B design spec:

- **Pair-level F1 >= 75 percent** (micro-aggregate across all labelled
  transcripts). A true positive is an extracted Q&A pair whose
  ``question_text`` matches a labelled pair at >= 90 percent
  ``SequenceMatcher`` similarity.
- **Per-class precision and recall >= 80 percent** for each of
  ``direct`` / ``partial`` / ``deflected``. Computed over the matched
  pairs only - a missing extraction does not penalise the class
  metric, the pair-F1 gate already covers misses.

Fixture corpus: 4 synthetic single-quarter transcripts plus the
cross-quarter NIMBUS pair, totalling 46 labelled Q&A pairs across 6
transcripts. See ``tests/fixtures/transcripts/`` for the raw data and
``tests/unit/_transcript_extract_helpers.py`` for the shared driver.

The extract LLM call is replayed from committed cassettes under
``tests/fixtures/cassettes/transcript_analyzer_f1/``. To regenerate the
cassettes after a prompt-body change, run with ``REC=1`` set in the
environment.

Caveat on per-class statistical power: the labelled corpus contains
only four ``deflected`` pairs, so a single misclassification drops
recall to 0.75 and trips the gate. The synthetic fixture set was
authored with that risk in mind - the deflected exemplars are
unambiguous enough that a competent extractor should score 4/4.
"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.unit._transcript_extract_helpers import (
    build_llm_client,
    iter_labelled_transcripts,
    known_classes,
    match_qa_pairs,
    run_extract,
)

_MIN_F1: float = 0.75
"""Minimum micro-aggregate F1 across all labelled transcripts."""

_MIN_PER_CLASS: float = 0.80
"""Minimum precision AND recall for each answer-class."""


async def test_qa_extraction_micro_f1_meets_gate() -> None:
    """Micro-aggregate pair-level F1 over all labelled transcripts >= 0.75.

    A true positive requires a 90 percent character-similarity match on
    ``question_text``; a missed pair is a false negative and a
    hallucinated extraction is a false positive. F1 is computed once
    over the global TP/FP/FN totals so each pair contributes equally
    regardless of transcript length.
    """
    llm = build_llm_client()
    tp = 0
    fp = 0
    fn = 0
    per_transcript: list[tuple[str, int, int, int]] = []
    for transcript in iter_labelled_transcripts():
        result = await run_extract(transcript, llm=llm)
        matched, unmatched_labels, unmatched_extracted = match_qa_pairs(
            extracted=result["qa_pairs"],
            labelled=transcript.qa_pairs,
        )
        local_tp = len(matched)
        local_fn = len(unmatched_labels)
        local_fp = len(unmatched_extracted)
        tp += local_tp
        fn += local_fn
        fp += local_fp
        per_transcript.append((transcript.name, local_tp, local_fp, local_fn))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    breakdown = ", ".join(
        f"{name}: tp={t} fp={fp_} fn={fn_}" for name, t, fp_, fn_ in per_transcript
    )
    assert f1 >= _MIN_F1, (
        f"Q&A pair F1 {f1:.3f} (precision {precision:.3f}, recall {recall:.3f}) "
        f"below gate {_MIN_F1}. Per-transcript: {breakdown}."
    )


async def test_per_class_precision_recall_meets_gate() -> None:
    """Each of direct/partial/deflected scores >= 0.80 precision AND recall.

    Confusion-matrix counts come from the matched-pair subset only -
    pairs the extractor missed entirely are accounted for by
    :func:`test_qa_extraction_micro_f1_meets_gate`. For each labelled
    class we compute:

    - **TP**: labelled class X, extracted class X
    - **FN**: labelled class X, extracted class Y != X (recall miss)
    - **FP**: labelled class Y != X, extracted class X (precision miss)
    """
    llm = build_llm_client()
    valid_classes = set(known_classes())

    tp = Counter[str]()
    fp = Counter[str]()
    fn = Counter[str]()
    class_support = Counter[str]()
    confusion: dict[tuple[str, str], int] = {}

    for transcript in iter_labelled_transcripts():
        result = await run_extract(transcript, llm=llm)
        matched, _, _ = match_qa_pairs(
            extracted=result["qa_pairs"],
            labelled=transcript.qa_pairs,
        )
        for pair in matched:
            labelled_class = str(pair.label.get("answer_class", ""))
            extracted_class = str(pair.extracted.get("answer_class", ""))
            if labelled_class not in valid_classes:
                # Defensive: labels were validated in test_phase4b_fixtures.
                continue
            class_support[labelled_class] += 1
            key = (labelled_class, extracted_class)
            confusion[key] = confusion.get(key, 0) + 1
            if extracted_class == labelled_class:
                tp[labelled_class] += 1
            else:
                fn[labelled_class] += 1
                if extracted_class in valid_classes:
                    fp[extracted_class] += 1

    failures: list[str] = []
    for cls in valid_classes:
        support = class_support[cls]
        if support == 0:
            failures.append(
                f"class {cls!r}: no matched labelled pairs, cannot evaluate"
            )
            continue
        denom_p = tp[cls] + fp[cls]
        denom_r = tp[cls] + fn[cls]
        precision = tp[cls] / denom_p if denom_p > 0 else 0.0
        recall = tp[cls] / denom_r if denom_r > 0 else 0.0
        if precision < _MIN_PER_CLASS or recall < _MIN_PER_CLASS:
            failures.append(
                f"class {cls!r}: precision {precision:.3f}, recall {recall:.3f} "
                f"(support {support}, tp {tp[cls]}, fp {fp[cls]}, fn {fn[cls]})"
            )

    if failures:
        confusion_summary = _format_confusion(confusion, sorted(valid_classes))
        pytest.fail(
            f"per-class gate {_MIN_PER_CLASS:.2f} missed for: "
            + "; ".join(failures)
            + f". Confusion matrix (rows=label, cols=extracted):\n{confusion_summary}"
        )


def _format_confusion(
    confusion: dict[tuple[str, str], int], classes: list[str]
) -> str:
    """Render the labelled-vs-extracted confusion matrix as a small table."""
    header = "label\\extract".ljust(16) + "".join(c.ljust(12) for c in classes)
    lines = [header]
    for row in classes:
        cells = "".join(
            str(confusion.get((row, col), 0)).ljust(12) for col in classes
        )
        lines.append(row.ljust(16) + cells)
    return "\n".join(lines)


__all__: list[str] = [
    "test_per_class_precision_recall_meets_gate",
    "test_qa_extraction_micro_f1_meets_gate",
]
