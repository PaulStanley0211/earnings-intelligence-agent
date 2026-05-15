"""The language-differ agent node.

This module is built incrementally. Task 12 lays down the deterministic
helpers (cosine similarity, greedy alignment, change classification). Task 13
wires the helpers to EDGAR fetching, the embeddings client, the repository,
and the LangGraph orchestrator.

The classifier thresholds are constants here so they can be tuned against
the recall-gate fixture before merge.
"""

from __future__ import annotations

import math
import re
from typing import Final

OWNER = "language_differ"

# Cosine similarity thresholds. Tuned against the 15 hand-labelled
# quarter-pairs in ``tests/fixtures/language_recall/``; do not adjust
# without re-running ``tests/unit/test_recall_gate.py``.
_SIMILARITY_MATCH_THRESHOLD: Final[float] = 0.65
_SIMILARITY_UNCHANGED_THRESHOLD: Final[float] = 0.97
_MAJOR_SIMILARITY_THRESHOLD: Final[float] = 0.85

# Length-based heuristic: a long unmatched paragraph is a major change,
# a short one is a minor one. Word count uses the simple whitespace
# tokeniser in :func:`_word_count`.
_MAJOR_WORD_COUNT_THRESHOLD: Final[int] = 30


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two equal-length vectors.

    Returns ``0.0`` when either vector is zero-length to avoid a divide by
    zero; callers treat that as "no match".
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


_WORD = re.compile(r"\b[\w'-]+\b")


def _word_count(text: str) -> int:
    """Count word-like tokens in ``text``."""
    return len(_WORD.findall(text))


def align_paragraphs(
    current_vectors: list[list[float]],
    prior_vectors: list[list[float]],
    *,
    threshold: float = _SIMILARITY_MATCH_THRESHOLD,
) -> list[tuple[int, int | None]]:
    """Greedy nearest-neighbour alignment of current paragraphs to prior.

    For each current index ``i`` returns ``(i, prior_index)`` where
    ``prior_index`` is the matched prior paragraph index, or ``None`` if no
    prior paragraph above ``threshold`` was available. Each prior paragraph
    is consumed by at most one current paragraph, picked greedily by
    similarity order.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, current in enumerate(current_vectors):
        for j, prior in enumerate(prior_vectors):
            sim = _cosine_similarity(current, prior)
            if sim >= threshold:
                candidates.append((sim, i, j))
    candidates.sort(reverse=True)

    paired_current: dict[int, int] = {}
    consumed_prior: set[int] = set()
    for _sim, i, j in candidates:
        if i in paired_current or j in consumed_prior:
            continue
        paired_current[i] = j
        consumed_prior.add(j)

    return [(i, paired_current.get(i)) for i in range(len(current_vectors))]


def _classify_pair(
    *,
    similarity: float | None,
    words: int,
    is_added: bool = True,
) -> tuple[str, str]:
    """Return ``(change_type, severity)`` for an aligned (or unmatched) pair.

    ``similarity is None`` means the paragraph was unmatched. ``is_added``
    distinguishes a current-side unmatched (``added``) from a prior-side
    unmatched (``removed``); ignored when ``similarity is not None``.
    """
    if similarity is not None:
        if similarity >= _SIMILARITY_UNCHANGED_THRESHOLD:
            return ("unchanged", "minor")
        severity = (
            "major" if similarity < _MAJOR_SIMILARITY_THRESHOLD else "minor"
        )
        return ("modified", severity)
    change_type = "added" if is_added else "removed"
    severity = "major" if words > _MAJOR_WORD_COUNT_THRESHOLD else "minor"
    return (change_type, severity)
