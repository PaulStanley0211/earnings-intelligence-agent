"""Unit tests for the language differ's pure-function helpers."""

from __future__ import annotations

import pytest

from app.agents.language_differ import (
    _classify_pair,
    _cosine_similarity,
    _word_count,
    align_paragraphs,
)


def test_cosine_similarity_orthogonal_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_parallel_is_one():
    assert _cosine_similarity([1.0, 0.5], [2.0, 1.0]) == pytest.approx(1.0)


def test_classify_pair_unchanged_when_similarity_above_unchanged_threshold():
    assert _classify_pair(similarity=0.99, words=10) == ("unchanged", "minor")


def test_classify_pair_minor_modified_when_similarity_between_0_85_and_unchanged():
    assert _classify_pair(similarity=0.90, words=10) == ("modified", "minor")


def test_classify_pair_major_modified_when_similarity_below_0_85():
    assert _classify_pair(similarity=0.74, words=10) == ("modified", "major")


def test_classify_pair_added_unmatched_major_when_long():
    assert _classify_pair(similarity=None, words=40, is_added=True) == ("added", "major")


def test_classify_pair_added_unmatched_minor_when_short():
    assert _classify_pair(similarity=None, words=10, is_added=True) == ("added", "minor")


def test_classify_pair_removed_unmatched_major_when_long():
    assert _classify_pair(similarity=None, words=40, is_added=False) == ("removed", "major")


def test_word_count_strips_punctuation():
    assert _word_count("Revenue grew, supported by demand.") == 5


def test_align_paragraphs_pairs_highest_similarity_greedy():
    # Current paragraphs: 0 is similar to prior 0; 1 has no match
    current_vecs = [[1.0, 0.0], [0.0, 1.0]]
    prior_vecs = [[0.99, 0.1], [0.5, 0.5]]
    pairs = align_paragraphs(current_vecs, prior_vecs, threshold=0.85)
    # Pair (0, 0) above threshold; (1, ?) below threshold so unmatched.
    assert pairs[0] == (0, 0)
    assert pairs[1] == (1, None)


def test_align_paragraphs_does_not_reuse_prior():
    current_vecs = [[1.0, 0.0], [1.0, 0.0]]
    prior_vecs = [[1.0, 0.0]]
    pairs = align_paragraphs(current_vecs, prior_vecs, threshold=0.5)
    matched = [p for p in pairs if p[1] is not None]
    assert len(matched) == 1
