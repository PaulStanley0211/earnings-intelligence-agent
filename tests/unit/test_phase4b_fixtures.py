"""Sanity checks on the synthetic transcript fixture set."""

from __future__ import annotations

import json
from pathlib import Path

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts" / "synthetic"


def test_all_four_transcripts_have_label_files() -> None:
    """Every synthetic .txt has a sibling .labels.json."""
    transcripts = sorted(_SYNTHETIC.glob("*.txt"))
    assert len(transcripts) >= 3
    for tx in transcripts:
        label = tx.with_suffix(".labels.json")
        assert label.is_file(), f"missing labels for {tx.name}"


def test_label_qa_text_is_verbatim_substring_of_transcript() -> None:
    """Every labelled question_text/answer_text/source_quote appears verbatim in its transcript."""
    for label_path in sorted(_SYNTHETIC.glob("*.labels.json")):
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        transcript_path = label_path.with_suffix("").with_suffix(".txt")
        transcript = transcript_path.read_text(encoding="utf-8")
        for pair in labels["qa_pairs"]:
            assert pair["question_text"] in transcript, (
                f"{label_path.name} qa_pair {pair['ordinal']}: question_text not in transcript"
            )
            assert pair["answer_text"] in transcript, (
                f"{label_path.name} qa_pair {pair['ordinal']}: answer_text not in transcript"
            )
        for commitment in labels["commitments"]:
            assert commitment["source_quote"] in transcript, (
                f"{label_path.name} commitment: source_quote not in transcript"
            )


def test_qa_pair_totals_meet_spec_band() -> None:
    """Total Q&A pairs across synthetic fixtures land in the 28-40 band (spec target ~30-35)."""
    total = 0
    for label_path in sorted(_SYNTHETIC.glob("*.labels.json")):
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        total += len(labels["qa_pairs"])
    assert 28 <= total <= 40, f"Q&A total {total} outside the 28-40 band"


def test_answer_class_distribution_is_balanced() -> None:
    """Each of direct/partial/deflected appears at least 4 times across the fixture set."""
    counts = {"direct": 0, "partial": 0, "deflected": 0}
    for label_path in sorted(_SYNTHETIC.glob("*.labels.json")):
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        for pair in labels["qa_pairs"]:
            counts[pair["answer_class"]] = counts.get(pair["answer_class"], 0) + 1
    for cls, n in counts.items():
        assert n >= 4, f"answer_class {cls!r} only appears {n} times (need >= 4)"


def test_commitments_totals_meet_spec_band() -> None:
    """Total commitments across synthetic fixtures land in the 12-20 band (spec target ~15)."""
    total = 0
    for label_path in sorted(_SYNTHETIC.glob("*.labels.json")):
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        total += len(labels["commitments"])
    assert 12 <= total <= 20, f"commitments total {total} outside 12-20 band"
