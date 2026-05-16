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


def test_cross_quarter_pair_present_and_consistent() -> None:
    """The Q2/Q3 cross-quarter pair exists, ties together cleanly, and exercises
    all three reconciliation verdicts."""
    real_dir = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts" / "real"
    q2_path = next(real_dir.glob("*_q2_2026.txt"))
    q3_path = next(real_dir.glob("*_q3_2026.txt"))
    q2_labels = json.loads(q2_path.with_suffix(".labels.json").read_text(encoding="utf-8"))
    q3_labels = json.loads(q3_path.with_suffix(".labels.json").read_text(encoding="utf-8"))

    # Both have Q&A pairs.
    assert len(q2_labels["qa_pairs"]) >= 5
    assert len(q3_labels["qa_pairs"]) >= 5
    total = len(q2_labels["qa_pairs"]) + len(q3_labels["qa_pairs"])
    assert 15 <= total <= 22, f"Q&A pair total {total} outside spec band 15-20"

    # Verbatim substrings hold for both transcripts.
    q2_text = q2_path.read_text(encoding="utf-8")
    q3_text = q3_path.read_text(encoding="utf-8")
    for pair in q2_labels["qa_pairs"]:
        assert pair["question_text"] in q2_text
        assert pair["answer_text"] in q2_text
    for pair in q3_labels["qa_pairs"]:
        assert pair["question_text"] in q3_text
        assert pair["answer_text"] in q3_text

    # Q3 labels carry the reconciliation_targets section.
    targets = q3_labels["reconciliation_targets"]
    assert len(targets) >= 3, "need >= 3 reconciliation targets to exercise all statuses"
    statuses_seen = {t["expected_new_status"] for t in targets}
    assert {"met", "missed", "still_open"}.issubset(statuses_seen), (
        f"missing status coverage; got {statuses_seen}"
    )

    # Every q2_commitment_text in reconciliation_targets must match a Q2 commitment.
    q2_commitment_texts = {c["commitment_text"] for c in q2_labels["commitments"]}
    for t in targets:
        assert t["q2_commitment_text"] in q2_commitment_texts, (
            f"reconciliation target references unknown q2 commitment: "
            f"{t['q2_commitment_text']!r}"
        )

    # For met/missed targets, q3_evidence_quote must be a verbatim substring of Q3.
    for t in targets:
        if t["expected_new_status"] in {"met", "missed"}:
            quote = t["q3_evidence_quote"]
            assert quote is not None, "met/missed must carry a q3_evidence_quote"
            assert quote in q3_text, (
                f"q3_evidence_quote not verbatim in Q3 transcript: {quote!r}"
            )
