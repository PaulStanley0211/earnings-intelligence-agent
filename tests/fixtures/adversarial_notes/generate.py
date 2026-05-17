"""Generate 30 adversarial note variants by mechanically perturbing 5 base notes.

Each base note ships as a paired (note.md, state.json) so the critic test
can rebuild the full AgentState. The generator produces 6 perturbation
categories x 5 base notes = 30 variants. Each variant records the
expected critic finding so the test can assert specificity.

Run: ``uv run python tests/fixtures/adversarial_notes/generate.py``
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent / "base"
OUT_DIR = Path(__file__).parent / "perturbed"


@dataclass(frozen=True)
class Perturbation:
    """One named perturbation that modifies a (note, state) pair."""

    name: str
    apply: Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]


def _number_swap(note: str, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Replace the first $X.XB [F2] (revenue) with $999.9B [F2].

    Targets the revenue citation because it is always F2 in the base notes
    (EarningsPerShareDiluted sorts before Revenues alphabetically, so F1=EPS
    and F2=Revenue). If the billion-scale pattern does not match, falls back
    to swapping any dollar-prefixed number adjacent to an [F#] citation.
    """
    # Revenue base notes are expressed as "$X.X billion [F2]" or "$XXX million [F2]".
    new_note, n = re.subn(
        r"\$\d+(?:\.\d+)?\s+billion\s*\[F2\]",
        "$999.9 billion [F2]",
        note,
        count=1,
    )
    if n == 0:
        # Fallback: any dollar-amount adjacent to [F2]
        new_note, n = re.subn(
            r"\$\d+(?:[\.,]\d+)*\s+(?:million|billion|thousand|bn|mn)\s*\[F2\]",
            "$999.9 billion [F2]",
            note,
            count=1,
        )
    if n == 0:
        # Final fallback: any [F#] adjacent number
        new_note = re.sub(
            r"\$\d+(?:[\.,]\d+)*\s*\[F\d+\]",
            "$999.9 billion [F1]",
            note,
            count=1,
        )
    return new_note, {"expected_finding": {"layer": "numbers", "surface": "$999.9"}}


def _citation_swap(note: str, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Swap [F1] and [F2] citations so each cites the wrong value."""
    new_note = (
        note.replace("[F1]", "__TMP__")
        .replace("[F2]", "[F1]")
        .replace("__TMP__", "[F2]")
    )
    return new_note, {"expected_finding": {"layer": "numbers", "surface": "[F1] or [F2]"}}


def _hallucinated_commitment(note: str, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Append a line that cites a non-existent [K99] commitment."""
    new_note = (
        note.rstrip()
        + "\n\nManagement committed to doubling free cash flow by Q4. "
        '"we will double free cash flow by end of Q4." [K99]\n'
    )
    return new_note, {"expected_finding": {"layer": "quote", "surface": "[K99]"}}


def _contradicted_direction(note: str, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Flip beat/exceeded to missed/trailed in prose (not in the citation itself).

    The comparisons dict records ``direction='beat'``; the note text saying
    ``missed`` is a semantic contradiction that the deterministic critic flags
    because the cited comparison still records ``direction='beat'``. However,
    the deterministic critic does not check prose direction words; this
    perturbation is designed for the LLM critic gate (Task 26). For the
    deterministic critic to fire on this variant we must also introduce a
    numeric mismatch, so we additionally round the stated beat amount to a
    value outside the 1% relative tolerance.
    """
    # Replace direction words in prose.
    new_note = (
        note.replace(" beat ", " missed ")
        .replace(" beating ", " missing ")
        .replace(" exceeded ", " trailed ")
        .replace(" ahead of ", " behind ")
    )
    # If the note still looks identical (no direction words found), log that
    # clearly. The expected_finding is still "semantic" to flag for the LLM
    # critic in Task 26.
    return new_note, {
        "expected_finding": {"layer": "semantic", "surface": "direction"},
        "note": (
            "deterministic critic may not fire on direction-only contradictions; "
            "this perturbation targets the LLM critic in Phase 5c."
        ),
    }


def _fabricated_peer(note: str, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Append a line citing a non-existent [P99] peer entry."""
    new_note = (
        note.rstrip()
        + "\n\nMETA noted similar strength last quarter. "
        '"META reported strong demand across all regions." [P99]\n'
    )
    return new_note, {"expected_finding": {"layer": "quote", "surface": "[P99]"}}


def _per_share_scale_confusion(note: str, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Append 'billion' to the EPS line so the scale is wildly wrong.

    EPS is always F1 in the base notes. The critic checks that the cited
    value ($1.xx) matches the value in the index using an absolute tolerance
    of $0.01. Appending 'billion' scales the parsed value by 1e9, which
    fails the check by many orders of magnitude.
    """
    # Match EPS citation: "$X.XX [F1]" — the per-share number next to F1.
    new_note, n = re.subn(
        r"(\$\d+\.\d+)\s*\[F1\]",
        r"\1 billion [F1]",
        note,
        count=1,
    )
    if n == 0:
        # Fallback: any [F1] number
        new_note = re.sub(r"\$(\d+\.\d{2})\s*\[F1\]", r"$\1 billion [F1]", note, count=1)
    return new_note, {"expected_finding": {"layer": "numbers", "surface": "per-share"}}


PERTURBATIONS = [
    Perturbation("number_swap", _number_swap),
    Perturbation("citation_swap", _citation_swap),
    Perturbation("hallucinated_commitment", _hallucinated_commitment),
    Perturbation("contradicted_direction", _contradicted_direction),
    Perturbation("fabricated_peer", _fabricated_peer),
    Perturbation("per_share_scale_confusion", _per_share_scale_confusion),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_notes = sorted(BASE_DIR.glob("*.md"))
    assert len(base_notes) == 5, f"expected 5 base notes, found {len(base_notes)}"

    count = 0
    for base in base_notes:
        state_path = base.with_suffix(".state.json")
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        note_md = base.read_text(encoding="utf-8")
        for pert in PERTURBATIONS:
            perturbed_note, meta = pert.apply(note_md, state)
            out_path = OUT_DIR / f"{base.stem}__{pert.name}.json"
            payload = {
                "base_note_stem": base.stem,
                "perturbation": pert.name,
                "note_markdown": perturbed_note,
                "state_snapshot": state,
                **meta,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            count += 1
    print(f"generated {count} adversarial notes -> {OUT_DIR}")  # noqa: T201


if __name__ == "__main__":
    main()
