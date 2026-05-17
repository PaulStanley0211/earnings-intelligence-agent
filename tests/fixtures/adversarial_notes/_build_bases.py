"""Build 5 base (note.md, state.json) pairs for adversarial note testing.

Each pair contains:
- A short Markdown note with valid [F#]/[C#]/[L#]/[Q#]/[K#] citations.
- A minimal AgentState snapshot (as JSON dict) that the critic can validate
  against.

All 5 base notes must pass the deterministic critic (CriticVerdict.ACCEPTED).
The builder verifies this before writing output.

Run: ``uv run python tests/fixtures/adversarial_notes/_build_bases.py``
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow import of app modules from repo root.
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agents.critic import critique_draft  # noqa: E402
from app.models.state import (  # noqa: E402
    AgentState,
    AnswerClass,
    CommitmentExtracted,
    CriticVerdict,
    FilingEvent,
    FilingEventSource,
    FilingForm,
    PeerContextEntry,
    QAPairPayload,
)

BASE_DIR = Path(__file__).parent / "base"


def _make_filing_event(
    ticker: str,
    accession: str,
    form: FilingForm = FilingForm.FORM_10Q,
) -> FilingEvent:
    """Construct a minimal FilingEvent for the given ticker and accession."""
    return FilingEvent(
        accession_number=accession,
        cik="0000000001",
        ticker=ticker,
        form=form,
        filed_at=datetime(2026, 5, 17, tzinfo=UTC),
        source_url=(
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={ticker}"
        ),
        source=FilingEventSource.UPLOAD,
    )


def _base_financials_revenue_eps(
    revenue_usd: int,
    eps_diluted: float,
    period_end: str,
    period_start: str,
) -> dict[str, Any]:
    """Minimal financials dict with Revenue + EPS.

    Citation order (alphabetical by concept name):
      F1 -> EarningsPerShareDiluted (eps_diluted)
      F2 -> Revenues (revenue_usd)
    """
    return {
        "by_concept": {
            "EarningsPerShareDiluted": [
                {
                    "value": eps_diluted,
                    "unit": "USD/share",
                    "period_end": period_end,
                    "period_start": period_start,
                    "fiscal_year": int(period_end[:4]),
                    "fiscal_period": "Q2",
                }
            ],
            "Revenues": [
                {
                    "value": revenue_usd,
                    "unit": "USD",
                    "period_end": period_end,
                    "period_start": period_start,
                    "fiscal_year": int(period_end[:4]),
                    "fiscal_period": "Q2",
                }
            ],
        }
    }


def _base_comparisons(revenue_usd: int, consensus_usd: int) -> dict[str, Any]:
    """Minimal comparisons dict with one revenue row.

    C1 -> revenue
    """
    surprise = revenue_usd - consensus_usd
    surprise_pct = round(surprise / consensus_usd * 100, 2)
    direction = "beat" if surprise > 0 else "miss"
    return {
        "metrics": [
            {
                "metric": "revenue",
                "reported_value": revenue_usd,
                "reported_unit": "USD",
                "consensus_value": consensus_usd,
                "consensus_source": "finnhub",
                "surprise_abs": surprise,
                "surprise_pct": surprise_pct,
                "direction": direction,
            }
        ]
    }


def _base_language_diffs(
    current_text: str,
    prior_text: str,
    section: str = "mda",
) -> list[dict[str, Any]]:
    """Minimal language diffs list.

    L1 -> current_text (modified diff)
    """
    return [
        {
            "section": section,
            "diffs": [
                {
                    "change_type": "modified",
                    "current_text": current_text,
                    "prior_text": prior_text,
                    "severity": "minor",
                }
            ],
        }
    ]


def _sha(char: str) -> str:
    """Return a valid 64-char sha256 placeholder using the given character."""
    return char * 64


def _build_nimbus_q2() -> tuple[str, AgentState]:
    """Nimbus Q2 2026 — no peers."""
    revenue = 612_000_000  # $612M
    consensus = 600_000_000  # $600M
    eps = 1.28
    period_end = "2026-03-31"
    period_start = "2026-01-01"

    financials = _base_financials_revenue_eps(revenue, eps, period_end, period_start)
    comparisons = _base_comparisons(revenue, consensus)
    lang_current = (
        "We continue to see strong demand from enterprise customers consolidating onto Stratus"
    )
    lang_prior = "We see demand from enterprise customers adopting Stratus"
    language_diffs = _base_language_diffs(lang_current, lang_prior)

    q_text = "Can you give us more color on the Cirrus Analytics integration plan?"
    a_text = (
        "We are very enthusiastic about the Cirrus combination. "
        "We expect native Cirrus query capabilities by the end of fiscal Q4 2026."
    )
    qa_pairs = [
        QAPairPayload(
            ordinal=1,
            analyst_name="Aaron Mitchell",
            question_text=q_text,
            answer_text=a_text,
            answer_class=AnswerClass.DIRECT,
            sha256_text=_sha("a"),
        )
    ]

    k_quote = "we expect native Cirrus query capabilities by the end of fiscal Q4 2026"
    commitments = [
        CommitmentExtracted(
            commitment_text=(
                "Deliver native Cirrus capabilities inside Stratus console by end of Q4 2026"
            ),
            target_period="Q4 2026",
            source_quote=k_quote,
        )
    ]

    note = f"""\
# Nimbus Systems Q2 FY2026 Earnings Note

Nimbus delivered a strong second quarter. Total revenue reached $612 million [F2], \
beating consensus and generating a beat of $12 million [C1]. \
EPS (diluted) was $1.28 [F1].

Management sharpened its demand commentary in the MD&A. \
"{lang_current}" [L1]

Aaron Mitchell asked about the Cirrus integration timeline. \
"{a_text}" [Q1]

Looking ahead, management committed to delivering Cirrus capabilities on schedule. \
"{k_quote}" [K1]
"""

    state = AgentState(
        trace_id="nimbus-q2-base-001",
        started_at=datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
        filing_event=_make_filing_event("NIMBUS", "upload-nimbus-q2-2026-aaaaaa"),
        financials=financials,
        comparisons=comparisons,
        language_diffs=language_diffs,
        qa_pairs=qa_pairs,
        commitments=commitments,
        draft_note=note,
    )
    return note, state


def _build_nimbus_q3() -> tuple[str, AgentState]:
    """Nimbus Q3 2026 — includes peer context so note must have [P0]."""
    revenue = 668_000_000  # $668M
    consensus = 655_000_000  # $655M
    eps = 1.41
    period_end = "2026-06-30"
    period_start = "2026-04-01"

    financials = _base_financials_revenue_eps(revenue, eps, period_end, period_start)
    comparisons = _base_comparisons(revenue, consensus)
    lang_current = "We now expect GAAP operating profitability in fiscal 2027"
    lang_prior = "We expect GAAP operating profitability in fiscal 2027"
    language_diffs = _base_language_diffs(lang_current, lang_prior)

    q_text = (
        "Are you still on track for the Stratus Copilot GA date at end of fiscal Q4 2026?"
    )
    a_text = (
        "We are not going to hit the end of fiscal Q4 2026 GA target for Stratus Copilot. "
        "We now expect Stratus Copilot to reach general availability in fiscal Q2 2027."
    )
    qa_pairs = [
        QAPairPayload(
            ordinal=1,
            analyst_name="Theo Bennett",
            question_text=q_text,
            answer_text=a_text,
            answer_class=AnswerClass.DIRECT,
            sha256_text=_sha("b"),
        )
    ]

    k_quote = "we now expect Stratus Copilot to reach general availability in fiscal Q2 2027"
    commitments = [
        CommitmentExtracted(
            commitment_text="Stratus Copilot GA moved to fiscal Q2 2027",
            target_period="Q2 2027",
            source_quote=k_quote,
        )
    ]

    peer_text = (
        "Similar cloud observability tailwinds noted in DATADOG Q3 language: "
        "customers consolidating onto unified platforms"
    )
    peer_context = [
        PeerContextEntry(
            peer_ticker="DDOG",
            kind="language_diff",
            text=peer_text,
            source_filing_accession="0000123456-26-000099",
            severity="minor",
        )
    ]

    note = f"""\
# Nimbus Systems Q3 FY2026 Earnings Note

Nimbus reported Q3 revenue of $668 million [F2], ahead of consensus by $13 million [C1]. \
EPS (diluted) came in at $1.41 [F1].

Management updated its GAAP profitability commentary. "{lang_current}" [L1]

The Stratus Copilot timeline slipped. Theo Bennett pressed on the original commitment: \
"{a_text}" [Q1]

Management set a revised commitment for the AI product. "{k_quote}" [K1]

Peer signals reinforce the consolidation theme. \
"{peer_text}" [P0]
"""

    state = AgentState(
        trace_id="nimbus-q3-base-001",
        started_at=datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
        filing_event=_make_filing_event("NIMBUS", "upload-nimbus-q3-2026-bbbbbb"),
        financials=financials,
        comparisons=comparisons,
        language_diffs=language_diffs,
        qa_pairs=qa_pairs,
        commitments=commitments,
        peer_context=peer_context,
        draft_note=note,
    )
    return note, state


def _build_synthetic_a() -> tuple[str, AgentState]:
    """Synthetic A — based on GOOGL Q3 2026 data."""
    revenue = 102_700_000_000  # $102.7B
    consensus = 101_000_000_000  # $101.0B
    eps = 2.15
    period_end = "2026-09-30"
    period_start = "2026-07-01"

    financials = _base_financials_revenue_eps(revenue, eps, period_end, period_start)
    comparisons = _base_comparisons(revenue, consensus)
    lang_current = (
        "AI-driven features in Search, including AI Overviews and AI Mode, "
        "continue to deliver strong engagement"
    )
    lang_prior = "AI-driven features in Search continue to show solid engagement"
    language_diffs = _base_language_diffs(lang_current, lang_prior)

    q_text = (
        "Can you walk through what is driving the capex increase to 88 billion, "
        "and how confident you are in the demand signals?"
    )
    a_text = (
        "The increase reflects stronger than expected Cloud demand pull-through, "
        "expanded build for our internal Gemini training fleet, "
        "and selective infrastructure pre-positioning. "
        "Cloud RPO grew 38 percent year over year."
    )
    qa_pairs = [
        QAPairPayload(
            ordinal=1,
            analyst_name="Brian Nowak",
            question_text=q_text,
            answer_text=a_text,
            answer_class=AnswerClass.DIRECT,
            sha256_text=_sha("c"),
        )
    ]

    k_quote = (
        "we now expect full-year 2026 capex to be approximately 88 billion dollars"
    )
    commitments = [
        CommitmentExtracted(
            commitment_text="Full-year 2026 capex guidance raised to approximately $88 billion",
            target_period="FY2026",
            source_quote=k_quote,
        )
    ]

    note = f"""\
# Alphabet Q3 2026 Earnings Note

Alphabet reported consolidated revenue of $102.7 billion [F2] for Q3 2026, \
beating consensus by $1.7 billion [C1]. \
EPS (diluted) came in at $2.15 [F1].

Management strengthened its Search AI commentary. "{lang_current}" [L1]

Brian Nowak from Morgan Stanley pressed on the capex step-up. \
"{a_text}" [Q1]

Management formalised the capex commitment for fiscal 2026. \
"{k_quote}" [K1]
"""

    state = AgentState(
        trace_id="synthetic-a-base-001",
        started_at=datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
        filing_event=_make_filing_event("GOOGL", "upload-googl-q3-2026-cccccc"),
        financials=financials,
        comparisons=comparisons,
        language_diffs=language_diffs,
        qa_pairs=qa_pairs,
        commitments=commitments,
        draft_note=note,
    )
    return note, state


def _build_synthetic_b() -> tuple[str, AgentState]:
    """Synthetic B — based on MSFT Q1 2026 data."""
    revenue = 70_100_000_000  # $70.1B
    consensus = 68_500_000_000  # $68.5B
    eps = 3.42
    period_end = "2026-09-30"
    period_start = "2026-07-01"

    financials = _base_financials_revenue_eps(revenue, eps, period_end, period_start)
    comparisons = _base_comparisons(revenue, consensus)
    lang_current = (
        "Microsoft Cloud revenue reached $45.3 billion, up 22 percent year over year"
    )
    lang_prior = "Microsoft Cloud revenue was up 21 percent year over year"
    language_diffs = _base_language_diffs(lang_current, lang_prior)

    q_text = (
        "Can you give us an update on Azure growth and the AI services contribution?"
    )
    a_text = (
        "Azure and other cloud services grew 33 percent in the quarter. "
        "AI services contributed 13 points of that growth, "
        "up from 9 points in the prior quarter."
    )
    qa_pairs = [
        QAPairPayload(
            ordinal=1,
            analyst_name="Keith Weiss",
            question_text=q_text,
            answer_text=a_text,
            answer_class=AnswerClass.DIRECT,
            sha256_text=_sha("d"),
        )
    ]

    k_quote = (
        "we expect Azure growth to accelerate in the second half of fiscal 2026 "
        "as our AI capacity investments come online"
    )
    commitments = [
        CommitmentExtracted(
            commitment_text="Azure growth expected to accelerate H2 FY2026 on AI capacity",
            target_period="H2 FY2026",
            source_quote=k_quote,
        )
    ]

    note = f"""\
# Microsoft Q1 FY2026 Earnings Note

Microsoft posted Q1 revenue of $70.1 billion [F2], beating consensus by $1.6 billion [C1]. \
EPS (diluted) reached $3.42 [F1].

The cloud narrative strengthened materially. "{lang_current}" [L1]

Keith Weiss asked about Azure's AI contribution. "{a_text}" [Q1]

Management committed to Azure acceleration in the second half. \
"{k_quote}" [K1]
"""

    state = AgentState(
        trace_id="synthetic-b-base-001",
        started_at=datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
        filing_event=_make_filing_event("MSFT", "upload-msft-q1-2026-dddddd"),
        financials=financials,
        comparisons=comparisons,
        language_diffs=language_diffs,
        qa_pairs=qa_pairs,
        commitments=commitments,
        draft_note=note,
    )
    return note, state


def _build_msft_8k() -> tuple[str, AgentState]:
    """MSFT 8-K based note — NVDA Q2 synthetic data as fallback."""
    revenue = 44_200_000_000  # $44.2B (NVDA Q2 2026 synthetic)
    consensus = 43_500_000_000  # $43.5B
    eps = 1.89
    period_end = "2026-07-28"
    period_start = "2026-04-28"

    financials = _base_financials_revenue_eps(revenue, eps, period_end, period_start)
    comparisons = _base_comparisons(revenue, consensus)
    lang_current = (
        "Data Center revenue reached $37.1 billion, driven by Blackwell architecture demand"
    )
    lang_prior = "Data Center revenue was up on strong GPU demand"
    language_diffs = _base_language_diffs(lang_current, lang_prior, section="risk_factors")

    q_text = (
        "Can you give us visibility into Blackwell supply-demand balance "
        "and when you expect supply to catch up to demand?"
    )
    a_text = (
        "Blackwell supply is ramping rapidly. "
        "We expect supply to be in balance with demand by the end of calendar 2026. "
        "Customer lead times have shortened from over 52 weeks to under 26 weeks."
    )
    qa_pairs = [
        QAPairPayload(
            ordinal=1,
            analyst_name="Stacy Rasgon",
            question_text=q_text,
            answer_text=a_text,
            answer_class=AnswerClass.DIRECT,
            sha256_text=_sha("e"),
        )
    ]

    k_quote = (
        "we expect supply to be in balance with demand by the end of calendar 2026"
    )
    commitments = [
        CommitmentExtracted(
            commitment_text="Blackwell supply-demand balance expected by end of calendar 2026",
            target_period="CY2026",
            source_quote=k_quote,
        )
    ]

    note = f"""\
# NVIDIA Q2 FY2026 Earnings Note

NVIDIA reported Q2 revenue of $44.2 billion [F2], exceeding consensus by $700 million [C1]. \
EPS (diluted) was $1.89 [F1].

The Data Center segment narrative updated significantly. "{lang_current}" [L1]

Stacy Rasgon from Bernstein pressed on Blackwell supply timing. "{a_text}" [Q1]

Management committed to resolving the supply-demand imbalance. \
"{k_quote}" [K1]
"""

    state = AgentState(
        trace_id="msft-8k-base-001",
        started_at=datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
        filing_event=_make_filing_event("NVDA", "upload-nvda-q2-2026-eeeeee", FilingForm.FORM_8K),
        financials=financials,
        comparisons=comparisons,
        language_diffs=language_diffs,
        qa_pairs=qa_pairs,
        commitments=commitments,
        draft_note=note,
    )
    return note, state


BUILDERS: list[tuple[str, Any]] = [
    ("nimbus_q2", _build_nimbus_q2),
    ("nimbus_q3", _build_nimbus_q3),
    ("synthetic_a", _build_synthetic_a),
    ("synthetic_b", _build_synthetic_b),
    ("msft_8k", _build_msft_8k),
]


def _state_to_json(state: AgentState) -> str:
    """Serialise AgentState to a JSON-compatible dict via model_dump."""
    raw = state.model_dump(mode="json")
    return json.dumps(raw, indent=2, default=str)


def main() -> None:
    """Build and validate all 5 base note pairs, then write them to BASE_DIR."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    for stem, builder in BUILDERS:
        note_md, state = builder()

        # Verify the critic accepts the base note.
        update = critique_draft(state)
        result_state = update.apply(state)
        verdict = result_state.critic_verdict

        if verdict != CriticVerdict.ACCEPTED:
            findings = result_state.critic_findings
            msgs = "\n  ".join(f"[{f.layer}/{f.severity}] {f.message}" for f in findings)
            print(f"FAIL {stem}: critic rejected with {len(findings)} findings:\n  {msgs}")  # noqa: T201
            sys.exit(1)

        # Write the pair.
        note_path = BASE_DIR / f"{stem}.md"
        state_path = BASE_DIR / f"{stem}.state.json"

        # Store state without draft_note (it lives in the separate .md file).
        state_for_snapshot = state.model_copy(update={"draft_note": None})
        note_path.write_text(note_md, encoding="utf-8")
        state_path.write_text(_state_to_json(state_for_snapshot), encoding="utf-8")
        print(f"OK {stem}: critic ACCEPTED, wrote {note_path.name} + {state_path.name}")  # noqa: T201

    print(f"\nAll {len(BUILDERS)} base notes accepted by the deterministic critic.")  # noqa: T201


if __name__ == "__main__":
    main()
