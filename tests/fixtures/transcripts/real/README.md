# Cross-quarter transcript pair (synthesized)

This directory holds a paired Q2/Q3 fiscal 2026 earnings-call transcript fixture
for the fictional ticker `NIMBUS` (Nimbus Systems Inc., a fictional observability
software vendor). The pair anchors the cross-quarter commitment-reconciliation
integration test for Phase 4B (Task 11 of the transcript-analyzer build).

## Why "real/" when the content is synthesized

The Phase 4B design spec at
`docs/superpowers/specs/2026-05-16-phase4b-transcript-analyzer-and-commitment-reconciliation.md`
reserves the directory name `real/` for transcripts intended to exercise the
end-to-end reconciliation flow (as opposed to the per-quarter single-transcript
fixtures in `synthetic/`). The product owner has approved synthesizing this
pair rather than sourcing two consecutive real earnings calls, because:

1. The reconciliation test needs precise, controllable cross-quarter linkage
   (one explicit `met`, one explicit `missed`, one true `still_open`).
2. Real consecutive calls rarely cover every status cleanly, and curating one
   that does would itself be a multi-day labelling exercise.
3. Using a fictional ticker keeps the fixture isolated from the real-ticker
   single-quarter fixtures in `synthetic/` (MSFT, NVDA, GOOGL, META), avoiding
   accidental contamination of those independent samples.

The directory name is kept as `real/` per the spec's naming convention so the
file layout matches what downstream code expects. The README here flags the
synthesized origin to anyone who reads the fixture.

## File set

- `transcript_nimbus_q2_2026.txt` + `.labels.json` -- Q2 fiscal 2026 call with
  seven forward-looking commitments and eight Q&A pairs.
- `transcript_nimbus_q3_2026.txt` + `.labels.json` -- Q3 fiscal 2026 call with
  seven Q&A pairs, five new commitments, and a `reconciliation_targets` array
  that documents the ground truth for the cross-quarter reconciler.

## reconciliation_targets ground truth

The Q3 labels file carries a `reconciliation_targets` array. Each entry maps a
Q2 commitment to the expected reconciler verdict (`met`, `missed`, or
`still_open`) and the verbatim Q3 quote that supports the verdict (or `null`
for the no-mention `still_open` case).

The pair exercises all three statuses:

- `met` -- Cirrus Analytics acquisition closed by end of Q3 (achieved on
  schedule); fiscal 2026 19 percent non-GAAP operating margin guide (on track).
- `missed` -- positive GAAP operating income for fiscal 2026 (explicitly
  acknowledged as missed in Q3 prepared remarks).
- `still_open` -- Stratus Copilot GA pushed from Q4 2026 to Q2 2027 (explicitly
  deferred); top-ten customer-concentration target (not mentioned in Q3 at
  all).

## Future swap to real transcripts

When real consecutive earnings calls with comparable status coverage become
available, the .txt files can be replaced and the .labels.json files re-labelled
against the new text. The schema and the integration test should not need to
change.
