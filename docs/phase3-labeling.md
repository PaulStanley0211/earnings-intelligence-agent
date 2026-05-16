# Phase 3 - language-differ recall labelling

Authoritative record of the 15 hand-labelled quarter pairs used as the
recall gate for the language differ. The recall test in
`tests/unit/test_recall_gate.py` reads from
`tests/fixtures/language_recall/labels.yaml` and asserts the differ
detects at least 80 percent of the labelled changes.

## Labeller and date

- Labeller: Paul Stanley Ganganapalli
- Initial labelling date: 2026-05-16
- Fixture origin: synthetic, authored to mirror real 10-Q language
  patterns. Can be replaced with real EDGAR HTML by overwriting the
  HTML files and updating `labels.yaml`; the schema and test logic
  stay the same.

## Tickers

| Ticker | Why |
|--------|-----|
| MSFT   | Cloud-focused; varied MD&A across quarters with AI infrastructure narrative evolving quarter to quarter. |
| AAPL   | Hardware and services mix; iPhone narrative shifts between quarters; sparse Risk Factors updates. |
| NVDA   | AI cycle at peak -- exercises change detection on material rewrites as Hopper transitions to Blackwell. |
| AMZN   | Multi-segment reporting (North America, International, AWS, Advertising) -- long MD&A with distinct sub-narratives. |

## Sections covered

Each ticker has 4 consecutive quarter MD&A files (q1 through q4). Risk
Factors files are present for a subset of quarters, reflecting the
real-world pattern where 10-Q filings often omit Item 1A when there are
no material updates:

- MSFT: q1 and q3 Risk Factors
- AAPL: q2 Risk Factors only
- NVDA: q2 and q4 Risk Factors
- AMZN: q1 and q3 Risk Factors

The 15 labelled pairs consist of 12 consecutive MD&A pairs (one per
consecutive quarter transition per ticker) plus 3 Risk Factors pairs.

## Rubric

A label is recorded when, reading two consecutive filings side by side, a
finance analyst would note one of:

1. **added**: a paragraph appears in the current quarter that has no close
   analogue in the prior quarter and conveys new substantive information.
   Formatting boilerplate, cross-references, and repeated legal language do
   not qualify.

2. **removed**: a paragraph from the prior quarter disappears in the current
   quarter and that paragraph conveyed substantive information. If the content
   migrated to a different section, it is still removed from this section.

3. **modified**: a paragraph maps to a prior paragraph but the wording has
   been rewritten in a way that changes the meaning. Synonym swaps do not
   count. Numerical updates inside an otherwise identical sentence do not count
   unless the surrounding sentence structure changes materially.

A paragraph whose only change is a single number (e.g., "grew 12 percent"
becoming "grew 14 percent") is intentionally NOT labelled -- the classifier
may or may not detect it as modified, and the recall gate should not penalise
a near-miss on cosmetic updates.

## Embedding and classifier context

The Task 19 recall-gate test (`tests/unit/test_recall_gate.py`) uses a
deterministic hash-based embedding (trigram bag-of-words projected to a
1536-dimension vector) rather than a live model. The classifier thresholds
in `app/agents/language_differ.py` are:

- Cosine >= 0.97 -- unchanged (not a diff)
- Cosine in [0.65, 0.97) -- modified
- Cosine < 0.65 (unmatched) -- added or removed

Fixtures are authored so that:

- Unchanged paragraphs are near-identical in wording (triggering cosine
  >= 0.97 under trigram similarity).
- Modified paragraphs share substantial vocabulary (~50 percent or more of
  trigrams) but differ enough to fall below the unchanged threshold.
- Added/removed paragraphs have minimal trigram overlap with any paragraph
  in the opposing section (cosine < 0.65).

The target recall is 80-85 percent, not 100 percent. A few labels represent
borderline cases where the differ may classify a removed+added pair as a
single modified, or vice versa. This makes the gate meaningful under
production conditions.

## Append-only policy

Labels are append-only. If the rubric evolves, new `id` entries are added to
`labels.yaml`; existing entries are not mutated. If a fixture file is replaced
with real EDGAR HTML, the associated labels must be re-validated and a new
entry with a distinct `id` suffix (e.g., `MSFT-q1-q2-mda-real`) should be
added rather than overwriting the synthetic entry.

## How to add labels

When adding a label to `labels.yaml`:

1. Choose or create a pair entry with a kebab-case `id` following the pattern
   `{TICKER}-q{N}-q{N+1}-{section}` (e.g., `MSFT-q1-q2-mda`).
2. Set `paragraph_excerpt` to a 5-15 word substring unique within the section,
   taken verbatim from the current-quarter fixture for `added`/`modified`
   labels, or from the prior-quarter fixture for `removed` labels.
   Use lowercase in excerpts for reliable substring matching.
3. Set `change_type` to `added`, `removed`, or `modified`.

After editing, run the recall gate:

```
uv run pytest tests/unit/test_recall_gate.py -v -m slow
```

The gate must remain above 80 percent recall after any label additions.

## File inventory

```
tests/fixtures/language_recall/
  labels.yaml                 -- 15 labelled pairs, 44 total labels
  MSFT/
    q1_mda.html
    q2_mda.html
    q3_mda.html
    q4_mda.html
    q1_risk_factors.html
    q3_risk_factors.html
  AAPL/
    q1_mda.html
    q2_mda.html
    q3_mda.html
    q4_mda.html
    q2_risk_factors.html
  NVDA/
    q1_mda.html
    q2_mda.html
    q3_mda.html
    q4_mda.html
    q2_risk_factors.html
    q4_risk_factors.html
  AMZN/
    q1_mda.html
    q2_mda.html
    q3_mda.html
    q4_mda.html
    q1_risk_factors.html
    q3_risk_factors.html
```

Total: 23 HTML fixture files.
