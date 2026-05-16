# Phase 4B - transcript-analyzer labelling protocol

Authoritative record of how the synthetic earnings-call transcripts under
`tests/fixtures/transcripts/synthetic/` are labelled. The Q&A F1 gate in
`tests/unit/test_transcript_f1_gate.py` reads each `*.labels.json` and
asserts the transcript analyzer extracts the Q&A pairs and commitments at
or above the spec recall/precision floors. This document is the human
reference labellers and reviewers consult when adding, revising, or
auditing a label set.

## Labeller and date

- Labeller: Paul Stanley Ganganapalli
- Initial labelling date: 2026-05-16
- Fixture origin: synthetic, authored in-house to mirror the cadence and
  shape of real US-large-cap earnings calls (MSFT, NVDA, GOOGL, META) but
  using fictional numbers and product details. No copyrighted excerpts.
- Replacement policy: a real transcript can drop in by overwriting the
  `.txt` and rebuilding the sibling `.labels.json` from scratch. The
  schema, the F1 gate, and this protocol stay the same.

## Purpose

Phase 4B introduces a transcript-analyzer node that runs Sonnet-4.6 over a
verbatim earnings-call transcript and emits (a) analyst Q&A pairs, each
labelled with one of three answer classes, and (b) forward-looking
management commitments with optional target periods. The labels here are
the ground truth those extractions are scored against. Both the analyzer
and the gate read the exact same `answer_class` rubric and `commitment`
definition that this document records below, so the rubric in
`prompts/transcript_analyzer/extract_v1.md` and the rubric here must move
together.

## What is a Q&A pair?

A Q&A pair is one **analyst question + the immediately following
management answer**, taken verbatim from the Q&A session block. The
prepared-remarks section at the top of the call is not labelled.

Worked guidance on borderline cases:

- **Multi-part question, single answer.** When an analyst asks two or
  three related questions in one turn (e.g. "what is the ad load on Reels
  versus Feed, and how much headroom do you see going forward?"), label
  this as a single Q&A pair. The `question_text` captures the analyst's
  full turn, and the `answer_text` captures management's full reply. The
  `answer_class` is decided by the rubric below on the question's
  **primary clause**.

- **Operator transitions.** Lines like "Operator: Our next question comes
  from..." are not part of any Q&A pair. They mark boundaries; ignore
  them when picking the verbatim spans for `question_text` and
  `answer_text`.

- **Analyst pleasantries.** "Thanks for taking my question" or "Great
  quarter" appearing before the actual question is not part of the
  question. Start the `question_text` at the first substantive sentence.
  When in doubt include slightly more context rather than less - the F1
  gate uses 90 percent character similarity so extra leading words do not
  hurt.

- **Follow-up clarifications.** If the same analyst asks a follow-up
  inside the same turn ("and just a quick follow-up - what about
  pricing?"), keep them as one pair. If the operator hands back to the
  same analyst on a separate turn after another analyst, that is a new
  pair with a new `ordinal`.

- **Prepared-remark quotes inside a Q&A answer.** Management often quotes
  a number from their own prepared remarks ("as I said earlier, capex
  was 22.1 billion"). That is fine - it is still part of the verbatim
  answer span.

- **Closing remarks.** The CEO's closing thank-you is not a Q&A pair.

## The three answer classes

Verbatim from the extract prompt rubric (these definitions must match
`prompts/transcript_analyzer/extract_v1.md`):

- **`direct`** - answers the question with a fact or number.
- **`partial`** - addresses the question but withholds a key piece
  (refuses to quantify, defers to next quarter, gives qualitative
  direction without magnitude).
- **`deflected`** - redirects to a different topic, declines, or punts to
  "we will update next quarter".

### Worked examples

`direct` example 1:
> Q: "Can you give us specific numbers on weekly Waymo trips?"
> A: "Waymo is now serving over 350,000 paid rider trips per week..."
The answer supplies the requested number. Direct.

`direct` example 2:
> Q: "What percentage of your largest hyperscale customers have deployed
> Blackwell?"
> A: "All five of our largest hyperscale customers are in production
> with Blackwell at meaningful scale today."
Specific fact, no withholding on the asked metric. Direct.

`partial` example 1:
> Q: "How does monetisation in AI Overviews compare to traditional
> Search, and what is the commercial query mix between the two?"
> A: "Monetisation in AI Overviews has converged with traditional
> Search...we are not yet ready to share the specific commercial query
> mix between the two formats."
First clause answered with magnitude; second clause withheld. Partial.

`partial` example 2:
> Q: "How should we think about opex growth in the second half?"
> A: "We expect non-GAAP operating expense to grow in the high teens
> sequentially in Q3...we will give updated full-year color when we
> report Q3 results."
Gives Q3 direction but defers the full-year piece the analyst implicitly
wanted. Partial.

`deflected` example 1:
> Q: "Can you give us specific dollar guidance on 2027 Reality Labs
> losses and tell us when Reality Labs will reach break-even?"
> A: "We are not providing specific dollar guidance on 2027 Reality Labs
> operating losses today...we do not see Reality Labs reaching segment
> break-even within the next several years."
Refuses the dollar number entirely and punts the break-even question to
a vague horizon. Deflected.

`deflected` example 2:
> Q: "Can you walk through Shorts ad RPMs?"
> A: "We are not breaking out specific RPM numbers today...we will share
> more detail at our investor day next year."
Declines to quantify and punts to a future event. Deflected.

## Tie-breaking rules

When a single answer is partly direct and partly deflected, classify by
the **primary clause** of the question - the metric or topic the analyst
led with. If the analyst's first sentence asks for a specific number and
management gives it, label `direct` even if a secondary clause is
deflected. If the analyst's first sentence asks for a number and
management punts on that number but answers a secondary clause, label
`partial` (the analyst still got something useful) or `deflected` (the
analyst got nothing on the primary ask).

Practical heuristic: read the analyst's first sentence. Imagine the
analyst is graded on whether they got an answer to that exact sentence.
That is the class.

A second tie-breaker: when an answer gives a magnitude on the asked
metric but qualifies it heavily ("approximately", "in the high teens",
"low 20 percent range"), still classify `direct`. Earnings-call answers
are almost never numerically precise; ranges and qualified magnitudes
are the standard form of a direct answer.

## What is a commitment?

A commitment is a **forward-looking statement where management asserts a
specific future outcome, target, or action with an implicit or explicit
time horizon**. The statement must come from management, not from an
analyst, and must reference the future, not the quarter just reported.

Examples that count:
- "We expect operating margins to expand by 100 basis points next
  quarter."
- "We will launch the next-generation Maia chip in the second half of
  calendar 2026."
- "Full-year 2027 Cloud operating margin will be in the low 20 percent
  range."
- "Networking will represent over 22 percent of data center system
  revenue by exit fiscal 2027."

Examples that do **not** count:
- "We are excited about our pipeline." - no specific outcome.
- "Q3 revenue was 67.8 billion dollars." - historical, not
  forward-looking.
- "The analyst asked about 2027 capex." - statement is from the analyst.
- "We have always been focused on customer success." - aspirational, no
  target or time horizon.

When a commitment is bundled with a related statement ("we expect to
ship 5 million units in 2027, and the next-generation device launches in
H2 2027"), it is usually fine to record one commitment with a
`source_quote` spanning both clauses, as long as the analyst can read
the quote and recognise both promises. Alternatively, split into two
records with overlapping source quotes. Either is acceptable - the F1
gate scores against the full set.

## `target_period` formats

`target_period` is a free-text string the analyzer (and you) attach to
each commitment, or `null` when the commitment carries no explicit time
horizon. The downstream reconciliation pre-filter does a
case-insensitive substring match against the next transcript, so prefer
short canonical strings that are likely to recur verbatim. Recommended:

- `"Q3 2026"`, `"Q4 2026"`, `"Q1 2027"` - calendar-quarter targets.
- `"FY2026"`, `"FY2027"` - fiscal-year targets (most US large caps use
  fiscal years that align with calendar quarters, but a few - MSFT and
  NVDA - run off-calendar fiscals).
- `"H1 2027"`, `"H2 2027"` - half-year targets.
- `"end of calendar 2026"` - when management explicitly says "calendar"
  not "fiscal".
- `"next 12 months"`, `"next 24 months"` - rolling-window targets where
  management does not anchor to a fixed period.
- `"next quarter"` - relative reference; the analyzer will resolve this
  to a concrete quarter at reconcile time.
- `null` - when no time horizon is stated and none is implied.

Do not invent a period that management did not state. If the speaker
says "in the coming years", record `target_period` as `null` rather
than guessing a fiscal year. The reconciliation step is conservative
enough to handle nulls.

## Workflow

Recommended labeller flow per transcript:

1. **Read once for structure.** Skim the full transcript end to end. Get
   a feel for the speakers, the segments, and the headline numbers.
2. **Find the Q&A boundary.** Locate the line that opens the Q&A session
   (usually "let us move to questions" or "now let us open the line").
   Everything before that line is prepared remarks and is not labelled
   as Q&A.
3. **Label Q&A pairs in order.** For each analyst turn, pick verbatim
   spans for `question_text` and `answer_text`, fill in `analyst_name`
   and `analyst_firm`, and assign `answer_class` using the rubric and
   tie-breaking rules above. Increment `ordinal` monotonically by
   transcript order, starting at 1.
4. **Separate pass for commitments.** Re-read the prepared remarks and
   each answer hunting for forward-looking management assertions. Each
   becomes a `commitment_text` (paraphrase allowed), a `target_period`
   (or `null`), and a `source_quote` (verbatim transcript substring).
5. **Verify substrings.** Run the unit test in
   `tests/unit/test_phase4b_fixtures.py`. The
   `test_label_qa_text_is_verbatim_substring_of_transcript` check fails
   loudly on any drift between transcript text and labels.

Treat the substring check as the labelling gate. If it fails, fix the
label - never edit the transcript to match a wrong label.

## Tooling

Every `.txt` file under `tests/fixtures/transcripts/synthetic/` has a
sibling `<name>.labels.json` with this shape:

```json
{
  "transcript_file": "<name>.txt",
  "qa_pairs": [
    {
      "ordinal": 1,
      "analyst_name": "Brent Thill",
      "analyst_firm": "Jefferies",
      "question_text": "<exact substring from the transcript>",
      "answer_text": "<exact substring from the transcript>",
      "answer_class": "direct"
    }
  ],
  "commitments": [
    {
      "commitment_text": "<paraphrased forward-looking statement>",
      "target_period": "Q3 2026",
      "source_quote": "<exact substring from the transcript>"
    }
  ]
}
```

Rules:

- `question_text`, `answer_text`, and `source_quote` must be exact
  substrings of the transcript. The F1 gate uses 90 percent character
  similarity, so trivial whitespace edits are forgiven, but the spans
  are otherwise compared verbatim. Write the file in UTF-8 with LF line
  endings.
- `ordinal` is 1-based and monotonic by transcript order.
- `analyst_firm` is part of the label set only - it is not in the
  analyzer's output schema, which is intentional. The firm is captured
  for human auditing of the fixture set.
- `target_period` is `null` for commitments without an explicit time
  horizon.
- `answer_class` is exactly one of `direct`, `partial`, `deflected`.

## Append-only policy

The label files are append-only in spirit: when the rubric evolves, do
not silently rewrite existing labels. Either (a) append new fixtures
under a fresh transcript file with the new rubric applied, or (b) bump
the surrounding documentation and the F1 gate's expected thresholds in
the same commit that rewrites labels, so the change is reviewable end
to end. If a transcript is replaced wholesale with a real recorded
transcript, the sibling label file is rebuilt from scratch under the
prevailing rubric.
