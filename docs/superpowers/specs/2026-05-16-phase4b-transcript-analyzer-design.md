# Phase 4B — Transcript analyzer + commitment reconciliation — design spec

- **Date:** 2026-05-16
- **Status:** approved by product owner; pending implementation plan
- **Author:** Paul Stanley (collaborating with Claude in IDE)
- **Phases affected:** Phase 4B (this spec). Pulls forward the commitment-status portion of Phase 5a.
- **Phases unchanged:** Phase 0, 1, 2, 3, 4A (all complete). Phase 5a is reduced to "remaining memory writes" since commitment-status transitions land in 4B. Phase 5b, 5c, 6, 7 unchanged in shape.
- **Supersedes:** the open Phase 4 items in [`2026-05-16-upload-first-pivot-design.md`](2026-05-16-upload-first-pivot-design.md) that were not delivered by Phase 4A.

## 1. Context

Phase 4 of the project was split into two cuts. Phase 4A (commit `4978f2a`, 2026-05-16) shipped the upload infrastructure: PDF / plain-text intake, document advisor against the EDGAR client, `upload_intake` node producing a `FilingEvent`, the API stubs (`/api/advise`, `/api/upload`, `/api/chat`), and the `uploaded_documents` table.

Phase 4B closes out the rest of Phase 4 from [`2026-05-16-upload-first-pivot-design.md`](2026-05-16-upload-first-pivot-design.md) §5: the transcript analyzer, the `qa_pairs` and `commitments` tables, the labelled fixtures, the synthesizer and critic updates to consume Q&A and commitment citations, and the advisor accuracy gate. It additionally pulls forward the **commitment-status reconciliation** work that the original PLAN.md placed in Phase 5a, because the product-owner directive is "everything Phase 4 gates require" — and the gate explicitly says commitments must persist across two consecutive quarters with open → met / missed transitions.

## 2. Scope

In scope for Phase 4B:

1. `transcript_analyzer` agent node — Q&A extraction, answer classification (direct / partial / deflected), commitment extraction, and reconciliation of prior open commitments against the current transcript.
2. Schema migration `0005_phase4b_transcripts_and_commitments` adding `qa_pairs` and `commitments` tables. Append-only except `commitments.status` and its companion fields.
3. `AgentState` extension with three new node-owned fields: `qa_pairs`, `commitments`, `commitment_updates`.
4. Filing-type-aware routing — `filings.form` accepts `TRANSCRIPT` as a valid value; financial-track nodes self-skip on transcripts; `transcript_analyzer` self-skips on 10-Q / 10-K / 8-K.
5. Synthesizer prompt v3 consuming Q&A and commitment context, emitting `[Q#]` and `[K#]` citations.
6. Critic citation index extension to resolve `[Q#]` and `[K#]`.
7. API: `POST /api/upload` accepts `filing_type=TRANSCRIPT`.
8. Test fixtures — 50 labelled Q&A pairs (hybrid synthetic + real) + 10 ticker/date pairs for the advisor accuracy gate + a 2-consecutive-quarter pair for the commitment-persistence gate.
9. Labelling protocol doc at `docs/phase4b-labeling.md`.

Out of scope (explicit deferrals):

- `POST /api/chat` real implementation — Phase 6.
- Streamlit / Next.js frontend — Phase 6.
- Peer reader — Phase 5b.
- Memory writes beyond commitments — Phase 5a (4B does not touch other append-only persistence paths beyond what is already in place).

## 3. Architecture

### 3.1 Data model

New migration `migrations/versions/20260516_HHMM_0005_phase4b_transcripts_and_commitments.py` (filename `HHMM` portion stamped at implementation time, matching the existing convention used by `0001`-`0004`):

```
qa_pairs
  id                pk
  filing_id         fk filings.id, indexed
  ordinal           int (1-based position within transcript)
  analyst_name      text, nullable
  question_text     text
  answer_text       text
  answer_class      enum(direct, partial, deflected)
  sha256_text       text (sha256 of question_text || "\n" || answer_text for cassette stability)
  created_at        timestamptz, default now()
  UNIQUE (filing_id, ordinal)
  INDEX (filing_id)

commitments
  id                  pk
  filing_id           fk filings.id (filing that MADE the commitment), indexed
  ticker              text, indexed
  commitment_text     text
  target_period       text (e.g. "Q3 2026", "FY2026", "next 12 months")
  source_quote        text (verbatim transcript span — anchor for [K#] citation)
  status              enum(open, met, missed, still_open), default open
  resolved_filing_id  fk filings.id, nullable (filing that CLOSED it)
  resolved_reason     text, nullable
  created_at          timestamptz, default now()
  updated_at          timestamptz, default now()
  INDEX (ticker, status)
```

`commitments.status` and the `resolved_*` columns are the only mutable fields anywhere in the system, consistent with the CLAUDE.md "memory is append-only" convention. Phase 5a inherits this schema unchanged.

`filings.form` is already a TEXT column; no enum migration is needed. The upload-intake validator's allowlist is extended to include `TRANSCRIPT`.

`AgentState` (in [`app/models/state.py`](../../../app/models/state.py)) gains three fields, each registered in `_FIELD_OWNERS` to `transcript_analyzer`:

- `qa_pairs: list[QAPair] = []`
- `commitments: list[CommitmentExtracted] = []`
- `commitment_updates: list[CommitmentStatusUpdate] = []`

DTOs in [`app/memory/schemas.py`](../../../app/memory/schemas.py); ORM models in [`app/memory/models.py`](../../../app/memory/models.py). New repository methods on [`app/memory/repository.py`](../../../app/memory/repository.py):

- `add_qa_pairs(filing_id, pairs)` — bulk insert, idempotent via `(filing_id, ordinal)`.
- `add_commitments(filing_id, ticker, commitments)` — bulk insert, idempotent if `(filing_id, source_quote)` is already present.
- `get_open_commitments(ticker)` — returns prior commitments with `status='open'` for reconciliation.
- `update_commitment_status(commitment_id, status, resolved_filing_id, resolved_reason)` — sets the four mutable fields atomically.

### 3.2 `transcript_analyzer` node

File: `app/agents/transcript_analyzer.py`. Pure function of `AgentState`, returns `StateUpdate`.

```
1. Guard:           if state.filing_event.form != "TRANSCRIPT": return empty update.
2. Extract:         single Sonnet call via LLMClient.acomplete using
                    prompts/transcript_analyzer/extract_v1.md. Transcript wrapped in
                    <source> tags. Temperature 0.0. JSON-mode response.
3. Reconcile:       fetch prior open commitments for state.filing_event.ticker via
                    Repository.get_open_commitments. Deterministic pre-filter on
                    keyword/period overlap; survivors batched into ONE Sonnet call
                    using prompts/transcript_analyzer/reconcile_v1.md, returning
                    {commitment_id, new_status, reason} per survivor.
4. Persist:         add_qa_pairs, add_commitments, then update_commitment_status for
                    each reconciled prior commitment, all inside one transaction.
5. Emit StateUpdate with qa_pairs, commitments, commitment_updates.
```

Cost: 2 Sonnet calls per transcript event (approximately $0.20 added to the Phase 2 / 3 per-event baseline of approximately $1.75; total approximately $1.95, within the $2/event target). When `form != "TRANSCRIPT"`, the node self-skips at step 1 and costs $0.

LLM calls go through the existing [`app/llm/client.py`](../../../app/llm/client.py); they participate in the cassette-replay layer and the Postgres-backed daily cost cap exactly like the synthesizer.

### 3.3 Graph topology

Updated [`app/graph.py`](../../../app/graph.py):

```
START
  -> upload_intake (or watcher when WATCHER_MODE_ENABLED)
  -> financial_extractor*
  -> [ comparator* || language_differ* || transcript_analyzer* ]
  -> synthesizer
  -> critic
  -> {synthesizer | END}
```

`*` = self-skips on inapplicable filing types. `financial_extractor`, `comparator`, and `language_differ` self-skip when `form == "TRANSCRIPT"`. `transcript_analyzer` self-skips when `form != "TRANSCRIPT"`. The parallel block is therefore safe regardless of upload type: every node either contributes its owned fields or yields an empty `StateUpdate`.

### 3.4 Synthesizer prompt v3

New prompt: `prompts/synthesizer/full_v1.md`. Extends the Phase 3 prompt by accepting two new blocks:

```
<source type="qa_pairs">
  Q1 (analyst: <name>): <question>
  A1 [<class>]: <answer>
  ...
</source>
<source type="commitments">
  K1 (target: <period>): <commitment_text>
  ...
</source>
```

New citation markers:

- `[Q#]` — Q&A pair. Resolves to `qa_pairs[# - 1]`.
- `[K#]` — commitment. Resolves to `commitments[# - 1]`.

The shared citation index in [`app/agents/citations.py`](../../../app/agents/citations.py) gains `Q` and `K` namespaces alongside the existing `F`, `C`, `L`. The critic continues to require an adjacent citation for every number and for every quoted phrase from the transcript. Tolerance for `[Q#]` and `[K#]` matches the existing `[L#]` rule: 90% character similarity between the quoted phrase in the draft and the resolved source text.

### 3.5 API surface

- `POST /api/upload` — `multipart/form-data` adds a required `filing_type` field accepting one of `8-K`, `10-Q`, `10-K`, `TRANSCRIPT`. Existing safety checks (content-type allowlist `application/pdf` + `text/plain`, magic-byte validation, size cap, scanned-PDF rejection) all unchanged. Upload-intake validator extends its `form` allowlist to include `TRANSCRIPT`.
- `POST /api/advise` — response unchanged in shape; `transcript_hint` text refined to point users to public IR pages.
- `POST /api/chat` — remains a 501 stub. Phase 6 territory.

### 3.6 Document advisor accuracy gate

New parametrized test `tests/unit/test_advisor_accuracy.py` over 10 ticker/date pairs. Each pair is a `(ticker, as_of_date, expected_latest_8k_accession)` tuple, backed by a cassette-recorded EDGAR submissions JSON in `tests/fixtures/edgar/advisor/`. Assertion: for each pair, `advise_for_ticker(...).suggested[0].accession_number == expected_latest_8k_accession`. Gate: 10/10, or 9/10 with a documented exception in the test docstring.

## 4. Prompts

Under `prompts/transcript_analyzer/`:

- `extract_v1.md` — Sonnet, temperature 0.0, JSON-mode. Body documents the answer-class rubric: `direct` answers the question with a fact or number; `partial` addresses the question but withholds a key piece; `deflected` redirects to a different topic, declines, or punts to "we will update next quarter". Returns `{qa_pairs: [...], commitments: [...]}` with `source_quote` anchors on every commitment.
- `reconcile_v1.md` — Sonnet, temperature 0.0, JSON-mode. Body takes a list of prior open commitments plus the current transcript text, returns per-id verdicts of `met` / `missed` / `still_open` with a short reason.

Both prompts carry frontmatter (`model`, `temperature`, `version`) and are body-SHA keyed by the existing prompt loader.

## 5. Testing

### 5.1 Fixtures

`tests/fixtures/transcripts/synthetic/` — 3 to 4 hand-written transcripts covering approximately 30 to 35 Q&A pairs across direct / partial / deflected, plus approximately 15 commitments. Labels in sibling `*.labels.json` files.

`tests/fixtures/transcripts/real/` — 2 real public transcripts for 1 ticker across 2 consecutive quarters, supplied by the product owner. Approximately 15 to 20 Q&A pairs labelled across the pair. The Q2 transcript anchors the reconciliation gate.

`tests/fixtures/edgar/advisor/` — 10 ticker/date pairs as cassette-recorded EDGAR submissions JSON responses.

`docs/phase4b-labeling.md` — the labelling protocol, mirroring `docs/phase3-labeling.md`.

### 5.2 Tests and gates

| Test | Location | Gate |
|---|---|---|
| Q&A F1 on labelled pairs | `tests/unit/test_transcript_analyzer_f1.py` | >= 75% F1 |
| Answer-classification per-class precision and recall | same file, separate assertion | >= 80% per-class |
| Commitment extraction recall | `tests/unit/test_commitment_extraction.py` | >= 80% recall |
| Commitment persistence Q1 to Q2 | `tests/integration/test_commitment_reconciliation.py` | >= 1 prior commitment closes (met or missed) on the Q2 run; zero false closes when transcript does not mention the commitment |
| Advisor accuracy on 10 pairs | `tests/unit/test_advisor_accuracy.py` | >= 95% (10/10 ideal; 9/10 acceptable with documented exception) |
| End-to-end transcript upload | `tests/integration/test_upload_transcript_e2e.py` | Upload -> pipeline -> analysis with `[Q#]` and `[K#]` citations resolving cleanly |
| Critic accepts valid `[Q#]` / `[K#]`, rejects invalid | `tests/unit/test_critic_transcript_citations.py` | Both directions pass |

### 5.3 Quality bar

`ruff` clean, `mypy` clean, `pip-audit` clean. Line coverage on `app/` >= 85% at Phase 4B close. All existing Phase 1 to 4A tests continue to pass without modification.

## 6. Failure modes

| Failure | Behavior |
|---|---|
| `extract_v1` returns malformed JSON | Retry once via LLMClient retry; on second failure, emit empty `qa_pairs` and `commitments` and continue to synthesizer with a `degraded=True` flag |
| Reconciliation LLM call fails | Prior commitments remain `open`; emit a structured warning; the pipeline still produces a note from current-quarter extraction |
| Prior open commitments list is empty | Reconciliation step skipped entirely; not an error |
| Transcript text exceeds the LLM's context window | Split on a Q&A boundary using the existing section-parser conventions; reassemble structured output. If splitting fails, return a `degraded=True` partial extraction |
| User uploads a transcript with `filing_type=10-Q` mistakenly | Pipeline runs the financial track and produces a note with empty financials; the critic does not reject because no numbers are claimed. Documented as a known user-error mode for Phase 6 to address with UI affordances |
| Critic rejects on `[Q#]` or `[K#]` similarity tolerance | Standard critic retry loop applies (bounded at 3); on `loop_exceeded` the note is held for manual review, consistent with Phase 2 behavior |

## 7. Cost and observability

- Per-transcript event: 2 Sonnet calls (extract + reconcile) + the existing synthesizer + critic Opus calls. Estimated approximately $1.95 worst case (baseline approximately $1.75 + approximately $0.20 added by the transcript analyzer), within the $2/event target.
- The daily cost cap in `app/llm/client.py` applies unchanged; transcript_analyzer participates via `acomplete`.
- Trace propagation: the node inherits `trace_id` from `AgentState`; per-call cost logged with the trace as for every other node.

## 8. Migration and rollback

Migration `0005_phase4b_transcripts_and_commitments` is forward-only and additive — it creates two new tables and touches no existing column. Rollback is a redeploy of the previous image tag; the two new tables are left in place (consistent with the Phase 7 "never drop in the same release that stops using it" convention).

## 9. Propagation plan after implementation

Once Phase 4B closes:

1. Update [`CLAUDE.md`](../../../CLAUDE.md) "Status" block with the Phase 4B "Added in" summary and commit / PR references.
2. Update [`PLAN.md`](../../../PLAN.md) §4 — flip the Phase 4 row to "complete" and note that Phase 5a inherits a reduced scope (commitment-status transitions already landed).
3. Update [`README.md`](../../../README.md) only if user-visible behavior changes warrant it.

No documentation changes are part of the implementation itself — they are the closing step.
