# Runbook

Operational playbook for the Earnings Intelligence Agent. One row per known
failure mode from `PLAN.md` section 8, plus anything we have learned in
production. Keep the table flat and the actions imperative.

## Operational playbook

### EDGAR returns 5xx

- **Symptom:** EDGAR watcher logs `edgar_5xx` repeatedly.
- **First response:** exponential backoff with jitter, up to 5 retries (automatic).
- **Escalation:** if all retries fail, the watcher logs `edgar_outage` and an
  alert fires in Slack. Confirm the outage at <https://www.sec.gov/cgi-bin/browse-edgar>;
  if confirmed, leave the watcher idle - it will resume on its own once the
  next poll succeeds.

### Finnhub rate-limit

- **Symptom:** `finnhub_rate_limited` in logs; consensus calls fail.
- **First response:** the consensus client automatically falls back to yfinance
  and tags the note with `consensus_source: yfinance`.
- **Escalation:** if both fail, the comparator returns no `vs_consensus` section
  and the note carries a `degraded: true` flag.

### Transcript unavailable

- **Symptom:** transcript node logs `transcript_missing`.
- **First response:** ship a partial note flagged `transcript_pending`; the
  watcher will reprocess the event when the transcript appears.

### XBRL malformed

- **Symptom:** `xbrl_extraction_failed` for a real filing.
- **First response:** fall back to LLM-driven extraction with reduced
  confidence. Note is tagged `extraction_source: llm`.

### LLM timeout

- **Symptom:** `llm_timeout` followed by retry.
- **First response:** retry once. If the retry still fails, ship a degraded
  note flagged `llm_partial` and continue.

### Critic loop exceeded

- **Symptom:** graph terminates with `critic_loop_exceeded` after three retries.
- **First response:** note is held for manual review. Slack alert fires.
  Inspect the `critic_findings` on the held note - if the critic is wrong,
  manually approve and file a regression test; if the synthesizer is wrong,
  add a rubric example and consider a prompt revision.

### Watcher restart

- **Symptom:** the watcher process restarts.
- **First response:** none required. Processed `accession_number` values are
  checkpointed in Postgres, so duplicate work is impossible.

### Daily LLM cost cap

- **Symptom:** `CostCapExceeded` raised; no further LLM calls happen today.
- **First response:** alert fires once daily spend crosses 80% of the cap. If
  the cap is hit, investigate the trace ids for the noisiest events - a
  prompt regression or a stuck retry loop is the usual cause.

## Reset procedures

### Replay a single filing

```bash
uv run python -m app.scripts.poll_once --ticker MSFT --accession <accession>
```

### Re-record a stale LLM cassette

```bash
REC=1 uv run pytest tests/integration/test_<scenario>.py
```

Re-record only when an intentional prompt or contract change makes the
cassette stale, not to make a flaky test pass.

## Phase 3 - language differ first-time setup

The differ requires a prior-quarter baseline in `filing_sections` to emit
non-degraded diffs. Backfill once per active ticker before the first live
event you want language coverage on:

    uv run python -m app.scripts.backfill_language --quarters 4

Properties:
- Idempotent: skips any filing already in `filing_sections`.
- Resumable: per-filing transaction boundary.
- Cost-bounded: enforces `MAX_DAILY_LLM_COST_USD` through the shared
  `daily_llm_spend` table.

Re-embedding gaps: if the daily cap blocked an embeddings call mid-run,
the affected `filing_sections` rows have `embedding=NULL`. Re-run the
backfill the next day; the no-op idempotency check skips the parsed
sections, but the embeddings update path will re-run for NULL rows
via a follow-up script (out of scope for Phase 3 launch).

## Phase 3 - language differ degraded paths

The language differ degrades gracefully on multiple failure modes:

| Symptom in logs | Cause | Action |
|---|---|---|
| `language_differ_short_circuit reason=primary_document_missing` | Filing row lacks `primary_document`. The watcher should set it. | Check the watcher; manually update the row if needed and re-run the graph. |
| `language_differ_short_circuit reason=fetch_failed` | EDGAR archives 4xx/5xx after retries. | Check EDGAR status; the note ships without language coverage for this event. |
| `language_differ_short_circuit reason=no_sections_parsed` | Section parser found no MD&A or Risk Factors. | Inspect the filing HTML; some 10-Q variants put MD&A in exhibits. |
| `language_differ_embed_failed` | OpenAI rate-limit or timeout after retries. | Paragraphs persist with `embedding=NULL`; re-embed via a follow-up. |
| Empty `language_diffs[i].diffs` with `degraded: true` | No prior quarter sections in memory. | Run the backfill CLI for this ticker. |
