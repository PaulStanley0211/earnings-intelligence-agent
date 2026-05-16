# Upload-first product pivot — design spec

- **Date:** 2026-05-16
- **Status:** approved by product owner; pending spec-review before propagation to PLAN.md / CLAUDE.md / README.md
- **Author:** Paul Stanley (collaborating with Claude in IDE)
- **Phases affected:** Phase 4 (next), Phase 5b, Phase 6, Phase 7
- **Phases unchanged:** Phase 0, 1, 2, 3 (all complete), Phase 5a, 5c

## 1. Context

PLAN.md as originally written positions this project as a fully autonomous multi-agent system: a watcher polls EDGAR on a 5-25 ticker watchlist, the system detects a fresh 10-Q / 8-K, kicks off the agent pipeline, and ships a fact-checked research note within 15 minutes — primarily delivered by email or Slack.

Phases 1-3 built and shipped the deterministic spine of that vision:

- EDGAR client, financial extractor, comparator with consensus, language differ with embeddings, deterministic critic with citation enforcement.
- All gates met: 105+ tests green at Phase 2 close, recall-gate met at Phase 3 close, ≥ 85% line coverage throughout.

Phase 4 as originally written would have built a transcript analyzer that scrapes earnings-call transcripts from third-party sources (Motley Fool, Seeking Alpha, IR pages). On review this turned out to be the weakest part of the plan: scraping is fragile, legally fuzzy, and licensed feeds are expensive. Plus the existing autonomous pipeline is hard to demo — the only way to show it off live is to wait for a real earnings event or pre-record one.

## 2. New vision

**The product becomes an interactive upload-and-chat agent.**

The user picks any ticker. The agent uses the existing EDGAR client to tell them exactly which documents to grab and where to download each from. The user uploads. The same Phase 1-3 pipeline runs over the uploaded documents. The user then chats with the resulting structured analysis via a citation-enforced chat surface.

**The autonomous watcher survives, demoted to opt-in eval / demo mode.** It is off by default in production. When enabled, it polls a small fixed eval-set of tickers and feeds the nightly eval pipeline, preserving the "autonomous research note within 15 minutes of an EDGAR filing" claim as a quantitative property of the system rather than its primary user-facing flow.

## 3. Architecture changes

### 3.1 New components (Phase 4 and later)

| Component | Phase | Purpose |
|---|---|---|
| `document_advisor` agent node | 4 | Given a ticker, query EDGAR for recent 8-K / 10-Q / 10-K and return a ranked "what to upload" checklist with direct EDGAR URLs. Point users to public sources (IR, Motley Fool) for transcripts — no scraping. |
| `app/tools/documents.py` | 4 | Extract text from uploaded PDFs (pypdf) and plain text. Reject scanned PDFs (`total_chars == 0`) with a clear "looks like a scanned image — paste the text instead" error. |
| `upload_intake` agent node | 4 | Accept uploaded PDF / plain text, produce a `FilingEvent` shaped the same way the watcher produces today so downstream agents are unchanged. |
| `transcript_analyzer` agent node | 4 | Q&A pair extraction + answer classification (direct / partial / deflected) + commitment extraction. Same Phase 4 gates as PLAN.md originally specified, now run on uploaded transcripts. |
| `POST /api/upload`, `POST /api/chat` | 4 | FastAPI routes. Phase 4 exercises these via pytest + curl. No UI yet. |
| `chat_agent` | 6 | Answers user questions over the structured analysis using citation rules borrowed from the critic. Scope intentionally narrow — see §7. |
| Upload-and-chat frontend | 6 | Web UI for upload section + checklist display + analysis view + chat surface. Streamlit vs Next.js — open question, defer to Phase 6 kickoff. |

### 3.2 Survives unchanged

- All Phase 1-3 agent nodes: `financial_extractor`, `comparator`, `language_differ`, `synthesizer`, `critic`, `citations`.
- Memory layer (`app/memory/`), EDGAR client (`app/tools/edgar.py`), companyfacts loader, consensus fetcher, embeddings client, section parser, LLM client with daily cost cap and cassette replay.
- Postgres schema for filings, financial_facts, watchlist, edgar_poll_log, daily_llm_spend, consensus_estimates, comparisons, filing_sections, language_diffs.

### 3.3 Demoted to eval / demo mode

- [app/agents/watcher.py](../../../app/agents/watcher.py) and [app/scripts/poll_once.py](../../../app/scripts/poll_once.py) become opt-in. Gated behind a new env flag `WATCHER_MODE_ENABLED` (default `false`).
- `/health` keeps its EDGAR-freshness check, but only enforces the 5-minute SLO when `WATCHER_MODE_ENABLED=true`. Otherwise the freshness check is reported as `not_applicable` and does not affect health status.
- Eval workflow (`evals/`) is the primary consumer of watcher mode going forward.

### 3.4 New tables (Phase 4)

| Table | Purpose |
|---|---|
| `uploaded_documents` | Append-only; one row per uploaded file (sha256, ticker, filing_type, parsed_text, uploaded_at). |
| `qa_pairs` | One row per analyst Q&A exchange extracted from a transcript (filing_id, analyst_name, question_text, answer_text, answer_class). |
| `commitments` | One row per forward-looking statement (filing_id, commitment_text, target_period, status: open / met / missed, resolved_filing_id). |

## 4. Phase reshuffle

| Phase | Old scope | New scope |
|---|---|---|
| **4 (next)** | Transcript analyzer scraping from third-party sources | Upload intake + document advisor + PDF/text parsing + transcript analyzer on uploaded transcripts. Gates: 75% F1 on 50 labelled Q&A pairs (from user-supplied transcripts); commitments persist across two consecutive quarters; document advisor returns correct latest filing on 5 test ticker/date pairs |
| 5a | Memory writes / commitment status | Unchanged |
| 5b | Peer reader | Unchanged in spirit; may de-scope if upload flow makes cross-ticker context unnatural — decide at Phase 5b kickoff |
| 5c | Full critic | Unchanged |
| **6** | Streamlit dashboard + email + Slack delivery | Upload-and-chat frontend (Streamlit vs Next.js — pick at kickoff) + chat agent + citation-enforced chat surface. Email/Slack delivery becomes secondary, used only by eval-mode autogenerated notes |
| 7 | Deployment | Unchanged in shape — three services: FastAPI web, RQ worker, optional watcher (env-gated). Multi-stage Docker, Fly.io or Railway. |

## 5. Phase 4 scope (next up)

This is what Phase 4 needs to ship, in priority order:

1. **PDF / plain-text intake** ([app/tools/documents.py](../../../app/tools/documents.py)). pypdf-based. Rejects scanned PDFs with a clear error. Returns extracted text + a content hash.
2. **Document advisor** ([app/agents/document_advisor.py](../../../app/agents/document_advisor.py), [app/tools/advisor.py](../../../app/tools/advisor.py)). Reuses the existing EDGAR client. Given a ticker, returns a ranked list of `(filing_type, accession_number, edgar_url, recommended_priority)` plus a hint for where to fetch the transcript.
3. **Upload intake node** ([app/agents/upload_intake.py](../../../app/agents/upload_intake.py)). Wraps an uploaded document into a `FilingEvent` for the existing graph.
4. **Transcript analyzer** ([app/agents/transcript_analyzer.py](../../../app/agents/transcript_analyzer.py)). Q&A extraction + answer classification + commitment extraction. Prompts under `prompts/transcript_analyzer/`.
5. **Alembic migration** adding `uploaded_documents`, `qa_pairs`, `commitments`.
6. **API routes**: `POST /api/upload` (multipart), `POST /api/advise` (ticker → checklist), `POST /api/chat` (Phase 4 ships a minimal version that exercises the route shape; full chat agent is Phase 6).
7. **Graph entry-point widened**: `START` accepts either an `upload_event` or a `watcher_event`, both producing the same `FilingEvent`.
8. **Test fixtures**: 50 labelled Q&A pairs from 4-6 user-supplied transcript texts. Labelling protocol documented in `docs/phase4-labeling.md`.

## 6. Updated project-level success criteria

- **End-to-end latency:** < 5 min from upload to delivered analysis on 90% of events. (Old: < 15 min from EDGAR filing.)
- **Cost:** < $2 per event. (Unchanged.)
- **Factuality:** > 0.9 on the golden eval set. (Unchanged.)
- **Intervention rate:** 5-20% on critic interventions. (Unchanged.)
- **Coverage:** ≥ 85% line coverage on `app/`. (Unchanged.)
- **Document advisor accuracy:** > 95% (correct latest filing identified) on a 10-pair test set. (New.)
- **Autonomous-watcher claim (retained, eval-mode only):** "The same agent autonomously generates a research note within 15 minutes of an EDGAR filing — verified by `evals/` against a fixed 5-ticker eval set."

## 7. Design risks — flagged for end-of-project review, NOT pre-built

These are real risks that the upload-first pivot introduces. The user has explicitly chosen to build phase by phase and address these at the end-of-project review rather than pre-build mitigations. Captured here so they are not forgotten.

### 7.1 The autonomous-research-agent story can get drowned out

"Chat with your earnings docs" is a crowded category (Hebbia, ChatGPT attachments, several fintech startups). If the README / dashboard / demo leads with "upload PDFs and ask questions", the project reads like another doc-chat app. The differentiator — multi-agent orchestration, deterministic critic, cross-quarter commitment tracking, prompt-injection-safe pipeline — is the autonomous-research-system story.

**Review action (Phase 7 polish):** verify the README leads with the multi-agent / autonomous claim and frames upload as the *interface*, not the *product*.

### 7.2 Chat scope is easy to get wrong

"What was revenue?" is trivial. "Why did margins compress?" requires real inference. "How does this compare to AWS?" needs peer data (Phase 5b). Without explicit scoping, the chat surface drifts into being a generic Claude wrapper.

**Review action (Phase 6 kickoff):** lock in chat scope before building. Constrain to *questions answerable from the structured analysis* (financials, comparisons, language diffs, Q&A pairs, draft note), with the same citation rules the critic enforces. Out-of-scope questions get a polite "I can't answer that — here's what I can do."

## 8. Out of scope (unchanged)

Trading advice, buy/sell recommendations, private companies, real-time intraday news, options or derivatives, foreign exchange, sell-side rating predictions, foreign filings, anything constituting investment advice.

## 9. Open questions

- **Frontend stack** (Phase 6 decision): Streamlit (closer to PLAN.md as written, faster to build, weaker as a portfolio piece) vs Next.js (heavier lift, better demo). Defer to Phase 6 kickoff.
- **Chat-agent persistence**: do chat sessions persist across reloads, or is each session ephemeral? Defer to Phase 6.
- **Multi-user / auth**: PLAN.md §7 says public deployment requires API-key gating. The upload model raises the surface area — defer concrete auth design to Phase 7.

## 10. Propagation plan after spec approval

Once this spec is approved by the user:

1. Update [PLAN.md](../../../PLAN.md) — rewrite §1 vision, §3 architecture, §4 phase table, §5 testing, §8 ops to reflect the upload-first model; preserve eval-mode watcher claims.
2. Update [CLAUDE.md](../../../CLAUDE.md) — update status block, common commands, conventions where they touch upload/chat.
3. Update [README.md](../../../README.md) — frame the project lead with the autonomous-multi-agent angle, upload-and-chat as the interface.
4. Invoke `writing-plans` skill to produce a detailed Phase 4 implementation plan.

No code changes are part of this propagation — that's Phase 4 implementation, which begins after the implementation plan is approved.
