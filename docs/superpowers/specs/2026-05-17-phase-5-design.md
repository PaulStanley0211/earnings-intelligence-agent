# Phase 5 design — memory writes, peer reader, full critic

Status: approved 2026-05-17
Author: paulstanleyganganapalli@gmail.com (via Claude)
Branch target: `phase-5-memory-peer-critic` (one branch, three sequential commits — one per subphase)
Supersedes: nothing; extends [PLAN.md](../../../PLAN.md) §4 Phase 5.

## 1. Context

Phase 4B closed with the synthesizer producing structured notes containing financial citations (`[F#]`), comparison citations (`[C#]`), language-change citations (`[L#]`), Q&A citations (`[Q#]`), and commitment citations (`[K#]`), all enforced by a deterministic critic. Three issues were carried over as documented `xfail`s; see [CLAUDE.md](../../../CLAUDE.md) "Phase 4B known limitations".

Phase 5 splits into three subphases per PLAN.md §4:

- **5a** — persistent memory writes after every event.
- **5b** — peer reader: surface cross-company commentary from previously-analyzed peers.
- **5c** — full critic: an LLM-driven fact-checker layered on top of the existing deterministic critic.

The upload-first product direction is locked in [`docs/superpowers/specs/2026-05-16-upload-first-pivot-design.md`](2026-05-16-upload-first-pivot-design.md). This Phase 5 spec respects that: peer reader is keyed on the user's chosen ticker rather than auto-discovered.

## 2. Scope

### 2.1 In scope

**5a — memory writes (minimal)**
- New append-only `notes` table; one row per accepted critic verdict.
- Repository methods `insert_note(...)` and `get_latest_note(ticker)`.
- New terminal `note_writer` agent node that runs after critic ACCEPTED.
- Fix 4B xfail #2: tighten `transcript_analyzer` extract + reconcile prompts so the strict per-target NIMBUS Q2→Q3 test passes.

**5b — peer reader**
- New `peers` table mapping `(ticker, peer_ticker, source, created_at)`, seeded manually via a new operator script.
- New `peer_reader` agent node, pure function of `AgentState`, no LLM calls.
- New AgentState field `peer_context: list[PeerContextEntry]` owned exclusively by `peer_reader`.
- New citation namespace `[P#]` resolved by the deterministic critic via the existing quote-citation machinery.
- New synthesizer prompt `full_with_peers_v1.md` extending `full_v1.md`.
- Peer-reader joins the existing parallel fan-out (alongside comparator, language_differ, transcript_analyzer).
- Pulls peer `language_diffs` (severity=major) and `commitments` (status=open) from the most recent peer filing only.

**5c — full critic**
- New `llm_critic` agent node running sequentially after the deterministic critic, only when the deterministic critic returns ACCEPTED.
- Single Opus call per pass, temp 0.0, daily-cost-cap-aware via the existing `LLMClient.acomplete` path.
- Adversarial test suite: 30 programmatically-perturbed notes (6 categories × 5 instances). Gate: ≥27/30 caught.
- Fix 4B xfail #3: relax `critic._language_match` to score only the first quoted substring when the line contains quotes.
- Shared retry budget with the deterministic critic (3 total attempts).

### 2.2 Out of scope

- xfail #1 (per-class answer-classification gate at 0.70 vs 0.80) — held over to Phase 6 fixture work; requires ≥25 real public-transcript labels per class.
- Persistent `critic_runs` and `event_runs` tables — deferred to Phase 6 / 7 alongside the SLO dashboards.
- Generic `agent_actions` audit log (PLAN.md §7) — deferred; not needed until the Phase 6 chat surface requires replayable history.
- Peer beat/miss directions, peer Q&A snippets — explicitly scoped out at brainstorming.
- LLM-driven peer discovery — curated table only.

### 2.3 Subphase sequencing

Three sequential commits on a single branch `phase-5-memory-peer-critic`. Each subphase passes its own gate before the next starts. PR opens after 5c.

## 3. Architecture

### 3.1 Graph topology

```
START
  └─> financial_extractor
        └─> {comparator | language_differ | transcript_analyzer | peer_reader}   (parallel)
              └─> synthesizer
                    └─> deterministic_critic
                          └─[ACCEPTED]─> llm_critic
                          │                └─[ACCEPTED]─> note_writer ─> END
                          │                └─[REJECTED, attempts < 3]─> synthesizer
                          │                └─[LOOP_EXCEEDED]─> END (no note persisted)
                          └─[REJECTED, attempts < 3]─> synthesizer
                          └─[LOOP_EXCEEDED]─> END
```

Notes:
- `peer_reader` is read-only over the persisted memory layer; adds zero LLM cost and minimal latency.
- The deterministic critic's ACCEPTED is now a checkpoint, not a terminal verdict. The LLM critic can still trigger a re-synth.
- The retry counter `state.critic_attempts` is incremented once per critic pass and shared across both critic layers. Worst case per event: 3 × (synth + det_critic + llm_critic) ≈ 6 Opus calls + 3 specialist fanouts ≈ $2-3, just over the per-event target. Daily cost cap protects the run-rate.
- `LOOP_EXCEEDED` skips `note_writer`. Filing row's `status` flips to `failed` via existing path; manual-review queue per the runbook stays unchanged.

### 3.2 Data model additions

**`notes` (new, Phase 5a)**

```sql
CREATE TABLE notes (
    id BIGSERIAL PRIMARY KEY,
    filing_accession VARCHAR(64) NOT NULL REFERENCES filings(accession_number) ON DELETE CASCADE,
    ticker VARCHAR(16) NOT NULL,
    markdown_body TEXT NOT NULL,
    prompt_template_name TEXT NOT NULL,
    prompt_template_sha CHAR(64) NOT NULL,
    critic_attempts INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (filing_accession)
);
CREATE INDEX ix_notes_ticker_created ON notes (ticker, created_at DESC);
```

Migration `0008_phase5a_notes`. Append-only by project convention; the UNIQUE on `filing_accession` means re-runs of the same filing get the existing id back via `ON CONFLICT DO NOTHING RETURNING id`. Per-event cost/latency are NOT stored here — the user explicitly scoped out a per-event metrics table (deferred to Phase 7 SLO work). `prompt_template_name` and `prompt_template_sha` are kept for eval traceability (which prompt version produced which note), independent of cost tracking.

**`peers` (new, Phase 5b)**

```sql
CREATE TABLE peers (
    ticker VARCHAR(16) NOT NULL,
    peer_ticker VARCHAR(16) NOT NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'curated',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, peer_ticker),
    CONSTRAINT peers_no_self_reference CHECK (ticker <> peer_ticker),
    CONSTRAINT peers_source_valid CHECK (source IN ('curated'))
);
CREATE INDEX ix_peers_ticker ON peers (ticker);
```

Migration `0009_phase5b_peers`. No FK to `watchlist.ticker` — peers are deliberately soft-referenced because users can upload arbitrary tickers that have never been on the watchlist. The check constraint forbids `(MSFT, MSFT)` style self-references. The `source` column is forward-compatible for an eventual auto-discovery option; the check currently allows only `curated`.

Seed script: `app/scripts/seed_peers.py` reads a YAML file at `data/peers.yaml` and upserts via `Repository.upsert_peer(ticker, peer_ticker, source)`. Initial seed: 3 pairs MSFT↔GOOGL, AAPL↔MSFT, JPM↔BAC (bidirectional rows, so 6 inserts).

### 3.3 AgentState contract

In `app/models/state.py`:

```python
class PeerContextEntry(BaseModel):
    """One peer signal surfaced by `peer_reader`.

    `kind='language_diff'`: a major MD&A/risk-factors change from the peer's
    most recent processed filing. `kind='commitment'`: an open management
    commitment from the peer's most recent transcript.
    """

    peer_ticker: str
    kind: Literal["language_diff", "commitment"]
    text: str
    source_filing_accession: str
    severity: Literal["major", "minor"] | None = None  # populated only for language_diff


class AgentState(BaseModel):
    # ...existing fields...
    peer_context: list[PeerContextEntry] = []          # owned by peer_reader
    persisted_note_id: int | None = None                # owned by note_writer
```

Field ownership additions to `_FIELD_OWNERS`:

| Field | Owner |
|---|---|
| `peer_context` | `peer_reader` |
| `persisted_note_id` | `note_writer` |

The LLM critic emits into the existing `critic_findings` / `critic_verdict` / `critic_attempts` fields under owner `critic` (it is composed inside the same critic stage; not a separate owner) — this keeps the existing field-ownership invariant unchanged.

### 3.4 Citation namespace

`app/agents/citations.py` adds:

```python
@dataclass(frozen=True)
class PeerCitation:
    identifier: str       # 'P0', 'P1', ...
    peer_ticker: str
    text: str
    kind: Literal["language_diff", "commitment"]


def build_peer_citations(peer_context: list[PeerContextEntry]) -> list[PeerCitation]: ...
```

`app/agents/critic.py` widens `_CITED_LANGUAGE` from `[LQK]\d+` to `[LQKP]\d+`. `_resolve_quote_citation` gets a `P` branch. `_namespace_label` maps `P` → `"peer commentary"`. Existing 90% character-similarity tolerance applies.

### 3.5 Prompt changes

- **`prompts/synthesizer/full_with_peers_v1.md`** (new). Extends `full_v1.md` with a `<source name="peers">` block containing each `PeerContextEntry` rendered with its `[P#]` id and source ticker. System prompt extension: "When peer commentary is materially relevant to the current filing's themes, cite it with `[P#]` exactly like other quote citations. Do not cite peers when their commentary is not directly relevant — empty peer context means no peer paragraph."
- **`prompts/transcript_analyzer/extract_v1.md`** (modified). Tightened to require explicit period markers ('Q3', 'next quarter', etc.) on every extracted commitment. Cassettes re-recorded.
- **`prompts/transcript_analyzer/reconcile_v1.md`** (modified). Reconcile heuristic tightened: a commitment may transition from `open` to `met` only when an unambiguous evidence quote with a numeric or boolean signal is present; otherwise it stays `still_open`. Cassettes re-recorded.
- **`prompts/critic/llm_v1.md`** (new). Opus, temp 0.0. Input: draft note + every `<source>`-wrapped citation index (financials, comparisons, language diffs, qa_pairs, commitments, peers). Output: JSON list of `{layer: "semantic", severity: "error"|"warning", claim: str, evidence: str, recommended_fix: str}`.

### 3.6 Repository methods

`app/memory/repository.py` adds:

- `insert_note(NoteCreate) -> int` — idempotent, returns id (existing or new).
- `get_latest_note(ticker: str) -> NoteRead | None`.
- `upsert_peer(ticker: str, peer_ticker: str, source: str = "curated") -> None`.
- `list_peers(ticker: str) -> list[str]` — returns `peer_ticker` values.
- `get_recent_peer_signals(peer_ticker: str, *, max_age_days: int = 180) -> PeerSignals` — returns:
  - `language_diffs`: from the peer's most recent processed 10-K or 10-Q filing (whichever is newest), filtered to `severity='major'`. Empty list if no such filing exists within `max_age_days`.
  - `commitments`: from the peer's most recent processed TRANSCRIPT filing, filtered to `status='open'`. Empty list if no such transcript exists within `max_age_days`.
  - The two are independently selected because language_diffs are populated only by 10-K/10-Q processing while commitments are populated only by transcripts; they typically come from different filings even for the same peer.
  - Returns an empty `PeerSignals` (both fields empty) for cold-start peers or peers with no fresh filings of either type.

## 4. Error handling

| Scenario | Layer | Behavior |
|---|---|---|
| `notes` insert fails after critic accepts | 5a | Log error, return note via API response, set `persisted_note_id=None`. Pipeline does not fail. |
| Duplicate `notes` insert for same filing | 5a | UNIQUE + `ON CONFLICT DO NOTHING RETURNING id` returns the existing id. |
| `LOOP_EXCEEDED` | 5a / 5c | `note_writer` skipped. `filings.status='failed'`. Manual-review queue per runbook. |
| No `peers` row for the uploaded ticker | 5b | `peer_context=[]`. Synthesizer omits peer paragraph. No error. |
| Peer ticker has no `language_diffs` and no open `commitments` in memory | 5b | Skip silently. If all peers are cold-start, `peer_context=[]`. |
| Peer's most recent filing is older than 180 days | 5b | Skip that peer; log at INFO with `stale_peer` event. Threshold is a single constant `_PEER_FRESHNESS_DAYS` in `peer_reader.py`. |
| `peer_reader` DB error | 5b | `peer_context=[]`. Emit a single `CriticFinding(layer="peer", severity="warning")` for trace visibility. Pipeline continues. |
| Synthesizer hallucinates a `[P#]` citation | 5b | Deterministic critic rejects with `"citation 'P#' references no known peer commentary"`. Same path as `[Q#]`/`[K#]` resolution failures. |
| LLM critic returns malformed JSON | 5c | Retry the LLM call once in-node. Still malformed → emit `CriticFinding(layer="semantic", severity="error", message="llm critic returned unparseable response")` and reject. Counts toward the bounded retry budget. |
| Daily cost cap exceeded mid-pipeline | 5c | Existing `acomplete` fail-closed path raises; graph's exception handling persists `filings.status='failed'` and bubbles to API as 503. |
| LLM critic accepts an obviously-wrong note | 5c | Caught by the 27/30 seeded-error gate. Below that gate, the prompt fails the eval and does not merge. |
| LLM critic over-eagerly rejects | 5c | Synthesizer re-runs with feedback appended (same path as deterministic-critic feedback). After 3 attempts → `LOOP_EXCEEDED`. Adversarial false-positive rate is part of the gate. |
| Det critic ACCEPTED + LLM critic emits only `severity="warning"` findings | 5c | Treated as ACCEPTED. Only `severity="error"` triggers a re-synth. |

## 5. Testing

### 5.1 Unit tests

| File | Coverage |
|---|---|
| `tests/unit/test_note_writer.py` (new) | happy path, idempotent re-run, DB-error degradation, LOOP_EXCEEDED skip |
| `tests/unit/test_peer_reader.py` (new) | no-peers, cold-start peer, stale peer (>180d), DB error → empty context, multi-peer ordering |
| `tests/unit/test_peer_citations.py` (new) | `build_peer_citations` returns `P0,P1,...`; round-trips through critic regex |
| `tests/unit/test_llm_critic.py` (new) | malformed JSON retry, accepts/rejects shape, daily-cap fail-closed |
| `tests/unit/test_adversarial_critic.py` (new) | the 30-note gate. 6 generators × 5 instances. Assert ≥27/30 caught. |
| `tests/unit/test_critic.py` (extend) | `[P#]` resolution; quote-substring relaxation; regression: full-line still scores when no quotes |
| `tests/unit/test_transcript_analyzer.py` (extend) | tightened extract+reconcile prompt regression cases |
| `tests/unit/test_recall_gate.py` (unchanged) | 80% language recall must still pass |

### 5.2 Integration tests

| File | Coverage |
|---|---|
| `tests/integration/test_notes_persistence.py` (new) | full graph run writes one `notes` row; re-run returns same id |
| `tests/integration/test_peer_reader_e2e.py` (new) | seed 2 peers + their language_diffs + commitments; uploaded filing produces note with ≥1 `[P#]` citation |
| `tests/integration/test_multi_quarter_synthetic_run.py` (new) | the 5a kickoff gate. 4 synthetic quarters end-to-end; assert (a) `notes` has 4 rows, (b) prior commitments correctly closed, (c) no orphan rows |
| `tests/integration/test_commitment_reconciliation.py` (modify) | strict per-target test must now pass; remove xfail marker |
| `tests/integration/test_upload_transcript_e2e.py` (modify) | E2E loop must now reach `final_note`; remove xfail marker |
| `tests/integration/test_phase5_cost_cap.py` (new) | run a 3-attempt critic loop with `$0.50/day` cap mid-run; assert third attempt fails closed, no `notes` row, `filings.status='failed'` |

### 5.3 Adversarial test set

Fixtures at `tests/fixtures/adversarial_notes/`. A generator script `tests/fixtures/adversarial_notes/generate.py` takes a correct synthesized note + its `AgentState` snapshot and produces 6 perturbed variants. Each variant carries `expected_finding: {layer: str, surface: str}` so the test asserts the critic catches THAT specific error, not just any error.

The 5 base notes:
- 4 from the synthetic transcript fixtures at `tests/fixtures/transcripts/`.
- 1 synthesized fresh from the MSFT 10-Q + transcript pair at `tests/fixtures/uploaded_pdfs/`.

The 6 perturbation categories:
1. Number swap (currency): replace a `$X [F#]` value with `$Y` where Y ≠ X.
2. Citation swap: swap `[F1]` and `[F2]` so each number cites the wrong source.
3. Hallucinated commitment: insert a sentence claiming a commitment that no `[K#]` covers.
4. Contradicted direction: claim "revenue beat consensus" when `comparisons[revenue].direction='miss'`.
5. Fabricated peer claim: insert "[P0] said X" where peer_context has no matching entry.
6. Per-share-vs-currency confusion: render an EPS value as `$X billion [F#]` (per-share concept mis-scaled).

5 base × 6 categories = 30 adversarial notes. Gate: ≥27/30 caught (90%).

### 5.4 Cassette strategy

- New cassette directories: `tests/fixtures/cassettes/llm_critic/`, `tests/fixtures/cassettes/synthesizer/full_with_peers/`.
- Both keyed by prompt-SHA so prompt edits force re-records via `REC=1 uv run pytest tests/integration -q`.
- Peer-reader is a pure DB read — no LLM cassettes needed.
- Existing transcript_analyzer cassettes will be re-recorded due to the prompt tightening for xfail #2.

## 6. Phase 5 gate evidence (at close)

- `ruff check app/ tests/` clean.
- `uv run mypy app/` clean. Projected ~52-54 source files.
- All xfails removed except #1 (per-class F1).
- Adversarial critic gate: ≥27/30 caught.
- Multi-quarter synthetic run: 4 `notes` rows persisted; all prior commitments correctly closed; no orphan rows in `qa_pairs`, `commitments`, `language_diffs`, `comparisons`.
- Peer-reader E2E: synthesized note includes ≥1 resolved `[P#]` citation.
- `uv run pytest --cov=app --cov-report=term` line coverage ≥85% (target: maintain ≥88%).
- `uv run pip-audit` clean.

## 7. Migration order

```
0008_phase5a_notes                  # creates `notes` table
0009_phase5b_peers                  # creates `peers` table
# 5c adds no migrations
```

Both forward-compatible: new tables only, no drops. Rollback is a redeploy of the previous image tag; the new tables are simply unused.

## 8. Open questions

None. All decisions locked at brainstorming.

## 9. Non-goals (explicit)

- Cross-language peer support (foreign-listed peers).
- Peer-reader auto-discovery via embedding similarity.
- LLM-generated peers from filing co-occurrence.
- Peer signal weighting / scoring beyond severity filtering.
- Critic findings persistence — deferred to Phase 6.
- Per-event cost/latency dashboards — deferred to Phase 7.
