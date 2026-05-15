# Phase 3 — Language differ design

Status: approved 2026-05-15. Implementation tracked separately via the writing-plans skill.

## 1. Goal

Build the language-differ specialist for the Earnings Intelligence Agent. The differ takes a freshly filed 10-Q (or 10-K), pulls its MD&A and Risk Factors sections, aligns each paragraph against the prior quarter's same section using paragraph-level embeddings, and emits typed `LanguageDiff` rows that the synthesiser quotes and the critic verifies.

The phase is complete when:

- The differ is wired into the LangGraph orchestrator in parallel with the comparator.
- The recall-gate test (`tests/unit/test_recall_gate.py`) hits ≥80% recall on 15 hand-labelled quarter-pairs of real EDGAR MD&A / Risk Factors text.
- 4 prior quarters of every active watchlist ticker have been backfilled via the new `app/scripts/backfill_language.py` CLI.
- `ruff`, `mypy`, the full pytest suite, `pip-audit`, and the 85% line-coverage gate all pass.

## 2. Architecture

A new specialist node `language_differ` joins the graph in parallel with `comparator`. Both depend on `financial_extractor`; both feed `synthesizer`. LangGraph fan-in waits for both.

```
START
  -> financial_extractor
       |
       +-> comparator -------+
       |                     |
       +-> language_differ --+
                             |
                             v
                         synthesizer
                             |
                             v
                          critic
                             |
                  (rejected -> synthesizer | accepted/loop_exceeded -> END)
```

Parallelism is safe transactionally:
- Each node opens its own `AsyncSession` and owns its own transaction (the existing pattern in `app/graph.py`).
- The two specialists own disjoint `AgentState` fields: `comparator` owns `comparisons`, `language_differ` owns `language_diffs`.
- The only shared mutable field is `cost_usd`, which the `StateUpdate.apply` reducer sums rather than overwrites (`app/models/state.py:170`).

`_FIELD_OWNERS` in `app/models/state.py` already reserves `language_diffs` for `language_differ`; the implementation only needs to remove the placeholder and emit `StateUpdate(owner="language_differ", changes={"language_diffs": ...})`.

## 3. Node responsibilities

`app/agents/language_differ.py` is a pure function of `AgentState` with three injected dependencies: an EDGAR client supporting `get_filing_document`, an embeddings client, and a `Repository` bound to the per-invocation session. The node does the following in order:

1. Resolve the filing's primary HTML document name. If the `Filing` row has `primary_document` set, use it; otherwise refresh from EDGAR's submissions API and persist for future runs.
2. Fetch the HTML body from `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_document}`.
3. Parse MD&A and Risk Factors sections; split each into paragraphs.
4. Persist every paragraph as a `filing_sections` row — even when the diff itself degrades, this filing must seed the next quarter's baseline.
5. Embed paragraph text in batches via the OpenAI embeddings client; back-fill the `embedding` column on the `filing_sections` rows.
6. For each section, look up the prior quarter's paragraphs for the same `(ticker, section_kind)`. If none, mark the section degraded with empty diffs and continue.
7. Run greedy nearest-neighbour alignment with cosine similarity, classify each pair, persist `language_diffs` rows.
8. Return a `StateUpdate` summarising the diff for the synthesiser and critic.

### Alignment + classification

Greedy nearest-neighbour. For each current paragraph, compute cosine similarity against every unconsumed prior paragraph (O(n*m); paragraph counts per section run 20-80, so quadratic is fine). Pair the highest match above threshold; mark the prior paragraph consumed. Then:

| Condition | Classification | Severity |
|---|---|---|
| similarity >= 0.97 | `unchanged` (not persisted) | n/a |
| 0.85 <= similarity < 0.97 | `modified` | `minor` |
| 0.65 <= similarity < 0.85 | `modified` | `major` |
| current unmatched, word_count > 30 | `added` | `major` |
| current unmatched, word_count <= 30 | `added` | `minor` |
| prior unmatched, word_count > 30 | `removed` | `major` |
| prior unmatched, word_count <= 30 | `removed` | `minor` |

Thresholds live as module constants with docstrings: `_SIMILARITY_MATCH_THRESHOLD = 0.65`, `_SIMILARITY_UNCHANGED_THRESHOLD = 0.97`, `_MAJOR_SIMILARITY_THRESHOLD = 0.85`, `_MAJOR_WORD_COUNT_THRESHOLD = 30`. They are tuned against the recall-gate fixture before merge.

### `StateUpdate` shape

```python
{
  "language_diffs": {
    "section": "mda",                              # iterated per section
    "prior_filing_accession": "0000950170-26-000010",
    "diff_count": 7,
    "major_count": 3,
    "diffs": [
      {"change_type": "added",    "text": "...", "severity": "major"},
      {"change_type": "modified", "current_text": "...", "prior_text": "...",
       "similarity": "0.7421", "severity": "major"},
      {"change_type": "removed",  "text": "...", "severity": "minor"},
      ...
    ],
    "degraded": false
  }
}
```

`AgentState.language_diffs` is `list[dict[str, Any]]` per `state.py:94`, so the node sets it to a list with one entry per section processed (typically two: `mda` and `risk_factors`).

## 4. Data model

Three additive changes to the memory layer, all in one Alembic migration:
`migrations/versions/20260515_NNNN_0003_phase3_schema.py`.

### 4.1 Enable pgvector

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The local Postgres image switches to `pgvector/pgvector:pg16` if it is not already (`docker-compose.yml`), and the CI integration job pulls the same image.

### 4.2 Extend `filings`

Add nullable `primary_document text` so the differ does not need to re-call the submissions API on every run. Backfilled lazily by the node itself on first encounter; never required to be non-null.

### 4.3 `filing_sections`

One row per paragraph per parsed section per filing.

| col | type | notes |
|---|---|---|
| `id` | bigserial PK | |
| `filing_accession` | varchar(32) FK -> filings | cascade delete |
| `cik` | varchar(10) | denorm for ticker-scoped baseline lookup |
| `ticker` | varchar(16) | denorm; the differ queries by `(ticker, section_kind)` |
| `section_kind` | varchar(16) | `'mda'` or `'risk_factors'` (CHECK) |
| `paragraph_index` | int | 0-based order within the section |
| `text` | text | the paragraph body, whitespace-normalised |
| `text_sha` | char(64) | sha256 of normalised text |
| `embedding` | `vector(1536)` | null when embedding call failed (degraded) |
| `embedding_model` | varchar(64) | e.g. `'openai/text-embedding-3-small'` |
| `created_at` | timestamptz | server default `now()` |

Constraints / indexes:
- Unique `(filing_accession, section_kind, paragraph_index)`.
- Index `(ticker, section_kind, filing_accession)` for prior-quarter lookup.
- Index `(cik, section_kind)` for the backfill script.

### 4.4 `language_diffs`

One row per material change. `unchanged` paragraphs are not persisted.

| col | type | notes |
|---|---|---|
| `id` | bigserial PK | |
| `filing_accession` | varchar(32) FK -> filings | the current filing |
| `prior_filing_accession` | varchar(32) FK -> filings, nullable | null only when no prior filing existed at all |
| `section_kind` | varchar(16) | matches `filing_sections.section_kind` |
| `change_type` | varchar(16) | `'added'` / `'removed'` / `'modified'` (CHECK) |
| `current_section_id` | bigint FK -> filing_sections, nullable | null for `removed` |
| `prior_section_id` | bigint FK -> filing_sections, nullable | null for `added` |
| `similarity` | numeric(6,4), nullable | cosine similarity for `modified`; null otherwise |
| `severity` | varchar(8) | `'major'` or `'minor'` |
| `created_at` | timestamptz | |

Unique `(filing_accession, section_kind, change_type, current_section_id, prior_section_id)` — re-running the differ is idempotent.

### 4.5 DTOs and repository

`app/memory/schemas.py` gains:
- `SectionKind: StrEnum` (`mda`, `risk_factors`)
- `ChangeType: StrEnum` (`added`, `removed`, `modified`)
- `Severity: StrEnum` (`major`, `minor`)
- `NewFilingSection`, `FilingSectionRecord`, `NewLanguageDiff`, `LanguageDiffRecord`

`app/memory/repository.py` gains:
- `insert_filing_sections(rows: Iterable[NewFilingSection]) -> int`
- `update_section_embeddings(rows: Iterable[tuple[int, list[float], str]]) -> int`
- `get_prior_quarter_sections(*, ticker: str, section_kind: SectionKind, before: date) -> Sequence[FilingSectionRecord]`
- `insert_language_diffs(rows: Iterable[NewLanguageDiff]) -> int`
- `list_language_diffs_for_filing(accession_number: str) -> Sequence[LanguageDiffRecord]`

Repository methods do not commit; the per-node session in `app/graph.py` owns the transaction boundary.

## 5. Components

### 5.1 EDGAR client extension

`app/tools/edgar.py` gains:

```python
async def get_filing_document(
    self, *, cik: str, accession_number: str, primary_document: str
) -> str:
    """Fetch the primary HTML body of a filing from EDGAR archives."""
```

Uses a separate base URL (`https://www.sec.gov`) because the archives are not on `data.sec.gov`. Reuses the same `_RateLimiter`, tenacity retry policy, and `User-Agent` header.

### 5.2 Section parser

`app/tools/sections.py` — pure function, no I/O:

```python
def parse_sections(html: str, *, form: str) -> list[ParsedSection]
```

Returns `ParsedSection(kind=SectionKind.MDA, paragraphs=[...])` etc. Strategy:

1. BeautifulSoup with `lxml` parser → flat text with paragraph boundaries preserved. `<table>` is collapsed to a single sentinel paragraph (financial tables are already captured via XBRL).
2. Anchor regex over the flat text:
   - 10-Q MD&A: `r"^\s*item\s+2\.?\s+management.?s discussion"`
   - 10-K MD&A: `r"^\s*item\s+7\.?\s+management.?s discussion"`
   - 10-Q Risk Factors updates: `r"^\s*item\s+1a\.?\s+risk factors"`
   - 10-K Risk Factors: `r"^\s*item\s+1a\.?\s+risk factors"`
3. End-of-section anchor: next `^\s*item\s+\d` line.
4. Paragraph split: blank-line separated, normalised whitespace.
5. Filter: drop paragraphs < 40 chars (boilerplate) or > 4000 chars (residual tables).

Note: 10-Q Item 1A only appears when the company has material updates to the prior 10-K's risk factors. When absent, the parser returns no Risk Factors `ParsedSection` and the differ silently skips that section for the filing — this is normal, not degraded.

### 5.3 Embeddings client

`app/tools/embeddings.py`:

```python
class EmbeddingsClient:
    def __init__(
        self, *, api_key: SecretStr,
        repository_factory: Callable[[], Repository],
        model: str = "text-embedding-3-small",
        cassette_dir: Path | None = None,
    ): ...

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches; charges daily_llm_spend; cassette-replays in tests."""
```

Behavior:
- **Batching.** Up to 100 inputs per OpenAI call. Empty input list returns `[]` without an API call.
- **Cost guard.** Pre-flights worst-case projection (`tiktoken cl100k_base` token count × published per-token price) against `Repository.get_daily_spend`. Raises `DailyCostCapExceeded` if it would exceed `MAX_DAILY_LLM_COST_USD`. After the call, writes actual spend via `Repository.add_daily_spend`. Reuses the existing `daily_llm_spend` table to keep one cap covering all paid AI.
- **Cassette replay.** SHA-keyed (`sha256((model, sorted(texts)))`) cassette under `tests/fixtures/cassettes/embeddings/`. `REC=1` re-records.
- **Retry.** `tenacity` with 3 attempts, exponential backoff with jitter, on `openai.RateLimitError`, `openai.APITimeoutError`, `httpx.RequestError`.

### 5.4 Language differ node

Already described in §3.

### 5.5 Synthesiser / critic integration

- New prompt template `prompts/synthesizer/numbers_with_language_v1.md`. Includes an optional `<language_diffs>` block wrapped in `<source>` tags. Instructs the model to cite `[L#]` for quoted language changes. Version bumped via frontmatter; cassette keys move with prompt content per `app/llm/prompts.py`.
- `app/agents/citations.py` adds a third citation family `[L#]` indexing into the differ's emitted diff list. For `added` and `modified`-new, the indexed text is `diffs[i].text` / `diffs[i].current_text`. For `removed`, the indexed text is `diffs[i].prior_text`.
- Critic's deterministic validation passes:
  - Number validation: unchanged.
  - Citation resolution: widened to accept `[L#]`. The cited text must appear in the indexed paragraph as a substring after whitespace normalisation, OR overlap ≥ 90% by character-level similarity (use `difflib.SequenceMatcher.ratio`).

## 6. Error handling and security

| Failure mode | Behavior |
|---|---|
| EDGAR 5xx / network error | Tenacity retry (5 attempts, exponential + jitter). Terminal failure → log + degrade, not crash. |
| EDGAR 4xx (e.g., document not found) | Surface immediately as `EdgarHTTPError`; node catches and degrades. |
| Section parser found nothing | `WARNING section_parser_empty`, persist nothing, degrade. |
| OpenAI 5xx / rate limit | Tenacity retry (3 attempts). Terminal failure → paragraphs persisted with `embedding=NULL`, degrade. |
| Daily cost cap exceeded | `DailyCostCapExceeded` raised by `EmbeddingsClient`; node degrades. |
| pgvector extension missing | Migration fails at deploy time with a clear error; not a runtime failure mode. |

**Prompt injection.** Filing text is wrapped in `<source>` tags. The system prompt instructs the model to treat tag content as data. Paragraphs rendered into the prompt are normalised (collapse whitespace, strip control chars) and capped at 800 chars per paragraph; full text stays in the database.

**Secret scrubbing.** The loguru scrubber in `app/observability/logging.py` adds `openai_api_key` to its pattern list.

**No raw SQL in agent code.** All persistence flows through `Repository` methods.

## 7. Backfill operational tool

`app/scripts/backfill_language.py` — operator-triggered CLI that seeds the prior-quarter baseline.

```bash
uv run python -m app.scripts.backfill_language --ticker MSFT --quarters 4
uv run python -m app.scripts.backfill_language --quarters 4    # all active watchlist
```

Per ticker:
1. Fetch submissions, take the most recent N 10-Q + 10-K filings.
2. For each filing not already represented in `filing_sections`:
   a. Fetch the filing HTML.
   b. Parse sections.
   c. Embed paragraphs.
   d. Persist `filing_sections` rows.
3. Each filing commits its own transaction (resumable).
4. Print summary: tickers processed, filings parsed, paragraphs stored, embedding cost USD, elapsed time.

Properties:
- **Idempotent.** Skips filings already in `filing_sections`.
- **Resumable.** Per-filing commit boundary.
- **Cost-bounded.** Reuses `EmbeddingsClient`, so the daily cap stops a runaway backfill.

The script is **not** invoked from the graph or any startup hook. `docs/runbook.md` gets a "Phase 3 — first-time setup" entry pointing at it.

## 8. Testing

### 8.1 Unit tests

- `tests/unit/test_section_parser.py` — synthetic HTML fixtures plus 4-6 real EDGAR 10-Q excerpts (~50-200KB each) under `tests/fixtures/edgar_html/`. Asserts: section found, paragraph count within ±10% of hand-counted ground truth, boilerplate dropped, financial tables collapsed.
- `tests/unit/test_embeddings_client.py` — cassette-replayed batches, cost-cap fail-closed, retry on `openai.RateLimitError`.
- `tests/unit/test_language_differ.py` — alignment determinism, classification thresholds, severity assignment, cold-start `degraded=True` path, parallel-branch state-ownership rules.
- `tests/unit/test_critic.py` — extends existing critic tests with `[L#]` citation resolution and the 90% character-similarity tolerance.
- `tests/unit/test_citations.py` — extends with the third citation family.

### 8.2 Integration tests

- `tests/integration/test_graph.py` — extends the existing Phase 2 test. New stubs for the EDGAR archives endpoint and `EmbeddingsClient` returning deterministic vectors. Verifies the parallel branch executes, both specialists land state updates, and the final note quotes a seeded language change with a `[L#]` citation.
- `tests/integration/test_backfill_language.py` — runs the CLI against a stubbed EDGAR client. Verifies idempotency (second run is a no-op) and resumability (failure on filing N preserves filings 1..N-1).
- `tests/integration/test_migrations.py` — extends to assert the `vector` extension exists and the new tables roundtrip through `Base.metadata`.

### 8.3 The 80% recall gate

`tests/unit/test_recall_gate.py`, marked `@pytest.mark.slow`.

**Fixture.** `tests/fixtures/language_recall/`:
- 4 tickers × 4 consecutive quarters of `mda.html` and (where present) `risk_factors.html` extracted from real EDGAR 10-Q archives — up to ~32 files. Each pair of consecutive quarters within a ticker produces one comparable pair per section. Across 4 tickers × 3 consecutive pairs × up to 2 sections, the universe is up to 24 candidate pairs; 15 are chosen for labelling.
- `labels.yaml` listing the 15 chosen quarter-pairs, each with ground-truth changes: `{pair_id, change_type, paragraph_excerpt}`.

**Test logic.** For each of the 15 pairs:
1. Build a synthetic `AgentState` for the current quarter.
2. Run the differ end-to-end against an embedding cassette captured once via `REC=1`.
3. Collect emitted diffs.
4. For each label, mark it matched when there is a detected diff of the same `change_type` whose paragraph contains the label's `paragraph_excerpt`.

**Assertion.** `matched / total_labels >= 0.80`.

**Labelling protocol.** `docs/phase3-labeling.md` records the 15 pairs, the labeller (Paul Stanley), the date, and a one-paragraph rubric. Labels are append-only; if the rubric evolves, new labels are added rather than mutating old ones.

CI runs the fast unit suite on every PR (`pytest tests/unit -q -m "not slow"`). The slow suite runs as a second CI step on every PR (`pytest tests/unit -m slow`); the recall gate is therefore enforced on every PR. This is a deterministic unit test using captured embedding cassettes, distinct from the LLM-judge eval in `evals/` which remains nightly per PLAN.md §5.

### 8.4 Coverage

Target stays at 85% line coverage on `app/`. Phase 3 adds approximately 600 LOC of source; unit tests are written before the source per project convention.

## 9. Configuration

New env vars in `.env.example` and `app/config.py`:

| Key | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | `SecretStr`; for the embeddings client |
| `EMBEDDINGS_MODEL` | no | `text-embedding-3-small` | Model rotation knob |

The existing `MAX_DAILY_LLM_COST_USD` covers both Anthropic and OpenAI spend through the shared `daily_llm_spend` table.

## 10. Dependencies

`pyproject.toml` adds:

- `openai>=1.40` — embedding API client.
- `beautifulsoup4>=4.12` — HTML parsing.
- `lxml>=5.2` — fast parser backend for BeautifulSoup.
- `tiktoken>=0.7` — token counting for the OpenAI cost guard.
- `pgvector>=0.3` — SQLAlchemy `Vector` type integration.

`docker-compose.yml` switches the Postgres service to `pgvector/pgvector:pg16` if it is not already on that image. CI integration job pulls the same image.

## 11. Rollout

1. Land the migration (`alembic upgrade head` adds the extension and tables; existing rows unaffected).
2. Add the new env vars in the deploy environment.
3. Run `uv run python -m app.scripts.backfill_language --quarters 4` once to warm the baseline.
4. Resume live processing. The first event for each ticker after backfill is the first event where the differ contributes language diffs.

Rollback: forward-compatible per project convention. The `comparator -> synthesizer` edge stays valid; an older revision of `graph.py` simply does not invoke the differ. The new tables and column are additive and can be dropped via `alembic downgrade` if absolutely necessary.

## 12. Out of scope (deferred to later phases)

- Embedding-based retrieval across all prior quarters (currently we only compare against the immediate prior quarter). Phase 5b (peer reader) extends this.
- Section parsing for 8-K item 2.02. Phase 4 covers transcript ingestion; 8-K earnings releases are short and rarely contain MD&A-style language.
- LLM-side change classification (current design uses pure embedding similarity). Worth revisiting if recall stalls below 80% on the gate.
- Language-diff dashboard UI. Phase 6.

## 13. References

- `PLAN.md` §4 phase 3 row — the canonical scope and definition of done.
- `CLAUDE.md` Phase 3 paragraph — the working-instructions summary.
- `app/agents/comparator.py` — the parallel-track sibling whose patterns the differ mirrors.
- `app/agents/critic.py` and `app/agents/citations.py` — the integration points for the new `[L#]` citation family.
- `app/graph.py` — the compiled LangGraph the new node joins.
