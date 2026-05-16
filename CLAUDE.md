# Earnings Intelligence Agent

Multi-agent equity-research assistant. Users pick a US ticker; the agent uses a live EDGAR query to tell them which SEC documents to upload and where to download each from; the user uploads; the same multi-agent pipeline (financial extractor, comparator, language differ, transcript analyzer, synthesizer, deterministic critic) runs over the upload; a citation-enforced chat surface lets the user query the resulting structured analysis. The autonomous EDGAR watcher built in Phase 1 survives as an opt-in eval / demo mode that preserves the "research note within 15 minutes of a filing" claim as a quantitative property verified nightly against a fixed eval set. Coordinated through LangGraph on public data only.

The product direction is locked in the upload-first design spec at [`docs/superpowers/specs/2026-05-16-upload-first-pivot-design.md`](docs/superpowers/specs/2026-05-16-upload-first-pivot-design.md). Read it before touching Phase 4+ scope.

## Status

Seven-phase build — see [`PLAN.md`](PLAN.md) for scope, architecture, and acceptance criteria.

**Phase 0 — project bootstrap: complete** (commit `0c228e9`, 2026-05-15).

**Phase 1 — Foundation: complete** (commit `ae2a5e2`, PR [#1](https://github.com/PaulStanley0211/earnings-intelligence-agent/pull/1), 2026-05-15).

**Phase 2 — Numbers track: complete** (2026-05-15).

**Phase 3 — Language differ: complete** (commit `ad3b159`, 2026-05-16).

**Phase 4A — Upload infrastructure: complete** (commit `cb6579d`, 2026-05-16).

**Phase 4 — Upload intake + transcript analyzer: in progress** (2026-05-16 onward, executing under the upload-first pivot design spec). Old Phase 4 scope (third-party transcript scraping) was scrapped in favor of user-supplied transcripts + a document-advisor agent that uses the Phase 1 EDGAR client to tell users exactly which filings to fetch.

In place from Phase 0:
- uv toolchain, `pyproject.toml`, `uv.lock`; ruff + mypy + pytest config; 85% coverage gate.
- Multi-stage [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml) (Postgres + pgvector on host port 5434 to avoid Windows host-Postgres collisions, Redis, FastAPI).
- Fail-fast Pydantic settings in [`app/config.py`](app/config.py) — every required env var validated at startup, including EDGAR User-Agent format.
- LLM client at [`app/llm/client.py`](app/llm/client.py) with SHA-keyed cassette replay and a daily cost cap that fails closed. Sync `complete()` uses an in-process counter; async `acomplete()` reads/writes the Postgres-backed `daily_llm_spend` table (Phase 2).
- `AgentState` contract with per-node `StateUpdate` field ownership in [`app/models/state.py`](app/models/state.py).
- Loguru JSON logging with trace-id propagation and secret scrubbing; OpenTelemetry tracing scaffolding in [`app/observability/`](app/observability/).
- GitHub Actions CI: ruff, mypy, unit, integration (services), pip-audit, plus a nightly eval workflow.
- Solo review prompt and ops playbook stubs: [`docs/review-prompt.md`](docs/review-prompt.md), [`docs/runbook.md`](docs/runbook.md).

Added in Phase 1:
- **Memory layer** ([`app/memory/`](app/memory/)). SQLAlchemy 2.x async ORM models for `filings`, `financial_facts`, `watchlist`, `edgar_poll_log`, `daily_llm_spend` ([`models.py`](app/memory/models.py)); detached Pydantic DTOs ([`schemas.py`](app/memory/schemas.py)); async engine + session factory ([`db.py`](app/memory/db.py)); a single :class:`Repository` ([`repository.py`](app/memory/repository.py)) — all DB access goes through it; Redis async wrapper ([`redis_client.py`](app/memory/redis_client.py)).
- **First Alembic migration** at [`migrations/versions/20260515_1933_0001_phase1_schema.py`](migrations/versions/20260515_1933_0001_phase1_schema.py) — hand-written, hand-reviewable. `migrations/env.py` now binds `target_metadata = Base.metadata`.
- **EDGAR client** at [`app/tools/edgar.py`](app/tools/edgar.py) — async httpx, token-bucket rate limit (≤10 rps), tenacity exponential backoff on 5xx and network errors, contact-email User-Agent validated at construction, typed responses (`SubmissionsResponse`, `CompanyFactsResponse`, `RecentFiling`).
- **XBRL track via companyfacts JSON** at [`app/tools/companyfacts.py`](app/tools/companyfacts.py). The `arelle` raw-XBRL fallback in [`docs/runbook.md`](docs/runbook.md) is deferred to Phase 2.
- **First agent node**: `financial_extractor` at [`app/agents/financial_extractor.py`](app/agents/financial_extractor.py), a pure function of `AgentState` returning a typed `StateUpdate`.
- **EDGAR watcher** at [`app/agents/watcher.py`](app/agents/watcher.py): `poll_once(...)` for one-shot use, `watch_forever(...)` for the production service. Idempotent (filings checkpointed by accession), records every cycle to `edgar_poll_log`.
- **LangGraph skeleton** at [`app/graph.py`](app/graph.py) — `START -> financial_extractor -> END`. Compiled and invoked end-to-end in tests/integration.
- **CLI**: `uv run python -m app.scripts.poll_once [--ticker T --cik C --company-name N]` ([`app/scripts/poll_once.py`](app/scripts/poll_once.py)).
- **`/health` upgraded** ([`app/api/health.py`](app/api/health.py)): real Postgres `SELECT 1`, Redis ping, and a 5-minute freshness check on the most recent EDGAR poll. DB outage → HTTP 503; Redis or stale watcher → 200 with `status: degraded`.

Gate evidence at Phase 1 close: ruff clean, mypy clean (28 source files), 62 tests green (44 unit + 18 integration), `coverage report` line coverage 85.22%.

Added in Phase 2:
- **Extended XBRL concept allowlist** ([`app/tools/companyfacts.py`](app/tools/companyfacts.py)). Beyond Phase 1's income-statement headline set, the synthesiser now sees share counts, operating-expense detail, balance-sheet liquidity (current assets/liabilities, long-term debt, inventory), and operating/investing/financing cash flow.
- **Consensus tables and migration** ([`migrations/versions/20260515_2230_0002_phase2_schema.py`](migrations/versions/20260515_2230_0002_phase2_schema.py)). Two new tables: `consensus_estimates` (one row per `(ticker, fiscal_year, fiscal_period, metric, source)`) and `comparisons` (one row per filing/metric capturing reported, consensus, surprise, and direction). ORM models in [`app/memory/models.py`](app/memory/models.py); DTOs in [`app/memory/schemas.py`](app/memory/schemas.py); repository methods on [`app/memory/repository.py`](app/memory/repository.py).
- **Consensus fetcher** ([`app/tools/consensus.py`](app/tools/consensus.py)). Two-tier strategy: Finnhub primary (`/stock/eps-estimate`, `/stock/revenue-estimate`), yfinance fallback. Both providers are Protocol-shaped for test injection. Per-source rows coexist so the comparator can prefer Finnhub when both are present.
- **Comparator node** ([`app/agents/comparator.py`](app/agents/comparator.py)). Maps us-gaap concepts to comparator metrics (`revenue`, `eps_diluted`, `eps_basic`, `net_income`), pulls consensus, persists `consensus_estimates` and `comparisons` rows, and emits a structured summary into `AgentState.comparisons`. Direction band: ±0.5 percent is `in_line`, beyond is `beat`/`miss`. Owns the `comparisons` field via `_FIELD_OWNERS` in [`app/models/state.py`](app/models/state.py).
- **Prompt template loader** ([`app/llm/prompts.py`](app/llm/prompts.py)). Parses YAML-ish frontmatter (`version`, `model`, `temperature`), computes a body-SHA so cassette keys move with prompt content, supports `{key}`-style placeholders.
- **First prompt templates**: `synthesizer/numbers_v1.md` (Opus, temperature 0.0, citation-first contract) and `critic/numbers_v0.md` (deterministic, documented for symmetry).
- **Synthesiser node** ([`app/agents/synthesizer.py`](app/agents/synthesizer.py)). Renders financials + comparisons into the Opus prompt; calls `LLMClient.acomplete` so the new Postgres-backed cost cap applies. Wraps source data in `<source>` tags per PLAN.md §3. On retry, appends previous critic findings to the user message.
- **Deterministic critic v0** ([`app/agents/critic.py`](app/agents/critic.py)). Parses every number from the draft note, demands an adjacent `[F#]`/`[C#]` citation, resolves it against a shared citation index ([`app/agents/citations.py`](app/agents/citations.py)) used by both synthesiser and critic, and validates the cited value within metric-appropriate tolerance (1 percent relative for currency, 0.01 absolute for per-share, 0.05 absolute for percentages). Bounded retry at 3 attempts; otherwise emits `loop_exceeded`.
- **Async LLM cost cap** ([`app/llm/client.py`](app/llm/client.py)). The new `acomplete` method reads `daily_llm_spend` via `Repository.get_daily_spend`, fails closed when adding the worst-case projection would exceed the configured cap, runs the sync Anthropic SDK in a worker thread to avoid blocking the event loop, then commits actual spend via `Repository.add_daily_spend`. The in-process counter remains for the sync `complete()` path used by lower-stakes tests.
- **Updated LangGraph** ([`app/graph.py`](app/graph.py)). Compiled as `START -> financial_extractor -> comparator -> synthesizer -> critic -> {synthesizer | END}`. The critic-to-synthesizer conditional edge enables the bounded retry loop without manual orchestration.

Gate evidence at Phase 2 close: ruff clean, mypy clean (34 source files), 105 tests green (81 unit + 24 integration), `coverage report` line coverage 86 percent. `pip-audit` reports no known vulnerabilities.

Added in Phase 3:
- **Section parser** for 10-Q / 10-K MD&A and Risk Factors ([`app/tools/sections.py`](app/tools/sections.py)). Heuristic BeautifulSoup + lxml flatten, regex anchors for `Item 2` / `Item 7` / `Item 1A`, 40-4000 char paragraph filter.
- **OpenAI embeddings client** ([`app/tools/embeddings.py`](app/tools/embeddings.py)) with SHA-keyed cassette replay, batching, tenacity retries, and a shared daily-cost cap via the existing `daily_llm_spend` table.
- **`language_differ` agent node** ([`app/agents/language_differ.py`](app/agents/language_differ.py)) running in parallel with `comparator`. Cold-start degrades cleanly with `degraded=True`.
- **Two new tables** `filing_sections` (pgvector `Vector(1536)`) and `language_diffs` plus the migration at [`migrations/versions/20260515_2330_0003_phase3_schema.py`](migrations/versions/20260515_2330_0003_phase3_schema.py).
- **Backfill CLI** at [`app/scripts/backfill_language.py`](app/scripts/backfill_language.py) — operator-triggered, idempotent, resumable.
- **Synthesiser prompt v2** with `[L#]` citations at [`prompts/synthesizer/numbers_with_language_v1.md`](prompts/synthesizer/numbers_with_language_v1.md); critic resolves them with 90% character-similarity tolerance.
- **80% recall gate** at [`tests/unit/test_recall_gate.py`](tests/unit/test_recall_gate.py) with 15 labelled quarter pairs (synthetic, can be replaced with real EDGAR HTML per [`docs/phase3-labeling.md`](docs/phase3-labeling.md)).

Gate evidence at Phase 3 close: ruff clean, mypy clean (38+ source files), all unit + integration tests green, `coverage report` line coverage >= 85 percent. `pip-audit` reports no known vulnerabilities.

Added in Phase 4A:
- **Document parser** at [`app/tools/documents.py`](app/tools/documents.py) — PDF + plain-text intake, scanned-PDF rejection, SHA-256 content hashing.
- **EDGAR advisor** at [`app/tools/advisor.py`](app/tools/advisor.py) plus the agent wrapper at [`app/agents/document_advisor.py`](app/agents/document_advisor.py).
- **Upload intake node** at [`app/agents/upload_intake.py`](app/agents/upload_intake.py) — idempotent on SHA-256 even under concurrent-insert races; emits `FilingEvent` with `source=FilingEventSource.UPLOAD`.
- **API routes**: `POST /api/advise` returns the upload checklist, `POST /api/upload` runs the full Phase 1-3 pipeline on the uploaded PDF/text and returns the structured analysis, `POST /api/chat` reserved with a 501 stub for Phase 6.
- **Watcher gated** behind `WATCHER_MODE_ENABLED` (default `false`); `/health` reports `edgar_watcher` as `not_applicable` when the flag is off.
- **Migration** `0004_phase4a_uploaded_documents` adds the append-only `uploaded_documents` table with SHA-256 dedupe.
- **Graceful shutdown**: `app/main.py`'s lifespan now closes the singleton EDGAR + Finnhub httpx clients on shutdown via `shutdown_compiled_graph()`.
- **Sample fixtures**: MSFT 8-Ks at [`tests/fixtures/uploaded_pdfs/`](tests/fixtures/uploaded_pdfs/).

Gate evidence at Phase 4A close: ruff clean, mypy clean (46 source files), 208 unit tests + 47 integration tests green (modulo the pre-existing `test_missing_anthropic_key_raises` env-leak flake), `coverage report` line coverage 88.15 percent. `pip-audit` reports no known vulnerabilities.

Empty stubs still awaiting later phases — do not assume contents exist:
`app/delivery/`, `evals/`.

## Tech stack

- Python 3.11+, managed with **uv** (not pip, not poetry)
- FastAPI, LangGraph
- Claude Opus (planner, synthesizer, critic) + Claude Sonnet (specialists)
- Postgres with pgvector, Redis + RQ
- SEC EDGAR, yfinance, Finnhub, arelle for XBRL
- pytest + hypothesis, loguru, OpenTelemetry
- Docker; deploys to Fly.io or Railway (Phase 7)

## When you start work

Read [`PLAN.md`](PLAN.md) first, then the upload-first design spec at [`docs/superpowers/specs/2026-05-16-upload-first-pivot-design.md`](docs/superpowers/specs/2026-05-16-upload-first-pivot-design.md) — that locks in the Phase 4+ direction. Then consult:

- `app/graph.py` — compiled LangGraph and orchestration entry point
- `app/agents/` — node implementations per specialist
- `app/models/state.py` — `AgentState` contract between nodes
- `app/llm/client.py` — single LLM client (traced, cached, cassette-replay)
- `prompts/` — versioned prompt templates
- `tests/fixtures/` — golden filings, transcripts, recorded LLM cassettes, sample uploaded PDFs (see `tests/fixtures/uploaded_pdfs/README.md`)
- `docs/runbook.md` — failure recovery playbook

## Common commands

```bash
# Install
uv sync --extra dev

# Local stack (Postgres + Redis + app)
docker compose up

# Tests
uv run pytest tests/unit -q                 # unit (uses cached cassettes)
uv run pytest tests/integration -q           # integration
REC=1 uv run pytest tests/integration -q     # re-record LLM cassettes
uv run python -m evals.run                   # system eval vs golden notes

# Lint and type-check
uv run ruff check app/ tests/
uv run mypy app/

# Migrations
uv run alembic upgrade head

# One-shot EDGAR poll (debug) - uses the persisted watchlist
uv run python -m app.scripts.poll_once

# Same, but seed/refresh a ticker first (idempotent)
uv run python -m app.scripts.poll_once \
    --ticker NVDA --cik 1045810 --company-name "NVIDIA Corp"

# Backfill 4 prior quarters of language sections (operator-run, once per ticker)
uv run python -m app.scripts.backfill_language --quarters 4
```

## Conventions

- **No Unicode emoji.** Anywhere — source, comments, commits, logs, notes, dashboard. The rule covers emoji from the Unicode emoji block. Functional typography like arrows, plus-minus, or box-drawing characters is permitted when design calls for it.
- **Quality bar.** Type hints on every public function. Docstrings on every module and non-trivial function. No dead code, no commented-out blocks, no `print` (use loguru). Functions under ~40 lines, modules under ~300. `ruff` and `mypy` must pass with zero warnings.
- **Code review on every change.** Solo mechanism: Claude in the IDE reviews using `docs/review-prompt.md`, then self-review after a 24-hour cooling-off. Reviewer verdict in the PR body. Full checklist in [`PLAN.md`](PLAN.md) §4.
- **uv only.** No `pip install` in scripts, Dockerfiles, or docs. Lock with `uv.lock`. Dockerfile uses `uv sync --frozen`.
- **Agent nodes are pure functions of `AgentState`.** Side effects only through `app/memory/` or `app/tools/`.
- **All LLM calls go through `app/llm/client.py`** — it traces, caches, enforces the daily cost cap, and supports cassette replay for tests. Never import the Anthropic SDK elsewhere.
- **Database access through `app/memory/` only.** Parameterized queries; no raw SQL in agent code.
- **Memory is append-only** for filings, transcripts, notes, and uploaded documents. Only `commitments.status` is mutable (open → met / missed).
- **The critic runs on every synthesizer output.** No bypass path.
- **EDGAR client sends `User-Agent`** with contact email — SEC policy. Missing → startup fails fast.
- **Prompt injection defense.** External content (filings, transcripts, uploaded documents) is wrapped in `<source>` tags; system prompts instruct the model to treat that content as data, not instructions.
- **Cost guard.** The LLM client enforces `MAX_DAILY_LLM_COST_USD` as a daily cap. Exceeded → calls fail closed.
- **Upload safety.** `POST /api/upload` accepts only `application/pdf` and `text/plain`, enforces a hard size cap, validates magic bytes, and rejects scanned PDFs with zero extractable characters via a clean error before invoking the pipeline.
- **Watcher mode is opt-in.** The Phase 1 EDGAR watcher only runs when `WATCHER_MODE_ENABLED=true`. It exists to feed `evals/`, not as the primary user-facing flow.

## Required environment variables

`ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `EDGAR_USER_AGENT` (format: `"<name> <email>"`), `MAX_DAILY_LLM_COST_USD`, `LOG_LEVEL`, `ENVIRONMENT` (dev/staging/prod), `LLM_CACHE_DIR`, `EDGAR_POLL_INTERVAL_SECONDS`.

Optional: `EMBEDDINGS_MODEL` (defaults to `text-embedding-3-small`), `WATCHER_MODE_ENABLED` (defaults to `false`; set `true` to enable the eval-mode EDGAR watcher), `MAX_UPLOAD_BYTES` (defaults to 25 MB).

Optional delivery: `SMTP_HOST` / `SMTP_USER` / `SMTP_PASS`, `SLACK_WEBHOOK_URL`.

See [`.env.example`](.env.example).
