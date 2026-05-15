# Earnings Intelligence Agent

Autonomous multi-agent system that produces a fact-checked equity research note within ~15 minutes of an SEC earnings filing. Watches EDGAR for 10-Q/8-K filings on a configurable watchlist, parses filings, diffs language against prior quarters, analyzes earnings-call transcripts, and synthesizes a structured note delivered to email, Slack, or a dashboard. Coordinated through LangGraph on public data only.

## Status

Seven-phase build — see [`PLAN.md`](PLAN.md) for scope, architecture, and acceptance criteria.

**Phase 0 — project bootstrap: complete** (commit `0c228e9`, 2026-05-15).

**Phase 1 — Foundation: complete** (commit `ae2a5e2`, PR [#1](https://github.com/PaulStanley0211/earnings-intelligence-agent/pull/1), 2026-05-15).

**Phase 2 — Numbers track: in progress.**

In place from Phase 0:
- uv toolchain, `pyproject.toml`, `uv.lock`; ruff + mypy + pytest config; 85% coverage gate.
- Multi-stage [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml) (Postgres + pgvector on host port 5434 to avoid Windows host-Postgres collisions, Redis, FastAPI).
- Fail-fast Pydantic settings in [`app/config.py`](app/config.py) — every required env var validated at startup, including EDGAR User-Agent format.
- LLM client at [`app/llm/client.py`](app/llm/client.py) with SHA-keyed cassette replay and daily cost cap that fails closed (in-process; Postgres-backed counter now exists for Phase 2 to adopt).
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

Empty stubs still awaiting later phases — do not assume contents exist:
`app/delivery/`, `prompts/`, `evals/`, `tests/fixtures/cassettes/`.

**Phase 2 scope.** Extend the financial extractor beyond Phase 1's allowlist, add a consensus fetcher (Finnhub primary, yfinance fallback) under `app/tools/`, build a comparator that diffs reported vs consensus, write the first synthesizer (Opus) and critic v0 (deterministic number checks). Migrate the LLM client's daily cost cap from in-process to the `daily_llm_spend` table that Phase 1 already created. Wire the new nodes into [`app/graph.py`](app/graph.py) so the graph becomes `START -> financial_extractor -> comparator -> synthesizer -> critic -> END`. Land the first prompt templates under `prompts/` with frontmatter (model, temperature, body-SHA) and the first LLM cassettes under `tests/fixtures/cassettes/`. Done when the system auto-generates a numbers-only note with zero unverified numbers — every figure in the note must trace to a row in `financial_facts` or to a recorded consensus value, and the critic blocks any draft that fails that check.

## Tech stack

- Python 3.11+, managed with **uv** (not pip, not poetry)
- FastAPI, LangGraph
- Claude Opus (planner, synthesizer, critic) + Claude Sonnet (specialists)
- Postgres with pgvector, Redis + RQ
- SEC EDGAR, yfinance, Finnhub, arelle for XBRL
- pytest + hypothesis, loguru, OpenTelemetry
- Docker; deploys to Fly.io or Railway (Phase 7)

## When you start work

Read [`PLAN.md`](PLAN.md) first. Then consult:

- `app/graph.py` — compiled LangGraph and orchestration entry point
- `app/agents/` — node implementations per specialist
- `app/models/state.py` — `AgentState` contract between nodes
- `app/llm/client.py` — single LLM client (traced, cached, cassette-replay)
- `prompts/` — versioned prompt templates
- `tests/fixtures/` — golden filings, transcripts, recorded LLM cassettes
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
```

## Conventions

- **No Unicode emoji.** Anywhere — source, comments, commits, logs, notes, dashboard. The rule covers emoji from the Unicode emoji block. Functional typography like arrows, plus-minus, or box-drawing characters is permitted when design calls for it.
- **Quality bar.** Type hints on every public function. Docstrings on every module and non-trivial function. No dead code, no commented-out blocks, no `print` (use loguru). Functions under ~40 lines, modules under ~300. `ruff` and `mypy` must pass with zero warnings.
- **Code review on every change.** Solo mechanism: Claude in the IDE reviews using `docs/review-prompt.md`, then self-review after a 24-hour cooling-off. Reviewer verdict in the PR body. Full checklist in [`PLAN.md`](PLAN.md) §4.
- **uv only.** No `pip install` in scripts, Dockerfiles, or docs. Lock with `uv.lock`. Dockerfile uses `uv sync --frozen`.
- **Agent nodes are pure functions of `AgentState`.** Side effects only through `app/memory/` or `app/tools/`.
- **All LLM calls go through `app/llm/client.py`** — it traces, caches, enforces the daily cost cap, and supports cassette replay for tests. Never import the Anthropic SDK elsewhere.
- **Database access through `app/memory/` only.** Parameterized queries; no raw SQL in agent code.
- **Memory is append-only** for filings, transcripts, and notes. Only `commitments.status` is mutable (open → met / missed).
- **The critic runs on every synthesizer output.** No bypass path.
- **EDGAR client sends `User-Agent`** with contact email — SEC policy. Missing → startup fails fast.
- **Prompt injection defense.** External content (filings, transcripts) is wrapped in `<source>` tags; system prompts instruct the model to treat that content as data, not instructions.
- **Cost guard.** The LLM client enforces `MAX_DAILY_LLM_COST_USD` as a daily cap. Exceeded → calls fail closed.

## Required environment variables

`ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `EDGAR_USER_AGENT` (format: `"<name> <email>"`), `MAX_DAILY_LLM_COST_USD`, `LOG_LEVEL`, `ENVIRONMENT` (dev/staging/prod), `LLM_CACHE_DIR`, `EDGAR_POLL_INTERVAL_SECONDS`.

Optional delivery: `SMTP_HOST` / `SMTP_USER` / `SMTP_PASS`, `SLACK_WEBHOOK_URL`.

See [`.env.example`](.env.example).
