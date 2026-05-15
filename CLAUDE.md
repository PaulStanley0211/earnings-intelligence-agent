# Earnings Intelligence Agent

Autonomous multi-agent system that produces a fact-checked equity research note within ~15 minutes of an SEC earnings filing. Watches EDGAR for 10-Q/8-K filings on a configurable watchlist, parses filings, diffs language against prior quarters, analyzes earnings-call transcripts, and synthesizes a structured note delivered to email, Slack, or a dashboard. Coordinated through LangGraph on public data only.

## Status

In development. Seven-phase build. See [`PLAN.md`](PLAN.md) — the source of truth for scope, architecture, and acceptance criteria.

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

# One-shot EDGAR poll (debug)
uv run python -m app.scripts.poll_once --ticker NVDA
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
