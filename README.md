# Earnings Intelligence Agent

Autonomous multi-agent system that produces a fact-checked equity research note within ~15 minutes of an SEC earnings filing. See [`PLAN.md`](PLAN.md) for full scope and [`CLAUDE.md`](CLAUDE.md) for development conventions.

## Status

Phase 0 (bootstrap) in progress.

## Local development

Prerequisites: Python 3.11+, `uv`, Docker Desktop, an Anthropic API key, a Finnhub API key.

```bash
# Install dependencies into .venv
uv sync --extra dev

# Configure environment
copy .env.example .env   # then fill in the required values

# Bring up Postgres + Redis + the app
docker compose up

# Run the unit tests
uv run pytest tests/unit -q

# Lint and type-check
uv run ruff check app/ tests/
uv run mypy app/
```

See [`PLAN.md`](PLAN.md) for the seven-phase build plan and acceptance gates.
