# Earnings Intelligence Agent

A multi-agent equity-research assistant. Pick a US-listed ticker; the agent uses a live SEC EDGAR query to tell you exactly which documents to upload and where to download each from; you upload them; a coordinated multi-agent pipeline (financial extractor, comparator, language differ, transcript analyzer, synthesizer, deterministic critic) produces a structured analysis; and a citation-enforced chat surface lets you ask follow-up questions.

The same pipeline runs autonomously against a fixed eval-set of tickers in an opt-in "eval / demo" mode, preserving the property that the system can deliver a research note within 15 minutes of an EDGAR filing — verified nightly.

See [`PLAN.md`](PLAN.md) for the full seven-phase build plan and acceptance gates. See [`docs/superpowers/specs/2026-05-16-upload-first-pivot-design.md`](docs/superpowers/specs/2026-05-16-upload-first-pivot-design.md) for the design rationale. See [`CLAUDE.md`](CLAUDE.md) for development conventions.

## Status

Phases 0-3 complete (foundation, numbers track, language differ). Phase 4 (upload intake + transcript analyzer) starting under the upload-first design.

## Local development

Prerequisites: Python 3.11+, `uv`, Docker Desktop, an Anthropic API key, an OpenAI API key (for embeddings), a Finnhub API key.

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
