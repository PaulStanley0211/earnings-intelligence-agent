# Code review prompt (Claude-in-IDE)

Paste this prompt into the IDE assistant to review a diff. Use it on the first
pass at submission time and again after the 24-hour cooling-off period.
The reviewer must produce a verdict line and that verdict goes in the PR body.

---

You are reviewing a pull request for the Earnings Intelligence Agent. The
project rules live in `CLAUDE.md` and the architecture and acceptance gates
live in `PLAN.md`. Read both before reviewing if you have not already.

Use the following checklist. For each item, return one of: **pass**, **fail**,
or **n/a**, and a single sentence of evidence. End with a single verdict line:
`Verdict: approve | request_changes | block`.

## Conventions

1. No Unicode emoji anywhere in source, comments, commit messages, logs, or docs.
2. No `print`; logging goes through `loguru` via `app/observability/logging.py`.
3. Type hints on every public function; docstring on every module and every non-trivial function.
4. Functions under ~40 lines, modules under ~300.
5. `ruff check app/ tests/` passes with zero warnings.
6. `mypy app/` passes with zero warnings.

## Architecture rules

7. The Anthropic SDK is imported only by `app/llm/client.py`.
8. Database access only through `app/memory/` (no raw SQL in agent code).
9. Memory is append-only except `commitments.status`.
10. Every LLM call routes through `app/llm/client.py` and so is cassette-recorded,
    cost-tracked, and prompt-version-tagged.
11. External content (filings, transcripts) is wrapped in `<source>` tags in prompts;
    the system prompt instructs the model to treat that content as data.
12. EDGAR client sends `User-Agent` with a real contact email; startup fails fast
    when it is missing or malformed.
13. New agent nodes are pure functions of `AgentState` that return a typed
    `StateUpdate` mutating only their owned fields.

## Tests

14. Unit + integration tests added for the new code path.
15. Tests are meaningful (assert behaviour, not just that the code runs).
16. No LLM or network call escapes `app/llm/` or `app/tools/`. In tests, all LLM
    calls hit a cassette.
17. Line coverage on touched modules is at or above 85%.

## Operational

18. New failure modes have a row in `docs/runbook.md`.
19. Any new prompt is committed to `prompts/` with versioned frontmatter
    (model, temperature, body-SHA).
20. The phase gate from `PLAN.md` section 4 is met with reproducible evidence.

## Final

`Verdict: approve | request_changes | block`
Include a one-paragraph summary of the most important finding, even when
approving.
