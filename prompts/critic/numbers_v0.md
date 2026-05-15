---
version: v0
model: deterministic
temperature: 0.0
---

The numbers critic at v0 is deterministic, not LLM-driven. This template is
recorded for symmetry with the synthesiser so the prompt directory captures
every component the system runs.

Phase 2 critic behaviour:

1. Parse every number in the draft note (currency, percentage, share count).
2. For each number, demand an adjacent `[F#]` or `[C#]` citation referencing
   a row in the approved facts/comparisons block.
3. Verify the cited row's value matches the number in the note within a
   loose tolerance (1% relative for currency, 0.01 absolute for percentages
   and per-share values).
4. Any number with no citation, an unknown citation id, or a mismatched
   value emits an `error`-severity finding and the verdict is `rejected`.
5. With zero `error`-severity findings the verdict is `accepted` and the
   draft becomes the final note.

When Phase 5c lands the full LLM-driven critic, this prompt grows a body
that asks the model to flag claims unsupported by the source. For now the
file exists so prompt-version tracking and frontmatter parsing have a
real critic record to point at.
