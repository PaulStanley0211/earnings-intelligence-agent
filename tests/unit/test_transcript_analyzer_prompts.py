"""Smoke tests for the transcript_analyzer prompt templates."""

from __future__ import annotations

from app.llm.prompts import load_prompt


def test_extract_v1_loads() -> None:
    """The extract_v1 template parses, frontmatter is correct, body renders."""
    tpl = load_prompt("transcript_analyzer/extract_v1")
    assert tpl.version == "v1"
    assert tpl.model == "claude-sonnet-4-6"
    assert tpl.temperature == 0.0
    rendered = tpl.render(transcript_text="Operator: Welcome.")
    assert "Operator: Welcome." in rendered
    # No stray unsubstituted placeholders.
    assert "{transcript_text}" not in rendered


def test_reconcile_v1_loads() -> None:
    """The reconcile_v1 template parses and renders with both placeholders."""
    tpl = load_prompt("transcript_analyzer/reconcile_v1")
    assert tpl.version == "v1"
    assert tpl.model == "claude-sonnet-4-6"
    assert tpl.temperature == 0.0
    rendered = tpl.render(
        transcript_text="Q&A session begins.",
        prior_commitments_block="1) Azure margin +100 bps next quarter",
    )
    assert "Q&A session begins." in rendered
    assert "Azure margin +100 bps next quarter" in rendered
    assert "{transcript_text}" not in rendered
    assert "{prior_commitments_block}" not in rendered


def test_both_prompts_have_stable_body_sha() -> None:
    """Cassette stability - sha256 of the body is deterministic per file content."""
    extract = load_prompt("transcript_analyzer/extract_v1")
    reconcile = load_prompt("transcript_analyzer/reconcile_v1")
    assert len(extract.body_sha) == 64
    assert len(reconcile.body_sha) == 64
    assert extract.body_sha != reconcile.body_sha
