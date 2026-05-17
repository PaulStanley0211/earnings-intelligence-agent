"""The LLM critic node.

Runs sequentially after the deterministic critic, ONLY when the
deterministic critic returned ACCEPTED. Catches semantic issues
(internal contradictions, unsupported causal claims, hallucinated peer
or commitment references) that the deterministic layer cannot see.

The bounded retry budget is shared with the deterministic critic via
``state.critic_attempts`` (incremented by the deterministic critic).
The LLM critic does NOT increment attempts; if it rejects and a
re-synth happens, the deterministic critic on the next pass bumps the
counter.

A malformed-JSON response gets one in-node retry. A second failure
emits an error finding and rejects the note.
"""

from __future__ import annotations

import json
from typing import Final

from app.agents.citations import (
    build_commitment_citations,
    build_comparison_citations,
    build_fact_citations,
    build_language_citations,
    build_peer_citations,
    build_qa_citations,
)
from app.llm.client import LLMClient, LLMResponse
from app.llm.prompts import PromptTemplate, load_prompt
from app.memory.repository import Repository
from app.models.state import AgentState, CriticFinding, CriticVerdict, StateUpdate
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER: Final[str] = "critic"

_PROMPT_NAME: Final[str] = "critic/llm_v1"
_MAX_TOKENS: Final[int] = 2048


async def llm_critique(
    state: AgentState,
    *,
    llm: LLMClient,
    repository: Repository,
) -> StateUpdate:
    """Validate ``state.final_note`` via an Opus call against every source.

    Skips immediately when the deterministic critic did not produce ACCEPTED
    so the LLM critic never runs on a note the det-critic already held.
    Returns an empty :class:`StateUpdate` on skip so the caller sees no
    changes.
    """
    if state.critic_verdict is not CriticVerdict.ACCEPTED:
        return StateUpdate(owner=OWNER, changes={})
    if state.final_note is None:
        return StateUpdate(owner=OWNER, changes={})

    template = load_prompt(_PROMPT_NAME)
    user_message = _render_user_message(state, template)

    raw = await _call_with_retry(llm, template, user_message, repository=repository)
    findings, parsed_ok = _parse_response(raw)

    if not parsed_ok:
        finding = CriticFinding(
            layer="semantic",
            severity="error",
            message="llm critic returned unparseable response",
        )
        _logger.bind(
            ticker=state.filing_event.ticker,
            accession=state.filing_event.accession_number,
            trace_id=current_trace_id(),
        ).warning("llm_critic_unparseable")
        return StateUpdate(
            owner=OWNER,
            changes={
                "critic_findings": [*state.critic_findings, finding],
                "critic_verdict": CriticVerdict.REJECTED,
                "final_note": None,
            },
        )

    has_errors = any(f.severity == "error" for f in findings)

    _logger.bind(
        ticker=state.filing_event.ticker,
        accession=state.filing_event.accession_number,
        finding_count=len(findings),
        accepted=not has_errors,
        trace_id=current_trace_id(),
    ).info("llm_critic_complete")

    if has_errors:
        return StateUpdate(
            owner=OWNER,
            changes={
                "critic_findings": [*state.critic_findings, *findings],
                "critic_verdict": CriticVerdict.REJECTED,
                "final_note": None,
            },
        )

    return StateUpdate(
        owner=OWNER,
        changes={
            "critic_findings": [*state.critic_findings, *findings],
            "critic_verdict": CriticVerdict.ACCEPTED,
        },
    )


async def _call_with_retry(
    llm: LLMClient,
    template: PromptTemplate,
    user_message: str,
    *,
    repository: Repository,
    attempts: int = 2,
) -> str:
    """Call the LLM up to ``attempts`` times; return raw text on first valid JSON."""
    last_raw = ""
    for _ in range(attempts):
        last_raw = await _invoke_llm(llm, template, user_message, repository=repository)
        try:
            json.loads(last_raw)
            return last_raw
        except json.JSONDecodeError:
            continue
    return last_raw


async def _invoke_llm(
    llm: LLMClient,
    template: PromptTemplate,
    user_message: str,
    *,
    repository: Repository,
) -> str:
    """Issue a single ``acomplete`` call and extract the response text."""
    result = await llm.acomplete(
        prompt_version=f"{template.prompt_version}#{template.body_sha[:8]}",
        messages=[{"role": "user", "content": user_message}],
        repository=repository,
        model=template.model,
        temperature=template.temperature,
        max_tokens=_MAX_TOKENS,
    )
    if isinstance(result, LLMResponse):
        return result.text
    return str(result)


def _parse_response(raw: str) -> tuple[list[CriticFinding], bool]:
    """Return ``(findings, parsed_ok)`` from the LLM's JSON output.

    ``parsed_ok`` is ``False`` when ``raw`` is not valid JSON or lacks the
    expected ``findings`` list; the caller uses this to trigger the
    unparseable-response rejection path.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ([], False)

    if not isinstance(payload, dict):
        return ([], False)

    findings_raw = payload.get("findings")
    if not isinstance(findings_raw, list):
        return ([], False)

    findings: list[CriticFinding] = []
    for entry in findings_raw:
        if not isinstance(entry, dict):
            continue
        claim = str(entry.get("claim") or "")
        evidence = str(entry.get("evidence") or "")
        message = f"{claim} :: {evidence}" if claim or evidence else "semantic issue"
        findings.append(
            CriticFinding(
                layer="semantic",
                severity=str(entry.get("severity") or "warning"),
                message=message,
            )
        )
    return (findings, True)


def _render_user_message(state: AgentState, template: PromptTemplate) -> str:
    """Render the prompt body with all available source blocks."""
    fact_citations = build_fact_citations(state.financials)
    comparison_citations = build_comparison_citations(state.comparisons)
    language_citations = build_language_citations(state.language_diffs)
    qa_citations = build_qa_citations(state.qa_pairs)
    commitment_citations = build_commitment_citations(state.commitments)
    peer_citations = build_peer_citations(state.peer_context)

    facts_block = _render_block(
        [(c.identifier, f"{c.concept} = {c.value} {c.unit}") for c in fact_citations]
    )
    comparisons_block = _render_block(
        [(c.identifier, c.metric) for c in comparison_citations]
    )
    language_block = _render_block(
        [(c.identifier, c.text) for c in language_citations]
    )
    qa_block = _render_block(
        [(c.identifier, c.source_text) for c in qa_citations]
    )
    commitments_block = _render_block(
        [(c.identifier, c.source_text) for c in commitment_citations]
    )
    peers_block = _render_block(
        [(c.identifier, c.text) for c in peer_citations]
    )

    # Use simple string replacement instead of str.format() to avoid
    # interpreting the JSON examples in the prompt body as format tokens.
    substitutions = {
        "draft_note": state.final_note or "",
        "facts_block": facts_block,
        "comparisons_block": comparisons_block,
        "language_block": language_block,
        "qa_block": qa_block,
        "commitments_block": commitments_block,
        "peers_block": peers_block,
    }
    body = template.body
    for key, value in substitutions.items():
        body = body.replace(f"{{{key}}}", value)
    return body


def _render_block(items: list[tuple[str, str]]) -> str:
    """Format citation items as ``[ID] text`` lines, or ``(empty)`` placeholder."""
    if not items:
        return "(empty)"
    return "\n".join(f"[{ident}] {text}" for ident, text in items)
