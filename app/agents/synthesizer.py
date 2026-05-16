"""The synthesiser agent node.

Reads :attr:`AgentState.financials`, :attr:`AgentState.comparisons`, and
:attr:`AgentState.language_diffs`, renders them into the
``synthesizer/numbers_with_language_v1`` prompt, calls Claude Opus through
:class:`~app.llm.client.LLMClient`, and writes the model's response to
:attr:`AgentState.draft_note`.

The synthesiser is the first node that consumes the database-backed daily
LLM spend cap: it always routes through :meth:`LLMClient.acomplete` with
the live :class:`~app.memory.repository.Repository`, so the cap survives
restarts and is shared across processes.

Phase 3 extends the prompt to include ``[L#]`` language-change citations
from the differ. Future phases swap the version string and add A/B
comparisons via ``evals/compare.py`` without changing this node.
"""

from __future__ import annotations

from typing import Any

from app.agents.citations import (
    CommitmentCitation,
    ComparisonCitation,
    FactCitation,
    LanguageCitation,
    QACitation,
    build_commitment_citations,
    build_comparison_citations,
    build_fact_citations,
    build_language_citations,
    build_qa_citations,
)
from app.llm.client import LLMClient, _SupportsDailySpend
from app.llm.prompts import load_prompt
from app.models.state import AgentState, StateUpdate
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER = "synthesizer"

_PROMPT_NUMBERS_ONLY = "synthesizer/numbers_v1"
_PROMPT_WITH_LANGUAGE = "synthesizer/numbers_with_language_v1"
_PROMPT_FULL = "synthesizer/full_v1"
_MAX_TOKENS = 1024


async def synthesize_note(
    state: AgentState,
    *,
    llm: LLMClient,
    repository: _SupportsDailySpend,
) -> StateUpdate:
    """Render the numbers-only note for the current filing.

    The function is a pure projection of ``state`` plus one Anthropic call;
    side effects on the database happen via ``repository.add_daily_spend``
    inside :meth:`LLMClient.acomplete`. The returned :class:`StateUpdate`
    increments ``cost_usd`` so the per-event cost ledger stays accurate
    even across retries from the critic.
    """
    prompt_name = _select_prompt(state)
    template = load_prompt(prompt_name)
    fact_citations = build_fact_citations(state.financials)
    comparison_citations = build_comparison_citations(state.comparisons)
    language_citations = build_language_citations(state.language_diffs)
    qa_citations = build_qa_citations(state.qa_pairs)
    commitment_citations = build_commitment_citations(state.commitments)
    facts_block = _render_facts_block(fact_citations)
    comparisons_block = _render_comparisons_block(comparison_citations)
    language_block = _render_language_block(language_citations)
    critic_feedback = _render_critic_feedback(state)

    substitutions: dict[str, str] = {
        "ticker": state.filing_event.ticker,
        "company_name": _company_name(state),
        "form": state.filing_event.form.value,
        "filed_at": state.filing_event.filed_at.isoformat(),
        "fiscal_year": str(_safe_get(state.comparisons, "fiscal_year") or ""),
        "fiscal_period": str(_safe_get(state.comparisons, "fiscal_period") or ""),
        "period_end": str(_safe_get(state.comparisons, "period_end") or ""),
        "facts_block": facts_block,
        "comparisons_block": comparisons_block,
        "language_block": language_block,
        "critic_feedback": critic_feedback,
    }
    if prompt_name == _PROMPT_FULL:
        substitutions["qa_pairs_block"] = _render_qa_pairs_block(qa_citations)
        substitutions["commitments_block"] = _render_commitments_block(
            commitment_citations
        )

    user_content = template.render(**substitutions)

    response = await llm.acomplete(
        prompt_version=f"{template.prompt_version}#{template.body_sha[:8]}",
        messages=[{"role": "user", "content": user_content}],
        repository=repository,
        model=template.model,
        temperature=template.temperature,
        max_tokens=_MAX_TOKENS,
    )

    _logger.bind(
        accession=state.filing_event.accession_number,
        ticker=state.filing_event.ticker,
        prompt_name=prompt_name,
        fact_citations=len(fact_citations),
        comparison_citations=len(comparison_citations),
        language_citations=len(language_citations),
        qa_citations=len(qa_citations),
        commitment_citations=len(commitment_citations),
        cost_usd=response.cost_usd,
        prompt_version=response.prompt_version,
        trace_id=current_trace_id(),
    ).info("synthesizer_complete")

    return StateUpdate(
        owner=OWNER,
        changes={
            "draft_note": response.text.strip(),
            "cost_usd": response.cost_usd,
        },
    )


def _select_prompt(state: AgentState) -> str:
    """Pick the synthesiser prompt for the data available on ``state``.

    Priority: transcript data wins over language-only, which wins over the
    numbers-only base. The selector is intentionally a small ladder; a
    fourth arm should be promoted to a registry rather than another
    elif branch.
    """
    if state.qa_pairs or state.commitments:
        return _PROMPT_FULL
    if state.language_diffs:
        return _PROMPT_WITH_LANGUAGE
    return _PROMPT_NUMBERS_ONLY


def _render_qa_pairs_block(citations: list[QACitation]) -> str:
    """Render Q&A citations as a numbered question/answer block."""
    if not citations:
        return "(no analyst Q&A pairs available)"
    lines: list[str] = []
    for c in citations:
        analyst = c.analyst_name or "unknown"
        lines.append(f"Q{c.identifier[1:]} (analyst: {analyst}): {c.question_text}")
        klass = c.answer_class or "unspecified"
        lines.append(f"A{c.identifier[1:]} [{klass}]: {c.answer_text}")
    return "\n".join(lines)


def _render_commitments_block(citations: list[CommitmentCitation]) -> str:
    """Render commitment citations as one numbered line per commitment."""
    if not citations:
        return "(no management commitments available)"
    lines: list[str] = []
    for c in citations:
        target = c.target_period or "not specified"
        lines.append(
            f"{c.identifier} (target: {target}): {c.commitment_text} "
            f'(source: "{c.source_quote}")'
        )
    return "\n".join(lines)


def _render_facts_block(citations: list[FactCitation]) -> str:
    """Render fact citations as a newline-joined markdown-friendly block."""
    if not citations:
        return "(no reported facts available)"
    return "\n".join(
        f"[{c.identifier}] {c.concept} = {c.value} {c.unit} "
        f"(period_end={c.period_end})"
        for c in citations
    )


def _render_comparisons_block(citations: list[ComparisonCitation]) -> str:
    """Render comparison citations as a newline-joined markdown-friendly block."""
    if not citations:
        return "(no consensus comparisons available)"
    lines: list[str] = []
    for c in citations:
        consensus = c.consensus_value if c.consensus_value is not None else "n/a"
        surprise = (
            f"{c.surprise_pct}%" if c.surprise_pct is not None else "no consensus"
        )
        lines.append(
            f"[{c.identifier}] {c.metric}: reported {c.reported_value} "
            f"{c.reported_unit}; consensus {consensus}; surprise {surprise}; "
            f"direction {c.direction or 'n/a'}"
        )
    return "\n".join(lines)


def _render_language_block(citations: list[LanguageCitation]) -> str:
    """Render language citations as a newline-joined markdown-friendly block."""
    if not citations:
        return "(no language changes detected this quarter)"
    lines: list[str] = []
    for c in citations:
        verb = {
            "added": "ADDED",
            "removed": "REMOVED",
            "modified": "MODIFIED",
        }.get(c.change_type, c.change_type.upper())
        lines.append(
            f"[{c.identifier}] section={c.section} change={verb} severity={c.severity}"
        )
        lines.append(f"    {c.text}")
    return "\n".join(lines)


def _render_critic_feedback(state: AgentState) -> str:
    """Format previous critic findings, if any, for the retry prompt."""
    if not state.critic_findings or state.critic_attempts == 0:
        return ""
    lines = ["Previous critic findings (you must address each):"]
    for finding in state.critic_findings:
        lines.append(f"- [{finding.severity}] {finding.message}")
    return "\n".join(lines) + "\n"


def _safe_get(payload: dict[str, Any] | None, key: str) -> Any:
    """Return ``payload[key]`` or ``None`` for absent payloads."""
    if not payload:
        return None
    return payload.get(key)


def _company_name(state: AgentState) -> str:
    """Best-effort company name pulled from synthesiser-visible context.

    Falls back to the ticker because Phase 2 does not pull the entity name
    onto :class:`~app.models.state.FilingEvent`; the watcher persists it on
    the watchlist row and a later phase adds it to the event.
    """
    return state.filing_event.ticker
