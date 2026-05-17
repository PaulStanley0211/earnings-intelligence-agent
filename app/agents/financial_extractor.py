"""The financial-extractor agent node.

Phase 1 wires this one node into the LangGraph skeleton. It pulls the
companyfacts JSON for the filing's CIK, filters to the
:data:`~app.tools.companyfacts.DEFAULT_CONCEPT_ALLOWLIST`, persists the
resulting facts through the memory repository, and emits a typed
:class:`StateUpdate` summarising what landed.

The node has no LLM dependency, so it stays cheap and deterministic. Phase 2
introduces the comparator node that consumes these facts and asks Sonnet to
generate the narrative.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from app.memory.repository import Repository
from app.models.state import AgentState, FilingForm, StateUpdate
from app.observability.logging import current_trace_id, get_logger
from app.tools.companyfacts import DEFAULT_CONCEPT_ALLOWLIST, parse_company_facts
from app.tools.edgar import CompanyFactsResponse

_logger = get_logger()

OWNER = "financial_extractor"


class _SupportsCompanyFacts(Protocol):
    """Minimal protocol covering what :func:`extract_financials` needs.

    Production passes an :class:`~app.tools.edgar.EdgarClient`; tests pass a
    stub with the same shape.
    """

    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse: ...


async def extract_financials(
    state: AgentState,
    *,
    edgar: _SupportsCompanyFacts,
    repository: Repository,
    concepts: Iterable[str] | None = None,
) -> StateUpdate:
    """Pull and persist XBRL facts for the filing referenced by ``state``.

    ``concepts`` defaults to :data:`DEFAULT_CONCEPT_ALLOWLIST`; pass an empty
    iterable to disable the filter. Returns a :class:`StateUpdate` that the
    LangGraph reducer applies to ``state.financials``.

    Self-skips on ``TRANSCRIPT`` filings: user-uploaded earnings-call
    transcripts have no XBRL companyfacts to pull, so the node yields an
    empty update and lets the parallel ``transcript_analyzer`` carry the
    payload for that branch of the graph.
    """
    filing = state.filing_event
    if filing.form == FilingForm.TRANSCRIPT:
        return StateUpdate(owner=OWNER, changes={})
    allowlist = (
        DEFAULT_CONCEPT_ALLOWLIST
        if concepts is None
        else frozenset(concepts) or None
    )

    facts_response = await edgar.get_company_facts(cik=filing.cik)
    parsed = parse_company_facts(
        facts_response,
        accession_number=filing.accession_number,
        concepts=allowlist,
    )
    inserted = await repository.insert_financial_facts(filing.accession_number, parsed)

    by_concept: dict[str, Any] = {}
    for fact in parsed:
        by_concept.setdefault(fact.concept, []).append(
            {
                "unit": fact.unit,
                "value": str(fact.value),
                "period_start": fact.period_start.isoformat() if fact.period_start else None,
                "period_end": fact.period_end.isoformat(),
                "fiscal_year": fact.fiscal_year,
                "fiscal_period": fact.fiscal_period,
            }
        )

    summary: dict[str, Any] = {
        "source": "companyfacts",
        "parsed_count": len(parsed),
        "inserted_count": inserted,
        "concepts": sorted(by_concept.keys()),
        "by_concept": by_concept,
    }
    _logger.bind(
        trace_id=current_trace_id(),
        accession=filing.accession_number,
        ticker=filing.ticker,
        parsed_count=len(parsed),
        inserted_count=inserted,
    ).info("financial_extractor_complete")
    return StateUpdate(owner=OWNER, changes={"financials": summary})
