"""LangGraph orchestrator.

Phase 3 expands the pipeline to a fan-out / fan-in topology::

    START
      -> financial_extractor
      -> [comparator, language_differ]   (parallel)
      -> synthesizer                     (waits for both)
      -> critic
      -> (accepted | loop_exceeded -> END, rejected -> synthesizer)

Each node is a pure function of :class:`AgentState`. The session lifecycle
is owned by the node closure: one session per invocation, committed on
success and rolled back on any raised exception so concurrent runs cannot
contaminate each other's transactions.

LangGraph fans out from ``financial_extractor`` to both ``comparator`` and
``language_differ`` automatically because both receive an edge from it. The
``synthesizer`` receives edges from both specialists; LangGraph fires it once
after both upstreams have delivered their partial state updates. The two
specialists own disjoint ``AgentState`` fields (``comparisons`` and
``language_diffs`` respectively) so the reducer never conflicts.

Constructed via :func:`build_graph` so callers can inject the EDGAR client,
LLM client, consensus fetcher, embeddings client, and session factory.
Production wires the live clients; tests wire stubs and recorded LLM
cassettes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.comparator import OWNER as COMPARATOR_OWNER
from app.agents.comparator import compare_against_consensus
from app.agents.critic import OWNER as CRITIC_OWNER
from app.agents.critic import critique_draft
from app.agents.financial_extractor import OWNER as FINANCIAL_EXTRACTOR_OWNER
from app.agents.financial_extractor import extract_financials
from app.agents.language_differ import OWNER as LANGUAGE_DIFFER_OWNER
from app.agents.language_differ import diff_language
from app.agents.synthesizer import OWNER as SYNTHESIZER_OWNER
from app.agents.synthesizer import synthesize_note
from app.llm.client import LLMClient
from app.memory.repository import Repository
from app.memory.schemas import NewConsensusEstimate
from app.models.state import AgentState, CriticVerdict
from app.tools.edgar import CompanyFactsResponse


class _SupportsFilingDocument(Protocol):
    """Minimal EDGAR interface required by both the extractor and the differ."""

    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse: ...

    async def get_filing_document(
        self, *, cik: str, accession_number: str, primary_document: str
    ) -> str: ...


class _SupportsConsensusFetch(Protocol):
    async def fetch(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
        period_end: Any,
    ) -> list[NewConsensusEstimate]: ...


class _SupportsEmbed(Protocol):
    """Minimal interface required by the language differ."""

    @property
    def model(self) -> str: ...

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]: ...


NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _make_financial_extractor_node(
    *,
    edgar: _SupportsFilingDocument,
    session_factory: async_sessionmaker[AsyncSession],
) -> NodeFn:
    """Return the LangGraph node closure for the financial-extractor."""

    async def node(state: AgentState) -> dict[str, Any]:
        async with session_factory() as session:
            try:
                update = await extract_financials(
                    state, edgar=edgar, repository=Repository(session)
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return update.changes

    return node


def _make_comparator_node(
    *,
    consensus_fetcher: _SupportsConsensusFetch,
    session_factory: async_sessionmaker[AsyncSession],
) -> NodeFn:
    """Return the LangGraph node closure for the comparator."""

    async def node(state: AgentState) -> dict[str, Any]:
        async with session_factory() as session:
            try:
                update = await compare_against_consensus(
                    state,
                    consensus_fetcher=consensus_fetcher,
                    repository=Repository(session),
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return update.changes

    return node


def _make_language_differ_node(
    *,
    edgar: _SupportsFilingDocument,
    embeddings: _SupportsEmbed,
    session_factory: async_sessionmaker[AsyncSession],
) -> NodeFn:
    """Return the LangGraph node closure for the language differ."""

    async def node(state: AgentState) -> dict[str, Any]:
        async with session_factory() as session:
            try:
                update = await diff_language(
                    state,
                    edgar=edgar,
                    embeddings=embeddings,
                    repository=Repository(session),
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return update.changes

    return node


def _make_synthesizer_node(
    *,
    llm: LLMClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> NodeFn:
    """Return the LangGraph node closure for the synthesiser.

    The synthesiser commits its own session because it advances the daily
    LLM spend counter through :meth:`LLMClient.acomplete`; committing keeps
    that counter consistent even when the critic later rejects.
    """

    async def node(state: AgentState) -> dict[str, Any]:
        async with session_factory() as session:
            try:
                update = await synthesize_note(
                    state, llm=llm, repository=Repository(session)
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return update.changes

    return node


def _make_critic_node() -> NodeFn:
    """Return the LangGraph node closure for the deterministic critic."""

    async def node(state: AgentState) -> dict[str, Any]:
        return critique_draft(state).changes

    return node


def _critic_router(state: AgentState) -> str:
    """Decide whether to retry the synthesiser or end the run."""
    if state.critic_verdict is CriticVerdict.REJECTED:
        return SYNTHESIZER_OWNER
    return END


def build_graph(
    *,
    edgar: _SupportsFilingDocument,
    consensus_fetcher: _SupportsConsensusFetch,
    embeddings: _SupportsEmbed,
    llm: LLMClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the Phase 3 graph with parallel comparator / language_differ branches.

    Topology::

        START -> financial_extractor -> [comparator, language_differ] ->
        synthesizer -> critic -> {synthesizer | END}

    The synthesiser/critic loop is bounded by
    :data:`~app.agents.critic._MAX_CRITIC_ATTEMPTS`; when the budget is
    spent the critic emits a ``LOOP_EXCEEDED`` verdict and the router
    routes to ``END`` so the note is held for manual review.

    LangGraph's built-in fan-in fires ``synthesizer`` exactly once after
    both ``comparator`` and ``language_differ`` deliver their partial state
    updates. The two nodes own disjoint :class:`AgentState` fields
    (``comparisons`` and ``language_diffs``), so the reducer merges without
    conflict.
    """
    builder: StateGraph[AgentState, Any, AgentState, AgentState] = StateGraph(AgentState)
    builder.add_node(  # type: ignore[call-overload]
        FINANCIAL_EXTRACTOR_OWNER,
        _make_financial_extractor_node(edgar=edgar, session_factory=session_factory),
    )
    builder.add_node(  # type: ignore[call-overload]
        COMPARATOR_OWNER,
        _make_comparator_node(
            consensus_fetcher=consensus_fetcher, session_factory=session_factory
        ),
    )
    builder.add_node(  # type: ignore[call-overload]
        LANGUAGE_DIFFER_OWNER,
        _make_language_differ_node(
            edgar=edgar,
            embeddings=embeddings,
            session_factory=session_factory,
        ),
    )
    builder.add_node(  # type: ignore[call-overload]
        SYNTHESIZER_OWNER,
        _make_synthesizer_node(llm=llm, session_factory=session_factory),
    )
    builder.add_node(  # type: ignore[call-overload]
        CRITIC_OWNER,
        _make_critic_node(),
    )
    builder.add_edge(START, FINANCIAL_EXTRACTOR_OWNER)
    builder.add_edge(FINANCIAL_EXTRACTOR_OWNER, COMPARATOR_OWNER)
    builder.add_edge(FINANCIAL_EXTRACTOR_OWNER, LANGUAGE_DIFFER_OWNER)
    builder.add_edge(COMPARATOR_OWNER, SYNTHESIZER_OWNER)
    builder.add_edge(LANGUAGE_DIFFER_OWNER, SYNTHESIZER_OWNER)
    builder.add_edge(SYNTHESIZER_OWNER, CRITIC_OWNER)
    builder.add_conditional_edges(
        CRITIC_OWNER,
        _critic_router,
        {SYNTHESIZER_OWNER: SYNTHESIZER_OWNER, END: END},
    )
    return builder.compile()
