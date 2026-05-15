"""LangGraph orchestrator.

Phase 1 wires a single specialist - the financial-extractor - so the graph
runtime, the :class:`AgentState` contract, and the per-node
:class:`StateUpdate` ownership all execute end-to-end. Phase 2 adds the
comparator and a minimal synthesiser; later phases bring in the language
differ, transcript analyser, peer reader, and critic.

The graph is constructed via :func:`build_graph` so callers can inject the
EDGAR client and a session factory at composition time - production wires
the real :class:`~app.tools.edgar.EdgarClient` and the process-wide session
factory; tests wire a stub.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.financial_extractor import OWNER as FINANCIAL_EXTRACTOR_OWNER
from app.agents.financial_extractor import extract_financials
from app.memory.repository import Repository
from app.models.state import AgentState
from app.tools.edgar import CompanyFactsResponse


class _SupportsCompanyFacts(Protocol):
    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse: ...


NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _make_financial_extractor_node(
    *,
    edgar: _SupportsCompanyFacts,
    session_factory: async_sessionmaker[AsyncSession],
) -> NodeFn:
    """Return the LangGraph-compatible node closure for financial extraction.

    The closure owns one session per invocation so concurrent node runs do
    not share a transaction. The session is committed on success and rolled
    back on any raised exception.
    """

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


def build_graph(
    *,
    edgar: _SupportsCompanyFacts,
    session_factory: async_sessionmaker[AsyncSession],
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the Phase 1 graph: ``START -> financial_extractor -> END``.

    The LangGraph generic parameters are intentionally ``Any``; the
    :class:`AgentState` schema is passed at construction time and the
    library's runtime checks enforce it. Pinning the four type variables
    here would tie the project to internal LangGraph types that still
    churn between minor releases.
    """
    builder: StateGraph[AgentState, Any, AgentState, AgentState] = StateGraph(AgentState)
    # LangGraph's ``StateNode`` union is a private alias that mypy cannot
    # reconcile with our plain ``async def node(state) -> dict[str, Any]``
    # signature, even though the runtime accepts it. The integration test in
    # ``tests/integration/test_graph.py`` exercises the wiring end-to-end.
    builder.add_node(  # type: ignore[call-overload]
        FINANCIAL_EXTRACTOR_OWNER,
        _make_financial_extractor_node(edgar=edgar, session_factory=session_factory),
    )
    builder.add_edge(START, FINANCIAL_EXTRACTOR_OWNER)
    builder.add_edge(FINANCIAL_EXTRACTOR_OWNER, END)
    return builder.compile()
