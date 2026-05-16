"""FastAPI dependency factories used by the upload-first routes."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.graph import build_graph
from app.llm.client import LLMClient
from app.memory.db import get_session_factory
from app.memory.repository import Repository
from app.tools.consensus import build_default_fetcher
from app.tools.edgar import EdgarClient
from app.tools.embeddings import EmbeddingsClient


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an ``AsyncSession`` bound to the request lifecycle.

    Commits on a clean exit; rolls back on exception so a failing route does
    not poison the connection pool.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_edgar_client() -> AsyncIterator[EdgarClient]:
    """Yield a request-scoped EDGAR client.

    ``EdgarClient`` is an async context manager - using a per-request scope
    avoids the singleton pitfalls of sharing one httpx client across the
    application lifetime.
    """
    settings = get_settings()
    async with EdgarClient(user_agent=settings.edgar_user_agent) as edgar:
        yield edgar


class _DailySpendAdapter:
    """Adapter satisfying ``EmbeddingsClient``'s ``_SupportsDailySpend`` protocol.

    The embeddings client expects a sync factory returning a repository
    instance; production needs a fresh session per spend operation so the
    long-lived embeddings singleton does not hold a single session open for
    the lifetime of the process. This adapter opens a one-shot session for
    each protocol call against the process-wide session factory.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Wire the session factory used to open one-shot sessions."""
        self._session_factory = session_factory

    async def get_daily_spend(self, day: date) -> Decimal:
        """Return today's recorded LLM spend (``0`` if no row yet)."""
        async with self._session_factory() as session:
            return await Repository(session).get_daily_spend(day)

    async def add_daily_spend(self, *, day: date, amount_usd: Decimal) -> Decimal:
        """Append ``amount_usd`` to the running total for ``day``."""
        async with self._session_factory() as session:
            try:
                total = await Repository(session).add_daily_spend(
                    day=day, amount_usd=amount_usd
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return total


_compiled_graph: CompiledStateGraph[Any, Any, Any, Any] | None = None
_unmanaged_clients: list[Any] = []


def get_compiled_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """Return the compiled graph singleton.

    Built lazily on first request using the production-configured clients.
    The :class:`EdgarClient` is instantiated without entering its async
    context manager - the underlying httpx client is built eagerly in
    ``__init__`` and lives for the lifetime of the application, mirroring
    the lifecycle of the compiled graph.

    The EDGAR and consensus clients hold long-lived ``httpx.AsyncClient``
    pools. Both are appended to ``_unmanaged_clients`` so the FastAPI
    lifespan shutdown can drain them via :func:`shutdown_compiled_graph`
    on graceful restart.

    Tests should override this dependency via
    ``app.dependency_overrides[get_compiled_graph]`` to inject a graph wired
    with stubs.
    """
    global _compiled_graph
    if _compiled_graph is None:
        settings = get_settings()
        session_factory = get_session_factory()
        edgar = EdgarClient(user_agent=settings.edgar_user_agent)
        consensus = build_default_fetcher(
            finnhub_api_key=settings.finnhub_api_key.get_secret_value()
        )
        spend_adapter = _DailySpendAdapter(session_factory)
        embeddings = EmbeddingsClient(
            api_key=settings.openai_api_key,
            repository_factory=lambda: spend_adapter,
            model=settings.embeddings_model,
            max_daily_cost_usd=settings.max_daily_llm_cost_usd,
        )
        llm = LLMClient()
        _compiled_graph = build_graph(
            edgar=edgar,
            consensus_fetcher=consensus,
            embeddings=embeddings,
            llm=llm,
            session_factory=session_factory,
        )
        _unmanaged_clients.append(edgar)
        _unmanaged_clients.append(consensus)
    return _compiled_graph


async def shutdown_compiled_graph() -> None:
    """Close the httpx clients owned by the lazy graph singleton.

    Called from ``app.main:_lifespan`` shutdown so production graceful-
    restart drains the EDGAR / consensus connection pools cleanly. Also
    resets the singleton so re-creating the application within a single
    process (e.g. integration test suites) yields a fresh graph rather
    than a stale one bound to closed clients.
    """
    global _compiled_graph
    for client in _unmanaged_clients:
        # Best-effort drain: a misbehaving client must not block shutdown.
        # Logging is intentionally avoided here because the logger may
        # already be torn down by this point.
        with contextlib.suppress(Exception):
            await client.aclose()
    _unmanaged_clients.clear()
    _compiled_graph = None
