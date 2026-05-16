"""Unit tests for the FastAPI dependency factories in ``app.api.dependencies``.

These tests target the production-singleton wiring that integration tests
override via ``app.dependency_overrides``. Coverage here keeps the module
above the per-file gate without standing up real HTTP clients or databases.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api import dependencies as deps


@pytest.mark.asyncio
async def test_get_edgar_client_yields_and_closes() -> None:
    """``get_edgar_client`` opens a context-managed EdgarClient per request."""
    gen = deps.get_edgar_client()
    edgar = await gen.__anext__()
    try:
        assert edgar is not None
        # The user-agent comes from settings; sanity-check the type contract.
        assert hasattr(edgar, "aclose")
    finally:
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()


@pytest.mark.asyncio
async def test_daily_spend_adapter_get_returns_repository_value() -> None:
    """The adapter opens a one-shot session and returns the repo's value."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=session)

    adapter = deps._DailySpendAdapter(factory)

    fake_repo = MagicMock()
    fake_repo.get_daily_spend = AsyncMock(return_value=Decimal("1.23"))
    with patch.object(deps, "Repository", return_value=fake_repo):
        result = await adapter.get_daily_spend(date(2026, 5, 16))

    assert result == Decimal("1.23")
    fake_repo.get_daily_spend.assert_awaited_once_with(date(2026, 5, 16))


@pytest.mark.asyncio
async def test_daily_spend_adapter_add_commits_on_success() -> None:
    """``add_daily_spend`` commits the session and returns the running total."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    factory = MagicMock(return_value=session)

    adapter = deps._DailySpendAdapter(factory)
    fake_repo = MagicMock()
    fake_repo.add_daily_spend = AsyncMock(return_value=Decimal("4.50"))

    with patch.object(deps, "Repository", return_value=fake_repo):
        total = await adapter.add_daily_spend(
            day=date(2026, 5, 16), amount_usd=Decimal("0.10")
        )

    assert total == Decimal("4.50")
    session.commit.assert_awaited_once()
    session.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_daily_spend_adapter_add_rolls_back_on_error() -> None:
    """An exception in ``add_daily_spend`` triggers rollback then re-raises."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    factory = MagicMock(return_value=session)

    adapter = deps._DailySpendAdapter(factory)
    fake_repo = MagicMock()
    fake_repo.add_daily_spend = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch.object(deps, "Repository", return_value=fake_repo),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await adapter.add_daily_spend(
            day=date(2026, 5, 16), amount_usd=Decimal("0.10")
        )

    session.rollback.assert_awaited_once()
    session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_get_compiled_graph_is_singleton_and_shutdown_resets() -> None:
    """The compiled graph is built lazily, cached, and cleared on shutdown."""
    # Stash existing state so we restore it after the test.
    prev_graph = deps._compiled_graph
    prev_clients = list(deps._unmanaged_clients)
    deps._compiled_graph = None
    deps._unmanaged_clients.clear()

    fake_graph = MagicMock(name="compiled_graph")
    fake_edgar = MagicMock(aclose=AsyncMock())
    fake_consensus = MagicMock(aclose=AsyncMock())
    fake_embeddings = MagicMock()
    fake_llm = MagicMock()

    with (
        patch.object(deps, "EdgarClient", return_value=fake_edgar),
        patch.object(deps, "build_default_fetcher", return_value=fake_consensus),
        patch.object(deps, "EmbeddingsClient", return_value=fake_embeddings),
        patch.object(deps, "LLMClient", return_value=fake_llm),
        patch.object(deps, "build_graph", return_value=fake_graph) as build,
        patch.object(deps, "get_session_factory", return_value=MagicMock()),
    ):
        first = deps.get_compiled_graph()
        second = deps.get_compiled_graph()
        assert first is second is fake_graph
        # Built exactly once across two invocations.
        assert build.call_count == 1
        # Both unmanaged clients registered for shutdown.
        assert fake_edgar in deps._unmanaged_clients
        assert fake_consensus in deps._unmanaged_clients

    try:
        await deps.shutdown_compiled_graph()
        fake_edgar.aclose.assert_awaited_once()
        fake_consensus.aclose.assert_awaited_once()
        assert deps._compiled_graph is None
        assert deps._unmanaged_clients == []
    finally:
        deps._compiled_graph = prev_graph
        deps._unmanaged_clients[:] = prev_clients


@pytest.mark.asyncio
async def test_shutdown_compiled_graph_suppresses_close_errors() -> None:
    """A misbehaving client must not block shutdown."""
    prev_graph = deps._compiled_graph
    prev_clients = list(deps._unmanaged_clients)
    deps._compiled_graph = MagicMock()
    deps._unmanaged_clients.clear()

    bad_client: Any = MagicMock()
    bad_client.aclose = AsyncMock(side_effect=RuntimeError("network gone"))
    deps._unmanaged_clients.append(bad_client)

    try:
        # Should not raise despite the underlying aclose failure.
        await deps.shutdown_compiled_graph()
        assert deps._compiled_graph is None
        assert deps._unmanaged_clients == []
    finally:
        deps._compiled_graph = prev_graph
        deps._unmanaged_clients[:] = prev_clients
