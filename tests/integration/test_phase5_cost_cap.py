"""Phase 5 cost-cap regression: acomplete fails closed when daily spend exceeds cap.

Verifies that :meth:`~app.llm.client.LLMClient.acomplete` raises
:class:`~app.llm.client.CostCapExceeded` before touching the Anthropic API when
pre-loaded daily spend would push the next call past the configured cap.  No live
API call is made, so this test is free to run and deterministic.

The test bypasses the cassette replay layer (which fires before the cost-cap check
in acomplete) by setting ENVIRONMENT=dev so the ``CassetteMiss`` guard is skipped,
and injecting a sentinel Anthropic client that fails loudly if ever invoked,
confirming the cap check fires before the API call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.client import CostCapExceeded, LLMClient
from app.memory.db import build_engine
from app.memory.models import Base
from app.memory.repository import Repository

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TODAY = datetime.now(UTC).date()

# For claude-sonnet-4-6: output price = 15.0/1000 per 1k tokens.
# With max_tokens=1024, worst_case_cost = 1024/1000 * 0.015 = ~$0.01536.
# We set cap=$0.50 and pre-load $0.49 so any acomplete call (even 1 token) trips the cap.
_CAP_USD = "0.50"
_PRE_LOADED_SPEND = Decimal("0.49")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentinel_anthropic() -> MagicMock:
    """Return a stub Anthropic client that fails loudly if messages.create is called.

    This confirms the cost-cap check fires before the API call.
    """
    client = MagicMock()
    client.messages.create.side_effect = AssertionError(
        "Anthropic API must NOT be called when CostCapExceeded should have fired first."
    )
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def session_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Fresh schema with the vector extension; no seeded rows needed."""
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acomplete_fails_closed_when_daily_spend_exceeds_cap(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-loaded near-cap spend causes acomplete to raise CostCapExceeded.

    The cap check in :meth:`~app.llm.client.LLMClient.acomplete` reads the
    Postgres-backed ``daily_llm_spend`` table via :class:`~app.memory.repository.Repository`
    and raises *before* the Anthropic SDK is invoked.

    The cassette replay guard (which precedes the cap check) is bypassed by
    setting ENVIRONMENT=dev.  The injected Anthropic stub fails loudly if called,
    proving the cap check fires first.
    """
    from app.config import reset_settings_cache

    monkeypatch.setenv("MAX_DAILY_LLM_COST_USD", _CAP_USD)
    # dev environment skips the CassetteMiss guard so execution reaches the cap check.
    monkeypatch.setenv("ENVIRONMENT", "dev")
    reset_settings_cache()

    # Pre-load daily spend so the next call would exceed the cap.
    async with session_factory() as session:
        repo = Repository(session)
        await repo.add_daily_spend(day=_TODAY, amount_usd=_PRE_LOADED_SPEND)
        await session.commit()

    # Verify the pre-loaded value is present before the test assertion.
    async with session_factory() as session:
        repo = Repository(session)
        recorded = await repo.get_daily_spend(_TODAY)
    assert recorded == _PRE_LOADED_SPEND

    llm = LLMClient(
        cassette_dir=cassette_dir,
        anthropic_client=_sentinel_anthropic(),
    )

    async with session_factory() as session:
        repo_for_llm = Repository(session)
        with pytest.raises(CostCapExceeded) as exc_info:
            await llm.acomplete(
                prompt_version="test/cost_cap_v0",
                messages=[{"role": "user", "content": "hello"}],
                repository=repo_for_llm,
                model="claude-sonnet-4-6",
                max_tokens=1024,
            )

    error_msg = str(exc_info.value)
    assert "0.50" in error_msg, "CostCapExceeded message should mention the configured cap"
    assert "0.49" in error_msg, "CostCapExceeded message should mention the pre-loaded spend"


@pytest.mark.asyncio
async def test_daily_spend_not_modified_when_cap_exceeded(
    session_factory: async_sessionmaker[AsyncSession],
    fresh_settings: None,
    cassette_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When CostCapExceeded fires, the daily_llm_spend table must not be modified.

    The cap check raises before the API call, so no additional spend row should be
    inserted and the recorded total remains exactly the pre-loaded amount.
    """
    from app.config import reset_settings_cache

    monkeypatch.setenv("MAX_DAILY_LLM_COST_USD", _CAP_USD)
    monkeypatch.setenv("ENVIRONMENT", "dev")
    reset_settings_cache()

    async with session_factory() as session:
        repo = Repository(session)
        await repo.add_daily_spend(day=_TODAY, amount_usd=_PRE_LOADED_SPEND)
        await session.commit()

    llm = LLMClient(
        cassette_dir=cassette_dir,
        anthropic_client=_sentinel_anthropic(),
    )

    async with session_factory() as session:
        repo_for_llm = Repository(session)
        with pytest.raises(CostCapExceeded):
            await llm.acomplete(
                prompt_version="test/cost_cap_v0",
                messages=[{"role": "user", "content": "hello"}],
                repository=repo_for_llm,
                model="claude-sonnet-4-6",
                max_tokens=1024,
            )

    # Confirm the DB was not modified after the cap check raised.
    async with session_factory() as session:
        repo = Repository(session)
        spend_after = await repo.get_daily_spend(_TODAY)

    assert spend_after == _PRE_LOADED_SPEND, (
        f"daily_llm_spend must remain {_PRE_LOADED_SPEND} after CostCapExceeded; "
        f"got {spend_after}"
    )
