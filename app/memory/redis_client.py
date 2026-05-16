"""Redis client wrapper.

Phase 1 only uses Redis for the ``/health`` readiness check; phase 2 picks
it up as the RQ backend. The wrapper keeps a single process-wide async
client so we have one connection pool to dispose at shutdown.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.config import get_settings

_client: Redis[str] | None = None


def get_redis() -> Redis[str]:
    """Return the process-wide :class:`redis.asyncio.Redis` (lazy)."""
    global _client
    if _client is None:
        _client = Redis.from_url(
            get_settings().redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _client


async def dispose_redis() -> None:
    """Close the cached Redis client so the next call rebuilds it."""
    global _client
    if _client is not None:
        # ``aclose`` is the documented async-shutdown call but the redis-py
        # type stubs only expose ``close``; the runtime ships both. mypy gets
        # an explicit hint here so the rest of the codebase stays untouched
        # when the stubs catch up.
        await _client.aclose()  # type: ignore[attr-defined]
    _client = None
