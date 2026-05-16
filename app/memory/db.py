"""Async database engine and session helpers.

Every consumer of the memory layer obtains an :class:`~sqlalchemy.ext.asyncio.AsyncSession`
through :func:`get_session` or by wiring its own session factory via
:func:`build_engine`. The engine itself is process-wide and lazy: it is built
on first request and re-used until :func:`dispose_engine` is called.

The DSN is read from :class:`~app.config.Settings`. We rewrite a plain
``postgresql://`` URL to ``postgresql+psycopg://`` so the async psycopg
driver is picked up; this matches the DSN format checked by the config
validator and used by docker-compose.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _async_url(url: str) -> str:
    """Promote a plain Postgres DSN to the async psycopg driver.

    Settings validation already restricts ``DATABASE_URL`` to a Postgres URL,
    so we only need to ensure SQLAlchemy picks the async driver.
    """
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def build_engine(*, url: str | None = None, echo: bool = False) -> AsyncEngine:
    """Construct a fresh :class:`AsyncEngine`.

    Tests use this directly so they can recreate the schema against a known
    DSN. Application code should call :func:`get_engine` instead so the
    process shares one connection pool.
    """
    effective = _async_url(url or get_settings().database_url)
    return create_async_engine(
        effective,
        echo=echo,
        future=True,
        pool_pre_ping=True,
    )


def get_engine() -> AsyncEngine:
    """Return the process-wide :class:`AsyncEngine` (lazy)."""
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the lazy session factory bound to :func:`get_engine`."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an :class:`AsyncSession` from the process-wide factory."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def dispose_engine() -> None:
    """Close the process-wide engine and drop cached factory references.

    Called from FastAPI's lifespan shutdown and from tests that need to reset
    pool state between cases.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
