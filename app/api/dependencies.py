"""FastAPI dependency factories used by the upload-first routes."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.memory.db import get_session_factory
from app.tools.edgar import EdgarClient


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
