"""FastAPI application entry point.

Wires the API router, configures structured logging and tracing, and exposes
the resulting ``app`` for Uvicorn (``uvicorn app.main:app``).

The agent graph itself is wired in by Phase 1; this module's only job in
Phase 0 is to give Docker something to serve so the deployment topology can be
validated end-to-end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.advise import router as advise_router
from app.api.health import router as health_router
from app.api.upload import router as upload_router
from app.config import get_settings
from app.observability.logging import configure_logging, get_logger
from app.observability.tracing import configure_tracing


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Configure observability on startup; close DB + Redis pools on shutdown."""
    settings = get_settings()
    configure_logging(level=settings.log_level)
    configure_tracing(environment=settings.environment.value)
    get_logger().bind(version=__version__, environment=settings.environment.value).info(
        "app_startup"
    )
    try:
        yield
    finally:
        from app.memory.db import dispose_engine
        from app.memory.redis_client import dispose_redis

        await dispose_engine()
        await dispose_redis()
        get_logger().info("app_shutdown")


def create_app() -> FastAPI:
    """Build and return the FastAPI application instance."""
    app = FastAPI(
        title="Earnings Intelligence Agent",
        version=__version__,
        description=(
            "Autonomous multi-agent system that produces a fact-checked equity "
            "research note within minutes of an SEC earnings filing."
        ),
        lifespan=_lifespan,
    )
    app.include_router(health_router)
    app.include_router(advise_router)
    app.include_router(upload_router)
    return app


app = create_app()
