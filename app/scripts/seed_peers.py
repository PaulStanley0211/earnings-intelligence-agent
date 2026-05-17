"""CLI: seed the ``peers`` table from data/peers.yaml.

Idempotent; safe to re-run.

Run: ``uv run python -m app.scripts.seed_peers``
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

from app.memory.db import get_session_factory
from app.memory.repository import Repository
from app.memory.schemas import PeerCreate
from app.observability.logging import configure_logging, get_logger

_logger = get_logger()


async def _seed(path: Path) -> int:
    """Insert peer rows from ``path`` and return an exit code."""
    if not path.exists():
        _logger.error(f"peers file not found: {path}")
        return 1
    rows = yaml.safe_load(path.read_text()) or []
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = Repository(session)
        for entry in rows:
            await repo.upsert_peer(
                PeerCreate(
                    ticker=entry["ticker"],
                    peer_ticker=entry["peer_ticker"],
                )
            )
        await session.commit()
    _logger.info(f"seeded {len(rows)} peer rows")
    return 0


def main() -> None:
    """Entry point for the seed_peers CLI."""
    configure_logging()
    path = Path("data/peers.yaml")
    code = asyncio.run(_seed(path))
    sys.exit(code)


if __name__ == "__main__":
    main()
