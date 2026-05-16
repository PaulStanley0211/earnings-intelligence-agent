r"""One-shot EDGAR poll for development and debugging.

Usage::

    # Use whatever watchlist is currently in the database.
    uv run python -m app.scripts.poll_once

    # Add (or refresh) a ticker, then poll. Idempotent.
    uv run python -m app.scripts.poll_once \
        --ticker NVDA --cik 1045810 --company-name "NVIDIA Corp"

The script wires a real :class:`~app.tools.edgar.EdgarClient` against the
project's Postgres test/dev database, runs one cycle of
:func:`app.agents.watcher.poll_once`, prints a one-line JSON summary, and
exits with a non-zero status on any failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.agents.watcher import poll_once
from app.config import get_settings
from app.memory.db import dispose_engine, get_session
from app.memory.repository import Repository
from app.observability.logging import configure_logging, get_logger, with_trace_id
from app.tools.edgar import EdgarClient

_logger = get_logger()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="poll_once",
        description="Run a single EDGAR poll cycle and dump the summary.",
    )
    parser.add_argument(
        "--ticker",
        help="If set, upsert this ticker into the watchlist before polling.",
    )
    parser.add_argument(
        "--cik",
        help="CIK for --ticker. Numeric; the script zero-pads to 10 digits.",
    )
    parser.add_argument(
        "--company-name",
        dest="company_name",
        help="Company name for --ticker.",
    )
    return parser.parse_args(argv)


async def _seed_watchlist_if_requested(args: argparse.Namespace) -> None:
    if args.ticker is None:
        return
    if not (args.cik and args.company_name):
        raise SystemExit("--ticker requires --cik and --company-name")
    cik_padded = args.cik.strip().lstrip("0").zfill(10)
    async with get_session() as session:
        await Repository(session).upsert_watchlist_entry(
            ticker=args.ticker.upper(),
            cik=cik_padded,
            company_name=args.company_name,
            active=True,
        )
        await session.commit()


async def _run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.log_level)

    await _seed_watchlist_if_requested(args)

    with with_trace_id() as trace_id:
        _logger.bind(trace_id=trace_id).info("poll_once_started")
        try:
            async with (
                EdgarClient(user_agent=settings.edgar_user_agent) as edgar,
                get_session() as session,
            ):
                result = await poll_once(edgar=edgar, session=session)
            sys.stdout.write(
                json.dumps(
                    {
                        "trace_id": trace_id,
                        "tickers_checked": result.tickers_checked,
                        "filings_found": result.filings_found,
                        "filings": [filing.model_dump() for filing in result.filings],
                    },
                    default=str,
                )
                + "\n"
            )
            return 0
        finally:
            await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    """Module-style entry point invoked via ``python -m app.scripts.poll_once``."""
    return asyncio.run(_run(argv))


if __name__ == "__main__":
    raise SystemExit(main())
