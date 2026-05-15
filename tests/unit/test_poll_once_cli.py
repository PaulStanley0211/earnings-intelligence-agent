"""Unit tests for the ``poll_once`` CLI argument surface.

The orchestration body of the script exercises real modules already covered
elsewhere (memory repository, EDGAR client, watcher). These tests focus on
the thin parts unique to the CLI: argument parsing and the early-exit
validation that prevents a half-specified watchlist seed from corrupting
the DB.
"""

from __future__ import annotations

import asyncio

import pytest

from app.scripts.poll_once import _parse_args, _seed_watchlist_if_requested


def test_parse_args_defaults_to_empty() -> None:
    args = _parse_args([])
    assert args.ticker is None
    assert args.cik is None
    assert args.company_name is None


def test_parse_args_reads_ticker_cik_and_name() -> None:
    args = _parse_args(
        ["--ticker", "NVDA", "--cik", "1045810", "--company-name", "NVIDIA Corp"]
    )
    assert args.ticker == "NVDA"
    assert args.cik == "1045810"
    assert args.company_name == "NVIDIA Corp"


def test_seed_watchlist_requires_cik_and_name_when_ticker_set() -> None:
    args = _parse_args(["--ticker", "NVDA"])
    with pytest.raises(SystemExit, match="--ticker requires --cik"):
        asyncio.run(_seed_watchlist_if_requested(args))


def test_seed_watchlist_noop_without_ticker() -> None:
    # Must complete without touching the database.
    asyncio.run(_seed_watchlist_if_requested(_parse_args([])))
