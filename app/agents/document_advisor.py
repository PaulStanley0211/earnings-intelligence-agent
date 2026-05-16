"""Document advisor agent node.

Wraps :func:`app.tools.advisor.advise_for_ticker` with the project's
repository pattern: tickers must already be registered in the watchlist
(so we have a verified CIK to query against EDGAR). Callers that want to
advise on a new ticker should add the watchlist entry first via the same
``poll_once.py --ticker T --cik C --company-name N`` route Phase 1 set up.
"""

from __future__ import annotations

from typing import Protocol

from app.memory.schemas import WatchlistRecord
from app.tools.advisor import AdvisorOutput, advise_for_ticker
from app.tools.edgar import SubmissionsResponse


class _SupportsWatchlist(Protocol):
    """Minimum repository surface the advisor depends on."""

    async def get_watchlist_entry_by_ticker(
        self, ticker: str
    ) -> WatchlistRecord | None: ...


class _SupportsSubmissions(Protocol):
    """Minimal EDGAR client surface required by the advisor."""

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse: ...


class UnknownTickerError(ValueError):
    """Raised when the advisor is asked about a ticker not on the watchlist.

    Surfaced verbatim to the API caller -- ``POST /api/advise`` turns this
    into a 404.
    """


async def advise(
    *,
    ticker: str,
    repository: _SupportsWatchlist,
    edgar: _SupportsSubmissions,
) -> AdvisorOutput:
    """Return the upload checklist for ``ticker``.

    Raises :class:`UnknownTickerError` if the ticker is not in the watchlist
    (the project never queries EDGAR by ticker because CIKs are the canonical
    identifier; the watchlist holds the ticker-to-CIK mapping).
    """
    entry = await repository.get_watchlist_entry_by_ticker(ticker.upper())
    if entry is None:
        raise UnknownTickerError(
            f"Ticker {ticker.upper()!r} is not on the watchlist. Add it first via "
            "`poll_once.py --ticker T --cik C --company-name N`."
        )
    return await advise_for_ticker(
        ticker=entry.ticker, cik=entry.cik, edgar=edgar
    )
