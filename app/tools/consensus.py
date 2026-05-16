"""Analyst-consensus fetcher.

The comparator needs a single ``(ticker, fiscal_year, fiscal_period)`` lookup
that returns the analyst-consensus values for the metrics the synthesiser
cites - EPS (diluted) and revenue in Phase 2. The fetcher routes through
Finnhub first because its analyst panels are richer and its quarterly
breakdown matches the SEC fiscal-period grid; yfinance is the fallback when
Finnhub is unreachable or rate-limited (per the runbook).

The provider integrations are wrapped behind two Protocol-shaped callables -
``_FinnhubProvider`` and ``_YFinanceProvider`` - so unit tests can inject
stubs without hitting either service. The default Finnhub provider uses
``httpx.AsyncClient``; the default yfinance provider imports the ``yfinance``
package lazily on first call so the dependency only matters when Finnhub
fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Final, Protocol

import httpx

from app.memory.schemas import ComparisonMetric, NewConsensusEstimate
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

_FINNHUB_BASE_URL: Final[str] = "https://finnhub.io/api/v1"


class ConsensusFetchError(RuntimeError):
    """Raised when both providers fail and no consensus rows are available."""


# Mapping between Finnhub period strings ("2026-03-31") and SEC fiscal periods
# is fragile: companies whose fiscal year does not align with the calendar
# year report "Q3" for periods ending in different calendar months. The
# fetcher therefore matches on ``period_end`` directly and lets the caller
# pass in the SEC-side ``fiscal_year`` and ``fiscal_period`` once.
@dataclass(frozen=True)
class _RawEstimate:
    metric: ComparisonMetric
    period_end: date
    value: Decimal
    analyst_count: int | None


class _FinnhubProvider(Protocol):
    async def fetch(
        self, *, ticker: str, period_end: date
    ) -> list[_RawEstimate]: ...


class _YFinanceProvider(Protocol):
    async def fetch(
        self, *, ticker: str, period_end: date
    ) -> list[_RawEstimate]: ...


class ConsensusFetcher:
    """Two-tier consensus fetcher (Finnhub primary, yfinance fallback).

    Construct with the production providers via :func:`build_default_fetcher`
    or inject stubs directly. The :meth:`fetch` method returns a list of
    :class:`NewConsensusEstimate` rows ready for the repository - one per
    metric the providers populated, tagged with the source that supplied it.
    """

    def __init__(
        self,
        *,
        finnhub: _FinnhubProvider,
        yfinance: _YFinanceProvider | None,
    ) -> None:
        """Wire the provider strategies; ``yfinance`` may be ``None`` in dev."""
        self._finnhub = finnhub
        self._yfinance = yfinance

    async def aclose(self) -> None:
        """Close any provider that owns a long-lived httpx client.

        The fetcher is used as a process-wide singleton by
        ``app.api.dependencies.get_compiled_graph``; the FastAPI lifespan
        shutdown calls this so the EDGAR / Finnhub connection pools are
        drained cleanly on graceful restart.
        """
        aclose = getattr(self._finnhub, "aclose", None)
        if callable(aclose):
            await aclose()
        if self._yfinance is not None:
            yaclose = getattr(self._yfinance, "aclose", None)
            if callable(yaclose):
                await yaclose()

    async def fetch(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
        period_end: date,
    ) -> list[NewConsensusEstimate]:
        """Return consensus rows for the requested period, source-tagged.

        Finnhub is tried first; on any provider failure or empty payload, the
        fetcher falls back to yfinance when configured. An empty list is a
        valid outcome - the comparator handles missing consensus by emitting
        a ``consensus_value=NULL`` comparison row.
        """
        try:
            raw = await self._finnhub.fetch(ticker=ticker, period_end=period_end)
            if raw:
                return _to_rows(
                    raw,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    source="finnhub",
                )
        except Exception as exc:
            _logger.bind(
                ticker=ticker,
                period_end=period_end.isoformat(),
                trace_id=current_trace_id(),
                error=str(exc),
            ).warning("consensus_finnhub_failed")
        if self._yfinance is None:
            return []
        try:
            raw = await self._yfinance.fetch(ticker=ticker, period_end=period_end)
        except Exception as exc:
            _logger.bind(
                ticker=ticker,
                period_end=period_end.isoformat(),
                trace_id=current_trace_id(),
                error=str(exc),
            ).warning("consensus_yfinance_failed")
            return []
        return _to_rows(
            raw,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            source="yfinance",
        )


def _to_rows(
    raw: list[_RawEstimate],
    *,
    ticker: str,
    fiscal_year: int,
    fiscal_period: str,
    source: str,
) -> list[NewConsensusEstimate]:
    """Convert raw provider rows into :class:`NewConsensusEstimate` payloads."""
    return [
        NewConsensusEstimate(
            ticker=ticker,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            metric=item.metric,
            value=item.value,
            analyst_count=item.analyst_count,
            source=source,  # type: ignore[arg-type]
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Default providers.
# ---------------------------------------------------------------------------


class FinnhubHTTPProvider:
    """Finnhub provider built on ``httpx.AsyncClient``.

    The class is constructible with an injected client so tests can route
    calls through ``httpx.MockTransport`` without monkey-patching the global
    httpx default.
    """

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Wire the API key and an optional pre-built httpx client."""
        if not api_key:
            raise ValueError("Finnhub API key is required.")
        self._api_key = api_key
        self._http = http_client or httpx.AsyncClient(
            base_url=_FINNHUB_BASE_URL, timeout=httpx.Timeout(timeout_seconds)
        )
        self._owns_http = http_client is None

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._owns_http:
            await self._http.aclose()

    async def fetch(self, *, ticker: str, period_end: date) -> list[_RawEstimate]:
        """Pull EPS and revenue estimates and pick the row matching ``period_end``."""
        eps_rows, revenue_rows = await asyncio.gather(
            self._get("/stock/eps-estimate", ticker=ticker),
            self._get("/stock/revenue-estimate", ticker=ticker),
        )
        results: list[_RawEstimate] = []
        eps = _match_finnhub_row(eps_rows, period_end=period_end, value_key="epsAvg")
        if eps is not None:
            value, analysts = eps
            results.append(
                _RawEstimate(
                    metric="eps_diluted",
                    period_end=period_end,
                    value=value,
                    analyst_count=analysts,
                )
            )
        rev = _match_finnhub_row(
            revenue_rows, period_end=period_end, value_key="revenueAvg"
        )
        if rev is not None:
            value, analysts = rev
            results.append(
                _RawEstimate(
                    metric="revenue",
                    period_end=period_end,
                    value=value,
                    analyst_count=analysts,
                )
            )
        return results

    async def _get(self, path: str, *, ticker: str) -> list[dict[str, Any]]:
        """Issue a single Finnhub GET and return the ``data`` array."""
        response = await self._http.get(
            path,
            params={"symbol": ticker, "freq": "quarterly", "token": self._api_key},
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            return []
        rows = body.get("data", [])
        return [row for row in rows if isinstance(row, dict)]


def _match_finnhub_row(
    rows: list[dict[str, Any]], *, period_end: date, value_key: str
) -> tuple[Decimal, int | None] | None:
    """Find the Finnhub row whose ``period`` exactly matches ``period_end``."""
    target = period_end.isoformat()
    for row in rows:
        if row.get("period") != target:
            continue
        raw_value = row.get(value_key)
        if raw_value is None:
            return None
        try:
            value = Decimal(str(raw_value))
        except (InvalidOperation, TypeError):
            return None
        analysts = row.get("numberAnalysts")
        try:
            analyst_count = int(analysts) if analysts is not None else None
        except (TypeError, ValueError):
            analyst_count = None
        return value, analyst_count
    return None


# ---------------------------------------------------------------------------
# yfinance fallback. yfinance is synchronous and slow; we wrap it in
# asyncio.to_thread so the watcher's event loop is not blocked.
# ---------------------------------------------------------------------------


class YFinanceProvider:
    """yfinance-backed fallback provider.

    Phase 2 only uses yfinance's per-quarter EPS estimate. Revenue is omitted
    because yfinance's analyst_estimate frame mixes annual and quarterly
    estimates in ways that have historically broken when yfinance updates its
    schema. The runbook flags a yfinance-sourced consensus as ``degraded``.
    """

    def __init__(
        self,
        *,
        ticker_factory: Callable[[str], Any] | None = None,
        runner: Callable[[Callable[[], Any]], Awaitable[Any]] | None = None,
    ) -> None:
        """Wire optional injection points for tests."""
        self._ticker_factory = ticker_factory
        self._runner = runner or (lambda fn: asyncio.to_thread(fn))

    async def fetch(self, *, ticker: str, period_end: date) -> list[_RawEstimate]:
        """Pull yfinance EPS estimate for the quarter ending ``period_end``."""
        ticker_factory = self._ticker_factory or _default_yfinance_ticker
        if ticker_factory is None:
            return []
        try:
            estimate_frame = await self._runner(
                lambda: getattr(ticker_factory(ticker), "earnings_estimate", None)
            )
        except Exception as exc:
            _logger.bind(
                ticker=ticker,
                period_end=period_end.isoformat(),
                trace_id=current_trace_id(),
                error=str(exc),
            ).warning("yfinance_call_failed")
            return []
        eps = _yfinance_quarter_eps(estimate_frame, period_end=period_end)
        if eps is None:
            return []
        return [
            _RawEstimate(
                metric="eps_diluted",
                period_end=period_end,
                value=eps,
                analyst_count=None,
            )
        ]


def _default_yfinance_ticker(ticker: str) -> Any:
    """Lazily import ``yfinance`` so the dep only matters at fallback time."""
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - import-time configuration
        raise ConsensusFetchError(
            "yfinance is not installed; the consensus fallback is unavailable."
        ) from exc
    return yfinance.Ticker(ticker)


def _yfinance_quarter_eps(frame: Any, *, period_end: date) -> Decimal | None:
    """Pluck the EPS estimate for ``period_end`` out of a yfinance dataframe.

    yfinance returns a pandas DataFrame indexed by period-end strings, with
    columns including ``avg``. The function tolerates either dataframe-like
    or dict-like shapes so tests can pass plain Python objects.
    """
    if frame is None:
        return None
    try:
        row = frame.loc[period_end.isoformat()]
    except Exception:
        try:
            row = frame[period_end.isoformat()]
        except Exception:
            return None
    raw = _extract_avg(row)
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, TypeError):
        return None


def _extract_avg(row: Any) -> Any:
    """Return the ``avg`` value from a dataframe row, a dict, or a scalar."""
    if hasattr(row, "get"):
        return row.get("avg")
    if hasattr(row, "loc"):
        try:
            return row.loc["avg"]
        except Exception:
            return None
    return row


def build_default_fetcher(
    *,
    finnhub_api_key: str,
    http_client: httpx.AsyncClient | None = None,
    enable_yfinance: bool = True,
) -> ConsensusFetcher:
    """Construct a :class:`ConsensusFetcher` wired to the live providers."""
    finnhub = FinnhubHTTPProvider(api_key=finnhub_api_key, http_client=http_client)
    yfinance: YFinanceProvider | None = YFinanceProvider() if enable_yfinance else None
    return ConsensusFetcher(finnhub=finnhub, yfinance=yfinance)
