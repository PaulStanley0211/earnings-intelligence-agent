"""Unit tests for :mod:`app.tools.consensus`."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import httpx
import pytest

from app.tools.consensus import (
    ConsensusFetcher,
    FinnhubHTTPProvider,
    YFinanceProvider,
    _RawEstimate,
)


class _StubFinnhub:
    def __init__(self, rows: list[_RawEstimate]) -> None:
        self.rows = rows
        self.calls = 0

    async def fetch(self, *, ticker: str, period_end: date) -> list[_RawEstimate]:
        self.calls += 1
        return self.rows


class _StubFinnhubError(_StubFinnhub):
    async def fetch(self, *, ticker: str, period_end: date) -> list[_RawEstimate]:
        self.calls += 1
        raise RuntimeError("rate limited")


class _StubYFinance:
    def __init__(self, rows: list[_RawEstimate]) -> None:
        self.rows = rows
        self.calls = 0

    async def fetch(self, *, ticker: str, period_end: date) -> list[_RawEstimate]:
        self.calls += 1
        return self.rows


async def test_fetch_prefers_finnhub_when_it_returns_rows() -> None:
    finnhub_rows = [
        _RawEstimate(
            metric="eps_diluted",
            period_end=date(2026, 3, 31),
            value=Decimal("1.30"),
            analyst_count=18,
        )
    ]
    yfinance_rows = [
        _RawEstimate(
            metric="eps_diluted",
            period_end=date(2026, 3, 31),
            value=Decimal("9.99"),
            analyst_count=None,
        )
    ]
    finnhub = _StubFinnhub(finnhub_rows)
    yfinance = _StubYFinance(yfinance_rows)
    fetcher = ConsensusFetcher(finnhub=finnhub, yfinance=yfinance)

    rows = await fetcher.fetch(
        ticker="MSFT",
        fiscal_year=2026,
        fiscal_period="Q3",
        period_end=date(2026, 3, 31),
    )

    assert [(r.source, str(r.value)) for r in rows] == [("finnhub", "1.30")]
    assert yfinance.calls == 0


async def test_fetch_falls_back_to_yfinance_on_finnhub_failure() -> None:
    yfinance_rows = [
        _RawEstimate(
            metric="eps_diluted",
            period_end=date(2026, 3, 31),
            value=Decimal("1.25"),
            analyst_count=None,
        )
    ]
    fetcher = ConsensusFetcher(
        finnhub=_StubFinnhubError([]), yfinance=_StubYFinance(yfinance_rows)
    )
    rows = await fetcher.fetch(
        ticker="MSFT",
        fiscal_year=2026,
        fiscal_period="Q3",
        period_end=date(2026, 3, 31),
    )
    assert len(rows) == 1
    assert rows[0].source == "yfinance"
    assert rows[0].value == Decimal("1.25")


async def test_fetch_returns_empty_when_both_providers_fail() -> None:
    fetcher = ConsensusFetcher(
        finnhub=_StubFinnhubError([]), yfinance=_StubFinnhubError([])
    )
    rows = await fetcher.fetch(
        ticker="MSFT",
        fiscal_year=2026,
        fiscal_period="Q3",
        period_end=date(2026, 3, 31),
    )
    assert rows == []


async def test_finnhub_http_provider_picks_period_end_match() -> None:
    handler_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        handler_calls.append(request)
        if request.url.path.endswith("/eps-estimate"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "period": "2025-12-31",
                            "epsAvg": "1.00",
                            "numberAnalysts": 10,
                        },
                        {
                            "period": "2026-03-31",
                            "epsAvg": "1.30",
                            "numberAnalysts": 20,
                        },
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "period": "2026-03-31",
                        "revenueAvg": "63000000000",
                        "numberAnalysts": 22,
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://finnhub.io/api/v1", transport=transport
    ) as http:
        provider = FinnhubHTTPProvider(api_key="token", http_client=http)
        rows = await provider.fetch(ticker="MSFT", period_end=date(2026, 3, 31))

    assert {r.metric for r in rows} == {"eps_diluted", "revenue"}
    eps_row = next(r for r in rows if r.metric == "eps_diluted")
    assert eps_row.value == Decimal("1.30")
    assert eps_row.analyst_count == 20
    revenue_row = next(r for r in rows if r.metric == "revenue")
    assert revenue_row.value == Decimal("63000000000")
    # Both endpoints reachable, both queried with token + ticker.
    assert {req.url.params.get("symbol") for req in handler_calls} == {"MSFT"}


async def test_finnhub_http_provider_returns_empty_when_no_period_match() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://finnhub.io/api/v1", transport=transport
    ) as http:
        provider = FinnhubHTTPProvider(api_key="token", http_client=http)
        rows = await provider.fetch(ticker="ZZZ", period_end=date(2026, 3, 31))
    assert rows == []


async def test_finnhub_http_provider_rejects_blank_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        FinnhubHTTPProvider(api_key="")


async def test_yfinance_provider_extracts_avg_from_dict_row() -> None:
    class _Frame:
        def __init__(self, rows: dict[str, dict[str, object]]) -> None:
            self._rows = rows

        # yfinance dataframes support both .loc and [] indexing; the dict
        # shape exercises the .loc branch with a dict-like row.
        @property
        def loc(self) -> dict[str, dict[str, object]]:
            return self._rows

    frame = _Frame({"2026-03-31": {"avg": "1.27"}})

    async def runner(fn: object) -> object:
        return fn()  # type: ignore[operator]

    provider = YFinanceProvider(
        ticker_factory=lambda ticker: type(
            "T", (), {"earnings_estimate": frame}
        )(),
        runner=runner,  # type: ignore[arg-type]
    )
    rows = await provider.fetch(ticker="MSFT", period_end=date(2026, 3, 31))
    assert rows == [
        _RawEstimate(
            metric="eps_diluted",
            period_end=date(2026, 3, 31),
            value=Decimal("1.27"),
            analyst_count=None,
        )
    ]


async def test_yfinance_provider_returns_empty_when_factory_raises() -> None:
    def factory(ticker: str) -> object:
        raise RuntimeError("missing yfinance")

    async def runner(fn: object) -> object:
        return fn()  # type: ignore[operator]

    provider = YFinanceProvider(ticker_factory=factory, runner=runner)  # type: ignore[arg-type]
    rows = await provider.fetch(ticker="MSFT", period_end=date(2026, 3, 31))
    assert rows == []
