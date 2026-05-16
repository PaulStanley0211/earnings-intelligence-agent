"""Unit tests for the EDGAR client.

The client is exercised through httpx ``MockTransport`` so the network never
moves. The rate-limit and retry pieces are unit-tested directly with short
budgets to keep the suite fast.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from app.tools.edgar import (
    CompanyFactsResponse,
    EdgarClient,
    EdgarHTTPError,
    EdgarServerError,
    RecentFiling,
    SubmissionsResponse,
    _RateLimiter,
)


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: object,
) -> EdgarClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://test")
    return EdgarClient(
        user_agent="Test Suite tests@example.com",
        http_client=http,
        rate_limit_rps=1000.0,
        backoff_initial=0.001,
        backoff_max=0.01,
        max_attempts=4,
        **kwargs,  # type: ignore[arg-type]
    )


# ---- rate limiter ----


async def test_rate_limiter_advances_next_slot_by_interval() -> None:
    """Each acquired slot pushes the next-available time forward by 1/rps.

    Wall-clock-based pacing is hard to assert under Windows' 15.6 ms timer
    resolution; the algorithmic guarantee here is what callers actually
    depend on - the limiter never lets ``_next_at`` regress.
    """
    limiter = _RateLimiter(rps=10.0)
    async with limiter:
        first_next = limiter._next_at
    async with limiter:
        second_next = limiter._next_at
    async with limiter:
        third_next = limiter._next_at
    assert second_next >= first_next + 0.099
    assert third_next >= second_next + 0.099


async def test_rate_limiter_rejects_non_positive_rps() -> None:
    with pytest.raises(ValueError, match="rps must be positive"):
        _RateLimiter(rps=0.0)


# ---- request behaviour ----


async def test_sends_user_agent_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, json={"cik": "0000789019", "name": "MSFT"})

    client = _make_client(handler)
    async with client:
        # Use the low-level _get_json to keep this test focused on headers.
        await client._get_json("/whatever")
    assert seen["ua"] == "Test Suite tests@example.com"


async def test_5xx_response_triggers_retry_until_success() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(503, text="EDGAR busy")
        return httpx.Response(200, json={"ok": True})

    client = _make_client(handler)
    async with client:
        body = await client._get_json("/retry-me")
    assert body == {"ok": True}
    assert attempts["count"] == 3


async def test_5xx_response_eventually_gives_up() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="boom")

    client = _make_client(handler)
    async with client:
        with pytest.raises(EdgarServerError):
            await client._get_json("/always-fails")


async def test_4xx_response_does_not_retry() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(404, text="not found")

    client = _make_client(handler)
    async with client:
        with pytest.raises(EdgarHTTPError) as info:
            await client._get_json("/missing")
    assert attempts["count"] == 1
    assert info.value.status_code == 404


# ---- high-level methods ----


async def test_get_submissions_parses_response() -> None:
    payload = {
        "cik": "789019",
        "name": "Microsoft Corp",
        "tickers": ["MSFT"],
        "sicDescription": "Services-Prepackaged Software",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000950170-26-000050",
                    "0000950170-26-000020",
                    "0000950170-25-000100",
                ],
                "form": ["10-Q", "8-K", "10-K"],
                "filingDate": ["2026-04-25", "2026-04-24", "2025-07-30"],
                "reportDate": ["2026-03-31", "", "2025-06-30"],
                "primaryDocument": ["msft-2026q3.htm", "msft-8k.htm", "msft-2025fy.htm"],
            }
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/submissions/CIK0000789019.json"
        return httpx.Response(200, json=payload)

    client = _make_client(handler)
    async with client:
        result = await client.get_submissions(cik="789019")
    assert isinstance(result, SubmissionsResponse)
    assert result.cik == "0000789019"
    assert result.entity_name == "Microsoft Corp"
    assert {f.form for f in result.recent_filings} == {"10-Q", "8-K", "10-K"}
    q3 = next(
        f for f in result.recent_filings if f.accession_number == "0000950170-26-000050"
    )
    assert isinstance(q3, RecentFiling)
    assert q3.report_date is not None
    assert q3.report_date.isoformat() == "2026-03-31"
    eight_k = next(f for f in result.recent_filings if f.form == "8-K")
    assert eight_k.report_date is None, "blank reportDate must map to None"


async def test_get_submissions_zero_pads_cik() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "cik": "789019",
                "name": "x",
                "tickers": [],
                "filings": {
                    "recent": {
                        "accessionNumber": [],
                        "form": [],
                        "filingDate": [],
                        "reportDate": [],
                        "primaryDocument": [],
                    }
                },
            },
        )

    client = _make_client(handler)
    async with client:
        await client.get_submissions(cik="789019")
    assert seen["path"] == "/submissions/CIK0000789019.json"


async def test_get_company_facts_returns_response_with_raw_payload() -> None:
    payload = {
        "cik": 789019,
        "entityName": "Microsoft Corp",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues",
                    "units": {
                        "USD": [
                            {
                                "start": "2026-01-01",
                                "end": "2026-03-31",
                                "val": 61858000000,
                                "accn": "0000950170-26-000050",
                                "fy": 2026,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2026-04-25",
                            }
                        ]
                    },
                }
            }
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/xbrl/companyfacts/CIK0000789019.json"
        return httpx.Response(200, json=payload)

    client = _make_client(handler)
    async with client:
        result = await client.get_company_facts(cik="789019")
    assert isinstance(result, CompanyFactsResponse)
    assert result.cik == "0000789019"
    assert result.entity_name == "Microsoft Corp"
    assert result.raw["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["val"] == 61858000000


async def test_user_agent_validation_at_construction() -> None:
    with pytest.raises(ValueError, match="EDGAR_USER_AGENT"):
        EdgarClient(user_agent="no-email-here", http_client=httpx.AsyncClient())


# ---- get_filing_document ----


async def test_get_filing_document_fetches_html_from_archives() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, text="<html><body>10-Q body</body></html>")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://www.sec.gov", transport=transport
    ) as http:
        edgar = EdgarClient(
            user_agent="Tester tester@example.com",
            http_client=http,
            rate_limit_rps=100.0,
        )
        body = await edgar.get_filing_document(
            cik="0000789019",
            accession_number="0000950170-26-000050",
            primary_document="msft-20260331.htm",
        )

    assert body == "<html><body>10-Q body</body></html>"
    assert (
        captured["url"]
        == "https://www.sec.gov/Archives/edgar/data/789019/000095017026000050/msft-20260331.htm"
    )
    assert captured["ua"] == "Tester tester@example.com"


async def test_get_filing_document_raises_on_4xx() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://www.sec.gov", transport=transport
    ) as http:
        edgar = EdgarClient(
            user_agent="Tester tester@example.com",
            http_client=http,
            rate_limit_rps=100.0,
        )
        with pytest.raises(EdgarHTTPError):
            await edgar.get_filing_document(
                cik="0000789019",
                accession_number="0000950170-26-000050",
                primary_document="msft-20260331.htm",
            )
