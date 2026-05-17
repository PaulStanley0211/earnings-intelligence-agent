"""Advisor accuracy gate - the document advisor must return the correct latest 8-K
on or before the as-of date for >= 9 of 10 reference tickers.

Cassettes in ``tests/fixtures/edgar/advisor/`` were recorded from live EDGAR
submissions responses for 10 tickers spanning multiple industries (tech,
banking, energy, pharma, consumer staples, retail) on dates in early-to-mid
2026. The index file ``_test_cases.json`` carries the
``(ticker, cik, as_of_date, expected_latest_8k_accession)`` tuples that ground
the assertion.

Each cassette is trimmed to filings filed on or before its ``as_of_date``, so
calling :func:`app.tools.advisor.advise_for_ticker` against a cassette-backed
EDGAR stub is equivalent to asking "what was the latest 8-K for this ticker as
of this date?" - which is exactly what the advisor must answer correctly to
unblock the upload-first user flow.

Per spec section 3.6 the gate is 10/10 ideal, or 9/10 with a documented
exception in this module docstring. Current state: 10/10 expected.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.tools.advisor import AdvisorOutput, advise_for_ticker
from app.tools.edgar import RecentFiling, SubmissionsResponse

_FIXTURES: Path = (
    Path(__file__).resolve().parents[1] / "fixtures" / "edgar" / "advisor"
)


def _load_cases() -> list[dict[str, str]]:
    """Load the (ticker, cik, as_of_date, expected_latest_8k_accession) tuples."""
    return list(
        json.loads((_FIXTURES / "_test_cases.json").read_text(encoding="utf-8"))
    )


def _load_cassette(ticker: str, as_of_date: str) -> dict[str, Any]:
    """Load the raw EDGAR-submissions cassette JSON for one ticker/as-of-date pair."""
    path = _FIXTURES / f"{ticker}_{as_of_date}.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _parse_as_submissions(payload: dict[str, Any]) -> SubmissionsResponse:
    """Convert a raw EDGAR-shape cassette body into a :class:`SubmissionsResponse`.

    Mirrors the parsing logic in :meth:`app.tools.edgar.EdgarClient.get_submissions`
    so the stub returns exactly what the real client would produce for the
    recorded body.
    """
    recent = payload.get("filings", {}).get("recent", {})
    filings = [
        RecentFiling(
            accession_number=accession,
            form=form,
            filing_date=date.fromisoformat(filing_date),
            report_date=date.fromisoformat(report_date) if report_date else None,
            primary_document=primary or None,
        )
        for accession, form, filing_date, report_date, primary in zip(
            recent.get("accessionNumber", []),
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("reportDate", []),
            recent.get("primaryDocument", []),
            strict=False,
        )
    ]
    return SubmissionsResponse(
        cik=str(payload.get("cik", "")),
        entity_name=str(payload.get("name", "")),
        tickers=list(payload.get("tickers", []) or []),
        sic_description=payload.get("sicDescription"),
        recent_filings=filings,
    )


class _CassetteEdgarClient:
    """Stub EDGAR client that returns a pre-recorded submissions response.

    The advisor calls ``edgar.get_submissions(cik=...)``; this stub simply
    hands back the cassette payload parsed via :func:`_parse_as_submissions`,
    bypassing the network and the rate limiter.
    """

    def __init__(self, cassette_payload: dict[str, Any]) -> None:
        """Capture the parsed :class:`SubmissionsResponse` for later retrieval."""
        self._response = _parse_as_submissions(cassette_payload)

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse:
        """Return the recorded submissions response regardless of ``cik``."""
        return self._response


@pytest.mark.parametrize(
    "case",
    _load_cases(),
    ids=lambda c: f"{c['ticker']}_{c['as_of_date']}",
)
@pytest.mark.asyncio
async def test_advisor_picks_correct_latest_8k(case: dict[str, str]) -> None:
    """For each reference (ticker, as_of_date), the advisor picks the right 8-K.

    The cassette is trimmed to filings on or before ``as_of_date``, so the
    advisor's "latest 8-K by filing date" rule must surface
    ``expected_latest_8k_accession`` at ``suggested[0]``.
    """
    cassette = _load_cassette(case["ticker"], case["as_of_date"])
    stub = _CassetteEdgarClient(cassette)
    result = await advise_for_ticker(
        ticker=case["ticker"], cik=case["cik"], edgar=stub
    )
    assert isinstance(result, AdvisorOutput)
    assert result.suggested, (
        f"advisor returned an empty suggestion list for "
        f"{case['ticker']} on {case['as_of_date']}"
    )
    top = result.suggested[0]
    assert top.filing_type == "8-K", (
        f"{case['ticker']} on {case['as_of_date']}: top suggestion is "
        f"{top.filing_type}, expected 8-K"
    )
    assert top.accession_number == case["expected_latest_8k_accession"], (
        f"{case['ticker']} on {case['as_of_date']}: got "
        f"{top.accession_number}, expected {case['expected_latest_8k_accession']}"
    )
