"""EDGAR REST client.

Two endpoints carry Phase 1: the submissions list at
``data.sec.gov/submissions/CIK{cik}.json`` for filing discovery, and the
companyfacts dump at
``data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`` for pre-parsed XBRL
values. arelle and the raw instance-document fallback land in Phase 2 as
``xbrl_malformed`` mitigation per the runbook.

The client enforces three constraints required by SEC policy and the
project plan:

* Every request carries a contact ``User-Agent`` header. Misformatted
  headers are rejected at construction so configuration mistakes surface
  at startup, not on the first poll.
* Requests are paced under 10/second by a token-bucket limiter.
* 5xx responses and network errors are retried with exponential backoff
  plus jitter; 4xx responses raise immediately.

All responses are returned as Pydantic models so callers do not need to
know JSON shapes - the schemas are the contract.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from types import TracebackType
from typing import Any, Final, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.observability.logging import get_logger

_EDGAR_DATA_BASE: Final[str] = "https://data.sec.gov"
_EDGAR_ARCHIVE_BASE: Final[str] = "https://www.sec.gov"
_EDGAR_UA_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[^<>]{2,}\s+\S+@\S+\.\S+$")

_logger = get_logger()


class EdgarHTTPError(RuntimeError):
    """A 4xx response from EDGAR - not retryable."""

    def __init__(self, status_code: int, url: str, body: str) -> None:
        """Capture the response status, URL, and a trimmed body for diagnostics."""
        super().__init__(f"EDGAR {status_code} for {url}: {body[:200]}")
        self.status_code = status_code
        self.url = url


class EdgarServerError(RuntimeError):
    """A 5xx response from EDGAR - retryable by :class:`EdgarClient`."""

    def __init__(self, status_code: int, url: str) -> None:
        """Capture the upstream status and URL."""
        super().__init__(f"EDGAR {status_code} for {url}")
        self.status_code = status_code
        self.url = url


class RecentFiling(BaseModel):
    """One row from the ``filings.recent`` array."""

    model_config = ConfigDict(frozen=True)

    accession_number: str
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str | None


class SubmissionsResponse(BaseModel):
    """Decoded ``submissions/CIK{cik}.json`` payload (recent filings only)."""

    model_config = ConfigDict(frozen=True)

    cik: str
    entity_name: str
    tickers: list[str] = Field(default_factory=list)
    sic_description: str | None = None
    recent_filings: list[RecentFiling] = Field(default_factory=list)


class CompanyFactsResponse(BaseModel):
    """Decoded ``api/xbrl/companyfacts/CIK{cik}.json`` payload.

    ``raw`` exposes the full JSON body so the parser in
    :mod:`app.tools.companyfacts` can walk the deeply nested
    ``facts.<taxonomy>.<concept>.units.<unit>`` arrays without us flattening
    the schema twice.
    """

    model_config = ConfigDict(frozen=True)

    cik: str
    entity_name: str
    raw: dict[str, Any]


class _RateLimiter:
    """Async token-bucket: at most ``rps`` calls per second across all tasks."""

    def __init__(self, rps: float) -> None:
        """Build a limiter pacing calls one every ``1/rps`` seconds."""
        if rps <= 0:
            raise ValueError("rps must be positive")
        self._interval = 1.0 / rps
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def __aenter__(self) -> Self:
        """Sleep until a fresh slot is available, then return."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            sleep_for = max(0.0, self._next_at - now)
            self._next_at = now + sleep_for + self._interval
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """No-op; the slot is consumed at acquire time."""
        return None


class EdgarClient:
    """Async EDGAR REST client.

    Use as an async context manager so the underlying httpx client is closed
    deterministically::

        async with EdgarClient(user_agent=ua) as edgar:
            sub = await edgar.get_submissions(cik="789019")
    """

    def __init__(
        self,
        *,
        user_agent: str,
        http_client: httpx.AsyncClient | None = None,
        rate_limit_rps: float = 10.0,
        max_attempts: int = 5,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
    ) -> None:
        """Wire dependencies; rejects a malformed ``user_agent``."""
        if not _EDGAR_UA_PATTERN.match(user_agent.strip()):
            raise ValueError(
                "EDGAR_USER_AGENT must be '<name> <email>'; SEC will block "
                "requests with a missing or malformed identifier."
            )
        self._user_agent = user_agent.strip()
        self._http = http_client or httpx.AsyncClient(
            base_url=_EDGAR_DATA_BASE, timeout=httpx.Timeout(20.0)
        )
        self._owns_http = http_client is None
        self._rate_limiter = _RateLimiter(rps=rate_limit_rps)
        self._max_attempts = max_attempts
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max

    async def __aenter__(self) -> Self:
        """Return ``self``; the http client is built eagerly in ``__init__``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying httpx client if we own it."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._owns_http:
            await self._http.aclose()

    # ---- low-level ----

    async def _get_json(self, path: str) -> Any:
        """Issue a rate-limited GET with retries on transient failures.

        ``path`` is appended to the configured base URL. Returns the decoded
        JSON body. Raises :class:`EdgarHTTPError` for 4xx and
        :class:`EdgarServerError` for any 5xx that survives the retry budget.
        """
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(
                initial=self._backoff_initial, max=self._backoff_max
            ),
            retry=retry_if_exception_type((httpx.RequestError, EdgarServerError)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                async with self._rate_limiter:
                    response = await self._http.get(
                        path, headers={"User-Agent": self._user_agent, "Accept": "application/json"}
                    )
                if 500 <= response.status_code < 600:
                    raise EdgarServerError(response.status_code, str(response.url))
                if 400 <= response.status_code < 500:
                    raise EdgarHTTPError(
                        response.status_code, str(response.url), response.text
                    )
                response.raise_for_status()
                return response.json()
        raise RuntimeError("unreachable: tenacity reraises on failure")

    async def _get_text(self, *, base_url: str, path: str) -> str:
        """Issue a rate-limited GET against ``base_url + path`` and return text.

        Retries 5xx and network errors with exponential backoff; surfaces
        4xx immediately as :class:`EdgarHTTPError`.
        """
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(
                initial=self._backoff_initial, max=self._backoff_max
            ),
            retry=retry_if_exception_type((httpx.RequestError, EdgarServerError)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                async with self._rate_limiter:
                    response = await self._http.get(
                        f"{base_url}{path}",
                        headers={"User-Agent": self._user_agent},
                    )
                if 500 <= response.status_code < 600:
                    raise EdgarServerError(response.status_code, str(response.url))
                if 400 <= response.status_code < 500:
                    raise EdgarHTTPError(
                        response.status_code, str(response.url), response.text
                    )
                response.raise_for_status()
                return response.text
        raise RuntimeError("unreachable: tenacity reraises on failure")

    # ---- high-level ----

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse:
        """Fetch the submissions index for ``cik`` and return the recent filings."""
        padded = _pad_cik(cik)
        body = await self._get_json(f"/submissions/CIK{padded}.json")
        recent = body.get("filings", {}).get("recent", {})
        filings = [
            RecentFiling(
                accession_number=accession,
                form=form,
                filing_date=date.fromisoformat(filing_date),
                report_date=_parse_optional_date(report_date),
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
            cik=padded,
            entity_name=str(body.get("name", "")),
            tickers=list(body.get("tickers", []) or []),
            sic_description=body.get("sicDescription"),
            recent_filings=filings,
        )

    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse:
        """Fetch the full XBRL companyfacts bundle for ``cik``.

        The body can run to several megabytes; the parser at
        :mod:`app.tools.companyfacts` walks it lazily.
        """
        padded = _pad_cik(cik)
        body = await self._get_json(f"/api/xbrl/companyfacts/CIK{padded}.json")
        return CompanyFactsResponse(
            cik=padded,
            entity_name=str(body.get("entityName", "")),
            raw=body,
        )

    async def get_filing_document(
        self,
        *,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> str:
        """Fetch the primary HTML body of a filing from EDGAR archives.

        The archives host is ``www.sec.gov`` rather than the JSON ``data.sec.gov``,
        so we override the per-request base URL. CIK is unpadded; accession
        number has dashes stripped per the archives URL convention.

        Raises ``ValueError`` when ``cik`` is non-numeric.
        """
        unpadded_cik = str(int(cik))
        accession_no_dashes = accession_number.replace("-", "")
        path = (
            f"/Archives/edgar/data/{unpadded_cik}/{accession_no_dashes}/{primary_document}"
        )
        return await self._get_text(base_url=_EDGAR_ARCHIVE_BASE, path=path)


def _pad_cik(cik: str) -> str:
    """Return ``cik`` left-padded to the 10-digit form EDGAR URLs require."""
    cleaned = cik.strip().lstrip("0") or "0"
    if not cleaned.isdigit():
        raise ValueError(f"CIK must be numeric, got {cik!r}")
    return cleaned.zfill(10)


def _parse_optional_date(value: str | None) -> date | None:
    """Map empty strings and ``None`` to ``None``; otherwise parse ISO-8601."""
    if not value:
        return None
    return date.fromisoformat(value)
