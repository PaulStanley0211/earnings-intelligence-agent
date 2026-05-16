"""Document advisor: given a ticker, return a ranked "what to upload" list.

The advisor consults the existing Phase 1 EDGAR client to enumerate recent
filings, then surfaces the latest 8-K (earnings release), latest 10-Q
(quarterly report), and latest 10-K (annual report) with direct EDGAR
archive URLs. The user clicks the link, downloads the PDF, and uploads it.

Transcripts are not on EDGAR; the advisor returns a hint pointing the user
to common public sources rather than attempting to fetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final, Protocol

from app.tools.edgar import RecentFiling, SubmissionsResponse


class _SupportsSubmissions(Protocol):
    """Minimal contract the advisor needs from an EDGAR-like client."""

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse: ...


_PRIORITY_FORMS: Final[tuple[str, ...]] = ("8-K", "10-Q", "10-K")

_TRANSCRIPT_HINT: Final[str] = (
    "Earnings-call transcripts are not on EDGAR. Try the company's investor-"
    "relations site (look for 'Earnings' or 'Quarterly Results') or a public "
    "transcript provider such as Motley Fool. Upload as plain text."
)


@dataclass(frozen=True)
class AdvisedFiling:
    """One row of the advisor's checklist."""

    filing_type: str
    accession_number: str
    filed_at: date
    edgar_index_url: str
    primary_document: str | None


@dataclass(frozen=True)
class AdvisorOutput:
    """Full advisor response for one ticker."""

    ticker: str
    cik: str
    suggested: list[AdvisedFiling]
    transcript_hint: str


def _edgar_index_url(cik: str, accession_number: str) -> str:
    """Build the canonical EDGAR archive index URL."""
    no_dashes = accession_number.replace("-", "")
    cik_stripped = cik.lstrip("0") or "0"
    return f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{no_dashes}/"


def _latest_for_form(
    filings: list[RecentFiling], form: str
) -> RecentFiling | None:
    """Return the most-recently-filed entry of ``form``, or ``None``."""
    matches = [f for f in filings if f.form == form]
    if not matches:
        return None
    return max(matches, key=lambda f: f.filing_date)


async def advise_for_ticker(
    *, ticker: str, cik: str, edgar: _SupportsSubmissions
) -> AdvisorOutput:
    """Build the upload checklist for ``ticker``.

    Queries EDGAR for the company's recent submissions and returns the latest
    filing for each priority form in :data:`_PRIORITY_FORMS`. Forms with no
    recent matches are silently omitted so the caller can render whatever
    the issuer actually has on file.
    """
    submissions = await edgar.get_submissions(cik=cik)
    suggested: list[AdvisedFiling] = []
    for form in _PRIORITY_FORMS:
        latest = _latest_for_form(submissions.recent_filings, form)
        if latest is None:
            continue
        suggested.append(
            AdvisedFiling(
                filing_type=form,
                accession_number=latest.accession_number,
                filed_at=latest.filing_date,
                edgar_index_url=_edgar_index_url(cik, latest.accession_number),
                primary_document=latest.primary_document,
            )
        )
    return AdvisorOutput(
        ticker=ticker.upper(),
        cik=cik,
        suggested=suggested,
        transcript_hint=_TRANSCRIPT_HINT,
    )
