"""EDGAR watcher.

The watcher is the deterministic head of the pipeline. It polls EDGAR for
every active watchlist ticker, persists newly-detected filings, and runs the
Phase 1 agent slice (currently just the financial-extractor) over each one.
Restarts are safe because the ``filings`` table checkpoints every accession
number we have ever processed.

The module exposes two entry points:

- :func:`poll_once` runs exactly one poll cycle. Used by the one-shot CLI
  (``python -m app.scripts.poll_once``) and by tests.
- :func:`watch_forever` runs :func:`poll_once` on the configured cadence.
  Used by the production watcher service.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.financial_extractor import extract_financials
from app.config import get_settings
from app.memory.db import get_session
from app.memory.repository import Repository
from app.memory.schemas import NewFiling, NewPollLog, PollStatus
from app.models.state import AgentState, FilingEvent, FilingForm
from app.observability.logging import current_trace_id, get_logger, new_trace_id, with_trace_id
from app.tools.edgar import CompanyFactsResponse, EdgarClient, RecentFiling, SubmissionsResponse

_logger = get_logger()

# SEC supports plenty of forms; we only process the ones the project plan
# enumerates as in-scope for the post-earnings pipeline.
_SUPPORTED_FORMS: Final[frozenset[str]] = frozenset(
    {form.value for form in FilingForm}
)


class WatcherDisabledError(RuntimeError):
    """Raised when ``watch_forever`` runs while ``watcher_mode_enabled`` is False.

    The upload-first product runs with the watcher off by default; enabling
    it is an explicit operator choice (eval/demo mode). Refusing to start
    prevents accidental EDGAR polling in production.
    """


def ensure_watcher_enabled(*, enabled: bool) -> None:
    """Raise :class:`WatcherDisabledError` if ``enabled`` is ``False``."""
    if not enabled:
        raise WatcherDisabledError(
            "Watcher mode is disabled. Set WATCHER_MODE_ENABLED=true to run "
            "the EDGAR watcher (eval/demo mode)."
        )


class _SupportsEdgar(Protocol):
    """Subset of :class:`~app.tools.edgar.EdgarClient` the watcher consumes."""

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse: ...
    async def get_company_facts(self, *, cik: str) -> CompanyFactsResponse: ...


class DetectedFiling(BaseModel):
    """Summary of one filing the watcher just processed."""

    model_config = ConfigDict(frozen=True)

    accession_number: str
    cik: str
    ticker: str
    form: FilingForm
    parsed_fact_count: int


class PollResult(BaseModel):
    """Outcome of one :func:`poll_once` cycle."""

    model_config = ConfigDict(frozen=True)

    tickers_checked: int
    filings_found: int
    filings: list[DetectedFiling]


async def poll_once(
    *,
    edgar: _SupportsEdgar,
    session: AsyncSession,
) -> PollResult:
    """Run one EDGAR poll cycle.

    Returns a :class:`PollResult` summarising what was found. On any
    unhandled exception the cycle records an ``error`` poll log entry,
    rolls back the session, and re-raises so the supervisor can decide.
    """
    repo = Repository(session)
    watchlist = await repo.list_active_watchlist()
    detected: list[DetectedFiling] = []
    try:
        for entry in watchlist:
            submissions = await edgar.get_submissions(cik=entry.cik)
            known = await repo.known_accession_numbers(cik=entry.cik)
            for filing_meta in submissions.recent_filings:
                if filing_meta.form not in _SUPPORTED_FORMS:
                    continue
                if filing_meta.accession_number in known:
                    continue
                summary = await _process_filing(
                    repo=repo,
                    edgar=edgar,
                    ticker=entry.ticker,
                    cik=entry.cik,
                    filing_meta=filing_meta,
                )
                if summary is not None:
                    detected.append(summary)
        await repo.record_poll(
            NewPollLog(
                tickers_checked=len(watchlist),
                filings_found=len(detected),
                status=PollStatus.OK,
            )
        )
        await session.commit()
        return PollResult(
            tickers_checked=len(watchlist),
            filings_found=len(detected),
            filings=detected,
        )
    except Exception as exc:
        await session.rollback()
        await repo.record_poll(
            NewPollLog(
                tickers_checked=len(watchlist),
                filings_found=0,
                status=PollStatus.ERROR,
                error_message=str(exc)[:1000],
            )
        )
        await session.commit()
        raise


async def _process_filing(
    *,
    repo: Repository,
    edgar: _SupportsEdgar,
    ticker: str,
    cik: str,
    filing_meta: RecentFiling,
) -> DetectedFiling | None:
    """Persist a new filing row and run the financial-extractor against it."""
    form = FilingForm(filing_meta.form)
    filed_at = datetime.combine(filing_meta.filing_date, datetime.min.time(), tzinfo=UTC)
    source_url = _filing_index_url(cik=cik, accession_number=filing_meta.accession_number)

    record = await repo.record_filing(
        filing=NewFiling(
            accession_number=filing_meta.accession_number,
            cik=cik,
            ticker=ticker,
            form=form,
            filed_at=filed_at,
            source_url=source_url,
            report_period_end=filing_meta.report_date,
        )
    )
    if record is None:
        return None  # raced with another poll, already known

    trace_id = current_trace_id() or new_trace_id()
    state = AgentState(
        trace_id=trace_id,
        started_at=datetime.now(UTC),
        filing_event=FilingEvent(
            accession_number=filing_meta.accession_number,
            cik=cik,
            ticker=ticker,
            form=form,
            filed_at=filed_at,
            source_url=source_url,
        ),
    )
    try:
        update = await extract_financials(state, edgar=edgar, repository=repo)
        await repo.mark_filing_processed(filing_meta.accession_number)
        new_state = update.apply(state)
        financials = new_state.financials or {}
        return DetectedFiling(
            accession_number=filing_meta.accession_number,
            cik=cik,
            ticker=ticker,
            form=form,
            parsed_fact_count=int(financials.get("parsed_count", 0)),
        )
    except Exception as exc:  # pragma: no cover - exercised via watch_forever
        await repo.mark_filing_failed(filing_meta.accession_number, error=str(exc))
        raise


def _filing_index_url(*, cik: str, accession_number: str) -> str:
    """Return the canonical EDGAR index URL for a filing.

    Accession numbers contain dashes in the form ``XXXXXXXXXX-YY-NNNNNN``;
    the directory path uses the dashless form.
    """
    bare = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={cik}&type=&dateb=&owner=include&count=40"
        f"&action=getcompany#{bare}"
    )


async def watch_forever(*, interval_seconds: int | None = None) -> None:
    """Poll EDGAR on an interval until the process is cancelled.

    Used by the production watcher service. Each cycle gets a fresh trace id
    so logs and OpenTelemetry spans can be grouped per poll.
    """
    settings = get_settings()
    ensure_watcher_enabled(enabled=settings.watcher_mode_enabled)
    cadence = interval_seconds or settings.edgar_poll_interval_seconds
    async with EdgarClient(user_agent=settings.edgar_user_agent) as edgar:
        while True:
            cycle_started = datetime.now(UTC)
            with with_trace_id() as trace_id:
                _logger.bind(trace_id=trace_id).info("edgar_poll_cycle_start")
                async with get_session() as session:
                    try:
                        result = await poll_once(edgar=edgar, session=session)
                        _logger.bind(
                            trace_id=trace_id,
                            tickers_checked=result.tickers_checked,
                            filings_found=result.filings_found,
                        ).info("edgar_poll_cycle_complete")
                    except Exception as exc:
                        _logger.bind(trace_id=trace_id, error=str(exc)).error(
                            "edgar_poll_cycle_failed"
                        )
            elapsed = (datetime.now(UTC) - cycle_started).total_seconds()
            await asyncio.sleep(max(0.0, cadence - elapsed))
