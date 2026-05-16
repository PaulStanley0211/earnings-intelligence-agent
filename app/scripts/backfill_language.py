"""Backfill the language baseline for one or more watchlist tickers.

For each ticker, fetch the most-recent ``N`` 10-Q / 10-K filings, parse
MD&A and Risk Factors, embed paragraphs, and persist ``filing_sections``.

The script is operator-triggered; it is not invoked by the graph or any
startup hook. Failure on filing N preserves the rows already committed
for filings 1..N-1 (per-filing transaction boundary).

Usage::

    uv run python -m app.scripts.backfill_language --ticker MSFT --quarters 4
    uv run python -m app.scripts.backfill_language --quarters 4

Note on live ``_main`` wiring: the :class:`~app.tools.embeddings.EmbeddingsClient`
requires a ``repository_factory: Callable[[], Repository]`` that produces a
fresh, *already-open* session per call. Wiring this cleanly across async
context managers is non-trivial; the live path below creates a one-shot
factory bound to a single long-lived session for the duration of the
backfill.  For production runs where per-call session isolation is required,
wrap ``run_backfill`` with an explicit ``EmbeddingsClient`` constructed in the
caller.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.memory.models import Filing
from app.memory.repository import Repository
from app.memory.schemas import (
    NewFiling,
    NewFilingSection,
    SectionKind,
)
from app.models.state import FilingForm
from app.observability.logging import get_logger
from app.tools.edgar import RecentFiling
from app.tools.sections import parse_sections

_logger = get_logger()


class _SupportsSubmissions(Protocol):
    """The shape the backfill needs from the EDGAR client."""

    async def get_submissions(self, *, cik: str) -> Any: ...

    async def get_filing_document(
        self, *, cik: str, accession_number: str, primary_document: str
    ) -> str: ...


class _SupportsEmbed(Protocol):
    """The shape the backfill needs from the embeddings client."""

    @property
    def model(self) -> str: ...

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]: ...


async def run_backfill(
    *,
    tickers: list[str] | None,
    quarters: int,
    edgar: _SupportsSubmissions,
    embeddings: _SupportsEmbed,
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, Any]:
    """Run the backfill across ``tickers`` (or the full watchlist when None).

    Returns a summary dict with keys ``tickers``, ``filings_parsed``,
    ``paragraphs_inserted``, and ``elapsed_seconds``.
    """
    started = time.time()
    target_tickers = await _resolve_tickers(tickers, session_factory)
    filings_parsed = 0
    paragraphs_inserted = 0

    for entry in target_tickers:
        submissions = await edgar.get_submissions(cik=entry["cik"])
        recent = _select_quarterly_filings(submissions.recent_filings, quarters)
        for recent_filing in recent:
            inserted = await _backfill_one(
                entry=entry,
                recent_filing=recent_filing,
                edgar=edgar,
                embeddings=embeddings,
                session_factory=session_factory,
            )
            if inserted is not None:
                filings_parsed += 1
                paragraphs_inserted += inserted

    return {
        "tickers": [e["ticker"] for e in target_tickers],
        "filings_parsed": filings_parsed,
        "paragraphs_inserted": paragraphs_inserted,
        "elapsed_seconds": round(time.time() - started, 2),
    }


async def _resolve_tickers(
    tickers: list[str] | None,
    session_factory: async_sessionmaker[AsyncSession],
) -> list[dict[str, str]]:
    """Return ``{ticker, cik}`` pairs for ``tickers`` or the active watchlist."""
    async with session_factory() as session:
        repo = Repository(session)
        active = await repo.list_active_watchlist()
    if tickers:
        wanted = set(tickers)
        return [{"ticker": w.ticker, "cik": w.cik} for w in active if w.ticker in wanted]
    return [{"ticker": w.ticker, "cik": w.cik} for w in active]


def _select_quarterly_filings(
    filings: list[RecentFiling], quarters: int
) -> list[RecentFiling]:
    """Pick the most-recent ``quarters`` 10-Q + 10-K filings."""
    eligible = [f for f in filings if f.form in {"10-Q", "10-K"}]
    return eligible[:quarters]


async def _backfill_one(
    *,
    entry: dict[str, str],
    recent_filing: RecentFiling,
    edgar: _SupportsSubmissions,
    embeddings: _SupportsEmbed,
    session_factory: async_sessionmaker[AsyncSession],
) -> int | None:
    """Process one filing. Returns paragraph count inserted, or None on skip.

    Opens and commits its own session so per-filing transaction boundaries
    are maintained: a failure on filing N does not roll back filings 1..N-1.
    """
    async with session_factory() as session:
        repo = Repository(session)
        existing = await repo.get_filing_sections(
            accession_number=recent_filing.accession_number,
            section_kind=SectionKind.MDA,
        )
        if existing:
            return None

        if await repo.get_filing(recent_filing.accession_number) is None:
            await repo.record_filing(
                filing=NewFiling(
                    accession_number=recent_filing.accession_number,
                    cik=entry["cik"],
                    ticker=entry["ticker"],
                    form=FilingForm(recent_filing.form),
                    filed_at=datetime.combine(
                        recent_filing.filing_date,
                        datetime.min.time(),
                        tzinfo=UTC,
                    ),
                    source_url=(
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{int(entry['cik'])}/{recent_filing.accession_number}.txt"
                    ),
                )
            )

        if recent_filing.primary_document:
            await session.execute(
                sa_update(Filing)
                .where(Filing.accession_number == recent_filing.accession_number)
                .values(primary_document=recent_filing.primary_document)
            )

        html = await edgar.get_filing_document(
            cik=entry["cik"],
            accession_number=recent_filing.accession_number,
            primary_document=recent_filing.primary_document or "",
        )
        sections = parse_sections(html, form=recent_filing.form)
        total_inserted = 0

        for section in sections:
            rows = [
                NewFilingSection(
                    filing_accession=recent_filing.accession_number,
                    cik=entry["cik"],
                    ticker=entry["ticker"],
                    section_kind=SectionKind(section.kind.value),
                    paragraph_index=i,
                    text=text,
                    text_sha=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    embedding=None,
                    embedding_model=None,
                )
                for i, text in enumerate(section.paragraphs)
            ]
            inserted = await repo.insert_filing_sections(rows)
            total_inserted += inserted

            reloaded = await repo.get_filing_sections(
                accession_number=recent_filing.accession_number,
                section_kind=SectionKind(section.kind.value),
            )
            vectors = await embeddings.aembed([r.text for r in reloaded])
            await repo.update_section_embeddings(
                updates=[
                    (r.id, v, embeddings.model)
                    for r, v in zip(reloaded, vectors, strict=True)
                ]
            )

        await session.commit()
        return total_inserted


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the backfill CLI."""
    parser = argparse.ArgumentParser(
        description="Backfill MD&A / Risk Factors sections for watchlist tickers."
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=None,
        help="Ticker to backfill (repeatable). Defaults to full active watchlist.",
    )
    parser.add_argument(
        "--quarters",
        type=int,
        default=4,
        help="Number of prior quarters to backfill per ticker (default: 4).",
    )
    return parser.parse_args()


async def _main() -> None:  # pragma: no cover
    """Live entry point wiring real EDGAR + OpenAI clients.

    The :class:`~app.tools.embeddings.EmbeddingsClient` cost guard needs a
    ``Repository`` with a live session. We create one shared session for the
    duration of the run and close it on completion. This is acceptable for a
    short-lived operator script but is not suitable for concurrent production
    workers.
    """
    from app.config import get_settings
    from app.memory.db import dispose_engine, get_session_factory
    from app.observability.logging import configure_logging
    from app.tools.edgar import EdgarClient
    from app.tools.embeddings import EmbeddingsClient

    args = _parse_args()
    settings = get_settings()
    configure_logging(level=settings.log_level)
    session_factory = get_session_factory()

    async with session_factory() as spend_session:
        spend_repo = Repository(spend_session)

        def _repo_factory() -> Repository:
            return spend_repo

        async with EdgarClient(user_agent=settings.edgar_user_agent) as edgar:
            embed_client = EmbeddingsClient(
                api_key=settings.openai_api_key,
                repository_factory=_repo_factory,
                model=settings.embeddings_model,
                max_daily_cost_usd=settings.max_daily_llm_cost_usd,
            )
            summary = await run_backfill(
                tickers=args.ticker,
                quarters=args.quarters,
                edgar=edgar,
                embeddings=embed_client,
                session_factory=session_factory,
            )

    _logger.bind(**summary).info("backfill_complete")
    await dispose_engine()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
