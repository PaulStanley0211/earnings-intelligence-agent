"""The peer_reader agent node.

Pure DB read over the curated ``peers`` table + each peer's most-recent
language diffs (10-K/10-Q) and open commitments (transcript). Emits a
typed ``peer_context`` list the synthesizer renders into the prompt
with ``[P#]`` citations.

No LLM call. No side-effects. On any DB error the node degrades to an
empty context so the pipeline continues without peer commentary.
"""

from __future__ import annotations

from app.memory.repository import Repository
from app.models.state import AgentState, PeerContextEntry, StateUpdate
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER = "peer_reader"

_PEER_FRESHNESS_DAYS = 180


async def read_peers(
    state: AgentState,
    *,
    repository: Repository,
) -> StateUpdate:
    """Return a StateUpdate populating ``peer_context``.

    Reads the curated peer list for the filing's ticker, then fetches
    recent language-diff and commitment signals for each peer. Degrades
    to an empty list on any DB error so the pipeline continues without
    peer commentary.
    """
    ticker = state.filing_event.ticker
    entries: list[PeerContextEntry] = []

    try:
        peer_tickers = await repository.list_peers(ticker=ticker)
        for peer_ticker in peer_tickers:
            signals = await repository.get_recent_peer_signals(
                peer_ticker=peer_ticker,
                max_age_days=_PEER_FRESHNESS_DAYS,
            )
            for diff in signals.language_diffs:
                entries.append(
                    PeerContextEntry(
                        peer_ticker=peer_ticker,
                        kind="language_diff",
                        text=diff.text,
                        source_filing_accession=diff.source_filing_accession,
                        severity=diff.severity,  # type: ignore[arg-type]
                    )
                )
            for commitment in signals.commitments:
                entries.append(
                    PeerContextEntry(
                        peer_ticker=peer_ticker,
                        kind="commitment",
                        text=commitment.text,
                        source_filing_accession=commitment.source_filing_accession,
                    )
                )
    except Exception as exc:  # degrade, don't crash
        _logger.bind(
            ticker=ticker,
            error=str(exc),
            trace_id=current_trace_id(),
        ).error("peer_reader_failed")
        return StateUpdate(owner=OWNER, changes={"peer_context": []})

    _logger.bind(
        ticker=ticker,
        peer_count=len({e.peer_ticker for e in entries}),
        entry_count=len(entries),
        trace_id=current_trace_id(),
    ).info("peer_reader_complete")
    return StateUpdate(owner=OWNER, changes={"peer_context": entries})
