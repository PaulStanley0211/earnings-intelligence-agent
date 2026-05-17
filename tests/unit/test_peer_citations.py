"""Unit tests for [P#] peer citations."""

from __future__ import annotations

from app.agents.citations import PeerCitation, build_peer_citations
from app.models.state import PeerContextEntry


def test_build_peer_citations_assigns_sequential_ids() -> None:
    entries = [
        PeerContextEntry(
            peer_ticker="GOOGL",
            kind="language_diff",
            text="Cloud pricing pressure.",
            source_filing_accession="x-1",
            severity="major",
        ),
        PeerContextEntry(
            peer_ticker="AAPL",
            kind="commitment",
            text="Margins to expand.",
            source_filing_accession="x-2",
        ),
    ]
    cits = build_peer_citations(entries)
    assert [c.identifier for c in cits] == ["P0", "P1"]
    assert isinstance(cits[0], PeerCitation)
    assert cits[0].peer_ticker == "GOOGL"


def test_build_peer_citations_empty() -> None:
    assert build_peer_citations([]) == []
