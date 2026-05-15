"""Parse 10-Q and 10-K HTML into MD&A and Risk Factors paragraph lists.

The parser is intentionally heuristic: SEC HTML varies widely across
filers and over time. We rely on three signals:

1. BeautifulSoup with the ``lxml`` backend converts the HTML to flat
   text with paragraph boundaries preserved.
2. A regex over the flat text finds the start anchors for the sections
   we care about (Item 2 / Item 7 / Item 1A).
3. The end of a section is the next ``Item <n>`` anchor.

A few sanity filters drop boilerplate that survives the strip (Table of
Contents headers, short cross-references) and collapse `<table>` elements
to a sentinel paragraph since their numeric content is already on the
XBRL track.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from bs4 import BeautifulSoup, Tag

_TABLE_SENTINEL: Final[str] = "[TABLE]"
_MIN_PARAGRAPH_CHARS: Final[int] = 40
_MAX_PARAGRAPH_CHARS: Final[int] = 4000


class SectionKind(StrEnum):
    """Kind of parsed filing section."""

    MDA = "mda"
    RISK_FACTORS = "risk_factors"


@dataclass(frozen=True)
class ParsedSection:
    """One section's worth of paragraphs."""

    kind: SectionKind
    paragraphs: list[str]


_MDA_10Q = re.compile(
    r"^\s*item\s+2\.?\s+management.{0,2}s discussion",
    re.IGNORECASE,
)
_MDA_10K = re.compile(
    r"^\s*item\s+7\.?\s+management.{0,2}s discussion",
    re.IGNORECASE,
)
_RISK_FACTORS = re.compile(
    r"^\s*item\s+1a\.?\s+risk factors",
    re.IGNORECASE,
)
_ITEM_HEADING = re.compile(r"^\s*item\s+\d", re.IGNORECASE)


def parse_sections(html: str, *, form: str) -> list[ParsedSection]:
    """Return MD&A and Risk Factors sections parsed out of ``html``.

    ``form`` is one of ``"10-Q"`` or ``"10-K"`` and selects the MD&A item
    number. Returns ``[]`` when neither section is found; this is a normal
    outcome for some filings and is handled by the caller as a degrade.
    """
    flat = _flatten_html(html)
    lines = [line for line in flat.split("\n") if line.strip()]
    out: list[ParsedSection] = []

    mda_anchor = _MDA_10K if form == "10-K" else _MDA_10Q
    mda_paragraphs = _extract_section(lines, mda_anchor)
    if mda_paragraphs:
        out.append(ParsedSection(kind=SectionKind.MDA, paragraphs=mda_paragraphs))

    risk_paragraphs = _extract_section(lines, _RISK_FACTORS)
    if risk_paragraphs:
        out.append(
            ParsedSection(kind=SectionKind.RISK_FACTORS, paragraphs=risk_paragraphs)
        )

    return out


def _flatten_html(html: str) -> str:
    """Render HTML as a flat string with one paragraph per line.

    Replaces ``<table>`` elements with the ``[TABLE]`` sentinel (always
    below the min-paragraph filter, so it is dropped). Block-level tags
    introduce a newline; inline whitespace is collapsed.
    """
    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table"):
        table.replace_with(_TABLE_SENTINEL)
    block_tags = {
        "p", "div", "li", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    }
    for tag in soup.find_all(True):
        if isinstance(tag, Tag) and tag.name in block_tags:
            tag.append("\n")
    text = soup.get_text(separator=" ")
    return _normalise_whitespace(text)


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces and tabs but preserve newlines."""
    lines = []
    for raw in text.split("\n"):
        cleaned = re.sub(r"[ \t]+", " ", raw).strip()
        lines.append(cleaned)
    return "\n".join(lines)


def _extract_section(lines: list[str], anchor: re.Pattern[str]) -> list[str]:
    """Return paragraph lines between ``anchor`` and the next ``Item N``."""
    start = _find_anchor(lines, anchor)
    if start is None:
        return []
    end = _find_end(lines, start + 1)
    candidates = lines[start + 1 : end]
    return [
        line
        for line in candidates
        if _MIN_PARAGRAPH_CHARS <= len(line) <= _MAX_PARAGRAPH_CHARS
    ]


def _find_anchor(lines: list[str], anchor: re.Pattern[str]) -> int | None:
    """Return the index of the first line matching ``anchor`` or ``None``."""
    for idx, line in enumerate(lines):
        if anchor.match(line):
            return idx
    return None


def _find_end(lines: list[str], start: int) -> int:
    """Return the index of the next ``Item N`` heading, or len(lines)."""
    for idx in range(start, len(lines)):
        if _ITEM_HEADING.match(lines[idx]):
            return idx
    return len(lines)
