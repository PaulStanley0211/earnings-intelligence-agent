"""The deterministic numbers critic (v0).

Phase 2 ships the cheapest possible critic that still meets the gate: every
figure the synthesiser writes in the draft note must be cited (``[F#]`` or
``[C#]``), the cited identifier must exist in the state-derived index, and
the cited row's value must match the figure in the note within a metric-
appropriate tolerance.

The critic uses no LLM, so it carries no cost and is fully deterministic.
A later phase layers an LLM-driven critic on top (see ``prompts/critic/``)
to catch claims that survive numeric validation but are unsupported by the
source.

If the third critic pass still rejects, the verdict becomes
``loop_exceeded`` and the note is held for manual review (see the runbook).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from app.agents.citations import (
    CommitmentCitation,
    ComparisonCitation,
    FactCitation,
    LanguageCitation,
    PeerCitation,
    QACitation,
    build_commitment_citations,
    build_comparison_citations,
    build_fact_citations,
    build_language_citations,
    build_peer_citations,
    build_qa_citations,
)
from app.models.state import (
    AgentState,
    CriticFinding,
    CriticVerdict,
    StateUpdate,
)
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER = "critic"

# After this many retries the critic gives up and holds the note for review.
_MAX_CRITIC_ATTEMPTS: Final[int] = 3

# Match a numeric token optionally preceded by a currency sign and followed
# by an optional scale word and optional percent sign. The named groups let
# the analyser reassemble the parsed value without re-running the regex.
_CITED_NUMBER: Final[re.Pattern[str]] = re.compile(
    r"(?P<value>-?\$?\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s+(?P<scale>billion|million|thousand|bn|mn))?"
    r"(?P<percent>\s*%)?"
    r"\s*\[(?P<cite>[FC]\d+)\]",
    re.IGNORECASE,
)

# Same shape, but used to find "meaningful" numbers that lack a citation.
# "Meaningful" means: currency-prefixed, scale-suffixed, or percent-suffixed.
# Bare integers and decimals are skipped to avoid flagging years and section
# numbers - the synthesiser is only required to cite figures that look like
# financial values.
_UNCITED_NUMBER: Final[re.Pattern[str]] = re.compile(
    r"(?P<value>-?\$\d+(?:,\d{3})*(?:\.\d+)?|"
    r"-?\d+(?:,\d{3})*(?:\.\d+)?\s+(?:billion|million|thousand|bn|mn)|"
    r"-?\d+(?:\.\d+)?\s*%)",
    re.IGNORECASE,
)

_CITED_LANGUAGE: Final[re.Pattern[str]] = re.compile(
    r"\[(?P<cite>[LQKP]\d+)\]",
    re.IGNORECASE,
)

_QUOTE_RX: Final[re.Pattern[str]] = re.compile(r'"([^"]+)"')

_SCALE_FACTOR: Final[dict[str, Decimal]] = {
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "million": Decimal("1000000"),
    "mn": Decimal("1000000"),
    "thousand": Decimal("1000"),
}

# Tolerances by citation kind. Currency uses a relative tolerance because
# real numbers swing many orders of magnitude; per-share and percentage use
# absolute tolerances because their magnitude is bounded.
_REL_TOLERANCE: Final[Decimal] = Decimal("0.01")  # 1 percent
_ABS_PER_SHARE: Final[Decimal] = Decimal("0.01")  # one cent
_ABS_PERCENT: Final[Decimal] = Decimal("0.05")  # five basis points


@dataclass(frozen=True)
class _ParsedNumber:
    """A number parsed out of the draft note with its surface form."""

    surface: str
    value: Decimal
    is_percent: bool


def critique_draft(state: AgentState) -> StateUpdate:
    """Validate ``state.draft_note`` against the citation index from ``state``.

    Returns a :class:`StateUpdate` carrying ``critic_findings``, the new
    ``critic_verdict``, an incremented ``critic_attempts`` counter, and -
    when accepted - the promoted ``final_note``.
    """
    if state.draft_note is None:
        finding = CriticFinding(
            layer="numbers",
            severity="error",
            message="critic invoked with no draft note",
        )
        return _result(state, [finding], accepted=False)

    fact_index = {c.identifier: c for c in build_fact_citations(state.financials)}
    comparison_index = {
        c.identifier: c for c in build_comparison_citations(state.comparisons)
    }
    language_index = {
        c.identifier: c for c in build_language_citations(state.language_diffs)
    }
    qa_index = {c.identifier: c for c in build_qa_citations(state.qa_pairs)}
    commitment_index = {
        c.identifier: c for c in build_commitment_citations(state.commitments)
    }
    peer_index = {c.identifier: c for c in build_peer_citations(state.peer_context)}

    findings: list[CriticFinding] = []
    cited_spans: list[tuple[int, int]] = []
    for match in _CITED_NUMBER.finditer(state.draft_note):
        cited_spans.append(match.span())
        validated = _validate_cited(match, fact_index, comparison_index)
        if validated is not None:
            findings.append(validated)
    findings.extend(_find_uncited(state.draft_note, cited_spans))
    findings.extend(
        _validate_quote_citations(
            state.draft_note,
            language_index=language_index,
            qa_index=qa_index,
            commitment_index=commitment_index,
            peer_index=peer_index,
        )
    )

    accepted = not any(f.severity == "error" for f in findings)
    _logger.bind(
        accession=state.filing_event.accession_number,
        ticker=state.filing_event.ticker,
        attempts=state.critic_attempts + 1,
        finding_count=len(findings),
        accepted=accepted,
        trace_id=current_trace_id(),
    ).info("critic_complete")
    return _result(state, findings, accepted=accepted)


def _validate_cited(
    match: re.Match[str],
    fact_index: dict[str, FactCitation],
    comparison_index: dict[str, ComparisonCitation],
) -> CriticFinding | None:
    """Resolve the citation and verify the cited value matches the surface."""
    cite_id = match.group("cite").upper()
    parsed = _parse_number(
        raw_value=match.group("value"),
        scale=match.group("scale"),
        percent=match.group("percent"),
    )
    if parsed is None:
        return CriticFinding(
            layer="numbers",
            severity="error",
            message=(
                f"cited number {match.group(0)!r} could not be parsed; "
                "the synthesiser must emit a recognisable numeric token"
            ),
        )
    if cite_id.startswith("F"):
        fact = fact_index.get(cite_id)
        if fact is None:
            return CriticFinding(
                layer="numbers",
                severity="error",
                message=(
                    f"citation {cite_id!r} references no known financial fact"
                ),
            )
        return _check_fact_match(parsed, fact, surface=match.group(0))
    fact = None
    comparison = comparison_index.get(cite_id)
    if comparison is None:
        return CriticFinding(
            layer="numbers",
            severity="error",
            message=f"citation {cite_id!r} references no known comparison",
        )
    return _check_comparison_match(parsed, comparison, surface=match.group(0))


def _parse_number(
    *, raw_value: str, scale: str | None, percent: str | None
) -> _ParsedNumber | None:
    """Convert a regex match into a Decimal-scaled value.

    Returns ``None`` only when the digit core cannot be parsed; scale and
    percent suffixes are otherwise tolerated.
    """
    cleaned = raw_value.replace(",", "").replace("$", "").strip()
    try:
        value = Decimal(cleaned)
    except Exception:
        return None
    if scale:
        factor = _SCALE_FACTOR.get(scale.lower())
        if factor is None:
            return None
        value = value * factor
    surface = (
        f"{raw_value}"
        f"{(' ' + scale) if scale else ''}"
        f"{percent if percent else ''}"
    )
    return _ParsedNumber(surface=surface, value=value, is_percent=percent is not None)


def _check_fact_match(
    parsed: _ParsedNumber,
    fact: FactCitation,
    *,
    surface: str,
) -> CriticFinding | None:
    """Compare ``parsed`` to a financial-fact citation within tolerance."""
    if parsed.is_percent:
        return CriticFinding(
            layer="numbers",
            severity="error",
            message=(
                f"{surface!r} cites a financial fact but is rendered as a "
                "percentage; only consensus comparisons are percentage-valued"
            ),
        )
    if _per_share(fact.concept):
        if (parsed.value - fact.value).copy_abs() <= _ABS_PER_SHARE:
            return None
        return _mismatch_finding(surface, fact.identifier, parsed.value, fact.value)
    return _relative_match(parsed.value, fact.value, surface, fact.identifier)


def _check_comparison_match(
    parsed: _ParsedNumber,
    comparison: ComparisonCitation,
    *,
    surface: str,
) -> CriticFinding | None:
    """Compare ``parsed`` to any of the comparison's exposed numeric fields."""
    candidates: list[Decimal] = [comparison.reported_value]
    for candidate in (
        comparison.consensus_value,
        comparison.surprise_abs,
        comparison.surprise_pct,
    ):
        if candidate is not None:
            candidates.append(candidate)
    if parsed.is_percent:
        tolerance = _ABS_PERCENT
        for candidate in candidates:
            if (parsed.value - candidate).copy_abs() <= tolerance:
                return None
        return _mismatch_finding(
            surface, comparison.identifier, parsed.value, candidates[0]
        )
    for candidate in candidates:
        if _within_relative(parsed.value, candidate):
            return None
        if (parsed.value - candidate).copy_abs() <= _ABS_PER_SHARE:
            return None
    return _mismatch_finding(
        surface, comparison.identifier, parsed.value, candidates[0]
    )


def _relative_match(
    parsed: Decimal,
    expected: Decimal,
    surface: str,
    identifier: str,
) -> CriticFinding | None:
    """Return ``None`` when ``parsed`` is within 1 percent of ``expected``."""
    if _within_relative(parsed, expected):
        return None
    return _mismatch_finding(surface, identifier, parsed, expected)


def _within_relative(parsed: Decimal, expected: Decimal) -> bool:
    """Return ``True`` when ``parsed`` is within 1 percent of ``expected``."""
    if expected == 0:
        return parsed == 0
    diff = (parsed - expected).copy_abs()
    return diff / expected.copy_abs() <= _REL_TOLERANCE


def _mismatch_finding(
    surface: str, identifier: str, parsed: Decimal, expected: Decimal
) -> CriticFinding:
    """Build the ``error`` finding for a number that did not match its citation."""
    return CriticFinding(
        layer="numbers",
        severity="error",
        message=(
            f"{surface!r} cites {identifier!r} but parsed value "
            f"{parsed} is outside tolerance of the cited value {expected}"
        ),
    )


def _per_share(concept: str) -> bool:
    """Identify per-share concepts so the critic uses an absolute tolerance."""
    return concept.startswith("EarningsPerShare")


def _find_uncited(text: str, cited_spans: list[tuple[int, int]]) -> list[CriticFinding]:
    """Flag any meaningful numeric token in ``text`` outside the cited spans."""
    findings: list[CriticFinding] = []
    for match in _UNCITED_NUMBER.finditer(text):
        if _covered_by_cited(match.span(), cited_spans):
            continue
        findings.append(
            CriticFinding(
                layer="numbers",
                severity="error",
                message=(
                    f"number {match.group(0)!r} has no [F#]/[C#] citation; "
                    "every numeric figure must trace to the supplied facts"
                ),
            )
        )
    return findings


def _covered_by_cited(
    span: tuple[int, int], cited_spans: list[tuple[int, int]]
) -> bool:
    """Return ``True`` when ``span`` falls inside any already-cited span."""
    start, end = span
    return any(
        start >= cited_start and end <= cited_end
        for cited_start, cited_end in cited_spans
    )


def _result(
    state: AgentState,
    findings: list[CriticFinding],
    *,
    accepted: bool,
) -> StateUpdate:
    """Assemble the ``StateUpdate`` mutating only critic-owned fields."""
    attempts = state.critic_attempts + 1
    changes: dict[str, object] = {
        "critic_findings": findings,
        "critic_attempts": attempts,
    }
    if accepted:
        changes["critic_verdict"] = CriticVerdict.ACCEPTED
        changes["final_note"] = state.draft_note
    elif attempts >= _MAX_CRITIC_ATTEMPTS:
        changes["critic_verdict"] = CriticVerdict.LOOP_EXCEEDED
    else:
        changes["critic_verdict"] = CriticVerdict.REJECTED
    return StateUpdate(owner=OWNER, changes=changes)


def _validate_quote_citations(
    text: str,
    *,
    language_index: dict[str, LanguageCitation],
    qa_index: dict[str, QACitation],
    commitment_index: dict[str, CommitmentCitation],
    peer_index: dict[str, PeerCitation],
) -> list[CriticFinding]:
    """Validate each ``[L#]``/``[Q#]``/``[K#]``/``[P#]`` quote citation in ``text``.

    For every quote-style citation marker the function resolves the id
    against the matching namespace index and verifies that the surrounding
    line text matches the resolved source within the standard 90%
    character-similarity tolerance.
    """
    findings: list[CriticFinding] = []
    for line in text.splitlines():
        for match in _CITED_LANGUAGE.finditer(line):
            cite_id = match.group("cite").upper()
            resolved = _resolve_quote_citation(
                cite_id,
                language_index=language_index,
                qa_index=qa_index,
                commitment_index=commitment_index,
                peer_index=peer_index,
            )
            if resolved is None:
                findings.append(
                    CriticFinding(
                        layer="quote",
                        severity="error",
                        message=(
                            f"citation {cite_id!r} references no known "
                            f"{_namespace_label(cite_id)}"
                        ),
                    )
                )
                continue
            quoted_part = _strip_citation_from_line(line, match.span())
            if not _language_match(quoted_part, resolved):
                findings.append(
                    CriticFinding(
                        layer="quote",
                        severity="error",
                        message=(
                            f"text near {cite_id!r} does not match the cited "
                            f"{_namespace_label(cite_id)} "
                            "(substring or 90% char similarity)"
                        ),
                    )
                )
    return findings


def _resolve_quote_citation(
    cite_id: str,
    *,
    language_index: dict[str, LanguageCitation],
    qa_index: dict[str, QACitation],
    commitment_index: dict[str, CommitmentCitation],
    peer_index: dict[str, PeerCitation],
) -> str | None:
    """Return the source text for ``cite_id`` or ``None`` when not found."""
    namespace = cite_id[:1]
    if namespace == "L":
        language = language_index.get(cite_id)
        return language.text if language is not None else None
    if namespace == "Q":
        qa = qa_index.get(cite_id)
        return qa.source_text if qa is not None else None
    if namespace == "K":
        commitment = commitment_index.get(cite_id)
        return commitment.source_text if commitment is not None else None
    if namespace == "P":
        peer = peer_index.get(cite_id)
        return peer.text if peer is not None else None
    return None


def _namespace_label(cite_id: str) -> str:
    """Human-readable label for the citation's namespace, used in messages."""
    return {
        "L": "language change",
        "Q": "Q&A pair",
        "K": "management commitment",
        "P": "peer commentary",
    }.get(cite_id[:1], "quote source")


def _strip_citation_from_line(line: str, span: tuple[int, int]) -> str:
    """Remove the citation token and bullet markup so we can compare prose."""
    start, end = span
    stripped = (line[:start] + line[end:]).strip()
    for prefix in ("- ", "* ", "+ "):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
    return stripped.strip(" .")


def _language_match(quoted: str, indexed_text: str) -> bool:
    """Return True when ``quoted`` is a substring or has >=90% char similarity.

    When ``quoted`` contains a ``"..."``-delimited substring, score only the
    first quoted substring; this avoids penalising editorial framing
    around a quoted line (``'Sarah Lee asked "..."'``). Lines without quotes
    score on the full line.
    """
    from difflib import SequenceMatcher

    if not quoted:
        return False
    q_match = _QUOTE_RX.search(quoted)
    candidate = q_match.group(1) if q_match else quoted
    q = _normalise(candidate)
    t = _normalise(indexed_text)
    if not q or not t:
        return False
    if q in t:
        return True
    return SequenceMatcher(a=q, b=t).ratio() >= 0.90


def _normalise(text: str) -> str:
    """Collapse whitespace, lowercase, strip trailing punctuation."""
    collapsed = re.sub(r"\s+", " ", text).strip().lower()
    return collapsed.strip(" .,;:!?")
