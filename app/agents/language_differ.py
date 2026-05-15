"""The language-differ agent node.

This module is built incrementally. Task 12 lays down the deterministic
helpers (cosine similarity, greedy alignment, change classification). Task 13
wires the helpers to EDGAR fetching, the embeddings client, the repository,
and the LangGraph orchestrator.

The classifier thresholds are constants here so they can be tuned against
the recall-gate fixture before merge.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, Protocol

from app.memory.repository import Repository
from app.memory.schemas import (
    ChangeType,
    FilingSectionRecord,
    NewFilingSection,
    NewLanguageDiff,
    SectionKind,
    Severity,
)
from app.models.state import AgentState, StateUpdate
from app.observability.logging import current_trace_id, get_logger
from app.tools.sections import ParsedSection, parse_sections

_logger = get_logger()

OWNER = "language_differ"

# Cosine similarity thresholds. Tuned against the 15 hand-labelled
# quarter-pairs in ``tests/fixtures/language_recall/``; do not adjust
# without re-running ``tests/unit/test_recall_gate.py``.
_SIMILARITY_MATCH_THRESHOLD: Final[float] = 0.65
_SIMILARITY_UNCHANGED_THRESHOLD: Final[float] = 0.97
_MAJOR_SIMILARITY_THRESHOLD: Final[float] = 0.85

# Length-based heuristic: a long unmatched paragraph is a major change,
# a short one is a minor one. Word count uses the simple whitespace
# tokeniser in :func:`_word_count`.
_MAJOR_WORD_COUNT_THRESHOLD: Final[int] = 30

_MAX_SUMMARY_DIFFS: Final[int] = 10
_PARAGRAPH_RENDER_CHAR_CAP: Final[int] = 800


class _SupportsFilingDocument(Protocol):
    """Minimal interface the differ needs from the EDGAR client."""

    async def get_filing_document(
        self, *, cik: str, accession_number: str, primary_document: str
    ) -> str: ...


class _SupportsEmbed(Protocol):
    """Minimal interface the differ needs from an embeddings client."""

    @property
    def model(self) -> str: ...

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class _DiffOutcome:
    """Persisted rows and summary dicts produced by :func:`_diff_section`."""

    persisted: list[NewLanguageDiff]
    summary: list[dict[str, Any]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two equal-length vectors.

    Returns ``0.0`` when either vector is zero-length to avoid a divide by
    zero; callers treat that as "no match".
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


_WORD = re.compile(r"\b[\w'-]+\b")


def _word_count(text: str) -> int:
    """Count word-like tokens in ``text``."""
    return len(_WORD.findall(text))


def align_paragraphs(
    current_vectors: list[list[float]],
    prior_vectors: list[list[float]],
    *,
    threshold: float = _SIMILARITY_MATCH_THRESHOLD,
) -> list[tuple[int, int | None]]:
    """Greedy nearest-neighbour alignment of current paragraphs to prior.

    For each current index ``i`` returns ``(i, prior_index)`` where
    ``prior_index`` is the matched prior paragraph index, or ``None`` if no
    prior paragraph above ``threshold`` was available. Each prior paragraph
    is consumed by at most one current paragraph, picked greedily by
    similarity order.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, current in enumerate(current_vectors):
        for j, prior in enumerate(prior_vectors):
            sim = _cosine_similarity(current, prior)
            if sim >= threshold:
                candidates.append((sim, i, j))
    candidates.sort(reverse=True)

    paired_current: dict[int, int] = {}
    consumed_prior: set[int] = set()
    for _sim, i, j in candidates:
        if i in paired_current or j in consumed_prior:
            continue
        paired_current[i] = j
        consumed_prior.add(j)

    return [(i, paired_current.get(i)) for i in range(len(current_vectors))]


def _classify_pair(
    *,
    similarity: float | None,
    words: int,
    is_added: bool = True,
) -> tuple[str, str]:
    """Return ``(change_type, severity)`` for an aligned (or unmatched) pair.

    ``similarity is None`` means the paragraph was unmatched. ``is_added``
    distinguishes a current-side unmatched (``added``) from a prior-side
    unmatched (``removed``); ignored when ``similarity is not None``.
    """
    if similarity is not None:
        if similarity >= _SIMILARITY_UNCHANGED_THRESHOLD:
            return ("unchanged", "minor")
        severity = (
            "major" if similarity < _MAJOR_SIMILARITY_THRESHOLD else "minor"
        )
        return ("modified", severity)
    change_type = "added" if is_added else "removed"
    severity = "major" if words > _MAJOR_WORD_COUNT_THRESHOLD else "minor"
    return (change_type, severity)


async def diff_language(
    state: AgentState,
    *,
    edgar: _SupportsFilingDocument,
    embeddings: _SupportsEmbed,
    repository: Repository,
) -> StateUpdate:
    """Parse, embed, persist, and diff the current filing's MD&A and Risk Factors.

    Always persists the current filing's parsed paragraphs so the next
    quarter has a baseline. Returns a :class:`StateUpdate` whose
    ``language_diffs`` payload is a list of per-section summaries.
    """
    filing = state.filing_event
    filing_row = await repository.get_filing(filing.accession_number)
    primary_document = getattr(filing_row, "primary_document", None)
    if not primary_document:
        return _empty_update(filing, reason="primary_document_missing")

    try:
        html = await edgar.get_filing_document(
            cik=filing.cik,
            accession_number=filing.accession_number,
            primary_document=primary_document,
        )
    except Exception as exc:
        _logger.bind(
            accession=filing.accession_number,
            trace_id=current_trace_id(),
        ).warning("language_differ_fetch_failed", extra={"error": str(exc)})
        return _empty_update(filing, reason="fetch_failed")

    sections = parse_sections(html, form=filing.form.value)
    if not sections:
        return _empty_update(filing, reason="no_sections_parsed")

    payloads: list[dict[str, Any]] = []
    for section in sections:
        paragraph_records = await _persist_paragraphs(
            section=section,
            filing=filing,
            repository=repository,
        )
        payload = await _process_section(
            section=section,
            paragraph_records=paragraph_records,
            embeddings=embeddings,
            filing=filing,
            repository=repository,
        )
        payloads.append(payload)

    _logger.bind(
        accession=filing.accession_number,
        ticker=filing.ticker,
        section_count=len(sections),
        trace_id=current_trace_id(),
    ).info("language_differ_complete")

    return StateUpdate(owner=OWNER, changes={"language_diffs": payloads})


async def _process_section(
    *,
    section: ParsedSection,
    paragraph_records: list[FilingSectionRecord],
    embeddings: _SupportsEmbed,
    filing: Any,
    repository: Repository,
) -> dict[str, Any]:
    """Embed paragraphs, load prior, align, persist diffs; return summary dict."""
    try:
        vectors = await embeddings.aembed([p.text for p in paragraph_records])
        await repository.update_section_embeddings(
            updates=[
                (record.id, vec, embeddings.model)
                for record, vec in zip(paragraph_records, vectors, strict=True)
            ]
        )
    except Exception as exc:
        _logger.bind(
            accession=filing.accession_number,
            section=section.kind.value,
            trace_id=current_trace_id(),
        ).warning("language_differ_embed_failed", extra={"error": str(exc)})
        return _degraded_payload(section.kind.value)

    kind = SectionKind(section.kind.value)
    prior = await repository.get_prior_quarter_sections(
        ticker=filing.ticker,
        section_kind=kind,
        before=filing.filed_at.date(),
    )
    if not prior or any(p.embedding is None for p in prior):
        return _degraded_payload(section.kind.value)

    prior_accession = prior[0].filing_accession
    diffs = _diff_section(
        current=paragraph_records,
        current_vectors=vectors,
        prior=prior,
        section_kind=kind,
        filing_accession=filing.accession_number,
        prior_filing_accession=prior_accession,
    )
    await repository.insert_language_diffs(diffs.persisted)
    return {
        "section": section.kind.value,
        "prior_filing_accession": prior_accession,
        "diff_count": len(diffs.summary),
        "major_count": sum(
            1 for d in diffs.summary if d.get("severity") == "major"
        ),
        "diffs": diffs.summary[:_MAX_SUMMARY_DIFFS],
        "degraded": False,
    }


def _diff_section(
    *,
    current: list[FilingSectionRecord],
    current_vectors: list[list[float]],
    prior: Sequence[FilingSectionRecord],
    section_kind: SectionKind,
    filing_accession: str,
    prior_filing_accession: str,
) -> _DiffOutcome:
    """Align current-vs-prior and classify; returns persisted + summary rows."""
    prior_vectors: list[list[float]] = [
        list(p.embedding) if p.embedding is not None else []
        for p in prior
    ]
    pairs = align_paragraphs(current_vectors, prior_vectors)

    persisted: list[NewLanguageDiff] = []
    summary: list[dict[str, Any]] = []
    consumed_prior: set[int] = set()

    for current_idx, prior_idx in pairs:
        current_para = current[current_idx]
        if prior_idx is not None:
            consumed_prior.add(prior_idx)
            sim = _cosine_similarity(
                current_vectors[current_idx], prior_vectors[prior_idx]
            )
            change_type, severity = _classify_pair(
                similarity=sim, words=_word_count(current_para.text)
            )
            if change_type == "unchanged":
                continue
            prior_para = prior[prior_idx]
            persisted.append(
                NewLanguageDiff(
                    filing_accession=filing_accession,
                    prior_filing_accession=prior_filing_accession,
                    section_kind=section_kind,
                    change_type=ChangeType(change_type),
                    current_section_id=current_para.id,
                    prior_section_id=prior_para.id,
                    similarity=Decimal(f"{sim:.4f}"),
                    severity=Severity(severity),
                )
            )
            summary.append(
                {
                    "change_type": "modified",
                    "current_text": _truncate(current_para.text),
                    "prior_text": _truncate(prior_para.text),
                    "similarity": f"{sim:.4f}",
                    "severity": severity,
                }
            )
        else:
            _change_type, severity = _classify_pair(
                similarity=None,
                words=_word_count(current_para.text),
                is_added=True,
            )
            persisted.append(
                NewLanguageDiff(
                    filing_accession=filing_accession,
                    prior_filing_accession=prior_filing_accession,
                    section_kind=section_kind,
                    change_type=ChangeType.ADDED,
                    current_section_id=current_para.id,
                    severity=Severity(severity),
                )
            )
            summary.append(
                {
                    "change_type": "added",
                    "text": _truncate(current_para.text),
                    "severity": severity,
                }
            )

    for prior_idx, prior_para in enumerate(prior):
        if prior_idx in consumed_prior:
            continue
        _change_type, severity = _classify_pair(
            similarity=None,
            words=_word_count(prior_para.text),
            is_added=False,
        )
        persisted.append(
            NewLanguageDiff(
                filing_accession=filing_accession,
                prior_filing_accession=prior_filing_accession,
                section_kind=section_kind,
                change_type=ChangeType.REMOVED,
                prior_section_id=prior_para.id,
                severity=Severity(severity),
            )
        )
        summary.append(
            {
                "change_type": "removed",
                "prior_text": _truncate(prior_para.text),
                "severity": severity,
            }
        )

    return _DiffOutcome(persisted=persisted, summary=summary)


async def _persist_paragraphs(
    *,
    section: ParsedSection,
    filing: Any,
    repository: Repository,
) -> list[FilingSectionRecord]:
    """Insert section paragraphs and return the resulting records with their ids."""
    rows = [
        NewFilingSection(
            filing_accession=filing.accession_number,
            cik=filing.cik,
            ticker=filing.ticker,
            section_kind=SectionKind(section.kind.value),
            paragraph_index=i,
            text=text,
            text_sha=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            embedding=None,
            embedding_model=None,
        )
        for i, text in enumerate(section.paragraphs)
    ]
    await repository.insert_filing_sections(rows)
    return list(
        await repository.get_filing_sections(
            accession_number=filing.accession_number,
            section_kind=SectionKind(section.kind.value),
        )
    )


def _truncate(text: str) -> str:
    """Cap paragraph text rendered into the synthesiser prompt."""
    if len(text) <= _PARAGRAPH_RENDER_CHAR_CAP:
        return text
    return text[: _PARAGRAPH_RENDER_CHAR_CAP - 3] + "..."


def _degraded_payload(section: str) -> dict[str, Any]:
    """Return the standard degraded payload for a section that could not produce diffs."""
    return {
        "section": section,
        "prior_filing_accession": None,
        "diff_count": 0,
        "major_count": 0,
        "diffs": [],
        "degraded": True,
    }


def _empty_update(filing: Any, *, reason: str) -> StateUpdate:
    """Emit an empty StateUpdate when the differ short-circuits."""
    _logger.bind(
        accession=filing.accession_number,
        reason=reason,
        trace_id=current_trace_id(),
    ).info("language_differ_short_circuit")
    return StateUpdate(
        owner=OWNER,
        changes={"language_diffs": []},
    )
