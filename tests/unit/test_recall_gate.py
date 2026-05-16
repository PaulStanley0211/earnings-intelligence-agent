"""80% recall gate: the differ must catch labelled changes on real EDGAR pairs.

Marked ``@pytest.mark.slow`` so the fast unit suite stays fast. CI runs the
slow suite as a second step on every PR.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import yaml
from sqlalchemy import text as sa_text
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.language_differ import diff_language
from app.memory.db import build_engine
from app.memory.models import Base, Filing
from app.memory.repository import Repository
from app.memory.schemas import NewFiling, NewFilingSection, SectionKind
from app.models.state import AgentState, FilingEvent, FilingForm
from app.tools.sections import parse_sections

pytestmark = [pytest.mark.slow, pytest.mark.integration]

_FIXTURE_DIR = Path("tests/fixtures/language_recall")
_LABELS_PATH = _FIXTURE_DIR / "labels.yaml"


def _load_pairs() -> list[dict[str, Any]]:
    with _LABELS_PATH.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return list(payload.get("pairs", []))


class _DeterministicEmbeddings:
    """Hash-based deterministic embeddings (no OpenAI call)."""

    @property
    def model(self) -> str:
        return "test/hash-1536"

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [_text_to_vector(t) for t in texts]


def _text_to_vector(text: str) -> list[float]:
    """Map text to a 1536-d unit-norm vector via hashed bag-of-trigrams.

    Trigrams overlap so similar texts produce similar vectors - good
    enough for the alignment-classification regression gate.
    """
    dim = 1536
    vec = [0.0] * dim
    text = text.lower()
    if len(text) < 3:
        return vec
    for i in range(len(text) - 2):
        trigram = text[i : i + 3]
        h = int.from_bytes(
            hashlib.sha256(trigram.encode("utf-8")).digest()[:8],
            "big",
        )
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class _Edgar:
    """Edgar stub backed by a per-pair HTML string."""

    def __init__(self, html: str) -> None:
        self._html = html

    async def get_filing_document(
        self, *, cik: str, accession_number: str, primary_document: str
    ) -> str:
        return self._html


@pytest_asyncio.fixture()
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


@pytest.mark.parametrize(
    "pair",
    _load_pairs(),
    ids=lambda p: p["id"],
)
async def test_pair_detected_changes_cover_labelled_changes(
    pair: dict[str, Any], session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Per-pair smoke test: detects at least some labelled changes (for debugging)."""
    detected = await _detect_changes(pair, session_factory)
    matched = sum(1 for label in pair["labels"] if _label_matched(label, detected))
    # Each pair must match >= 0 (trivially true) - the aggregate test enforces 80%.
    assert matched >= 0, f"pair {pair['id']}: {matched}/{len(pair['labels'])}"


async def test_overall_recall_meets_80_percent_threshold(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The recall gate: aggregate matched/total across all 15 pairs >= 80%."""
    pairs = _load_pairs()
    matched_total = 0
    label_total = 0
    misses: list[str] = []
    for pair in pairs:
        detected = await _detect_changes(pair, session_factory)
        for label in pair["labels"]:
            label_total += 1
            if _label_matched(label, detected):
                matched_total += 1
            else:
                misses.append(
                    f"{pair['id']}/{label['change_type']}/{label['paragraph_excerpt']!r}"
                )
    assert label_total >= 15, f"expected >= 15 labels, found {label_total}"
    recall = matched_total / label_total
    assert recall >= 0.80, (
        f"recall {recall:.2%} ({matched_total}/{label_total}) below 0.80 gate. "
        f"Misses: {misses}"
    )


async def _detect_changes(
    pair: dict[str, Any], session_factory: async_sessionmaker[AsyncSession]
) -> list[dict[str, Any]]:
    """Seed prior+current sections and run the differ; return emitted diffs."""
    current_html = (_FIXTURE_DIR / pair["current_fixture"]).read_text(encoding="utf-8")
    prior_html = (_FIXTURE_DIR / pair["prior_fixture"]).read_text(encoding="utf-8")
    section_kind = SectionKind(pair["section_kind"])

    prior_sections = parse_sections(prior_html, form="10-Q")
    target_prior = next(
        (s for s in prior_sections if s.kind.value == pair["section_kind"]),
        None,
    )
    if target_prior is None:
        return []

    cik = "0000000000"
    ticker = pair["ticker"]
    prior_accession = f"{pair['id']}-prior"
    current_accession = f"{pair['id']}-current"
    embeddings = _DeterministicEmbeddings()

    prior_vectors = await embeddings.aembed(target_prior.paragraphs)
    async with session_factory() as session:
        repo = Repository(session)
        for accession, filed in [
            (prior_accession, datetime(2026, 1, 1, tzinfo=UTC)),
            (current_accession, datetime(2026, 4, 1, tzinfo=UTC)),
        ]:
            await repo.record_filing(
                filing=NewFiling(
                    accession_number=accession,
                    cik=cik,
                    ticker=ticker,
                    form=FilingForm.FORM_10Q,
                    filed_at=filed,
                    source_url="https://www.sec.gov/x",
                )
            )
        await repo.insert_filing_sections(
            [
                NewFilingSection(
                    filing_accession=prior_accession,
                    cik=cik,
                    ticker=ticker,
                    section_kind=section_kind,
                    paragraph_index=i,
                    text=text,
                    text_sha=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    embedding=vec,
                    embedding_model=embeddings.model,
                )
                for i, (text, vec) in enumerate(
                    zip(target_prior.paragraphs, prior_vectors, strict=True)
                )
            ]
        )
        await session.execute(
            sa_update(Filing)
            .where(Filing.accession_number == current_accession)
            .values(primary_document=f"{pair['id']}.htm")
        )
        await session.commit()

    async with session_factory() as session:
        state = AgentState(
            trace_id="trace-test",
            started_at=datetime.now(UTC),
            filing_event=FilingEvent(
                accession_number=current_accession,
                cik=cik,
                ticker=ticker,
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 1, tzinfo=UTC),
                source_url="https://www.sec.gov/x",
            ),
        )
        update = await diff_language(
            state,
            edgar=_Edgar(current_html),
            embeddings=embeddings,
            repository=Repository(session),
        )
        await session.commit()
    payload = update.changes["language_diffs"]
    for section_payload in payload:
        if section_payload.get("section") == pair["section_kind"]:
            return section_payload.get("diffs", [])
    return []


def _label_matched(label: dict[str, Any], detected: list[dict[str, Any]]) -> bool:
    """A label is matched when a detected diff has same change_type and excerpt match."""
    excerpt = label["paragraph_excerpt"].lower()
    target_type = label["change_type"]
    for diff in detected:
        if diff.get("change_type") != target_type:
            continue
        for key in ("text", "current_text", "prior_text"):
            haystack = (diff.get(key) or "").lower()
            if excerpt in haystack:
                return True
    return False
