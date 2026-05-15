"""Integration tests for the language-differ node.

These tests require a live Postgres at ``DATABASE_URL``. The schema is
rebuilt from ``Base.metadata`` per test so they do not depend on alembic
having run. The ``vector`` extension must be enabled; the fixture does this
with ``CREATE EXTENSION IF NOT EXISTS vector``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete, text
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.language_differ import OWNER, diff_language
from app.memory.db import build_engine
from app.memory.models import Base, Filing, FilingSection
from app.memory.repository import Repository
from app.memory.schemas import (
    NewFiling,
    NewFilingSection,
    SectionKind,
)
from app.models.state import AgentState, FilingEvent, FilingForm

pytestmark = pytest.mark.integration

_MDA_PRIOR = [
    "Revenue grew supported by enterprise demand for our cloud platform.",
    "Operating expenses rose as we expanded research and development headcount.",
    "We expect macroeconomic conditions to remain volatile through the period.",
]

_MDA_CURRENT = [
    "Revenue grew supported by enterprise demand for our cloud platform.",
    "Operating expenses rose substantially as we accelerated AI infrastructure investment.",
    "A new geopolitical risk has emerged that may impact international sales.",
]

_PRIOR_ACCESSION = "0000000000-26-000001"
_CURRENT_ACCESSION = "0000000000-26-000002"
_CIK = "0000789019"
_TICKER = "MSFT"

# The FilingSection ORM column is Vector(1536). Stub vectors must match that
# dimension. We use unit-basis vectors with zeros padded to 1536 dims so the
# cosine-similarity math works the same as in the 3-dim conceptual example.
_DIM = 1536


def _basis(idx: int) -> list[float]:
    """Return a unit-basis vector of dimension _DIM with 1.0 at position idx."""
    v = [0.0] * _DIM
    v[idx] = 1.0
    return v


def _stub_html(paragraphs: list[str]) -> str:
    body = "\n".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        "<html><body>"
        "<p>Item 2. Management's Discussion and Analysis</p>"
        f"{body}"
        "<p>Item 3. Quantitative and Qualitative Disclosures</p>"
        "</body></html>"
    )


class _EdgarStub:
    """Returns canned HTML without hitting the network."""

    def __init__(self, html: str) -> None:
        self.html = html
        self.calls: list[dict[str, str]] = []

    async def get_filing_document(
        self, *, cik: str, accession_number: str, primary_document: str
    ) -> str:
        self.calls.append(
            {"cik": cik, "accession": accession_number, "doc": primary_document}
        )
        return self.html


class _EmbeddingsStub:
    """Returns 1536-dim vectors so current[0] matches prior[0] exactly, current[1] modified.

    Vectors use unit-basis directions so cosine similarity is exact:
    - current[0] == prior[0] -> similarity 1.0 (unchanged)
    - current[1] has partial overlap with prior[1] -> similarity ~0.95 (modified)
    - current[2] has no strong match to any prior -> added
    - prior[2] is left unmatched -> removed
    """

    def __init__(self) -> None:
        # Build a modified version of basis[1] for current[1]: mostly axis-1
        # with a significant axis-0 component. Cosine similarity with prior[1]
        # (axis-1 unit vector) is 0.9/sqrt(0.4^2+0.9^2) ~= 0.914, which is
        # above the match threshold (0.65) but below unchanged (0.97) -> "modified".
        _modified = [0.0] * _DIM
        _modified[0] = 0.4
        _modified[1] = 0.9

        # current[2] lives entirely on axis 3 so its cosine similarity with
        # prior paragraphs on axes 0/1/2 is exactly 0.0 -> "added".
        # prior[2] on axis 2 is left unmatched -> "removed".
        self._table: dict[str, list[float]] = {
            _MDA_PRIOR[0]: _basis(0),
            _MDA_PRIOR[1]: _basis(1),
            _MDA_PRIOR[2]: _basis(2),
            _MDA_CURRENT[0]: _basis(0),
            _MDA_CURRENT[1]: _modified,
            _MDA_CURRENT[2]: _basis(3),
        }

    @property
    def model(self) -> str:
        return "openai/text-embedding-3-small"

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


@pytest_asyncio.fixture()
async def session_factory_with_prior() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Rebuild schema, seed two filings + the prior's sections, stamp primary_document."""
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with factory() as session:
        repo = Repository(session)
        await repo.record_filing(
            filing=NewFiling(
                accession_number=_PRIOR_ACCESSION,
                cik=_CIK,
                ticker=_TICKER,
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 1, 25, tzinfo=UTC),
                source_url="https://www.sec.gov/x",
            )
        )
        await repo.insert_filing_sections(
            [
                NewFilingSection(
                    filing_accession=_PRIOR_ACCESSION,
                    cik=_CIK,
                    ticker=_TICKER,
                    section_kind=SectionKind.MDA,
                    paragraph_index=i,
                    text=paragraph_text,
                    text_sha=f"{i:064d}",
                    embedding=_basis(i),
                    embedding_model="openai/text-embedding-3-small",
                )
                for i, paragraph_text in enumerate(_MDA_PRIOR)
            ]
        )
        await repo.record_filing(
            filing=NewFiling(
                accession_number=_CURRENT_ACCESSION,
                cik=_CIK,
                ticker=_TICKER,
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 25, tzinfo=UTC),
                source_url="https://www.sec.gov/y",
            )
        )
        # Stamp primary_document on the current filing so the differ can fetch.
        await session.execute(
            sa_update(Filing)
            .where(Filing.accession_number == _CURRENT_ACCESSION)
            .values(primary_document="msft-20260331.htm")
        )
        await session.commit()

    yield factory
    await engine.dispose()


async def test_diff_language_emits_state_update_with_owner_and_diffs(
    session_factory_with_prior: async_sessionmaker[AsyncSession],
) -> None:
    """Happy path: differ produces added/modified/removed diffs for the MDA section."""
    edgar = _EdgarStub(_stub_html(_MDA_CURRENT))
    embeddings = _EmbeddingsStub()
    async with session_factory_with_prior() as session:
        state = AgentState(
            trace_id="trace-test",
            started_at=datetime.now(UTC),
            filing_event=FilingEvent(
                accession_number=_CURRENT_ACCESSION,
                cik=_CIK,
                ticker=_TICKER,
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 25, tzinfo=UTC),
                source_url="https://www.sec.gov/y",
            ),
        )
        update = await diff_language(
            state,
            edgar=edgar,
            embeddings=embeddings,
            repository=Repository(session),
        )
        await session.commit()

    assert update.owner == OWNER
    payload = update.changes["language_diffs"]
    assert isinstance(payload, list)
    mda_payload = next(s for s in payload if s["section"] == "mda")
    assert mda_payload["degraded"] is False
    diff_types = sorted(d["change_type"] for d in mda_payload["diffs"])
    assert diff_types == ["added", "modified", "removed"]


async def test_diff_language_degrades_when_no_prior_quarter(
    session_factory_with_prior: async_sessionmaker[AsyncSession],
) -> None:
    """Degrade path: no prior sections means the differ emits degraded=True."""
    edgar = _EdgarStub(_stub_html(_MDA_CURRENT))
    embeddings = _EmbeddingsStub()
    # Drop the prior filing's sections to simulate cold start.
    async with session_factory_with_prior() as session:
        await session.execute(delete(FilingSection))
        await session.commit()

    async with session_factory_with_prior() as session:
        state = AgentState(
            trace_id="trace-test",
            started_at=datetime.now(UTC),
            filing_event=FilingEvent(
                accession_number=_CURRENT_ACCESSION,
                cik=_CIK,
                ticker=_TICKER,
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2026, 4, 25, tzinfo=UTC),
                source_url="https://www.sec.gov/y",
            ),
        )
        update = await diff_language(
            state,
            edgar=edgar,
            embeddings=embeddings,
            repository=Repository(session),
        )
        await session.commit()

    payload = update.changes["language_diffs"]
    mda_payload = next(s for s in payload if s["section"] == "mda")
    assert mda_payload["degraded"] is True
    assert mda_payload["diffs"] == []
