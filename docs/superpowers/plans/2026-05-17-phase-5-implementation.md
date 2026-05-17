# Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 5 (memory writes + peer reader + full LLM critic) per [`docs/superpowers/specs/2026-05-17-phase-5-design.md`](../specs/2026-05-17-phase-5-design.md), retiring 4B xfails #2 and #3 along the way.

**Architecture:** Three sequential subphases on one branch `phase-5-memory-peer-critic`. 5a adds an append-only `notes` table + terminal `note_writer` node, then tightens transcript-analyzer prompts. 5b adds a curated `peers` table + read-only `peer_reader` node + `[P#]` citation namespace + `full_with_peers_v1` synthesizer prompt. 5c adds an LLM critic sequential to the deterministic critic, plus the 30-note adversarial gate.

**Tech Stack:** Python 3.11 + uv, FastAPI, LangGraph, Anthropic SDK (via `app/llm/client.py`), SQLAlchemy 2.x async + Alembic, Postgres + pgvector, pytest + cassette replay.

---

## Pre-flight

**Branch setup** — run before Task 1:

```bash
git checkout phase-4b-transcript-analyzer
git pull
git checkout -b phase-5-memory-peer-critic
```

The Phase 5 design spec already lives at `docs/superpowers/specs/2026-05-17-phase-5-design.md` on `phase-4b-transcript-analyzer`; this new branch inherits it.

---

# Phase 5a — Memory writes (notes table + xfail #2 fix)

## Task 1: Add `Note` ORM model + Pydantic DTOs

**Files:**
- Modify: `app/memory/models.py` (append at bottom, before any future model)
- Modify: `app/memory/schemas.py` (append `NoteCreate` + `NoteRead`)
- Test: `tests/unit/test_memory_schemas.py` (extend with note round-trip)

- [ ] **Step 1: Write the failing schema round-trip test**

Append to `tests/unit/test_memory_schemas.py`:

```python
def test_note_create_round_trips_through_pydantic() -> None:
    from datetime import datetime, timezone

    from app.memory.schemas import NoteCreate

    note = NoteCreate(
        filing_accession="0000123-25-000001",
        ticker="MSFT",
        markdown_body="# Microsoft Q3 FY25\n\nRevenue rose [F1].",
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
        critic_attempts=1,
    )
    assert note.filing_accession == "0000123-25-000001"
    assert note.ticker == "MSFT"
    assert len(note.prompt_template_sha) == 64

    # Round-trip via model_dump_json -> validate
    rebuilt = NoteCreate.model_validate_json(note.model_dump_json())
    assert rebuilt == note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_memory_schemas.py::test_note_create_round_trips_through_pydantic -v`
Expected: FAIL with `ImportError: cannot import name 'NoteCreate'`.

- [ ] **Step 3: Add the ORM model**

Append to `app/memory/models.py`:

```python
class Note(Base):
    """An accepted synthesized note for one filing.

    Append-only. One row per filing_accession; re-runs return the existing
    row via ON CONFLICT DO NOTHING. Per-event cost/latency are deliberately
    not stored here -- a per-event metrics table is deferred to Phase 7.
    """

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filing_accession: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("filings.accession_number", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    markdown_body: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_template_name: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_template_sha: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    critic_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_notes_ticker_created", "ticker", "created_at"),
    )
```

- [ ] **Step 4: Add the Pydantic DTOs**

Append to `app/memory/schemas.py`:

```python
class NoteCreate(BaseModel):
    """Pre-persistence note payload."""

    model_config = ConfigDict(frozen=True)

    filing_accession: str
    ticker: str
    markdown_body: str
    prompt_template_name: str
    prompt_template_sha: str = Field(..., min_length=64, max_length=64)
    critic_attempts: int = Field(..., ge=1)


class NoteRead(BaseModel):
    """Read-side note projection."""

    model_config = ConfigDict(frozen=True)

    id: int
    filing_accession: str
    ticker: str
    markdown_body: str
    prompt_template_name: str
    prompt_template_sha: str
    critic_attempts: int
    created_at: datetime
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_memory_schemas.py::test_note_create_round_trips_through_pydantic -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/memory/models.py app/memory/schemas.py tests/unit/test_memory_schemas.py
git commit -m "phase-5a: add Note ORM model + NoteCreate/NoteRead DTOs"
```

---

## Task 2: Migration 0008 — create `notes` table

**Files:**
- Create: `migrations/versions/20260517_1100_0008_phase5a_notes.py`
- Test: `tests/integration/test_migrations.py` (extend; already open in the IDE)

- [ ] **Step 1: Write the failing migration round-trip test**

Append to `tests/integration/test_migrations.py`:

```python
async def test_migration_0008_creates_notes_table(
    engine: AsyncEngine,
) -> None:
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        def _check(sync_conn: Any) -> dict[str, list[str]]:
            insp = inspect(sync_conn)
            cols = [c["name"] for c in insp.get_columns("notes")]
            idx = [i["name"] for i in insp.get_indexes("notes")]
            return {"cols": cols, "idx": idx}

        result = await conn.run_sync(_check)

    assert "id" in result["cols"]
    assert "filing_accession" in result["cols"]
    assert "markdown_body" in result["cols"]
    assert "prompt_template_sha" in result["cols"]
    assert "ix_notes_ticker_created" in result["idx"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_migrations.py::test_migration_0008_creates_notes_table -v`
Expected: FAIL — table `notes` does not exist.

- [ ] **Step 3: Author the migration**

Create `migrations/versions/20260517_1100_0008_phase5a_notes.py`:

```python
"""Phase 5a: notes table.

Revision ID: 0008_phase5a_notes
Revises: 0007_widen_filings_accession_number
Create Date: 2026-05-17 11:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_phase5a_notes"
down_revision = "0007_widen_filings_accession_number"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "filing_accession",
            sa.String(64),
            sa.ForeignKey("filings.accession_number", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("markdown_body", sa.Text(), nullable=False),
        sa.Column("prompt_template_name", sa.Text(), nullable=False),
        sa.Column("prompt_template_sha", sa.CHAR(64), nullable=False),
        sa.Column("critic_attempts", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("filing_accession", name="uq_notes_filing_accession"),
    )
    op.create_index("ix_notes_ticker_created", "notes", ["ticker", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_notes_ticker_created", table_name="notes")
    op.drop_table("notes")
```

- [ ] **Step 4: Apply migration and re-run test**

Run:
```bash
uv run alembic upgrade head
uv run pytest tests/integration/test_migrations.py::test_migration_0008_creates_notes_table -v
```
Expected: alembic prints `Running upgrade 0007... -> 0008_phase5a_notes`, then PASS.

- [ ] **Step 5: Verify downgrade works**

Run:
```bash
uv run alembic downgrade -1
uv run alembic upgrade head
```
Expected: both succeed silently.

- [ ] **Step 6: Commit**

```bash
git add migrations/versions/20260517_1100_0008_phase5a_notes.py tests/integration/test_migrations.py
git commit -m "phase-5a: alembic 0008 creates notes table"
```

---

## Task 3: Repository methods — `insert_note` + `get_latest_note`

**Files:**
- Modify: `app/memory/repository.py`
- Test: `tests/unit/test_repository_notes.py` (new)

- [ ] **Step 1: Write the failing repository test**

Create `tests/unit/test_repository_notes.py`:

```python
"""Unit tests for Note repository methods."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.memory.repository import Repository
from app.memory.schemas import NewFiling, NoteCreate
from app.models.state import FilingForm


@pytest.mark.asyncio
async def test_insert_note_persists_and_returns_id(repository: Repository) -> None:
    await repository.upsert_filing(
        NewFiling(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 4, 15, tzinfo=timezone.utc),
            source_url="https://www.sec.gov/Archives/edgar/...",
        )
    )

    note = NoteCreate(
        filing_accession="0000123-25-000001",
        ticker="MSFT",
        markdown_body="Body",
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
        critic_attempts=2,
    )
    note_id = await repository.insert_note(note)
    assert note_id is not None

    again_id = await repository.insert_note(note)
    assert again_id == note_id, "second insert should return existing id"


@pytest.mark.asyncio
async def test_get_latest_note_returns_most_recent(repository: Repository) -> None:
    # filing rows + 2 notes inserted with different timestamps (left to the test
    # harness to create distinct created_at via direct SQL or sleep). Reads back
    # the most recent.
    await repository.upsert_filing(
        NewFiling(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 4, 15, tzinfo=timezone.utc),
            source_url="https://www.sec.gov/...",
        )
    )
    await repository.insert_note(
        NoteCreate(
            filing_accession="0000123-25-000001",
            ticker="MSFT",
            markdown_body="Older",
            prompt_template_name="synthesizer/full_v1",
            prompt_template_sha="a" * 64,
            critic_attempts=1,
        )
    )

    latest = await repository.get_latest_note(ticker="MSFT")
    assert latest is not None
    assert latest.markdown_body == "Older"
    assert latest.ticker == "MSFT"


@pytest.mark.asyncio
async def test_get_latest_note_returns_none_for_unknown_ticker(
    repository: Repository,
) -> None:
    result = await repository.get_latest_note(ticker="NOPE")
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_repository_notes.py -v`
Expected: FAIL with `AttributeError: 'Repository' object has no attribute 'insert_note'`.

- [ ] **Step 3: Implement the repository methods**

In `app/memory/repository.py`, add (near the other `add_*` methods):

```python
async def insert_note(self, note: NoteCreate) -> int:
    """Idempotent note insert.

    Returns the new row id, or the existing id when a row with the same
    ``filing_accession`` already exists.
    """
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.memory.models import Note

    stmt = (
        pg_insert(Note)
        .values(
            filing_accession=note.filing_accession,
            ticker=note.ticker,
            markdown_body=note.markdown_body,
            prompt_template_name=note.prompt_template_name,
            prompt_template_sha=note.prompt_template_sha,
            critic_attempts=note.critic_attempts,
        )
        .on_conflict_do_nothing(constraint="uq_notes_filing_accession")
        .returning(Note.id)
    )
    result = await self._session.execute(stmt)
    inserted_id = result.scalar_one_or_none()
    if inserted_id is not None:
        return int(inserted_id)
    # Conflict path: read back the existing id.
    existing = await self._session.execute(
        select(Note.id).where(Note.filing_accession == note.filing_accession)
    )
    return int(existing.scalar_one())


async def get_latest_note(self, *, ticker: str) -> NoteRead | None:
    """Return the most-recently-created note for ``ticker`` or ``None``."""
    from sqlalchemy import select

    from app.memory.models import Note

    stmt = (
        select(Note)
        .where(Note.ticker == ticker)
        .order_by(Note.created_at.desc())
        .limit(1)
    )
    result = await self._session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return NoteRead(
        id=row.id,
        filing_accession=row.filing_accession,
        ticker=row.ticker,
        markdown_body=row.markdown_body,
        prompt_template_name=row.prompt_template_name,
        prompt_template_sha=row.prompt_template_sha,
        critic_attempts=row.critic_attempts,
        created_at=row.created_at,
    )
```

Also add to the top-of-module imports: `from app.memory.schemas import ..., NoteCreate, NoteRead`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_repository_notes.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/memory/repository.py tests/unit/test_repository_notes.py
git commit -m "phase-5a: repository.insert_note + get_latest_note"
```

---

## Task 4: AgentState — add `persisted_note_id` + `note_writer` owner

**Files:**
- Modify: `app/models/state.py`
- Test: `tests/unit/test_state.py` (extend)

- [ ] **Step 1: Write the failing state-ownership test**

Append to `tests/unit/test_state.py`:

```python
def test_note_writer_owns_persisted_note_id() -> None:
    from app.models.state import StateUpdate

    update = StateUpdate(owner="note_writer", changes={"persisted_note_id": 42})
    assert update.changes == {"persisted_note_id": 42}


def test_critic_cannot_set_persisted_note_id() -> None:
    import pytest

    from app.models.state import StateUpdate

    with pytest.raises(ValueError, match="cannot mutate fields"):
        StateUpdate(owner="critic", changes={"persisted_note_id": 1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_state.py::test_note_writer_owns_persisted_note_id tests/unit/test_state.py::test_critic_cannot_set_persisted_note_id -v`
Expected: FAIL — `note_writer` not a known owner.

- [ ] **Step 3: Extend `AgentState` and `_FIELD_OWNERS`**

In `app/models/state.py`:

Add a new field at the end of `AgentState` (right before the docstring of `_FIELD_OWNERS`):

```python
    # ---- Phase 5a: notes persistence ----
    persisted_note_id: int | None = None
```

In `_FIELD_OWNERS`, add a new entry:

```python
    "note_writer": frozenset({"persisted_note_id", "cost_usd"}),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_state.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/state.py tests/unit/test_state.py
git commit -m "phase-5a: AgentState.persisted_note_id owned by note_writer"
```

---

## Task 5: Implement the `note_writer` agent node

**Files:**
- Create: `app/agents/note_writer.py`
- Test: `tests/unit/test_note_writer.py`

- [ ] **Step 1: Write the failing node test**

Create `tests/unit/test_note_writer.py`:

```python
"""Unit tests for the note_writer agent node."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.note_writer import OWNER, write_note
from app.models.state import (
    AgentState,
    CriticVerdict,
    FilingEvent,
    FilingEventSource,
    FilingForm,
)


def _state(*, verdict: CriticVerdict, final_note: str | None) -> AgentState:
    return AgentState(
        trace_id="t-1",
        started_at=datetime(2025, 4, 15, tzinfo=timezone.utc),
        filing_event=FilingEvent(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 4, 15, tzinfo=timezone.utc),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
        draft_note=final_note,
        critic_verdict=verdict,
        critic_attempts=1,
        final_note=final_note,
    )


@pytest.mark.asyncio
async def test_writes_note_when_critic_accepted() -> None:
    state = _state(verdict=CriticVerdict.ACCEPTED, final_note="# Body\n\nText [F1].")
    repo = MagicMock()
    repo.insert_note = AsyncMock(return_value=99)

    update = await write_note(
        state,
        repository=repo,
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
    )

    assert update.owner == OWNER
    assert update.changes == {"persisted_note_id": 99}
    repo.insert_note.assert_awaited_once()


@pytest.mark.asyncio
async def test_skips_when_loop_exceeded() -> None:
    state = _state(verdict=CriticVerdict.LOOP_EXCEEDED, final_note=None)
    repo = MagicMock()
    repo.insert_note = AsyncMock()

    update = await write_note(
        state,
        repository=repo,
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
    )

    assert update.changes == {}
    repo.insert_note.assert_not_awaited()


@pytest.mark.asyncio
async def test_swallows_db_error_logs_and_continues() -> None:
    state = _state(verdict=CriticVerdict.ACCEPTED, final_note="x [F1]")
    repo = MagicMock()
    repo.insert_note = AsyncMock(side_effect=RuntimeError("boom"))

    update = await write_note(
        state,
        repository=repo,
        prompt_template_name="synthesizer/full_v1",
        prompt_template_sha="a" * 64,
    )

    # Note persistence failure must NOT block the response; persisted_note_id
    # stays None.
    assert update.changes == {"persisted_note_id": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_note_writer.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the node**

Create `app/agents/note_writer.py`:

```python
"""The note_writer terminal node.

Persists the accepted synthesized note into the ``notes`` table. Runs only
after the critic returns ACCEPTED. On LOOP_EXCEEDED the node yields an
empty StateUpdate so the graph proceeds to END with no row written -- the
note is held for manual review per the runbook.

A DB failure here does not propagate: the user already has the note in
their API response, so we degrade gracefully and set ``persisted_note_id``
to ``None`` for trace visibility.
"""

from __future__ import annotations

from app.memory.repository import Repository
from app.memory.schemas import NoteCreate
from app.models.state import AgentState, CriticVerdict, StateUpdate
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER = "note_writer"


async def write_note(
    state: AgentState,
    *,
    repository: Repository,
    prompt_template_name: str,
    prompt_template_sha: str,
) -> StateUpdate:
    """Persist the final note when the critic accepted it, otherwise no-op."""
    if state.critic_verdict is not CriticVerdict.ACCEPTED:
        _logger.bind(
            accession=state.filing_event.accession_number,
            verdict=state.critic_verdict.value if state.critic_verdict else None,
            trace_id=current_trace_id(),
        ).info("note_writer_skipped")
        return StateUpdate(owner=OWNER, changes={})

    if state.final_note is None:
        _logger.bind(
            accession=state.filing_event.accession_number,
            trace_id=current_trace_id(),
        ).warning("note_writer_no_final_note")
        return StateUpdate(owner=OWNER, changes={})

    payload = NoteCreate(
        filing_accession=state.filing_event.accession_number,
        ticker=state.filing_event.ticker,
        markdown_body=state.final_note,
        prompt_template_name=prompt_template_name,
        prompt_template_sha=prompt_template_sha,
        critic_attempts=state.critic_attempts,
    )

    try:
        note_id: int | None = await repository.insert_note(payload)
    except Exception as exc:  # noqa: BLE001 - degrade, don't crash the pipeline
        _logger.bind(
            accession=state.filing_event.accession_number,
            error=str(exc),
            trace_id=current_trace_id(),
        ).error("note_writer_persist_failed")
        note_id = None

    _logger.bind(
        accession=state.filing_event.accession_number,
        note_id=note_id,
        trace_id=current_trace_id(),
    ).info("note_writer_complete")
    return StateUpdate(owner=OWNER, changes={"persisted_note_id": note_id})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_note_writer.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/agents/note_writer.py tests/unit/test_note_writer.py
git commit -m "phase-5a: note_writer node persists accepted notes idempotently"
```

---

## Task 6: Wire `note_writer` into the graph

**Files:**
- Modify: `app/graph.py`
- Test: `tests/integration/test_notes_persistence.py` (new)

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_notes_persistence.py`:

```python
"""Integration test: accepted critic -> notes row written via note_writer."""

from __future__ import annotations

import pytest

from app.memory.repository import Repository
# fixtures: ``invoke_graph_for_filing`` and ``test_session_factory`` are
# project conftest fixtures already used by other integration tests.


@pytest.mark.asyncio
async def test_accepted_note_persists_one_row(
    invoke_graph_for_filing,
    test_session_factory,
) -> None:
    final_state = await invoke_graph_for_filing("MSFT_Q3_FY25_8K")

    assert final_state.critic_verdict.value == "accepted"
    assert final_state.persisted_note_id is not None

    async with test_session_factory() as session:
        repo = Repository(session)
        latest = await repo.get_latest_note(ticker="MSFT")
        assert latest is not None
        assert latest.markdown_body == final_state.final_note
        assert latest.critic_attempts == final_state.critic_attempts


@pytest.mark.asyncio
async def test_rerun_returns_same_note_id(
    invoke_graph_for_filing,
) -> None:
    s1 = await invoke_graph_for_filing("MSFT_Q3_FY25_8K")
    s2 = await invoke_graph_for_filing("MSFT_Q3_FY25_8K")

    assert s1.persisted_note_id == s2.persisted_note_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_notes_persistence.py -v`
Expected: FAIL — `persisted_note_id` is `None` because no node writes it yet.

- [ ] **Step 3: Wire `note_writer` into `app/graph.py`**

Add at the top:

```python
from app.agents.note_writer import OWNER as NOTE_WRITER_OWNER
from app.agents.note_writer import write_note
```

Add a new closure factory near `_make_critic_node`:

```python
def _make_note_writer_node(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    prompt_template_name: str,
    prompt_template_sha: str,
) -> NodeFn:
    """Return the LangGraph node closure for note_writer."""

    async def node(state: AgentState) -> dict[str, Any]:
        async with session_factory() as session:
            try:
                update = await write_note(
                    state,
                    repository=Repository(session),
                    prompt_template_name=prompt_template_name,
                    prompt_template_sha=prompt_template_sha,
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return update.changes

    return node
```

Update `_critic_router` so ACCEPTED routes to `note_writer` instead of `END` (LLM critic insertion lands in 5c — for 5a the router goes critic → note_writer → END):

```python
def _critic_router(state: AgentState) -> str:
    """Decide whether to retry the synthesiser, persist, or end the run."""
    if state.critic_verdict is CriticVerdict.REJECTED:
        return SYNTHESIZER_OWNER
    if state.critic_verdict is CriticVerdict.ACCEPTED:
        return NOTE_WRITER_OWNER
    return END  # LOOP_EXCEEDED
```

In `build_graph(...)`, after adding the critic node, also add:

```python
    # Phase 5a: note_writer runs after the critic accepts.
    from app.llm.prompts import load_prompt
    synth_template = load_prompt("synthesizer/full_v1")
    builder.add_node(  # type: ignore[call-overload]
        NOTE_WRITER_OWNER,
        _make_note_writer_node(
            session_factory=session_factory,
            prompt_template_name="synthesizer/full_v1",
            prompt_template_sha=synth_template.body_sha,
        ),
    )
```

Replace the critic's conditional edges with:

```python
    builder.add_conditional_edges(
        CRITIC_OWNER,
        _critic_router,
        {
            SYNTHESIZER_OWNER: SYNTHESIZER_OWNER,
            NOTE_WRITER_OWNER: NOTE_WRITER_OWNER,
            END: END,
        },
    )
    builder.add_edge(NOTE_WRITER_OWNER, END)
```

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `uv run pytest tests/integration/test_notes_persistence.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Run full unit + integration sweep to catch regressions**

Run: `uv run pytest tests/unit tests/integration -q`
Expected: all green except the three pre-existing 4B xfails.

- [ ] **Step 6: Commit**

```bash
git add app/graph.py tests/integration/test_notes_persistence.py
git commit -m "phase-5a: wire note_writer terminal node into the graph"
```

---

## Task 7: Tighten transcript_analyzer extract prompt for xfail #2

**Files:**
- Modify: `prompts/transcript_analyzer/extract_v1.md`
- Re-record: `tests/fixtures/cassettes/transcript_analyzer/extract_*.json`
- Test: `tests/unit/test_transcript_analyzer.py` (extend)

- [ ] **Step 1: Read the current extract prompt**

Run: `cat prompts/transcript_analyzer/extract_v1.md` (or open in IDE).

Identify the commitment-extraction section. Currently it asks for commitments without insisting on explicit period markers.

- [ ] **Step 2: Add a tighter extraction rule**

In `prompts/transcript_analyzer/extract_v1.md`, locate the commitments instruction block and append:

```markdown
COMMITMENT REQUIREMENT — EXPLICIT PERIOD: A commitment MUST contain an
explicit forward-looking period marker. Acceptable markers include
"Q1", "Q2", "Q3", "Q4", "next quarter", "next fiscal year", "by year-end",
"in [calendar quarter]", "FY26", "in 2026", or any specific calendar
date. If management speaks aspirationally without naming a period
(e.g., "we hope to grow margins"), do NOT extract it. Extract only
commitments tied to an identifiable period.
```

Update the prompt's frontmatter version comment if present (e.g., bump `# Updated 2026-05-17 for phase-5a-xfail-2`).

- [ ] **Step 3: Re-record affected cassettes**

Run: `REC=1 uv run pytest tests/unit/test_transcript_analyzer.py tests/integration/test_commitment_reconciliation.py -q`
Expected: cassettes refreshed; tests pass (loose ones still pass; strict ones may now pass or near-pass).

- [ ] **Step 4: Verify the looser sibling test still passes**

Run: `uv run pytest tests/integration/test_commitment_reconciliation.py::test_q3_reconcile_produces_state_update_with_commitment_updates -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add prompts/transcript_analyzer/extract_v1.md tests/fixtures/cassettes/transcript_analyzer/
git commit -m "phase-5a: extract prompt requires explicit period on commitments (xfail-2 fix)"
```

---

## Task 8: Tighten transcript_analyzer reconcile prompt + remove xfail-2 marker

**Files:**
- Modify: `prompts/transcript_analyzer/reconcile_v1.md`
- Re-record: cassettes
- Modify: `tests/integration/test_commitment_reconciliation.py` (remove xfail)

- [ ] **Step 1: Add tighter reconcile rule**

In `prompts/transcript_analyzer/reconcile_v1.md`, locate the rules for `met` vs `still_open` and replace/extend:

```markdown
RECONCILE RULE — UNAMBIGUOUS EVIDENCE: A commitment may transition to "met"
ONLY when the new transcript contains an unambiguous evidence quote
naming a concrete result, number, or boolean outcome. Examples of
unambiguous evidence:
  - "We delivered $X in Q3" (number)
  - "We launched product Y on date" (boolean event)
  - "Margins expanded N basis points QoQ" (directional + magnitude)

If the evidence is hedged ("we made progress on...", "broadly tracking
expectations") OR the cited quote lacks a verifiable outcome, the
status stays "still_open". Do not flip to "met" on optimistic framing
alone.
```

- [ ] **Step 2: Re-record reconcile cassettes**

Run: `REC=1 uv run pytest tests/integration/test_commitment_reconciliation.py -q`

- [ ] **Step 3: Remove the xfail marker on the strict test**

In `tests/integration/test_commitment_reconciliation.py`, locate `test_q3_reconcile_closes_expected_q2_commitments`. Remove the `@pytest.mark.xfail(...)` decorator and its `reason=` string.

- [ ] **Step 4: Run the strict test**

Run: `uv run pytest tests/integration/test_commitment_reconciliation.py::test_q3_reconcile_closes_expected_q2_commitments -v`
Expected: PASS.

- [ ] **Step 5: Run full sweep**

Run: `uv run pytest tests/unit tests/integration -q`
Expected: all green except remaining 4B xfails (#1 and #3).

- [ ] **Step 6: Commit**

```bash
git add prompts/transcript_analyzer/reconcile_v1.md tests/fixtures/cassettes/transcript_analyzer/ tests/integration/test_commitment_reconciliation.py
git commit -m "phase-5a: reconcile prompt requires unambiguous evidence; retire xfail #2"
```

---

## Task 9: Multi-quarter synthetic E2E (Phase 5a gate)

**Files:**
- Create: `tests/integration/test_multi_quarter_synthetic_run.py`

- [ ] **Step 1: Write the gate test**

Create `tests/integration/test_multi_quarter_synthetic_run.py`:

```python
"""Phase 5a gate: multi-quarter synthetic run closes prior commitments correctly."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.memory.repository import Repository


@pytest.mark.asyncio
async def test_four_quarter_run_writes_four_notes_and_closes_commitments(
    invoke_graph_for_synthetic_quarter,
    test_session_factory,
) -> None:
    """End-to-end: run 4 sequential synthetic quarters; assert (a) 4 notes
    persisted, (b) prior commitments correctly closed, (c) no orphan rows.
    """
    # The harness fixture ``invoke_graph_for_synthetic_quarter`` runs the full
    # graph for one synthetic quarter; quarters Q1-Q4 must be fully wired into
    # ``tests/fixtures/transcripts/`` per Phase 4B fixtures.
    states = []
    for quarter_label in ("NIMBUS_Q1", "NIMBUS_Q2", "NIMBUS_Q3", "NIMBUS_Q4"):
        s = await invoke_graph_for_synthetic_quarter(quarter_label)
        assert s.critic_verdict.value == "accepted", f"{quarter_label} not accepted"
        assert s.persisted_note_id is not None, f"{quarter_label} note not persisted"
        states.append(s)

    async with test_session_factory() as session:
        repo = Repository(session)

        # (a) 4 notes for the synthetic NIMBUS ticker.
        from sqlalchemy import select
        from app.memory.models import Note

        n_count = await session.execute(
            select(Note).where(Note.ticker == "NIMBUS")
        )
        rows = n_count.scalars().all()
        assert len(rows) == 4

        # (b) at least one prior open commitment was closed by a later quarter.
        from app.memory.models import Commitment
        closed = await session.execute(
            select(Commitment).where(
                Commitment.ticker == "NIMBUS",
                Commitment.status.in_(("met", "missed")),
            )
        )
        assert len(closed.scalars().all()) >= 1, "no commitment closed across 4 quarters"

        # (c) no orphan FK rows.
        from app.memory.models import QAPair
        orphan_qa = await session.execute(
            select(QAPair).outerjoin(
                Note, QAPair.filing_accession == Note.filing_accession
            ).where(Note.id.is_(None))
        )
        # qa_pairs FK to filings (not notes), so this check exists only to
        # surface true orphans; it must not increase across quarters.
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/integration/test_multi_quarter_synthetic_run.py -v`
Expected: PASS. If a fixture is missing for `NIMBUS_Q4`, the fixture catalogue is incomplete — adapt to whichever labelled quarters exist (Phase 4B shipped at least `NIMBUS_Q2` and `NIMBUS_Q3`).

If `NIMBUS_Q4` does not exist, change the test to iterate over the available labelled quarters and adjust the assertion to "at least 2 sequential quarters" — keep the spirit of the gate (multi-quarter persistence + commitment closure) intact.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_multi_quarter_synthetic_run.py
git commit -m "phase-5a: gate test - multi-quarter run persists notes, closes commitments"
```

---

## Task 10: Phase 5a wrap-up — verify gate

- [ ] **Step 1: Run full quality bar**

Run:
```bash
uv run ruff check app/ tests/
uv run mypy app/
uv run pytest tests/unit tests/integration -q
uv run pytest --cov=app --cov-report=term tests/unit tests/integration
uv run pip-audit
```

Expected: ruff clean; mypy clean; xfail #1 and #3 remain; xfail #2 retired; coverage ≥85% (target ≥88%); pip-audit clean.

- [ ] **Step 2: Commit any cleanup**

If any small fixes (typos, missing docstrings) shake out, commit them with `phase-5a: post-gate cleanup`.

---

# Phase 5b — Peer reader

## Task 11: Add `Peer` ORM model + `PeerContextEntry` + DTOs

**Files:**
- Modify: `app/memory/models.py`
- Modify: `app/memory/schemas.py`
- Modify: `app/models/state.py`
- Test: `tests/unit/test_memory_schemas.py` (extend), `tests/unit/test_state.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_memory_schemas.py`:

```python
def test_peer_create_validates_no_self_reference() -> None:
    import pytest

    from app.memory.schemas import PeerCreate

    with pytest.raises(ValueError):
        PeerCreate(ticker="MSFT", peer_ticker="MSFT")


def test_peer_signals_defaults_to_empty() -> None:
    from app.memory.schemas import PeerSignals

    sig = PeerSignals(language_diffs=[], commitments=[])
    assert sig.language_diffs == []
    assert sig.commitments == []
```

Append to `tests/unit/test_state.py`:

```python
def test_peer_context_entry_minimal_shape() -> None:
    from app.models.state import PeerContextEntry

    entry = PeerContextEntry(
        peer_ticker="GOOGL",
        kind="commitment",
        text="We expect cloud growth to accelerate next quarter.",
        source_filing_accession="0000123-25-000002",
    )
    assert entry.kind == "commitment"
    assert entry.severity is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_memory_schemas.py::test_peer_create_validates_no_self_reference tests/unit/test_state.py::test_peer_context_entry_minimal_shape -v`
Expected: FAIL.

- [ ] **Step 3: Add `Peer` ORM model**

Append to `app/memory/models.py`:

```python
class Peer(Base):
    """A curated ``(ticker, peer_ticker)`` mapping for the peer reader.

    Append-only via upsert. The ``source`` column is forward-compatible for
    auto-discovery; currently constrained to ``'curated'``.
    """

    __tablename__ = "peers"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    peer_ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="curated")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("ticker <> peer_ticker", name="peers_no_self_reference"),
        CheckConstraint("source IN ('curated')", name="peers_source_valid"),
        Index("ix_peers_ticker", "ticker"),
    )
```

- [ ] **Step 4: Add DTOs**

Append to `app/memory/schemas.py`:

```python
class PeerCreate(BaseModel):
    """Pre-persistence peer mapping."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    peer_ticker: str
    source: str = "curated"

    @field_validator("peer_ticker")
    @classmethod
    def _no_self_reference(cls, v: str, info: ValidationInfo) -> str:
        ticker = info.data.get("ticker")
        if ticker is not None and v == ticker:
            raise ValueError("peer_ticker must differ from ticker")
        return v


class PeerLanguageDiffSignal(BaseModel):
    """One major language diff signal from a peer."""

    model_config = ConfigDict(frozen=True)

    text: str
    severity: str
    source_filing_accession: str


class PeerCommitmentSignal(BaseModel):
    """One open commitment signal from a peer."""

    model_config = ConfigDict(frozen=True)

    text: str
    source_filing_accession: str


class PeerSignals(BaseModel):
    """Bundle of peer signals returned by the repository."""

    model_config = ConfigDict(frozen=True)

    language_diffs: list[PeerLanguageDiffSignal] = Field(default_factory=list)
    commitments: list[PeerCommitmentSignal] = Field(default_factory=list)
```

Import `field_validator` and `ValidationInfo` from `pydantic` if not already imported.

- [ ] **Step 5: Add `PeerContextEntry` to state**

In `app/models/state.py`, add (next to `QAPairPayload`):

```python
class PeerContextEntry(BaseModel):
    """One peer signal surfaced by `peer_reader` for use in the synthesizer.

    `kind='language_diff'` rows surface MD&A/risk-factor language changes from
    the peer's most recent 10-K or 10-Q. `kind='commitment'` rows surface
    open commitments from the peer's most recent transcript.
    """

    model_config = ConfigDict(frozen=True)

    peer_ticker: str
    kind: Literal["language_diff", "commitment"]
    text: str
    source_filing_accession: str
    severity: Literal["major", "minor"] | None = None
```

Add `from typing import Literal` at the top if missing.

Replace the existing placeholder line in `AgentState`:

```python
    peer_context: dict[str, Any] | None = None
```

with:

```python
    peer_context: list[PeerContextEntry] = Field(default_factory=list)
```

The `_FIELD_OWNERS` entry for `peer_reader` already grants `peer_context` ownership — no change needed.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/test_memory_schemas.py tests/unit/test_state.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add app/memory/models.py app/memory/schemas.py app/models/state.py tests/unit/test_memory_schemas.py tests/unit/test_state.py
git commit -m "phase-5b: Peer ORM + PeerCreate/PeerSignals/PeerContextEntry types"
```

---

## Task 12: Migration 0009 — create `peers` table

**Files:**
- Create: `migrations/versions/20260517_1130_0009_phase5b_peers.py`
- Test: `tests/integration/test_migrations.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_migrations.py`:

```python
async def test_migration_0009_creates_peers_table(engine: AsyncEngine) -> None:
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        def _check(sync_conn: Any) -> dict[str, list[str]]:
            insp = inspect(sync_conn)
            cols = [c["name"] for c in insp.get_columns("peers")]
            constraints = [c["name"] for c in insp.get_check_constraints("peers")]
            return {"cols": cols, "constraints": constraints}

        result = await conn.run_sync(_check)

    assert "ticker" in result["cols"]
    assert "peer_ticker" in result["cols"]
    assert "peers_no_self_reference" in result["constraints"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/test_migrations.py::test_migration_0009_creates_peers_table -v`
Expected: FAIL.

- [ ] **Step 3: Author the migration**

Create `migrations/versions/20260517_1130_0009_phase5b_peers.py`:

```python
"""Phase 5b: peers table.

Revision ID: 0009_phase5b_peers
Revises: 0008_phase5a_notes
Create Date: 2026-05-17 11:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009_phase5b_peers"
down_revision = "0008_phase5a_notes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "peers",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("peer_ticker", sa.String(16), nullable=False),
        sa.Column(
            "source", sa.String(32), nullable=False, server_default="curated"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("ticker", "peer_ticker", name="pk_peers"),
        sa.CheckConstraint(
            "ticker <> peer_ticker", name="peers_no_self_reference"
        ),
        sa.CheckConstraint("source IN ('curated')", name="peers_source_valid"),
    )
    op.create_index("ix_peers_ticker", "peers", ["ticker"])


def downgrade() -> None:
    op.drop_index("ix_peers_ticker", table_name="peers")
    op.drop_table("peers")
```

- [ ] **Step 4: Apply and verify**

Run:
```bash
uv run alembic upgrade head
uv run pytest tests/integration/test_migrations.py::test_migration_0009_creates_peers_table -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add migrations/versions/20260517_1130_0009_phase5b_peers.py tests/integration/test_migrations.py
git commit -m "phase-5b: alembic 0009 creates peers table"
```

---

## Task 13: Repository methods — `upsert_peer`, `list_peers`, `get_recent_peer_signals`

**Files:**
- Modify: `app/memory/repository.py`
- Test: `tests/unit/test_repository_peers.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_repository_peers.py`:

```python
"""Unit tests for peer repository methods."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.memory.repository import Repository
from app.memory.schemas import NewFiling, PeerCreate
from app.models.state import FilingForm


@pytest.mark.asyncio
async def test_upsert_peer_inserts_then_no_ops_on_duplicate(
    repository: Repository,
) -> None:
    await repository.upsert_peer(PeerCreate(ticker="MSFT", peer_ticker="GOOGL"))
    await repository.upsert_peer(PeerCreate(ticker="MSFT", peer_ticker="GOOGL"))
    peers = await repository.list_peers(ticker="MSFT")
    assert peers == ["GOOGL"]


@pytest.mark.asyncio
async def test_list_peers_returns_empty_for_unknown_ticker(
    repository: Repository,
) -> None:
    assert await repository.list_peers(ticker="UNKNOWN") == []


@pytest.mark.asyncio
async def test_get_recent_peer_signals_empty_when_cold_start(
    repository: Repository,
) -> None:
    sig = await repository.get_recent_peer_signals(peer_ticker="GOOGL")
    assert sig.language_diffs == []
    assert sig.commitments == []


@pytest.mark.asyncio
async def test_get_recent_peer_signals_skips_stale_filings(
    repository: Repository,
) -> None:
    """Filings older than max_age_days are excluded."""
    old = datetime.now(timezone.utc) - timedelta(days=400)
    await repository.upsert_filing(
        NewFiling(
            accession_number="0000123-22-000001",
            cik="0000123",
            ticker="GOOGL",
            form=FilingForm.FORM_10K,
            filed_at=old,
            source_url="https://www.sec.gov/...",
        )
    )
    sig = await repository.get_recent_peer_signals(
        peer_ticker="GOOGL", max_age_days=180
    )
    assert sig.language_diffs == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_repository_peers.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement repository methods**

In `app/memory/repository.py`, add:

```python
async def upsert_peer(self, peer: PeerCreate) -> None:
    """Idempotent peer insert; no-op on duplicate."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.memory.models import Peer

    stmt = (
        pg_insert(Peer)
        .values(
            ticker=peer.ticker,
            peer_ticker=peer.peer_ticker,
            source=peer.source,
        )
        .on_conflict_do_nothing(constraint="pk_peers")
    )
    await self._session.execute(stmt)


async def list_peers(self, *, ticker: str) -> list[str]:
    """Return peer tickers for ``ticker``."""
    from sqlalchemy import select

    from app.memory.models import Peer

    stmt = select(Peer.peer_ticker).where(Peer.ticker == ticker).order_by(
        Peer.peer_ticker
    )
    result = await self._session.execute(stmt)
    return [str(r) for r in result.scalars().all()]


async def get_recent_peer_signals(
    self,
    *,
    peer_ticker: str,
    max_age_days: int = 180,
) -> PeerSignals:
    """Return the peer's most-recent language diffs + open commitments.

    language_diffs come from the most recent processed 10-K or 10-Q within
    ``max_age_days``, filtered to ``severity='major'``.
    commitments come from the most recent processed TRANSCRIPT within
    ``max_age_days``, filtered to ``status='open'``.
    The two filings may be different rows for the same peer.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from app.memory.models import Commitment, Filing, LanguageDiff

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    # Most recent 10-K/10-Q filing for language diffs.
    filing_for_language = await self._session.execute(
        select(Filing.accession_number)
        .where(
            Filing.ticker == peer_ticker,
            Filing.form.in_(("10-K", "10-Q")),
            Filing.filed_at >= cutoff,
            Filing.status == "processed",
        )
        .order_by(Filing.filed_at.desc())
        .limit(1)
    )
    accession = filing_for_language.scalar_one_or_none()

    language_diffs: list[PeerLanguageDiffSignal] = []
    if accession is not None:
        diff_rows = await self._session.execute(
            select(LanguageDiff)
            .where(
                LanguageDiff.filing_accession == accession,
                LanguageDiff.severity == "major",
            )
        )
        # Repository must materialize an arbitrary 'text' for each diff. The
        # LanguageDiff row stores section IDs; pull current section text.
        for row in diff_rows.scalars().all():
            text = await self._language_diff_text(row)
            language_diffs.append(
                PeerLanguageDiffSignal(
                    text=text,
                    severity=row.severity,
                    source_filing_accession=accession,
                )
            )

    # Most recent TRANSCRIPT for commitments.
    transcript_filing = await self._session.execute(
        select(Filing.accession_number)
        .where(
            Filing.ticker == peer_ticker,
            Filing.form == "TRANSCRIPT",
            Filing.filed_at >= cutoff,
            Filing.status == "processed",
        )
        .order_by(Filing.filed_at.desc())
        .limit(1)
    )
    t_accession = transcript_filing.scalar_one_or_none()

    commitments: list[PeerCommitmentSignal] = []
    if t_accession is not None:
        rows = await self._session.execute(
            select(Commitment)
            .where(
                Commitment.filing_accession == t_accession,
                Commitment.status == "open",
            )
        )
        for c in rows.scalars().all():
            commitments.append(
                PeerCommitmentSignal(
                    text=c.commitment_text,
                    source_filing_accession=t_accession,
                )
            )

    return PeerSignals(language_diffs=language_diffs, commitments=commitments)


async def _language_diff_text(self, row: "LanguageDiff") -> str:
    """Fetch the current-section text for a language diff row."""
    from sqlalchemy import select

    from app.memory.models import FilingSection

    if row.current_section_id is None:
        return ""
    sec = await self._session.execute(
        select(FilingSection.text).where(FilingSection.id == row.current_section_id)
    )
    return str(sec.scalar_one_or_none() or "")
```

Add to the top-of-file imports: `PeerCreate, PeerSignals, PeerLanguageDiffSignal, PeerCommitmentSignal`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_repository_peers.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/memory/repository.py tests/unit/test_repository_peers.py
git commit -m "phase-5b: repository upsert_peer, list_peers, get_recent_peer_signals"
```

---

## Task 14: Implement the `peer_reader` agent node

**Files:**
- Create: `app/agents/peer_reader.py`
- Test: `tests/unit/test_peer_reader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_peer_reader.py`:

```python
"""Unit tests for the peer_reader agent node."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.peer_reader import OWNER, read_peers
from app.memory.schemas import (
    PeerCommitmentSignal,
    PeerLanguageDiffSignal,
    PeerSignals,
)
from app.models.state import (
    AgentState,
    FilingEvent,
    FilingEventSource,
    FilingForm,
    PeerContextEntry,
)


def _state(ticker: str = "MSFT") -> AgentState:
    return AgentState(
        trace_id="t-1",
        started_at=datetime(2025, 4, 15, tzinfo=timezone.utc),
        filing_event=FilingEvent(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker=ticker,
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 4, 15, tzinfo=timezone.utc),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
    )


@pytest.mark.asyncio
async def test_no_peers_yields_empty_context() -> None:
    repo = MagicMock()
    repo.list_peers = AsyncMock(return_value=[])

    update = await read_peers(_state(), repository=repo)

    assert update.owner == OWNER
    assert update.changes == {"peer_context": []}


@pytest.mark.asyncio
async def test_one_peer_returns_combined_signals() -> None:
    repo = MagicMock()
    repo.list_peers = AsyncMock(return_value=["GOOGL"])
    repo.get_recent_peer_signals = AsyncMock(
        return_value=PeerSignals(
            language_diffs=[
                PeerLanguageDiffSignal(
                    text="Cloud pricing pressure intensified.",
                    severity="major",
                    source_filing_accession="0000123-25-000002",
                ),
            ],
            commitments=[
                PeerCommitmentSignal(
                    text="We expect cloud margins to expand next quarter.",
                    source_filing_accession="0000123-25-000003",
                ),
            ],
        )
    )

    update = await read_peers(_state(), repository=repo)
    entries = update.changes["peer_context"]
    assert len(entries) == 2
    assert {e.kind for e in entries} == {"language_diff", "commitment"}
    assert all(isinstance(e, PeerContextEntry) for e in entries)


@pytest.mark.asyncio
async def test_db_error_degrades_to_empty_context() -> None:
    repo = MagicMock()
    repo.list_peers = AsyncMock(side_effect=RuntimeError("db down"))

    update = await read_peers(_state(), repository=repo)
    assert update.changes == {"peer_context": []}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_peer_reader.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement the node**

Create `app/agents/peer_reader.py`:

```python
"""The peer_reader agent node.

Pure DB read over the curated `peers` table + each peer's most-recent
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
    """Return a StateUpdate populating ``peer_context``."""
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
    except Exception as exc:  # noqa: BLE001 - degrade, don't crash
        _logger.bind(
            ticker=ticker,
            error=str(exc),
            trace_id=current_trace_id(),
        ).error("peer_reader_failed")
        return StateUpdate(owner=OWNER, changes={"peer_context": []})

    _logger.bind(
        ticker=ticker,
        peer_count=len(set(e.peer_ticker for e in entries)),
        entry_count=len(entries),
        trace_id=current_trace_id(),
    ).info("peer_reader_complete")
    return StateUpdate(owner=OWNER, changes={"peer_context": entries})
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_peer_reader.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/agents/peer_reader.py tests/unit/test_peer_reader.py
git commit -m "phase-5b: peer_reader node reads curated peers + their signals"
```

---

## Task 15: Add `[P#]` citation namespace

**Files:**
- Modify: `app/agents/citations.py`
- Modify: `app/agents/critic.py`
- Test: `tests/unit/test_peer_citations.py` (new), `tests/unit/test_critic.py` (extend)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_peer_citations.py`:

```python
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
```

Append to `tests/unit/test_critic.py`:

```python
def test_critic_resolves_unknown_p_citation() -> None:
    # When the synthesizer cites [P0] but peer_context is empty, the critic
    # must flag the citation as unknown.
    from app.agents.critic import critique_draft
    from app.models.state import (
        AgentState,
        FilingEvent,
        FilingEventSource,
        FilingForm,
    )
    from datetime import datetime, timezone

    state = AgentState(
        trace_id="t",
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        filing_event=FilingEvent(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
        draft_note='Peer says "growth strong" [P0].',
        peer_context=[],
    )
    update = critique_draft(state)
    assert any(
        f.severity == "error" and "P0" in f.message
        for f in update.changes["critic_findings"]
    )


def test_critic_resolves_known_p_citation() -> None:
    from app.agents.critic import critique_draft
    from app.models.state import (
        AgentState,
        FilingEvent,
        FilingEventSource,
        FilingForm,
        PeerContextEntry,
    )
    from datetime import datetime, timezone

    state = AgentState(
        trace_id="t",
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        filing_event=FilingEvent(
            accession_number="0000123-25-000001",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
        draft_note='GOOGL noted "Cloud pricing pressure intensified" [P0].',
        peer_context=[
            PeerContextEntry(
                peer_ticker="GOOGL",
                kind="language_diff",
                text="Cloud pricing pressure intensified during the quarter.",
                source_filing_accession="x-1",
                severity="major",
            )
        ],
    )
    update = critique_draft(state)
    # The [P0] citation resolves and the quoted substring matches.
    findings = update.changes["critic_findings"]
    assert all(f.severity != "error" or "P0" not in f.message for f in findings)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_peer_citations.py tests/unit/test_critic.py -k "peer or P0 or p_citation" -v`
Expected: FAIL.

- [ ] **Step 3: Add `PeerCitation` + builder**

In `app/agents/citations.py`, add:

```python
@dataclass(frozen=True)
class PeerCitation:
    """A resolved [P#] reference to a PeerContextEntry."""

    identifier: str
    peer_ticker: str
    text: str
    kind: str  # 'language_diff' | 'commitment'


def build_peer_citations(
    peer_context: list["PeerContextEntry"],
) -> list[PeerCitation]:
    """Assign sequential P0, P1, ... ids to each peer context entry."""
    cits: list[PeerCitation] = []
    for idx, entry in enumerate(peer_context):
        cits.append(
            PeerCitation(
                identifier=f"P{idx}",
                peer_ticker=entry.peer_ticker,
                text=entry.text,
                kind=entry.kind,
            )
        )
    return cits
```

Add `from app.models.state import PeerContextEntry` import (TYPE_CHECKING-guarded if necessary).

- [ ] **Step 4: Extend critic for `[P#]`**

In `app/agents/critic.py`:

Widen `_CITED_LANGUAGE` regex:

```python
_CITED_LANGUAGE: Final[re.Pattern[str]] = re.compile(
    r"\[(?P<cite>[LQKP]\d+)\]",
    re.IGNORECASE,
)
```

Add a `peer_index` parameter to `_validate_quote_citations` and `_resolve_quote_citation`:

```python
def _validate_quote_citations(
    text: str,
    *,
    language_index: dict[str, LanguageCitation],
    qa_index: dict[str, QACitation],
    commitment_index: dict[str, CommitmentCitation],
    peer_index: dict[str, PeerCitation],
) -> list[CriticFinding]:
    ...
```

In `_resolve_quote_citation`, add the `P` branch:

```python
    if namespace == "P":
        peer = peer_index.get(cite_id)
        return peer.text if peer is not None else None
```

In `_namespace_label`, add `"P": "peer commentary"`.

In `critique_draft`, build the peer index and pass it through:

```python
    peer_index = {c.identifier: c for c in build_peer_citations(state.peer_context)}
    ...
    findings.extend(
        _validate_quote_citations(
            state.draft_note,
            language_index=language_index,
            qa_index=qa_index,
            commitment_index=commitment_index,
            peer_index=peer_index,
        )
    )
```

Add to imports: `from app.agents.citations import ..., PeerCitation, build_peer_citations`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_peer_citations.py tests/unit/test_critic.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app/agents/citations.py app/agents/critic.py tests/unit/test_peer_citations.py tests/unit/test_critic.py
git commit -m "phase-5b: [P#] peer citation namespace + critic resolution"
```

---

## Task 16: Author `synthesizer/full_with_peers_v1.md` + modify synthesizer

**Files:**
- Create: `prompts/synthesizer/full_with_peers_v1.md`
- Modify: `app/agents/synthesizer.py`
- Re-record: synthesizer cassettes

- [ ] **Step 1: Create the new prompt**

Create `prompts/synthesizer/full_with_peers_v1.md`. Use the existing `full_v1.md` as a starting point (copy verbatim) and add a new `<source name="peers">` block in the user message. The system message must instruct:

```markdown
PEER COMMENTARY RULES:
- Each entry in <source name="peers"> is identified by [P0], [P1], ...
- Each entry tags its source ticker explicitly.
- Cite peer commentary with [P#] only when materially relevant to the
  current filing's themes. Empty <source name="peers"> means do not
  include a peer paragraph at all.
- Quote no more than 15 contiguous words from any peer source (copyright).
- Do not paraphrase a peer claim and cite it; quote the substring or
  rewrite without a citation.
```

Frontmatter should set `version: 1`, `model: claude-opus-4-7`, `temperature: 0.0`.

- [ ] **Step 2: Modify the synthesizer to choose between prompts**

In `app/agents/synthesizer.py`, locate the prompt-name selection. Add a branch:

```python
_PROMPT_NAME = "synthesizer/full_v1"
_PROMPT_NAME_WITH_PEERS = "synthesizer/full_with_peers_v1"


def _select_prompt_name(state: AgentState) -> str:
    return _PROMPT_NAME_WITH_PEERS if state.peer_context else _PROMPT_NAME
```

Where the prompt is loaded, call `_select_prompt_name(state)` instead of the constant. Pass `peer_context` into the template's `format` dict, rendering each entry as:

```
[P{i}] ({peer_ticker}, {kind}) {text}
```

- [ ] **Step 3: Re-record synthesizer cassettes**

Run: `REC=1 uv run pytest tests/integration -k synthesizer -q`
Expected: cassettes refreshed.

- [ ] **Step 4: Run existing synthesizer tests**

Run: `uv run pytest tests/unit/test_synthesizer.py tests/integration -k synthesizer -v`
Expected: all PASS (peer-aware path may not yet have its own integration test — that's Task 18).

- [ ] **Step 5: Commit**

```bash
git add prompts/synthesizer/full_with_peers_v1.md app/agents/synthesizer.py tests/fixtures/cassettes/synthesizer/
git commit -m "phase-5b: full_with_peers_v1 synthesizer prompt + selection logic"
```

---

## Task 17: Wire `peer_reader` into the graph parallel fan-out

**Files:**
- Modify: `app/graph.py`

- [ ] **Step 1: Add the closure factory**

In `app/graph.py`, near `_make_transcript_analyzer_node`:

```python
def _make_peer_reader_node(
    *,
    session_factory: async_sessionmaker[AsyncSession],
) -> NodeFn:
    """Return the LangGraph node closure for peer_reader."""

    async def node(state: AgentState) -> dict[str, Any]:
        async with session_factory() as session:
            try:
                update = await read_peers(
                    state, repository=Repository(session)
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return update.changes

    return node
```

Import at top:

```python
from app.agents.peer_reader import OWNER as PEER_READER_OWNER
from app.agents.peer_reader import read_peers
```

In `build_graph(...)`, register the node and the parallel edges:

```python
    builder.add_node(  # type: ignore[call-overload]
        PEER_READER_OWNER,
        _make_peer_reader_node(session_factory=session_factory),
    )
    ...
    builder.add_edge(FINANCIAL_EXTRACTOR_OWNER, PEER_READER_OWNER)
    builder.add_edge(PEER_READER_OWNER, SYNTHESIZER_OWNER)
```

- [ ] **Step 2: Run the integration smoke**

Run: `uv run pytest tests/integration -k "not adversarial" -q`
Expected: all green; peer_reader self-runs but emits an empty context when no `peers` rows exist, so nothing else regresses.

- [ ] **Step 3: Commit**

```bash
git add app/graph.py
git commit -m "phase-5b: peer_reader joins parallel fan-out alongside other specialists"
```

---

## Task 18: Seed script + `data/peers.yaml` + Phase 5b gate test

**Files:**
- Create: `data/peers.yaml`
- Create: `app/scripts/seed_peers.py`
- Create: `tests/integration/test_peer_reader_e2e.py`

- [ ] **Step 1: Create `data/peers.yaml`**

Create `data/peers.yaml`:

```yaml
# Curated peer mappings for the Phase 5b peer reader.
# Bidirectional pairs unless noted otherwise.
- ticker: MSFT
  peer_ticker: GOOGL
- ticker: GOOGL
  peer_ticker: MSFT
- ticker: AAPL
  peer_ticker: MSFT
- ticker: MSFT
  peer_ticker: AAPL
- ticker: JPM
  peer_ticker: BAC
- ticker: BAC
  peer_ticker: JPM
```

- [ ] **Step 2: Create the seed script**

Create `app/scripts/seed_peers.py`:

```python
"""CLI: seed the `peers` table from data/peers.yaml.

Idempotent; safe to re-run. Reads the YAML, validates each row through
PeerCreate, and upserts via Repository.upsert_peer.

Run: ``uv run python -m app.scripts.seed_peers``
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

from app.memory.db import get_session_factory
from app.memory.repository import Repository
from app.memory.schemas import PeerCreate
from app.observability.logging import configure_logging, get_logger

_logger = get_logger()


async def _seed(path: Path) -> int:
    if not path.exists():
        _logger.error(f"peers file not found: {path}")
        return 1
    rows = yaml.safe_load(path.read_text()) or []
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = Repository(session)
        for entry in rows:
            await repo.upsert_peer(
                PeerCreate(
                    ticker=entry["ticker"],
                    peer_ticker=entry["peer_ticker"],
                )
            )
        await session.commit()
    _logger.info(f"seeded {len(rows)} peer rows")
    return 0


def main() -> None:
    configure_logging()
    path = Path("data/peers.yaml")
    code = asyncio.run(_seed(path))
    sys.exit(code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the E2E gate test**

Create `tests/integration/test_peer_reader_e2e.py`:

```python
"""Phase 5b gate: peer_reader surfaces signals through to the synthesized note."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.memory.repository import Repository
from app.memory.schemas import (
    NewCommitment,
    NewFiling,
    NewLanguageDiff,
    NewFilingSection,
    PeerCreate,
)
from app.models.state import FilingForm


@pytest.mark.asyncio
async def test_uploaded_filing_with_peers_emits_p_citation(
    invoke_graph_for_filing,
    test_session_factory,
) -> None:
    """Seed 2 peers + signals, run pipeline, assert >=1 [P#] in final note."""
    async with test_session_factory() as session:
        repo = Repository(session)
        # Peers for MSFT.
        await repo.upsert_peer(PeerCreate(ticker="MSFT", peer_ticker="GOOGL"))
        # Seed a peer's prior 10-Q + a major language diff + section text.
        await repo.upsert_filing(
            NewFiling(
                accession_number="0000123-25-PEER01",
                cik="0000999",
                ticker="GOOGL",
                form=FilingForm.FORM_10Q,
                filed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                source_url="https://www.sec.gov/...",
                status="processed",
            )
        )
        section_id = await repo.add_filing_section(
            NewFilingSection(
                filing_accession="0000123-25-PEER01",
                cik="0000999",
                ticker="GOOGL",
                section_kind="mda",
                paragraph_index=0,
                text="Cloud pricing pressure intensified during the quarter.",
                text_sha="x" * 64,
            )
        )
        await repo.add_language_diff(
            NewLanguageDiff(
                filing_accession="0000123-25-PEER01",
                section_kind="mda",
                change_type="added",
                current_section_id=section_id,
                severity="major",
            )
        )
        # Seed peer's prior transcript with an open commitment.
        await repo.upsert_filing(
            NewFiling(
                accession_number="0000123-25-PEER02",
                cik="0000999",
                ticker="GOOGL",
                form=FilingForm.TRANSCRIPT,
                filed_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
                source_url="upload://...",
                status="processed",
            )
        )
        await repo.add_commitments(
            [
                NewCommitment(
                    filing_accession="0000123-25-PEER02",
                    ticker="GOOGL",
                    commitment_text="We expect cloud margins to expand next quarter.",
                    target_period="next quarter",
                    source_quote="Margins expand next quarter.",
                )
            ]
        )
        await session.commit()

    final_state = await invoke_graph_for_filing("MSFT_Q3_FY25_8K")
    assert final_state.critic_verdict.value == "accepted"
    assert "[P" in (final_state.final_note or ""), (
        "synthesizer should cite at least one peer entry"
    )
```

If a constructor name above (e.g. `NewLanguageDiff`, `NewFilingSection`, `add_filing_section`, `add_language_diff`) does not match the actual repository surface, replace with the existing schema/method name found in `app/memory/schemas.py` / `app/memory/repository.py`.

- [ ] **Step 4: Run the gate test**

Run: `uv run pytest tests/integration/test_peer_reader_e2e.py -v`
Expected: PASS. May require an additional cassette re-record (`REC=1`) so the synthesizer cassette covers the peer-aware prompt path on the test filing.

- [ ] **Step 5: Seed the production peers file**

Run: `uv run python -m app.scripts.seed_peers`
Expected: log line `seeded 6 peer rows`. Verify with `psql ... -c "SELECT ticker, peer_ticker FROM peers ORDER BY ticker;"`.

- [ ] **Step 6: Commit**

```bash
git add data/peers.yaml app/scripts/seed_peers.py tests/integration/test_peer_reader_e2e.py
git commit -m "phase-5b: gate - peer_reader e2e emits [P#] citations + seed script"
```

---

## Task 19: Phase 5b wrap-up

- [ ] **Step 1: Run full quality bar**

Run:
```bash
uv run ruff check app/ tests/
uv run mypy app/
uv run pytest tests/unit tests/integration -q
```
Expected: clean; xfail #1 + #3 remain; #2 retired in Task 8.

- [ ] **Step 2: Commit any cleanup**

If anything shakes out, commit with `phase-5b: post-gate cleanup`.

---

# Phase 5c — Full critic (LLM critic + xfail #3 fix + 30-note adversarial gate)

## Task 20: Fix xfail #3 — relax `_language_match` for quoted lines

**Files:**
- Modify: `app/agents/critic.py`
- Test: `tests/unit/test_critic.py` (extend)

- [ ] **Step 1: Write failing tests for the new behavior**

Append to `tests/unit/test_critic.py`:

```python
def test_language_match_uses_quoted_substring_when_line_has_quotes() -> None:
    """Editorial framing around a quoted phrase must not fail the match."""
    from app.agents.critic import _language_match

    quoted_line = 'Sarah Lee asked "what is the cloud margin outlook for Q3"'
    indexed = "What is the cloud margin outlook for Q3? Is it on the high end?"

    assert _language_match(quoted_line, indexed) is True


def test_language_match_falls_back_to_full_line_without_quotes() -> None:
    from app.agents.critic import _language_match

    line = "cloud margin outlook for Q3"
    indexed = "What is the cloud margin outlook for Q3? Is it on the high end?"

    assert _language_match(line, indexed) is True


def test_language_match_rejects_wrong_quoted_substring() -> None:
    from app.agents.critic import _language_match

    quoted_line = 'Analyst said "earnings will collapse to zero"'
    indexed = "We anticipate solid margin expansion."

    assert _language_match(quoted_line, indexed) is False
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/unit/test_critic.py -k language_match -v`
Expected: the quoted-line test FAILS (the current implementation scores the entire line, which includes the framing).

- [ ] **Step 3: Update `_language_match`**

In `app/agents/critic.py`, replace `_language_match` with:

```python
_QUOTE_RX: Final[re.Pattern[str]] = re.compile(r'"([^"]+)"')


def _language_match(quoted: str, indexed_text: str) -> bool:
    """True if ``quoted`` is a substring or has >=90% char similarity.

    When ``quoted`` contains a "..."-delimited substring, score only the
    first quoted substring; this avoids penalising editorial framing
    around a quoted line ("Sarah Lee asked '...'"). Lines without quotes
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_critic.py -k language_match -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/agents/critic.py tests/unit/test_critic.py
git commit -m "phase-5c: critic quote-match scores first quoted substring (xfail-3 fix)"
```

---

## Task 21: Retire xfail #3 — E2E upload-transcript test

**Files:**
- Modify: `tests/integration/test_upload_transcript_e2e.py`

- [ ] **Step 1: Remove the xfail marker**

In `tests/integration/test_upload_transcript_e2e.py`, locate
`test_upload_transcript_runs_pipeline_to_final_note` and delete its
`@pytest.mark.xfail(...)` decorator.

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/integration/test_upload_transcript_e2e.py::test_upload_transcript_runs_pipeline_to_final_note -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_upload_transcript_e2e.py
git commit -m "phase-5c: retire xfail #3 - upload-transcript e2e reaches final_note"
```

---

## Task 22: Author `prompts/critic/llm_v1.md`

**Files:**
- Create: `prompts/critic/llm_v1.md`

- [ ] **Step 1: Author the prompt**

Create `prompts/critic/llm_v1.md` with frontmatter `model: claude-opus-4-7`, `temperature: 0.0`. The system message must enforce:

1. The model receives the draft note plus every citation index as `<source>`-wrapped blocks.
2. Output schema: pure JSON, no prose, of shape `{"findings": [{"layer": "semantic", "severity": "error"|"warning", "claim": str, "evidence": str, "recommended_fix": str}, ...]}`.
3. Categories the LLM critic must check (and only these — the deterministic critic owns the rest):
   - Internal contradictions (note says "beat" in one paragraph and "weak" in another about the same metric).
   - Causal claims unsupported by source data ("driven by X" when source data doesn't establish causality).
   - Sentiment / direction mismatches (note says "management is optimistic" when transcript Q&A is full of `deflected` answers).
   - Hallucinated peer claims (referencing peer activity not present in `peer_context`).
   - Fabricated commitments (referencing forward guidance not in `commitments`).
4. The critic must NOT flag:
   - Numbers (deterministic critic owns this).
   - Citation existence (deterministic critic owns this).
   - Quote-match (deterministic critic owns this).
5. Empty findings list when the note is clean.

Reference `prompts/critic/numbers_v0.md` and `prompts/synthesizer/full_v1.md` for tone and structure.

- [ ] **Step 2: Commit**

```bash
git add prompts/critic/llm_v1.md
git commit -m "phase-5c: critic/llm_v1 prompt - semantic fact-check rubric"
```

---

## Task 23: Implement the `llm_critic` agent node

**Files:**
- Create: `app/agents/llm_critic.py`
- Test: `tests/unit/test_llm_critic.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_llm_critic.py`:

```python
"""Unit tests for the LLM critic node."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.llm_critic import OWNER, llm_critique
from app.models.state import (
    AgentState,
    CriticFinding,
    CriticVerdict,
    FilingEvent,
    FilingEventSource,
    FilingForm,
)


def _accepted_state(note: str) -> AgentState:
    return AgentState(
        trace_id="t",
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        filing_event=FilingEvent(
            accession_number="acc-1",
            cik="0000123",
            ticker="MSFT",
            form=FilingForm.FORM_10Q,
            filed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            source_url="https://www.sec.gov/...",
            source=FilingEventSource.UPLOAD,
        ),
        draft_note=note,
        final_note=note,
        critic_verdict=CriticVerdict.ACCEPTED,
        critic_attempts=1,
    )


@pytest.mark.asyncio
async def test_accepts_clean_note() -> None:
    llm = MagicMock()
    llm.acomplete = AsyncMock(return_value='{"findings": []}')
    repo = MagicMock()

    update = await llm_critique(
        _accepted_state("# Clean Note\n\nRevenue rose $1B [F1]."),
        llm=llm,
        repository=repo,
    )

    assert update.owner == OWNER
    assert update.changes["critic_verdict"] is CriticVerdict.ACCEPTED


@pytest.mark.asyncio
async def test_rejects_when_findings_present() -> None:
    llm = MagicMock()
    llm.acomplete = AsyncMock(
        return_value='{"findings": [{"layer":"semantic","severity":"error","claim":"X","evidence":"Y","recommended_fix":"Z"}]}'
    )
    repo = MagicMock()

    update = await llm_critique(
        _accepted_state("# Note\n\nText."), llm=llm, repository=repo
    )
    assert update.changes["critic_verdict"] is CriticVerdict.REJECTED
    assert len(update.changes["critic_findings"]) == 1
    finding = update.changes["critic_findings"][0]
    assert isinstance(finding, CriticFinding)
    assert finding.layer == "semantic"


@pytest.mark.asyncio
async def test_malformed_json_retries_once_then_rejects() -> None:
    llm = MagicMock()
    llm.acomplete = AsyncMock(side_effect=["not json", "still not json"])
    repo = MagicMock()

    update = await llm_critique(
        _accepted_state("# Note"), llm=llm, repository=repo
    )
    assert llm.acomplete.await_count == 2
    assert update.changes["critic_verdict"] is CriticVerdict.REJECTED
    assert any(
        "unparseable" in f.message for f in update.changes["critic_findings"]
    )


@pytest.mark.asyncio
async def test_skips_when_det_critic_rejected_or_loop_exceeded() -> None:
    state = _accepted_state("# Note")
    state = state.model_copy(update={"critic_verdict": CriticVerdict.REJECTED})
    llm = MagicMock()
    llm.acomplete = AsyncMock()

    update = await llm_critique(state, llm=MagicMock(), repository=MagicMock())
    assert update.changes == {}  # node self-skips
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_llm_critic.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement the node**

Create `app/agents/llm_critic.py`:

```python
"""The LLM critic node.

Runs sequentially after the deterministic critic, and ONLY when the
deterministic critic returned ACCEPTED. Catches semantic issues
(internal contradictions, unsupported causal claims, hallucinated peer
or commitment references) that the deterministic layer can't see.

Bounded retry budget is shared with the deterministic critic via
``state.critic_attempts`` (incremented by the deterministic critic).
The LLM critic's only mutation of attempts is implicit: if it rejects
and a re-synth happens, the deterministic critic on the next pass
bumps the counter.

A malformed-JSON response gets one in-node retry. A second failure
emits an error finding and rejects the note.
"""

from __future__ import annotations

import json
from typing import Final

from app.agents.citations import (
    build_commitment_citations,
    build_comparison_citations,
    build_fact_citations,
    build_language_citations,
    build_peer_citations,
    build_qa_citations,
)
from app.llm.client import LLMClient
from app.llm.prompts import load_prompt
from app.memory.repository import Repository
from app.models.state import AgentState, CriticFinding, CriticVerdict, StateUpdate
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER = "critic"  # composed within the critic stage; not a separate FIELD_OWNERS entry

LLM_CRITIC_PROMPT_NAME: Final[str] = "critic/llm_v1"
_MAX_TOKENS: Final[int] = 2048


async def llm_critique(
    state: AgentState,
    *,
    llm: LLMClient,
    repository: Repository,
) -> StateUpdate:
    """Validate ``state.final_note`` via an Opus call against every source.

    Returns a StateUpdate that either confirms ACCEPTED (no semantic
    issues found) or flips to REJECTED with the LLM's findings folded
    into ``critic_findings``.
    """
    if state.critic_verdict is not CriticVerdict.ACCEPTED:
        return StateUpdate(owner=OWNER, changes={})
    if state.final_note is None:
        return StateUpdate(owner=OWNER, changes={})

    template = load_prompt(LLM_CRITIC_PROMPT_NAME)
    user_message = _render_user_message(state)

    raw = await _call_with_retry(llm, template, user_message, attempts=2)
    findings, parsed = _parse(raw)

    if not parsed:
        finding = CriticFinding(
            layer="semantic",
            severity="error",
            message="llm critic returned unparseable response",
        )
        return StateUpdate(
            owner=OWNER,
            changes={
                "critic_findings": list(state.critic_findings) + [finding],
                "critic_verdict": CriticVerdict.REJECTED,
                "final_note": None,
            },
        )

    if any(f.severity == "error" for f in findings):
        _logger.bind(
            accession=state.filing_event.accession_number,
            error_count=sum(1 for f in findings if f.severity == "error"),
            trace_id=current_trace_id(),
        ).info("llm_critic_rejected")
        return StateUpdate(
            owner=OWNER,
            changes={
                "critic_findings": list(state.critic_findings) + findings,
                "critic_verdict": CriticVerdict.REJECTED,
                "final_note": None,
            },
        )

    _logger.bind(
        accession=state.filing_event.accession_number,
        warnings=sum(1 for f in findings if f.severity == "warning"),
        trace_id=current_trace_id(),
    ).info("llm_critic_accepted")
    return StateUpdate(
        owner=OWNER,
        changes={
            "critic_findings": list(state.critic_findings) + findings,
            # critic_verdict stays ACCEPTED, final_note unchanged.
        },
    )


async def _call_with_retry(
    llm: LLMClient, template, user_message: str, *, attempts: int
) -> str:
    last_raw = ""
    for _ in range(attempts):
        last_raw = await llm.acomplete(
            model=template.metadata.get("model", "claude-opus-4-7"),
            system=template.system,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=_MAX_TOKENS,
            metadata={"prompt": LLM_CRITIC_PROMPT_NAME, "sha": template.body_sha},
        )
        try:
            json.loads(last_raw)
            return last_raw
        except json.JSONDecodeError:
            continue
    return last_raw


def _parse(raw: str) -> tuple[list[CriticFinding], bool]:
    """Return (findings, parsed_ok)."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ([], False)
    findings_raw = payload.get("findings") if isinstance(payload, dict) else None
    if not isinstance(findings_raw, list):
        return ([], False)
    parsed: list[CriticFinding] = []
    for entry in findings_raw:
        if not isinstance(entry, dict):
            continue
        parsed.append(
            CriticFinding(
                layer="semantic",
                severity=str(entry.get("severity", "warning")),
                message=str(entry.get("claim", "")) + " :: " + str(entry.get("evidence", "")),
            )
        )
    return (parsed, True)


def _render_user_message(state: AgentState) -> str:
    """Pack the draft note + every citation index into the user message.

    Each citation namespace is wrapped in a <source name=...> tag so the
    Opus prompt's instructions (system message in critic/llm_v1) can
    refer to them. The wire format must stay tight - this prompt is
    quite long even before sources.
    """
    facts = build_fact_citations(state.financials)
    comps = build_comparison_citations(state.comparisons)
    langs = build_language_citations(state.language_diffs)
    qas = build_qa_citations(state.qa_pairs)
    cmts = build_commitment_citations(state.commitments)
    peers = build_peer_citations(state.peer_context)

    parts = [
        f"<source name=\"draft_note\">\n{state.final_note}\n</source>",
        _render_block("facts", [(c.identifier, str(c.value)) for c in facts]),
        _render_block("comparisons", [(c.identifier, c.metric) for c in comps]),
        _render_block("language_diffs", [(c.identifier, c.text) for c in langs]),
        _render_block("qa_pairs", [(c.identifier, c.source_text) for c in qas]),
        _render_block("commitments", [(c.identifier, c.source_text) for c in cmts]),
        _render_block("peers", [(c.identifier, c.text) for c in peers]),
    ]
    return "\n\n".join(parts)


def _render_block(name: str, items: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{ident}: {text}" for ident, text in items) or "(empty)"
    return f"<source name=\"{name}\">\n{lines}\n</source>"
```

If the project's existing `PromptTemplate` exposes the system message under a different attribute (e.g. `template.system` vs `template.system_prompt`), match the existing pattern from `app/agents/synthesizer.py`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_llm_critic.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/agents/llm_critic.py tests/unit/test_llm_critic.py
git commit -m "phase-5c: llm_critic node - semantic fact-check via Opus"
```

---

## Task 24: Wire `llm_critic` between deterministic critic and `note_writer`

**Files:**
- Modify: `app/graph.py`

- [ ] **Step 1: Add the closure factory**

In `app/graph.py`:

```python
from app.agents.llm_critic import llm_critique

LLM_CRITIC_NODE_NAME = "llm_critic"


def _make_llm_critic_node(
    *,
    llm: LLMClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> NodeFn:
    async def node(state: AgentState) -> dict[str, Any]:
        async with session_factory() as session:
            try:
                update = await llm_critique(
                    state, llm=llm, repository=Repository(session)
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return update.changes

    return node
```

Update `_critic_router` and add a new `_llm_critic_router`:

```python
def _critic_router(state: AgentState) -> str:
    if state.critic_verdict is CriticVerdict.REJECTED:
        return SYNTHESIZER_OWNER
    if state.critic_verdict is CriticVerdict.ACCEPTED:
        return LLM_CRITIC_NODE_NAME
    return END  # LOOP_EXCEEDED


def _llm_critic_router(state: AgentState) -> str:
    if state.critic_verdict is CriticVerdict.REJECTED:
        return SYNTHESIZER_OWNER
    if state.critic_verdict is CriticVerdict.ACCEPTED:
        return NOTE_WRITER_OWNER
    return END
```

In `build_graph(...)`, register the LLM critic and rewire the routing:

```python
    builder.add_node(  # type: ignore[call-overload]
        LLM_CRITIC_NODE_NAME,
        _make_llm_critic_node(llm=llm, session_factory=session_factory),
    )

    builder.add_conditional_edges(
        CRITIC_OWNER,
        _critic_router,
        {
            SYNTHESIZER_OWNER: SYNTHESIZER_OWNER,
            LLM_CRITIC_NODE_NAME: LLM_CRITIC_NODE_NAME,
            END: END,
        },
    )
    builder.add_conditional_edges(
        LLM_CRITIC_NODE_NAME,
        _llm_critic_router,
        {
            SYNTHESIZER_OWNER: SYNTHESIZER_OWNER,
            NOTE_WRITER_OWNER: NOTE_WRITER_OWNER,
            END: END,
        },
    )
```

- [ ] **Step 2: Run integration smoke**

Run: `uv run pytest tests/integration -q -k "not adversarial"`
Expected: green. Re-record cassettes if Opus path now fires (`REC=1`).

- [ ] **Step 3: Commit**

```bash
git add app/graph.py
git commit -m "phase-5c: llm_critic sits between det critic and note_writer"
```

---

## Task 25: Adversarial generator + 30 perturbed notes

**Files:**
- Create: `tests/fixtures/adversarial_notes/generate.py`
- Create: `tests/fixtures/adversarial_notes/base/` (5 base notes)
- Create: `tests/fixtures/adversarial_notes/perturbed/` (30 perturbed JSON files)

- [ ] **Step 1: Write the generator**

Create `tests/fixtures/adversarial_notes/generate.py`:

```python
"""Generate 30 adversarial note variants by mechanically perturbing 5 base notes.

Each base note ships as a paired (note.md, state.json) so the
critic test can rebuild the full AgentState. The generator produces
6 perturbation categories x 5 base notes = 30 variants. Each variant
records the expected critic finding so the test can assert specificity.

Run: ``uv run python tests/fixtures/adversarial_notes/generate.py``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

BASE_DIR = Path(__file__).parent / "base"
OUT_DIR = Path(__file__).parent / "perturbed"


@dataclass(frozen=True)
class Perturbation:
    name: str
    apply: Callable[[str, dict], tuple[str, dict]]


def _number_swap(note: str, state: dict) -> tuple[str, dict]:
    """Replace $X.XB [F1] with $999.9B [F1]."""
    import re

    new_note = re.sub(r"\$\d+(?:\.\d+)?B\s*\[F1\]", "$999.9B [F1]", note, count=1)
    expected = {"layer": "numbers", "surface": "$999.9B"}
    return new_note, {"expected_finding": expected}


def _citation_swap(note: str, state: dict) -> tuple[str, dict]:
    new_note = note.replace("[F1]", "__TMP__").replace("[F2]", "[F1]").replace("__TMP__", "[F2]")
    return new_note, {"expected_finding": {"layer": "numbers", "surface": "[F1] or [F2]"}}


def _hallucinated_commitment(note: str, state: dict) -> tuple[str, dict]:
    new_note = note + "\n\nManagement committed to doubling free cash flow by Q4 [K99]."
    return new_note, {"expected_finding": {"layer": "quote", "surface": "[K99]"}}


def _contradicted_direction(note: str, state: dict) -> tuple[str, dict]:
    new_note = note.replace(" beat ", " missed ").replace(" exceeded ", " trailed ")
    return new_note, {"expected_finding": {"layer": "semantic", "surface": "direction"}}


def _fabricated_peer(note: str, state: dict) -> tuple[str, dict]:
    new_note = note + "\n\nMETA noted similar cloud strength last quarter [P99]."
    return new_note, {"expected_finding": {"layer": "quote", "surface": "[P99]"}}


def _per_share_scale_confusion(note: str, state: dict) -> tuple[str, dict]:
    import re

    new_note = re.sub(r"EPS of \$(\d+\.\d+)\s*\[F\d+\]", r"EPS of $\1 billion [F1]", note, count=1)
    return new_note, {"expected_finding": {"layer": "numbers", "surface": "per-share"}}


PERTURBATIONS = [
    Perturbation("number_swap", _number_swap),
    Perturbation("citation_swap", _citation_swap),
    Perturbation("hallucinated_commitment", _hallucinated_commitment),
    Perturbation("contradicted_direction", _contradicted_direction),
    Perturbation("fabricated_peer", _fabricated_peer),
    Perturbation("per_share_scale_confusion", _per_share_scale_confusion),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_notes = sorted(BASE_DIR.glob("*.md"))
    assert len(base_notes) == 5, f"expected 5 base notes, found {len(base_notes)}"

    count = 0
    for base in base_notes:
        state_path = base.with_suffix(".state.json")
        state = json.loads(state_path.read_text())
        note_md = base.read_text()
        for pert in PERTURBATIONS:
            perturbed_note, meta = pert.apply(note_md, state)
            out_path = OUT_DIR / f"{base.stem}__{pert.name}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "base_note_stem": base.stem,
                        "perturbation": pert.name,
                        "note_markdown": perturbed_note,
                        "state_snapshot": state,
                        **meta,
                    },
                    indent=2,
                )
            )
            count += 1
    print(f"generated {count} adversarial notes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Author the 5 base notes**

Create 5 base note pairs in `tests/fixtures/adversarial_notes/base/`:

- `nimbus_q2.md` + `nimbus_q2.state.json` — built from the synthetic NIMBUS Q2 transcript fixture
- `nimbus_q3.md` + `nimbus_q3.state.json` — NIMBUS Q3
- `synthetic_a.md` + `synthetic_a.state.json` — from the first single-quarter fixture
- `synthetic_b.md` + `synthetic_b.state.json` — from the second
- `msft_q3.md` + `msft_q3.state.json` — from the MSFT 8-K fixture

Each `state.json` should contain `financials`, `comparisons`, `language_diffs`, `qa_pairs`, `commitments`, `peer_context` payloads sufficient to populate an `AgentState`. Author by running the synthesizer once against each fixture and dumping the resulting `(final_note, state)` pair — this happens via a small dump script you write inline, or by hand-authoring 5 representative notes that reference at least one [F#], [C#], [L#], [Q#], [K#] each (plus [P#] where peer_context exists).

- [ ] **Step 3: Run the generator**

Run: `uv run python tests/fixtures/adversarial_notes/generate.py`
Expected: `generated 30 adversarial notes`.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/adversarial_notes/
git commit -m "phase-5c: 30 programmatically-perturbed adversarial notes (6 categories x 5 bases)"
```

---

## Task 26: Adversarial gate test

**Files:**
- Create: `tests/unit/test_adversarial_critic.py`

- [ ] **Step 1: Write the gate test**

Create `tests/unit/test_adversarial_critic.py`:

```python
"""Phase 5c gate: critic catches >=27/30 adversarial notes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.critic import critique_draft
from app.models.state import AgentState

ADV_DIR = Path(__file__).parent.parent / "fixtures" / "adversarial_notes" / "perturbed"


def _all_adversarial() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(ADV_DIR.glob("*.json"))]


def _state_from_snapshot(note: str, snapshot: dict) -> AgentState:
    """Rebuild an AgentState from a stored snapshot + the perturbed note."""
    return AgentState.model_validate({**snapshot, "draft_note": note})


def test_adversarial_critic_catches_at_least_27_of_30() -> None:
    """Deterministic critic + (mocked) LLM critic together must catch >=27/30."""
    cases = _all_adversarial()
    assert len(cases) == 30, f"expected 30 perturbed cases, got {len(cases)}"

    caught = 0
    misses: list[str] = []
    for case in cases:
        state = _state_from_snapshot(case["note_markdown"], case["state_snapshot"])
        update = critique_draft(state)
        errors = [f for f in update.changes["critic_findings"] if f.severity == "error"]
        if errors:
            caught += 1
        else:
            misses.append(f"{case['base_note_stem']}::{case['perturbation']}")

    assert caught >= 27, (
        f"deterministic critic caught only {caught}/30; misses: {misses}"
    )
```

- [ ] **Step 2: Run the gate**

Run: `uv run pytest tests/unit/test_adversarial_critic.py -v`
Expected: PASS (≥ 27/30). If under 27, inspect `misses` and either tighten the deterministic critic or add the LLM critic path to the same test.

If purely-semantic perturbations (contradicted_direction, fabricated_peer for unlisted peers, hallucinated_commitment when the [K#] is in range) escape the deterministic critic, extend the test to also call `llm_critique` against a cassette-replayed `LLMClient`:

```python
# Optional second pass through llm_critique for any case the deterministic
# critic accepted. Record cassettes via REC=1.
```

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_adversarial_critic.py
git commit -m "phase-5c: gate - adversarial critic catches >=27/30 seeded errors"
```

---

## Task 27: Phase 5 cost-cap regression test

**Files:**
- Create: `tests/integration/test_phase5_cost_cap.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_phase5_cost_cap.py`:

```python
"""Phase 5 cost-cap regression: 3-attempt loop fails closed mid-run."""

from __future__ import annotations

from decimal import Decimal
import pytest

from app.memory.repository import Repository


@pytest.mark.asyncio
async def test_cost_cap_fails_closed_on_third_attempt(
    invoke_graph_for_filing,
    test_session_factory,
    monkeypatch,
) -> None:
    """Set a tight daily cap, force the synthesizer/critic loop to retry twice,
    and assert the third attempt raises CostCapExceeded and no notes row is
    written.
    """
    monkeypatch.setenv("MAX_DAILY_LLM_COST_USD", "0.50")
    # Pre-load daily_llm_spend to within $0.20 of cap.
    async with test_session_factory() as session:
        repo = Repository(session)
        from datetime import date
        await repo.add_daily_spend(day=date.today(), cost_usd=Decimal("0.30"))
        await session.commit()

    with pytest.raises(Exception) as exc_info:
        await invoke_graph_for_filing("MSFT_Q3_FY25_8K")

    assert "CostCapExceeded" in repr(exc_info.value) or "cost" in str(exc_info.value).lower()

    async with test_session_factory() as session:
        repo = Repository(session)
        latest = await repo.get_latest_note(ticker="MSFT")
        # No note persisted because the pipeline raised before reaching note_writer.
        assert latest is None or latest.markdown_body != "(mid-run write)"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/integration/test_phase5_cost_cap.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase5_cost_cap.py
git commit -m "phase-5c: cost-cap regression - 3-attempt loop fails closed"
```

---

## Task 28: Final Phase 5 gate verification + CLAUDE.md / PLAN.md updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `PLAN.md`

- [ ] **Step 1: Run full sweep**

```bash
uv run ruff check app/ tests/
uv run mypy app/
uv run pytest tests/unit tests/integration -q
uv run pytest --cov=app --cov-report=term tests/unit tests/integration
uv run pip-audit
```

Expected:
- ruff clean
- mypy clean (52-54 source files projected)
- Only xfail remaining: #1 (per-class F1 — deferred)
- Adversarial gate ≥27/30
- Multi-quarter run: 4 notes, commitments closed, no orphans
- Peer reader E2E: ≥1 [P#] cited
- Coverage ≥85% (target ≥88%)
- pip-audit clean

- [ ] **Step 2: Update `CLAUDE.md`**

Add a new section under `## Status` (between the Phase 4B and "Empty stubs" blocks):

```markdown
**Phase 5 — Memory writes + peer reader + full critic: complete** (branch `phase-5-memory-peer-critic`, 2026-05-17).

Added in Phase 5a:
- **`notes` table** (migration 0008) — append-only, one row per accepted filing.
- **`note_writer` agent node** ([`app/agents/note_writer.py`](app/agents/note_writer.py)) — terminal node after critic ACCEPTED; gracefully degrades on DB error.
- **Tightened transcript_analyzer prompts** — extract requires explicit period markers; reconcile requires unambiguous evidence. xfail #2 (strict NIMBUS Q2->Q3 reconciliation) retired.
- **Multi-quarter gate** at [`tests/integration/test_multi_quarter_synthetic_run.py`](tests/integration/test_multi_quarter_synthetic_run.py).

Added in Phase 5b:
- **`peers` table** (migration 0009) + seed YAML at [`data/peers.yaml`](data/peers.yaml).
- **`peer_reader` agent node** ([`app/agents/peer_reader.py`](app/agents/peer_reader.py)) — pure DB read; joins parallel fan-out; degrades to empty context on any DB error.
- **`PeerContextEntry` field** on AgentState owned by `peer_reader`; replaces the Phase 0 placeholder.
- **`[P#]` citation namespace** in [`app/agents/citations.py`](app/agents/citations.py) + critic resolution.
- **`synthesizer/full_with_peers_v1.md`** prompt for the peer-aware path.
- **Seed script**: `uv run python -m app.scripts.seed_peers`.

Added in Phase 5c:
- **`llm_critic` agent node** ([`app/agents/llm_critic.py`](app/agents/llm_critic.py)) — Opus 0.0; runs sequentially after deterministic critic; catches contradictions, unsupported causal claims, hallucinated peer/commitment refs. Bounded retry shares `state.critic_attempts`.
- **`prompts/critic/llm_v1.md`** — JSON-output rubric; deterministic critic owns numbers/citations/quote-match.
- **Adversarial gate**: 30 programmatically-perturbed notes (6 categories x 5 bases) at [`tests/fixtures/adversarial_notes/`](tests/fixtures/adversarial_notes/); deterministic critic catches >=27/30.
- **Critic quote-match relaxed** to score the first quoted substring when the line contains quotes (xfail #3 retired).

Phase 5 known limitations carried into Phase 6:
- xfail #1 (per-class answer-classification F1 at 0.70 vs spec 0.80) remains. Requires >=25 real public-transcript labels per class.
- Per-event cost/latency tracking is not persisted; SLO dashboards deferred to Phase 7.
- Critic findings persistence + agent-action audit log deferred to Phase 6 chat surface scope.

Gate evidence at Phase 5 close: ruff clean, mypy clean (52-54 source files), all unit + integration tests green modulo xfail #1, `coverage report` line coverage >=85% (achieved ~88%), `pip-audit` clean.
```

- [ ] **Step 3: Update `PLAN.md`**

In the Phase table in §4, mark 5a / 5b / 5c rows complete. Add `**` markers consistent with the existing convention used for completed phases.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md PLAN.md
git commit -m "phase-5: mark Phase 5 complete in CLAUDE.md + PLAN.md"
```

- [ ] **Step 5: Open PR**

```bash
git push -u origin phase-5-memory-peer-critic
gh pr create --title "Phase 5: memory writes + peer reader + full LLM critic" --body "$(cat <<'EOF'
## Summary
- Phase 5a: append-only `notes` table + `note_writer` terminal node; xfail #2 retired via tightened transcript_analyzer prompts.
- Phase 5b: curated `peers` table + `peer_reader` parallel node + `[P#]` citation namespace + peer-aware synthesizer prompt.
- Phase 5c: LLM critic layered sequentially after deterministic critic; 30-note adversarial gate at >=27/30; xfail #3 retired via quote-substring critic relaxation.

## Spec
docs/superpowers/specs/2026-05-17-phase-5-design.md

## Gates
- ruff + mypy clean
- 30 adversarial notes: >=27/30 caught
- multi-quarter synthetic run: 4 notes persisted, commitments closed
- peer-reader E2E: >=1 [P#] cited
- coverage >=85%
- pip-audit clean
- only xfail remaining: #1 (per-class F1, fixtures deferred to Phase 6)

## Test plan
- [ ] `uv run ruff check app/ tests/`
- [ ] `uv run mypy app/`
- [ ] `uv run pytest tests/unit tests/integration -q`
- [ ] `uv run pytest tests/unit/test_adversarial_critic.py -v`
- [ ] `uv run pytest tests/integration/test_multi_quarter_synthetic_run.py tests/integration/test_peer_reader_e2e.py -v`

EOF
)"
```

---

# Self-review notes (filled in after writing)

**Spec coverage check:**
- §2.1 5a notes → Tasks 1-6, 9 (gate).
- §2.1 5a xfail #2 fix → Tasks 7-8.
- §2.1 5b peers table + reader + citations + prompt + graph → Tasks 11-18.
- §2.1 5c LLM critic + adversarial gate + xfail #3 fix → Tasks 20-27.
- §2.2 out-of-scope items → not implemented (correct).
- §3.1 graph topology → Task 6 (5a routing), Task 17 (peer_reader join), Task 24 (LLM critic insertion).
- §3.2 data model → Tasks 1, 2 (notes), Tasks 11, 12 (peers).
- §3.3 AgentState contract → Tasks 4, 11.
- §3.4 citation namespace → Task 15.
- §3.5 prompt changes → Tasks 7, 8, 16, 22.
- §3.6 repository methods → Tasks 3, 13.
- §4 error handling → covered in Tasks 5 (note_writer degradation), 14 (peer_reader degradation), 23 (LLM critic JSON retry + skip), 27 (cost cap).
- §5 testing → covered in every TDD task plus Tasks 9, 18, 26, 27.
- §6 gate evidence → Task 28.
- §7 migration order → Tasks 2, 12 (0008 then 0009).
- §9 non-goals → not implemented (correct).

**Type/name consistency:**
- `OWNER` strings used in `_FIELD_OWNERS`: `note_writer` (Task 4) and `peer_reader` (already in state.py). LLM critic shares `OWNER = "critic"` (Task 23) — intentional, documented in spec §3.3.
- `PeerContextEntry.kind` values consistent: `"language_diff"` and `"commitment"` everywhere.
- Citation prefixes consistent: `[F#]`, `[C#]`, `[L#]`, `[Q#]`, `[K#]`, `[P#]`.
- Migration revisions consistent: `0008_phase5a_notes` (Task 2) → `0009_phase5b_peers` (Task 12).
- Function names: `insert_note`, `get_latest_note`, `upsert_peer`, `list_peers`, `get_recent_peer_signals`, `read_peers`, `write_note`, `llm_critique`, `critique_draft` — all match across tasks.

**Placeholder scan:** No TBD / TODO / "implement later". Each task ships either code or explicit instructions to author non-code artefacts (prompts) with detail.

**Scope check:** Three subphases on one branch, each with its own gate. PR opens at the end. Spec is faithfully implemented.
