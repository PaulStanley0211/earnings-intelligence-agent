# Phase 4A: Upload Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the upload-and-advise foundation so a user can `POST /api/advise` with a ticker to get a download checklist, then `POST /api/upload` with the resulting PDF, and the existing Phase 1-3 pipeline (financial extractor → comparator → language differ → synthesizer → critic) runs over the uploaded content end-to-end. The autonomous EDGAR watcher is gated behind `WATCHER_MODE_ENABLED`.

**Architecture:** A new `upload_intake` agent node converts an uploaded PDF or plain-text file into a `FilingEvent` (the same shape the watcher produces today) and persists the raw text to a new `uploaded_documents` table. The downstream graph is unchanged. A separate `document_advisor` agent node queries the existing Phase 1 EDGAR client and returns a ranked "what to upload" checklist plus a transcript-source hint. Three new FastAPI routes (`/api/advise`, `/api/upload`, `/api/chat`) expose the capability; the `/api/chat` route ships as a 501 stub in 4A (full chat agent is Phase 6). The watcher is opt-in behind `WATCHER_MODE_ENABLED` so production deploys run upload-only by default.

**Tech Stack:** FastAPI, LangGraph, SQLAlchemy 2.x async, Alembic, Pydantic v2, pydantic-settings, pypdf, httpx, pytest + pytest-asyncio.

---

## Conventions used throughout this plan

- All file paths are repository-relative (project root: `c:/Users/pauls/Projects/Earnings Intellegence Agent/`).
- The project uses **uv only** — never `pip install`. Add deps with `uv add <pkg>` (or `uv add --dev <pkg>` for test-only).
- Run tests with `uv run pytest <path> -q`. Run a single test with `uv run pytest tests/unit/test_foo.py::test_bar -v`.
- Lint with `uv run ruff check app/ tests/`, type-check with `uv run mypy app/`. Both must pass before commit.
- Commit after each task. Commit-message style mirrors recent history: `phase-4a: <topic in lowercase>` (see `git log` for examples).
- Database migrations follow the file pattern `migrations/versions/YYYYMMDD_HHMM_NNNN_phase4a_<topic>.py`. The next sequence number is `0004`.
- Do not skip `ruff` or `mypy` failures. If a hook fails, fix the underlying issue.
- Memory remains append-only — no `UPDATE` on `uploaded_documents`.

---

## File structure (new + modified)

**New files:**
- `app/tools/documents.py` — PDF + plain-text intake with hard size cap and scanned-PDF rejection.
- `app/tools/advisor.py` — EDGAR helper returning the ranked filing checklist for a ticker.
- `app/agents/document_advisor.py` — agent wrapper around the advisor tool.
- `app/agents/upload_intake.py` — agent node converting a parsed upload into a `FilingEvent` and persisting the row.
- `app/api/advise.py` — `POST /api/advise` route.
- `app/api/upload.py` — `POST /api/upload` route.
- `app/api/chat.py` — `POST /api/chat` minimal stub returning 501.
- `migrations/versions/20260516_<HHMM>_0004_phase4a_uploaded_documents.py` — new Alembic migration.
- `tests/unit/test_documents_parser.py`
- `tests/unit/test_advisor.py`
- `tests/unit/test_document_advisor_agent.py`
- `tests/unit/test_upload_intake.py`
- `tests/integration/test_upload_api.py` — covers advise/upload/chat routes end-to-end.

**Modified files:**
- `app/config.py` — add `watcher_mode_enabled: bool` (default `False`) and `max_upload_bytes: int` (default 26214400).
- `app/memory/models.py` — add `UploadedDocumentORM`.
- `app/memory/schemas.py` — add `UploadedDocumentDTO` and `NewUploadedDocument`.
- `app/memory/repository.py` — add `add_uploaded_document`, `get_uploaded_document_by_sha256`, `get_uploaded_document`.
- `app/api/health.py` — short-circuit the watcher freshness check when `watcher_mode_enabled` is `False` (return `not_applicable`).
- `app/agents/watcher.py` — `watch_forever` raises `WatcherDisabled` when `watcher_mode_enabled` is `False`.
- `app/scripts/poll_once.py` — read the flag and log a warning when running with watcher disabled (still allowed for ad-hoc operator use).
- `app/api/__init__.py` — wire the new routers.
- `tests/unit/test_config.py` — cover the new settings.
- `tests/integration/test_health.py` — cover both flag states.

---

## Task 1: Add settings for `watcher_mode_enabled` and `max_upload_bytes`

**Files:**
- Modify: `app/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config.py`:

```python
def test_settings_default_watcher_disabled(monkeypatch_env_with_required):
    """``watcher_mode_enabled`` defaults to False so production runs upload-only."""
    monkeypatch_env_with_required.delenv("WATCHER_MODE_ENABLED", raising=False)
    from app.config import Settings, reset_settings_cache
    reset_settings_cache()
    settings = Settings()  # type: ignore[call-arg]
    assert settings.watcher_mode_enabled is False


def test_settings_watcher_can_be_enabled(monkeypatch_env_with_required):
    monkeypatch_env_with_required.setenv("WATCHER_MODE_ENABLED", "true")
    from app.config import Settings, reset_settings_cache
    reset_settings_cache()
    settings = Settings()  # type: ignore[call-arg]
    assert settings.watcher_mode_enabled is True


def test_settings_max_upload_bytes_default(monkeypatch_env_with_required):
    monkeypatch_env_with_required.delenv("MAX_UPLOAD_BYTES", raising=False)
    from app.config import Settings, reset_settings_cache
    reset_settings_cache()
    settings = Settings()  # type: ignore[call-arg]
    assert settings.max_upload_bytes == 26214400
```

Reuse the existing `monkeypatch_env_with_required` fixture in this test module — do not invent a new one.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -q -k "watcher or max_upload"`
Expected: `FAILED` with `AttributeError: 'Settings' object has no attribute 'watcher_mode_enabled'`.

- [ ] **Step 3: Implement the new fields**

Edit `app/config.py`. Add these two fields inside the `Settings` class, in the `# ---- EDGAR ----` and `# ---- Runtime ----` regions respectively:

In the `# ---- EDGAR ----` block (after `edgar_poll_interval_seconds`):

```python
    watcher_mode_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in flag for the eval/demo EDGAR watcher. The upload-first product "
            "runs with this off; turn on for nightly evals or demo recordings."
        ),
    )
```

In the `# ---- Runtime ----` block (after `environment`):

```python
    max_upload_bytes: int = Field(
        default=26_214_400,
        gt=0,
        description="Hard upper bound on /api/upload payload size. Default 25 MiB.",
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py -q -k "watcher or max_upload"`
Expected: 3 passed.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check app/ tests/` then `uv run mypy app/`. Both must be clean.

Commit:

```bash
git add app/config.py tests/unit/test_config.py
git commit -m "phase-4a: add watcher_mode_enabled and max_upload_bytes settings"
```

---

## Task 2: Alembic migration for `uploaded_documents`

**Files:**
- Create: `migrations/versions/20260516_<HHMM>_0004_phase4a_uploaded_documents.py` (substitute the current `HHMM` UTC at the moment of creation)
- Test: `tests/integration/test_migrations.py` (extend if a stamp/upgrade test exists, otherwise the migration is exercised by the next task's integration tests)

- [ ] **Step 1: Create the migration file**

Use the existing migration files (`migrations/versions/20260515_*.py`) as the template for header, imports, and revision-id conventions. The new file must:

- `revision = "0004_phase4a_uploaded_documents"`
- `down_revision = "0003_phase3_schema"` (verify by reading the head of `migrations/versions/20260515_2330_0003_phase3_schema.py`)

Body:

```python
"""Phase 4A: uploaded_documents table for user-supplied filings.

Append-only. One row per accepted upload. SHA-256 is unique to deduplicate
re-uploads of the same content.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_phase4a_uploaded_documents"
down_revision = "0003_phase3_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "uploaded_documents",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("upload_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("filing_type", sa.String(length=16), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("parsed_text", sa.Text, nullable=False),
        sa.Column("parsed_char_count", sa.Integer, nullable=False),
        sa.Column("page_count", sa.Integer, nullable=True),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_uploaded_documents_content_sha256",
        "uploaded_documents",
        ["content_sha256"],
        unique=True,
    )
    op.create_index(
        "ix_uploaded_documents_ticker",
        "uploaded_documents",
        ["ticker"],
    )


def downgrade() -> None:
    op.drop_index("ix_uploaded_documents_ticker", table_name="uploaded_documents")
    op.drop_index("ix_uploaded_documents_content_sha256", table_name="uploaded_documents")
    op.drop_table("uploaded_documents")
```

- [ ] **Step 2: Run the migration against the local Postgres**

Run: `uv run alembic upgrade head`
Expected: `Running upgrade 0003_phase3_schema -> 0004_phase4a_uploaded_documents`, no error.

- [ ] **Step 3: Verify the schema**

Run: `uv run python -c "import asyncio; from sqlalchemy import text; from app.memory.db import get_engine; async def main():
    async with get_engine().connect() as c:
        result = await c.execute(text(\"select column_name from information_schema.columns where table_name='uploaded_documents' order by ordinal_position\"))
        print([row[0] for row in result.fetchall()])
asyncio.run(main())"`

Expected output starts with: `['id', 'upload_id', 'ticker', 'filing_type', 'original_filename', 'content_sha256', 'parsed_text', 'parsed_char_count', 'page_count', 'uploaded_at']`.

- [ ] **Step 4: Commit**

```bash
git add migrations/versions/20260516_*0004_phase4a_uploaded_documents.py
git commit -m "phase-4a: alembic migration for uploaded_documents"
```

---

## Task 3: ORM model, DTOs, and repository methods

**Files:**
- Modify: `app/memory/models.py`
- Modify: `app/memory/schemas.py`
- Modify: `app/memory/repository.py`
- Test: `tests/integration/test_repository.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/integration/test_repository.py`:

```python
@pytest.mark.asyncio
async def test_add_and_fetch_uploaded_document(session_factory):
    from app.memory.repository import Repository
    from app.memory.schemas import NewUploadedDocument

    new = NewUploadedDocument(
        upload_id="test-upload-001",
        ticker="MSFT",
        filing_type="8-K",
        original_filename="msft-8k-q2.pdf",
        content_sha256="a" * 64,
        parsed_text="Microsoft reported revenue of $XX billion.",
        parsed_char_count=42,
        page_count=14,
    )
    async with session_factory() as session:
        repo = Repository(session)
        stored = await repo.add_uploaded_document(new)
        await session.commit()
        assert stored.upload_id == "test-upload-001"
        assert stored.ticker == "MSFT"

        by_sha = await repo.get_uploaded_document_by_sha256("a" * 64)
        assert by_sha is not None
        assert by_sha.original_filename == "msft-8k-q2.pdf"


@pytest.mark.asyncio
async def test_uploaded_document_sha256_unique(session_factory):
    """Re-uploading the same content (same sha256) fails on insert."""
    from app.memory.repository import Repository
    from app.memory.schemas import NewUploadedDocument
    import sqlalchemy.exc

    base = NewUploadedDocument(
        upload_id="upload-a",
        ticker="MSFT",
        filing_type="8-K",
        original_filename="a.pdf",
        content_sha256="b" * 64,
        parsed_text="hi",
        parsed_char_count=2,
        page_count=1,
    )
    duplicate = base.model_copy(update={"upload_id": "upload-b"})
    async with session_factory() as session:
        repo = Repository(session)
        await repo.add_uploaded_document(base)
        await session.commit()

    async with session_factory() as session:
        repo = Repository(session)
        await repo.add_uploaded_document(duplicate)
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            await session.commit()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/integration/test_repository.py::test_add_and_fetch_uploaded_document -v`
Expected: `FAILED` with `ImportError: cannot import name 'NewUploadedDocument'`.

- [ ] **Step 3: Add the ORM model in `app/memory/models.py`**

Follow the existing model style (consult `FilingORM` and `FilingSectionORM` in the same file). Add:

```python
class UploadedDocumentORM(Base):
    """Append-only record of a user-uploaded filing."""

    __tablename__ = "uploaded_documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    upload_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    filing_type: Mapped[str] = mapped_column(String(16), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    parsed_text: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
```

Import `Text`, `BigInteger`, `Integer`, `func`, and `DateTime` from SQLAlchemy if not already imported in this file.

- [ ] **Step 4: Add DTOs in `app/memory/schemas.py`**

Follow the existing DTO style (e.g. `FilingDTO`). Add:

```python
class NewUploadedDocument(BaseModel):
    """Input shape for inserting a new ``uploaded_documents`` row."""

    model_config = ConfigDict(frozen=True)

    upload_id: str
    ticker: str
    filing_type: str
    original_filename: str
    content_sha256: str
    parsed_text: str
    parsed_char_count: int
    page_count: int | None = None


class UploadedDocumentDTO(BaseModel):
    """Detached view of an ``uploaded_documents`` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    upload_id: str
    ticker: str
    filing_type: str
    original_filename: str
    content_sha256: str
    parsed_text: str
    parsed_char_count: int
    page_count: int | None
    uploaded_at: datetime
```

- [ ] **Step 5: Add repository methods in `app/memory/repository.py`**

Add to the `Repository` class:

```python
    async def add_uploaded_document(
        self, new: NewUploadedDocument
    ) -> UploadedDocumentDTO:
        """Insert a new ``uploaded_documents`` row and return its detached DTO."""
        row = UploadedDocumentORM(
            upload_id=new.upload_id,
            ticker=new.ticker,
            filing_type=new.filing_type,
            original_filename=new.original_filename,
            content_sha256=new.content_sha256,
            parsed_text=new.parsed_text,
            parsed_char_count=new.parsed_char_count,
            page_count=new.page_count,
        )
        self._session.add(row)
        await self._session.flush()
        return UploadedDocumentDTO.model_validate(row)

    async def get_uploaded_document_by_sha256(
        self, content_sha256: str
    ) -> UploadedDocumentDTO | None:
        """Return the document with the given content hash, or ``None``."""
        result = await self._session.execute(
            select(UploadedDocumentORM).where(
                UploadedDocumentORM.content_sha256 == content_sha256
            )
        )
        row = result.scalar_one_or_none()
        return UploadedDocumentDTO.model_validate(row) if row is not None else None

    async def get_uploaded_document(
        self, upload_id: str
    ) -> UploadedDocumentDTO | None:
        """Return the document with the given upload_id, or ``None``."""
        result = await self._session.execute(
            select(UploadedDocumentORM).where(
                UploadedDocumentORM.upload_id == upload_id
            )
        )
        row = result.scalar_one_or_none()
        return UploadedDocumentDTO.model_validate(row) if row is not None else None
```

Import `UploadedDocumentORM` from `app.memory.models` and `NewUploadedDocument`, `UploadedDocumentDTO` from `app.memory.schemas` at the top of the file alongside the existing imports.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_repository.py -q -k uploaded`
Expected: 2 passed.

- [ ] **Step 7: Lint, type-check, commit**

Run: `uv run ruff check app/ tests/` then `uv run mypy app/`. Both clean.

```bash
git add app/memory/ tests/integration/test_repository.py
git commit -m "phase-4a: UploadedDocument ORM + DTOs + repository methods"
```

---

## Task 4: PDF + plain-text intake tool with scanned-PDF rejection

**Files:**
- Create: `app/tools/documents.py`
- Test: `tests/unit/test_documents_parser.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_documents_parser.py`:

```python
"""Unit tests for the document intake tool."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.tools.documents import (
    DocumentParseError,
    ParsedDocument,
    parse_pdf,
    parse_plain_text,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "uploaded_pdfs"


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_parse_pdf_extracts_text_from_real_8k():
    """The small Jan 28 2026 MSFT 8-K extracts a non-empty body."""
    raw = _read("0001193125-26-027198.pdf")
    parsed = parse_pdf(raw)
    assert isinstance(parsed, ParsedDocument)
    assert parsed.char_count > 1000
    assert "Microsoft" in parsed.text
    assert parsed.page_count >= 1
    assert parsed.content_sha256 == hashlib.sha256(raw).hexdigest()


def test_parse_pdf_rejects_zero_extracted_text():
    """A PDF whose text extraction yields nothing is treated as scanned-image."""
    # Construct a minimal valid PDF with no text content.
    empty_pdf = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000055 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n156\n%%EOF"
    )
    with pytest.raises(DocumentParseError, match="scanned|no extractable text"):
        parse_pdf(empty_pdf)


def test_parse_pdf_rejects_wrong_magic_bytes():
    with pytest.raises(DocumentParseError, match="not a PDF"):
        parse_pdf(b"hello, world")


def test_parse_plain_text_decodes_utf8():
    raw = "Microsoft reported revenue of $X.".encode("utf-8")
    parsed = parse_plain_text(raw)
    assert parsed.text == "Microsoft reported revenue of $X."
    assert parsed.char_count == 33
    assert parsed.page_count is None
    assert parsed.content_sha256 == hashlib.sha256(raw).hexdigest()


def test_parse_plain_text_rejects_empty():
    with pytest.raises(DocumentParseError, match="empty"):
        parse_plain_text(b"   \n\t  ")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_documents_parser.py -q`
Expected: `FAILED` with `ModuleNotFoundError: No module named 'app.tools.documents'`.

- [ ] **Step 3: Implement `app/tools/documents.py`**

Create the file:

```python
"""Upload-document parser.

Handles two content types accepted by ``POST /api/upload``:

* ``application/pdf`` -- decoded with ``pypdf``. PDFs whose pages contain no
  embedded text (typical of scanned images) are rejected with a clear error;
  OCR is intentionally out of scope.
* ``text/plain`` -- decoded as UTF-8. Whitespace-only payloads are rejected.

Every successful parse returns a :class:`ParsedDocument` carrying the
extracted text, character count, page count (PDFs only), and the SHA-256
of the raw bytes. The hash dedupes re-uploads of identical content via
:meth:`Repository.get_uploaded_document_by_sha256`.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Final

from pypdf import PdfReader

_PDF_MAGIC: Final[bytes] = b"%PDF-"


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be parsed.

    The message is user-facing -- it goes back through the API to the
    uploader, so it must be specific and actionable (e.g. "this looks like
    a scanned image, paste the text instead" rather than "parse failed").
    """


@dataclass(frozen=True)
class ParsedDocument:
    """Normalised view of an uploaded document."""

    text: str
    char_count: int
    page_count: int | None
    content_sha256: str


def parse_pdf(raw: bytes) -> ParsedDocument:
    """Extract text from a PDF byte string.

    Raises :class:`DocumentParseError` if the bytes do not begin with the
    ``%PDF-`` magic header, or if the extracted text is empty (a strong
    signal the PDF is a scan and would require OCR).
    """
    if not raw.startswith(_PDF_MAGIC):
        raise DocumentParseError(
            "Uploaded file is not a PDF (missing %PDF- magic bytes)."
        )
    reader = PdfReader(io.BytesIO(raw))
    page_count = len(reader.pages)
    body_chunks: list[str] = []
    for page in reader.pages:
        body_chunks.append(page.extract_text() or "")
    text = "\n".join(chunk for chunk in body_chunks if chunk).strip()
    if not text:
        raise DocumentParseError(
            "This PDF has no extractable text -- it looks like a scanned image. "
            "Paste the text directly or supply a text-extractable PDF."
        )
    return ParsedDocument(
        text=text,
        char_count=len(text),
        page_count=page_count,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def parse_plain_text(raw: bytes, *, encoding: str = "utf-8") -> ParsedDocument:
    """Decode a plain-text upload.

    Raises :class:`DocumentParseError` if the decoded body is empty after
    stripping whitespace.
    """
    try:
        decoded = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise DocumentParseError(
            f"Plain-text upload is not valid {encoding}: {exc!s}."
        ) from exc
    text = decoded.strip()
    if not text:
        raise DocumentParseError("Plain-text upload is empty.")
    return ParsedDocument(
        text=text,
        char_count=len(text),
        page_count=None,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )
```

- [ ] **Step 4: Move `pypdf` from dev dep to runtime dep**

`pypdf` was added as a dev dep earlier during PDF inspection. For Phase 4A it becomes a runtime dependency.

Run: `uv remove --dev pypdf && uv add pypdf`
Expected: `pyproject.toml` now lists `pypdf` under `[project] dependencies`, not `[dependency-groups.dev]`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_documents_parser.py -q`
Expected: 5 passed.

- [ ] **Step 6: Lint, type-check, commit**

Run: `uv run ruff check app/ tests/` and `uv run mypy app/`. Both clean.

```bash
git add app/tools/documents.py tests/unit/test_documents_parser.py pyproject.toml uv.lock
git commit -m "phase-4a: PDF + plain-text intake tool with scanned-PDF rejection"
```

---

## Task 5: EDGAR advisor tool

**Files:**
- Create: `app/tools/advisor.py`
- Test: `tests/unit/test_advisor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_advisor.py`:

```python
"""Unit tests for the document advisor tool."""
from __future__ import annotations

from datetime import datetime, UTC

import pytest

from app.tools.advisor import (
    AdvisedFiling,
    AdvisorOutput,
    advise_for_ticker,
)


class _FakeEdgar:
    """Stub EDGAR client returning a canned recent-filings list."""

    def __init__(self) -> None:
        from app.tools.edgar import RecentFiling, SubmissionsResponse

        self._filings = SubmissionsResponse(
            cik="0000789019",
            ticker="MSFT",
            name="Microsoft Corp",
            recent=[
                RecentFiling(
                    accession_number="0001193125-26-191457",
                    filing_date="2026-04-29",
                    primary_document="msft-20260429.htm",
                    form="8-K",
                ),
                RecentFiling(
                    accession_number="0001193125-26-027207",
                    filing_date="2026-01-28",
                    primary_document="msft-20260128.htm",
                    form="10-Q",
                ),
                RecentFiling(
                    accession_number="0001193125-26-027198",
                    filing_date="2026-01-28",
                    primary_document="msft-20260128b.htm",
                    form="8-K",
                ),
                RecentFiling(
                    accession_number="0000950170-25-100235",
                    filing_date="2025-08-15",
                    primary_document="msft-20250630.htm",
                    form="10-K",
                ),
            ],
        )

    async def get_submissions(self, *, cik: str):
        return self._filings


@pytest.mark.asyncio
async def test_advise_returns_latest_per_type():
    edgar = _FakeEdgar()
    output = await advise_for_ticker(
        ticker="MSFT", cik="0000789019", edgar=edgar
    )
    assert isinstance(output, AdvisorOutput)
    forms = [f.filing_type for f in output.suggested]
    assert "8-K" in forms
    assert "10-Q" in forms
    assert "10-K" in forms
    # Latest 8-K must be the Apr 29 one.
    eight_k = next(f for f in output.suggested if f.filing_type == "8-K")
    assert eight_k.accession_number == "0001193125-26-191457"
    # Every suggestion exposes the canonical EDGAR archive URL.
    for filing in output.suggested:
        assert filing.edgar_index_url.startswith(
            "https://www.sec.gov/Archives/edgar/data/789019/"
        )
    # Transcript hint is plain text, not a fetched URL.
    assert "transcript" in output.transcript_hint.lower()


@pytest.mark.asyncio
async def test_advise_orders_8k_before_10q_before_10k():
    """The order of ``suggested`` reflects upload priority for an earnings analysis."""
    edgar = _FakeEdgar()
    output = await advise_for_ticker(
        ticker="MSFT", cik="0000789019", edgar=edgar
    )
    assert [f.filing_type for f in output.suggested] == ["8-K", "10-Q", "10-K"]
```

If the existing `RecentFiling`/`SubmissionsResponse` shapes differ from what's used above, adapt the stub to the real schemas without changing the test intent. Read `app/tools/edgar.py` to confirm field names.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_advisor.py -q`
Expected: `FAILED` with `ModuleNotFoundError: No module named 'app.tools.advisor'`.

- [ ] **Step 3: Implement `app/tools/advisor.py`**

```python
"""Document advisor: given a ticker, return a ranked "what to upload" list.

The advisor consults the existing Phase 1 EDGAR client to enumerate recent
filings, then surfaces the latest 8-K (earnings release), latest 10-Q
(quarterly report), and latest 10-K (annual report) with direct EDGAR
archive URLs. The user clicks the link, downloads the PDF, and uploads it.

Transcripts are not on EDGAR; the advisor returns a hint pointing the user
to common public sources rather than attempting to fetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, Sequence

from app.tools.edgar import SubmissionsResponse


class _SupportsSubmissions(Protocol):
    async def get_submissions(self, *, cik: str) -> SubmissionsResponse: ...


_PRIORITY_FORMS: Final[tuple[str, ...]] = ("8-K", "10-Q", "10-K")

_TRANSCRIPT_HINT: Final[str] = (
    "Earnings-call transcripts are not on EDGAR. Try the company's investor-"
    "relations site (look for 'Earnings' or 'Quarterly Results') or a public "
    "transcript provider such as Motley Fool. Upload as plain text."
)


@dataclass(frozen=True)
class AdvisedFiling:
    """One row of the advisor's checklist."""

    filing_type: str
    accession_number: str
    filed_at: str
    edgar_index_url: str
    primary_document: str


@dataclass(frozen=True)
class AdvisorOutput:
    """Full advisor response for one ticker."""

    ticker: str
    cik: str
    suggested: list[AdvisedFiling]
    transcript_hint: str


def _edgar_index_url(cik: str, accession_number: str) -> str:
    """Build the canonical EDGAR archive index URL."""
    no_dashes = accession_number.replace("-", "")
    cik_stripped = cik.lstrip("0") or "0"
    return f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{no_dashes}/"


def _latest_for_form(
    filings: Sequence, form: str
) -> object | None:
    """Return the most-recently-filed entry of ``form``, or ``None``."""
    matches = [f for f in filings if f.form == form]
    if not matches:
        return None
    return max(matches, key=lambda f: f.filing_date)


async def advise_for_ticker(
    *, ticker: str, cik: str, edgar: _SupportsSubmissions
) -> AdvisorOutput:
    """Build the upload checklist for ``ticker``.

    Queries EDGAR for the company's recent submissions and returns the latest
    filing for each priority form in :data:`_PRIORITY_FORMS`. Forms with no
    recent matches are silently omitted so the caller can render whatever the
    issuer actually has on file.
    """
    submissions = await edgar.get_submissions(cik=cik)
    suggested: list[AdvisedFiling] = []
    for form in _PRIORITY_FORMS:
        latest = _latest_for_form(submissions.recent, form)
        if latest is None:
            continue
        suggested.append(
            AdvisedFiling(
                filing_type=form,
                accession_number=latest.accession_number,
                filed_at=latest.filing_date,
                edgar_index_url=_edgar_index_url(cik, latest.accession_number),
                primary_document=latest.primary_document,
            )
        )
    return AdvisorOutput(
        ticker=ticker.upper(),
        cik=cik,
        suggested=suggested,
        transcript_hint=_TRANSCRIPT_HINT,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_advisor.py -q`
Expected: 2 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add app/tools/advisor.py tests/unit/test_advisor.py
git commit -m "phase-4a: EDGAR advisor tool returning ranked upload checklist"
```

---

## Task 6: AgentState extensions (upload-aware metadata)

The graph still pivots on `FilingEvent`, which is shared between watcher and upload modes. We only need to add a `source` discriminator to `FilingEvent` so downstream nodes can distinguish "this came from a user upload" from "this came from EDGAR autopoll". No new state field; one new enum.

**Files:**
- Modify: `app/models/state.py`
- Test: `tests/unit/test_state.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_state.py`:

```python
def test_filing_event_defaults_to_watcher_source():
    """Existing code that builds FilingEvent without ``source`` keeps working."""
    from app.models.state import FilingEvent, FilingEventSource, FilingForm
    from datetime import UTC, datetime

    event = FilingEvent(
        accession_number="0001193125-26-027198",
        cik="0000789019",
        ticker="MSFT",
        form=FilingForm.FORM_8K,
        filed_at=datetime(2026, 1, 28, tzinfo=UTC),
        source_url="https://example.com",
    )
    assert event.source is FilingEventSource.WATCHER


def test_filing_event_accepts_upload_source():
    from app.models.state import FilingEvent, FilingEventSource, FilingForm
    from datetime import UTC, datetime

    event = FilingEvent(
        accession_number="upload-001",
        cik="0000789019",
        ticker="MSFT",
        form=FilingForm.FORM_8K,
        filed_at=datetime(2026, 1, 28, tzinfo=UTC),
        source_url="https://example.com",
        source=FilingEventSource.UPLOAD,
    )
    assert event.source is FilingEventSource.UPLOAD
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_state.py -q -k filing_event_`
Expected: `FAILED` with `ImportError: cannot import name 'FilingEventSource'`.

- [ ] **Step 3: Add the new enum and field**

In `app/models/state.py`, add this enum next to `FilingForm`:

```python
class FilingEventSource(StrEnum):
    """Where a FilingEvent originated."""

    WATCHER = "watcher"
    UPLOAD = "upload"
```

Add `source` to `FilingEvent` (still frozen) with a default of `WATCHER`:

```python
    source: FilingEventSource = Field(
        default=FilingEventSource.WATCHER,
        description="Whether this event came from the EDGAR watcher or a user upload.",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_state.py -q`
Expected: all tests in the file pass (including pre-existing ones).

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add app/models/state.py tests/unit/test_state.py
git commit -m "phase-4a: add FilingEventSource discriminator (watcher | upload)"
```

---

## Task 7: Document advisor agent node

**Files:**
- Create: `app/agents/document_advisor.py`
- Test: `tests/unit/test_document_advisor_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_document_advisor_agent.py`:

```python
"""Unit tests for the document_advisor agent node."""
from __future__ import annotations

import pytest

from app.agents.document_advisor import advise as advise_node


class _FakeEdgar:
    async def get_submissions(self, *, cik: str):
        from app.tools.edgar import RecentFiling, SubmissionsResponse

        return SubmissionsResponse(
            cik=cik,
            ticker="MSFT",
            name="Microsoft Corp",
            recent=[
                RecentFiling(
                    accession_number="0001193125-26-191457",
                    filing_date="2026-04-29",
                    primary_document="msft.htm",
                    form="8-K",
                )
            ],
        )


class _FakeRepository:
    """Stand-in for Repository: looks up CIK by ticker."""

    async def get_watchlist_entry_by_ticker(self, ticker: str):
        from app.memory.schemas import WatchlistEntryDTO

        if ticker.upper() != "MSFT":
            return None
        return WatchlistEntryDTO(
            id=1,
            ticker="MSFT",
            cik="0000789019",
            company_name="Microsoft Corp",
            created_at=None,
        )


@pytest.mark.asyncio
async def test_advise_for_known_ticker():
    output = await advise_node(
        ticker="MSFT", repository=_FakeRepository(), edgar=_FakeEdgar()
    )
    assert output.ticker == "MSFT"
    assert len(output.suggested) == 1
    assert output.suggested[0].filing_type == "8-K"


@pytest.mark.asyncio
async def test_advise_for_unknown_ticker_raises():
    """Tickers not in the watchlist need to be added before advising."""
    from app.agents.document_advisor import UnknownTickerError

    with pytest.raises(UnknownTickerError):
        await advise_node(
            ticker="ZZZZ", repository=_FakeRepository(), edgar=_FakeEdgar()
        )
```

If the existing `WatchlistEntryDTO` shape uses different field names, adapt the stub. Read `app/memory/schemas.py` to confirm.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_document_advisor_agent.py -q`
Expected: `FAILED` with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `app/agents/document_advisor.py`**

```python
"""Document advisor agent node.

Wraps :func:`app.tools.advisor.advise_for_ticker` with the project's
repository pattern: tickers must already be registered in the watchlist
(so we have a verified CIK to query against EDGAR). Callers that want to
advise on a new ticker should add the watchlist entry first via the same
``poll_once.py --ticker T --cik C --company-name N`` route Phase 1 set up.
"""

from __future__ import annotations

from typing import Protocol

from app.memory.schemas import WatchlistEntryDTO
from app.tools.advisor import AdvisorOutput, advise_for_ticker
from app.tools.edgar import SubmissionsResponse


class _SupportsWatchlist(Protocol):
    async def get_watchlist_entry_by_ticker(
        self, ticker: str
    ) -> WatchlistEntryDTO | None: ...


class _SupportsSubmissions(Protocol):
    async def get_submissions(self, *, cik: str) -> SubmissionsResponse: ...


class UnknownTickerError(ValueError):
    """Raised when the advisor is asked about a ticker not on the watchlist.

    Surfaced verbatim to the API caller -- the API turns this into a 404.
    """


async def advise(
    *,
    ticker: str,
    repository: _SupportsWatchlist,
    edgar: _SupportsSubmissions,
) -> AdvisorOutput:
    """Return the upload checklist for ``ticker``.

    Raises :class:`UnknownTickerError` if the ticker is not in the watchlist
    (the project never queries EDGAR by ticker because CIKs are the canonical
    identifier; the watchlist holds the ticker -> CIK mapping).
    """
    entry = await repository.get_watchlist_entry_by_ticker(ticker.upper())
    if entry is None:
        raise UnknownTickerError(
            f"Ticker {ticker!r} is not on the watchlist. Add it first via "
            "`poll_once.py --ticker T --cik C --company-name N`."
        )
    return await advise_for_ticker(ticker=entry.ticker, cik=entry.cik, edgar=edgar)
```

If `Repository.get_watchlist_entry_by_ticker` does not yet exist, add a thin method in `app/memory/repository.py` and a corresponding unit/integration test before this task is considered complete:

```python
    async def get_watchlist_entry_by_ticker(
        self, ticker: str
    ) -> WatchlistEntryDTO | None:
        result = await self._session.execute(
            select(WatchlistEntryORM).where(WatchlistEntryORM.ticker == ticker.upper())
        )
        row = result.scalar_one_or_none()
        return WatchlistEntryDTO.model_validate(row) if row is not None else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_document_advisor_agent.py -q`
Expected: 2 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add app/agents/document_advisor.py app/memory/repository.py tests/unit/test_document_advisor_agent.py
git commit -m "phase-4a: document_advisor agent node + watchlist lookup helper"
```

---

## Task 8: Upload intake agent node

**Files:**
- Create: `app/agents/upload_intake.py`
- Test: `tests/unit/test_upload_intake.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_upload_intake.py`:

```python
"""Unit tests for the upload_intake agent node."""
from __future__ import annotations

import pytest

from app.agents.upload_intake import intake_upload
from app.tools.documents import ParsedDocument


class _FakeRepository:
    def __init__(self) -> None:
        self.saved: list = []

    async def add_uploaded_document(self, new):
        from app.memory.schemas import UploadedDocumentDTO
        from datetime import UTC, datetime
        dto = UploadedDocumentDTO(
            id=len(self.saved) + 1,
            uploaded_at=datetime.now(UTC),
            **new.model_dump(),
        )
        self.saved.append(dto)
        return dto

    async def get_uploaded_document_by_sha256(self, content_sha256: str):
        for dto in self.saved:
            if dto.content_sha256 == content_sha256:
                return dto
        return None

    async def get_watchlist_entry_by_ticker(self, ticker: str):
        from app.memory.schemas import WatchlistEntryDTO
        return WatchlistEntryDTO(
            id=1,
            ticker="MSFT",
            cik="0000789019",
            company_name="Microsoft Corp",
            created_at=None,
        )


@pytest.mark.asyncio
async def test_intake_creates_filing_event_with_upload_source():
    from app.models.state import FilingEventSource

    repo = _FakeRepository()
    parsed = ParsedDocument(
        text="Microsoft reported revenue of $X billion.",
        char_count=42,
        page_count=14,
        content_sha256="c" * 64,
    )
    event = await intake_upload(
        ticker="MSFT",
        filing_type="8-K",
        original_filename="msft-8k.pdf",
        parsed=parsed,
        repository=repo,
    )
    assert event.source is FilingEventSource.UPLOAD
    assert event.ticker == "MSFT"
    assert event.form.value == "8-K"
    # The upload row is persisted exactly once.
    assert len(repo.saved) == 1
    assert repo.saved[0].content_sha256 == "c" * 64


@pytest.mark.asyncio
async def test_intake_idempotent_on_duplicate_sha256():
    """Re-uploading the same content returns the existing row, no duplicate insert."""
    repo = _FakeRepository()
    parsed = ParsedDocument(
        text="hi", char_count=2, page_count=1, content_sha256="d" * 64
    )
    e1 = await intake_upload(
        ticker="MSFT",
        filing_type="8-K",
        original_filename="x.pdf",
        parsed=parsed,
        repository=repo,
    )
    e2 = await intake_upload(
        ticker="MSFT",
        filing_type="8-K",
        original_filename="x.pdf",
        parsed=parsed,
        repository=repo,
    )
    assert len(repo.saved) == 1
    assert e1.accession_number == e2.accession_number
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_upload_intake.py -q`
Expected: `FAILED` with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `app/agents/upload_intake.py`**

```python
"""Upload intake agent node.

Persists an uploaded document and produces the canonical
:class:`~app.models.state.FilingEvent` that drives the downstream graph.
A duplicate upload (same SHA-256) is returned idempotently: we look up the
existing row and reuse its ``upload_id`` as the synthetic accession number.

The graph downstream is identical to the watcher-driven path; only the
``FilingEvent.source`` discriminator differs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from app.memory.schemas import (
    NewUploadedDocument,
    UploadedDocumentDTO,
    WatchlistEntryDTO,
)
from app.models.state import FilingEvent, FilingEventSource, FilingForm
from app.tools.documents import ParsedDocument


class _SupportsUploadStorage(Protocol):
    async def add_uploaded_document(
        self, new: NewUploadedDocument
    ) -> UploadedDocumentDTO: ...

    async def get_uploaded_document_by_sha256(
        self, content_sha256: str
    ) -> UploadedDocumentDTO | None: ...

    async def get_watchlist_entry_by_ticker(
        self, ticker: str
    ) -> WatchlistEntryDTO | None: ...


def _filing_form(filing_type: str) -> FilingForm:
    """Map the user-supplied filing-type string to the enum the graph uses."""
    try:
        return FilingForm(filing_type)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported filing_type {filing_type!r}; expected one of "
            f"{[m.value for m in FilingForm]}."
        ) from exc


async def intake_upload(
    *,
    ticker: str,
    filing_type: str,
    original_filename: str,
    parsed: ParsedDocument,
    repository: _SupportsUploadStorage,
) -> FilingEvent:
    """Persist (or recover) the uploaded document and return its FilingEvent."""
    ticker_upper = ticker.upper()
    entry = await repository.get_watchlist_entry_by_ticker(ticker_upper)
    if entry is None:
        raise ValueError(
            f"Ticker {ticker_upper!r} not on watchlist; add it before uploading."
        )

    existing = await repository.get_uploaded_document_by_sha256(parsed.content_sha256)
    if existing is not None:
        upload_id = existing.upload_id
    else:
        upload_id = uuid4().hex
        await repository.add_uploaded_document(
            NewUploadedDocument(
                upload_id=upload_id,
                ticker=ticker_upper,
                filing_type=_filing_form(filing_type).value,
                original_filename=original_filename,
                content_sha256=parsed.content_sha256,
                parsed_text=parsed.text,
                parsed_char_count=parsed.char_count,
                page_count=parsed.page_count,
            )
        )

    return FilingEvent(
        accession_number=f"upload-{upload_id}",
        cik=entry.cik,
        ticker=ticker_upper,
        form=_filing_form(filing_type),
        filed_at=datetime.now(UTC),
        source_url=f"upload://{upload_id}",
        source=FilingEventSource.UPLOAD,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_upload_intake.py -q`
Expected: 2 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add app/agents/upload_intake.py tests/unit/test_upload_intake.py
git commit -m "phase-4a: upload_intake agent node with idempotent SHA-based dedupe"
```

---

## Task 9: Watcher gating behind `watcher_mode_enabled`

**Files:**
- Modify: `app/agents/watcher.py`
- Modify: `app/scripts/poll_once.py`
- Test: `tests/unit/test_poll_once_cli.py` (extend) and an inline test in `tests/unit/test_watcher_gating.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_watcher_gating.py`:

```python
"""Watcher gating behind ``watcher_mode_enabled``."""
from __future__ import annotations

import pytest

from app.agents.watcher import WatcherDisabledError, ensure_watcher_enabled


def test_ensure_watcher_enabled_when_off():
    with pytest.raises(WatcherDisabledError):
        ensure_watcher_enabled(enabled=False)


def test_ensure_watcher_enabled_when_on():
    # No exception when enabled.
    ensure_watcher_enabled(enabled=True)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_watcher_gating.py -q`
Expected: `FAILED` with `ImportError`.

- [ ] **Step 3: Add the gating helper and wire it into `watch_forever`**

In `app/agents/watcher.py`, add near the top:

```python
class WatcherDisabledError(RuntimeError):
    """Raised when ``watch_forever`` runs while ``watcher_mode_enabled`` is False.

    The upload-first product runs with the watcher off by default; enabling it
    is an explicit operator choice (eval/demo mode). Refusing to start prevents
    accidental EDGAR polling in production.
    """


def ensure_watcher_enabled(*, enabled: bool) -> None:
    """Raise :class:`WatcherDisabledError` if ``enabled`` is False."""
    if not enabled:
        raise WatcherDisabledError(
            "Watcher mode is disabled. Set WATCHER_MODE_ENABLED=true to run the "
            "EDGAR watcher (eval/demo mode)."
        )
```

Then in the existing `watch_forever(...)` function body, as the first line after the docstring:

```python
    from app.config import get_settings
    ensure_watcher_enabled(enabled=get_settings().watcher_mode_enabled)
```

`poll_once` (one-shot operator command) should not be blocked but **should log a warning** when the flag is off. In `app/scripts/poll_once.py`, immediately after the existing settings load:

```python
    if not settings.watcher_mode_enabled:
        logger.warning(
            "watcher_mode_disabled_but_poll_once_invoked: "
            "you are running an ad-hoc EDGAR poll while WATCHER_MODE_ENABLED=false. "
            "The continuous watcher will not start."
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_watcher_gating.py tests/unit/test_poll_once_cli.py -q`
Expected: all green.

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add app/agents/watcher.py app/scripts/poll_once.py tests/unit/test_watcher_gating.py
git commit -m "phase-4a: gate the EDGAR watcher behind watcher_mode_enabled"
```

---

## Task 10: Health endpoint update for watcher gating

**Files:**
- Modify: `app/api/health.py`
- Test: `tests/integration/test_health.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_health.py` (use the existing async test client fixture):

```python
@pytest.mark.asyncio
async def test_health_reports_not_applicable_when_watcher_disabled(
    async_client, monkeypatch
):
    """With watcher mode off, the watcher freshness check is informational only."""
    monkeypatch.setenv("WATCHER_MODE_ENABLED", "false")
    from app.config import reset_settings_cache
    reset_settings_cache()

    response = await async_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["checks"]["edgar_watcher"] == "not_applicable"
    # Overall status is not degraded by the watcher when it's intentionally off.
    assert payload["status"] in {"ok", "degraded"}
    if payload["status"] == "degraded":
        # Any 'degraded' must be due to Redis or DB, NOT the watcher.
        assert payload["checks"]["edgar_watcher"] != "stale"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/integration/test_health.py::test_health_reports_not_applicable_when_watcher_disabled -v`
Expected: `FAILED` (`"not_applicable"` is not currently a valid value).

- [ ] **Step 3: Update `app/api/health.py`**

Extend the `CheckStatus` literal:

```python
CheckStatus = Literal["ok", "stale", "unknown", "error", "not_applicable"]
```

Update `_check_watcher` to short-circuit when the flag is off:

```python
async def _check_watcher(details: HealthDetails) -> CheckStatus:
    from app.config import get_settings
    if not get_settings().watcher_mode_enabled:
        return "not_applicable"
    try:
        engine = get_engine()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            last = await Repository(session).last_successful_poll_at()
        if last is None:
            return "unknown"
        age = datetime.now(UTC) - last.polled_at
        details.last_poll_age_seconds = age.total_seconds()
        return "ok" if age <= _POLL_FRESHNESS_THRESHOLD else "stale"
    except Exception as exc:
        details.database_error = (details.database_error or "") + f"; watcher: {exc!s}"
        _logger.bind(error=str(exc)).warning("health_watcher_check_failed")
        return "error"
```

Adjust the overall-status logic so `not_applicable` is treated as healthy:

```python
    elif (
        redis_status != "ok"
        or watcher_status in {"stale", "unknown", "error"}
    ):
        overall = _STATUS_DEGRADED
```

(`not_applicable` is intentionally absent from that set so it does not flag degraded.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/integration/test_health.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add app/api/health.py tests/integration/test_health.py
git commit -m "phase-4a: /health reports edgar_watcher as not_applicable when disabled"
```

---

## Task 11: Graph entry-point widening

The graph's `START -> financial_extractor` edge already works for both `FilingEvent.source = WATCHER` and `FilingEvent.source = UPLOAD` because the downstream nodes are agnostic to source. What the graph **does** need is a tiny no-op `upload_intake` graph node that simply forwards the state when the event is upload-sourced (so the topology mirrors the design spec and tracing/logging can identify the upload path).

For Phase 4A we keep the graph as-is. The `upload_intake` function from Task 8 is invoked by `POST /api/upload` **before** the graph is started — building the `FilingEvent` happens at the API boundary, not as a graph node. This matches how `watcher.watch_forever` calls into the graph: the event is constructed first, then the graph runs.

**No code changes for Task 11 beyond documenting this decision.** Add a paragraph to the module docstring of `app/graph.py`:

- [ ] **Step 1: Update `app/graph.py` module docstring**

Insert this paragraph after the existing "LangGraph fans out..." paragraph:

```
Entry points. The graph is started identically by both the EDGAR watcher and
the ``POST /api/upload`` route: each constructs a :class:`FilingEvent` and
calls ``compiled_graph.ainvoke(initial_state)``. The ``FilingEvent.source``
discriminator distinguishes the two for tracing and logging; downstream
nodes do not branch on it.
```

- [ ] **Step 2: Commit**

```bash
git add app/graph.py
git commit -m "phase-4a: document upload vs watcher graph entry-point parity"
```

---

## Task 12: `POST /api/advise` route

**Files:**
- Create: `app/api/advise.py`
- Modify: `app/api/__init__.py` (or the module that wires `app/main.py`'s routers)
- Test: `tests/integration/test_upload_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_upload_api.py`:

```python
"""End-to-end tests for /api/advise, /api/upload, /api/chat."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "uploaded_pdfs"


@pytest.mark.asyncio
async def test_advise_for_msft_returns_checklist(async_client, seed_watchlist_msft):
    response = await async_client.post("/api/advise", json={"ticker": "MSFT"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert any(s["filing_type"] == "8-K" for s in payload["suggested"])
    assert "transcript" in payload["transcript_hint"].lower()


@pytest.mark.asyncio
async def test_advise_unknown_ticker_404(async_client):
    response = await async_client.post("/api/advise", json={"ticker": "ZZZZZZ"})
    assert response.status_code == 404
    assert "watchlist" in response.json()["detail"].lower()
```

`seed_watchlist_msft` is a new fixture you'll add to `tests/conftest.py` (or the integration conftest) that inserts MSFT into the watchlist and returns a stub EDGAR client returning canned MSFT submissions. The EDGAR stub should follow the same pattern used in `tests/unit/test_advisor.py`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/integration/test_upload_api.py::test_advise_for_msft_returns_checklist -v`
Expected: 404 (route not registered).

- [ ] **Step 3: Implement `app/api/advise.py`**

```python
"""``POST /api/advise``: given a ticker, return the upload checklist."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.document_advisor import UnknownTickerError, advise
from app.api.dependencies import get_edgar_client, get_session  # see Step 4
from app.memory.repository import Repository
from app.tools.advisor import AdvisedFiling, AdvisorOutput
from app.tools.edgar import EdgarClient

router = APIRouter(prefix="/api", tags=["advise"])


class AdviseRequest(BaseModel):
    """Request body for ``POST /api/advise``."""

    ticker: str


class AdviseFilingResponse(BaseModel):
    filing_type: str
    accession_number: str
    filed_at: str
    edgar_index_url: str
    primary_document: str


class AdviseResponse(BaseModel):
    ticker: str
    cik: str
    suggested: list[AdviseFilingResponse]
    transcript_hint: str


def _to_response(output: AdvisorOutput) -> AdviseResponse:
    return AdviseResponse(
        ticker=output.ticker,
        cik=output.cik,
        suggested=[
            AdviseFilingResponse(
                filing_type=f.filing_type,
                accession_number=f.accession_number,
                filed_at=f.filed_at,
                edgar_index_url=f.edgar_index_url,
                primary_document=f.primary_document,
            )
            for f in output.suggested
        ],
        transcript_hint=output.transcript_hint,
    )


@router.post("/advise", response_model=AdviseResponse)
async def post_advise(
    body: AdviseRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    edgar: Annotated[EdgarClient, Depends(get_edgar_client)],
) -> AdviseResponse:
    try:
        output = await advise(
            ticker=body.ticker, repository=Repository(session), edgar=edgar
        )
    except UnknownTickerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(output)
```

- [ ] **Step 4: Add `app/api/dependencies.py`**

If `app/api/dependencies.py` does not exist yet, create it. The exact bindings depend on how the project currently composes the FastAPI app (look at `app/api/health.py` and the entry point that mounts `health.router`). The dependencies needed here:

```python
"""FastAPI dependency factories used by the upload-first routes."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.memory.db import get_session_factory
from app.tools.edgar import EdgarClient


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an ``AsyncSession`` bound to the request lifecycle."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_edgar_client() -> EdgarClient:
    """Return a singleton EDGAR client configured from settings."""
    settings = get_settings()
    return EdgarClient(user_agent=settings.edgar_user_agent)
```

If existing code already provides equivalent helpers, reuse them rather than duplicating.

- [ ] **Step 5: Wire the router**

In whatever module currently does `app.include_router(health.router)`, add an `app.include_router(advise.router)` line alongside it.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_upload_api.py -q -k advise`
Expected: 2 passed.

- [ ] **Step 7: Lint, type-check, commit**

```bash
git add app/api/ tests/integration/test_upload_api.py
git commit -m "phase-4a: POST /api/advise route + EDGAR dependency wiring"
```

---

## Task 13: `POST /api/upload` route + end-to-end smoke test

**Files:**
- Create: `app/api/upload.py`
- Modify: `app/api/__init__.py` (or the router-wiring module)
- Test: `tests/integration/test_upload_api.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_upload_api.py`:

```python
@pytest.mark.asyncio
async def test_upload_msft_8k_runs_pipeline_to_final_note(
    async_client, seed_watchlist_msft, stub_llm_cassettes, stub_consensus, stub_embeddings
):
    pdf_bytes = (FIXTURES / "0001193125-26-027198.pdf").read_bytes()
    response = await async_client.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "8-K"},
        files={"file": ("msft-8k.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["upload_id"].startswith("") is False  # Some non-empty id
    assert payload["analysis"]["final_note"] is not None


@pytest.mark.asyncio
async def test_upload_rejects_too_large(async_client, seed_watchlist_msft, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    from app.config import reset_settings_cache
    reset_settings_cache()

    big = b"%PDF-1.4\n" + b"A" * 2048
    response = await async_client.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "8-K"},
        files={"file": ("big.pdf", big, "application/pdf")},
    )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_upload_rejects_wrong_content_type(async_client, seed_watchlist_msft):
    response = await async_client.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "8-K"},
        files={"file": ("evil.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_upload_rejects_scanned_pdf(async_client, seed_watchlist_msft):
    empty_pdf = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000055 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n156\n%%EOF"
    )
    response = await async_client.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "8-K"},
        files={"file": ("scan.pdf", empty_pdf, "application/pdf")},
    )
    assert response.status_code == 422
    assert "scanned" in response.json()["detail"].lower() or "extractable" in response.json()["detail"].lower()
```

The test-only fixtures (`stub_llm_cassettes`, `stub_consensus`, `stub_embeddings`) already exist for the Phase 2/3 graph tests — reuse them. Read `tests/integration/test_graph.py` to confirm fixture names; rename if needed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_upload_api.py -q -k upload`
Expected: all four FAIL with 404 (route not registered).

- [ ] **Step 3: Implement `app/api/upload.py`**

```python
"""``POST /api/upload``: ingest a user-supplied PDF or plain-text filing,
persist it, build a :class:`FilingEvent`, run the graph end-to-end, and
return the resulting structured analysis.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.upload_intake import intake_upload
from app.api.dependencies import (
    get_compiled_graph,
    get_edgar_client,
    get_session,
    get_session_factory_dep,
)
from app.config import get_settings
from app.memory.repository import Repository
from app.models.state import AgentState
from app.tools.documents import (
    DocumentParseError,
    ParsedDocument,
    parse_pdf,
    parse_plain_text,
)

router = APIRouter(prefix="/api", tags=["upload"])

_ACCEPTED_PDF_TYPES: Final[set[str]] = {"application/pdf"}
_ACCEPTED_TEXT_TYPES: Final[set[str]] = {"text/plain"}


class UploadResponse(BaseModel):
    """Response shape for ``POST /api/upload``."""

    upload_id: str
    trace_id: str
    status: Literal["completed", "failed"]
    analysis: "AnalysisPayload"


class AnalysisPayload(BaseModel):
    """The structured analysis distilled from the upload."""

    financials: dict | None
    comparisons: dict | None
    language_diffs: list[dict]
    draft_note: str | None
    final_note: str | None
    critic_verdict: str | None


def _parse_or_400(file: UploadFile, raw: bytes) -> ParsedDocument:
    try:
        if file.content_type in _ACCEPTED_PDF_TYPES:
            return parse_pdf(raw)
        if file.content_type in _ACCEPTED_TEXT_TYPES:
            return parse_plain_text(raw)
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type {file.content_type!r}. Accepted: "
                "application/pdf, text/plain."
            ),
        )
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/upload", response_model=UploadResponse)
async def post_upload(
    ticker: Annotated[str, Form()],
    filing_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    session: Annotated[AsyncSession, Depends(get_session)],
    session_factory: Annotated[
        async_sessionmaker[AsyncSession], Depends(get_session_factory_dep)
    ],
    graph=Depends(get_compiled_graph),
) -> UploadResponse:
    settings = get_settings()
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes} bytes)."
            ),
        )
    parsed = _parse_or_400(file, raw)
    repository = Repository(session)
    filing_event = await intake_upload(
        ticker=ticker,
        filing_type=filing_type,
        original_filename=file.filename or "upload.bin",
        parsed=parsed,
        repository=repository,
    )
    trace_id = uuid4().hex

    from datetime import UTC, datetime

    initial_state = AgentState(
        trace_id=trace_id,
        started_at=datetime.now(UTC),
        filing_event=filing_event,
    )
    final_state_dict = await graph.ainvoke(initial_state)
    final_state = AgentState.model_validate(final_state_dict)

    return UploadResponse(
        upload_id=filing_event.accession_number.replace("upload-", ""),
        trace_id=trace_id,
        status="completed",
        analysis=AnalysisPayload(
            financials=final_state.financials,
            comparisons=final_state.comparisons,
            language_diffs=final_state.language_diffs,
            draft_note=final_state.draft_note,
            final_note=final_state.final_note,
            critic_verdict=(
                final_state.critic_verdict.value
                if final_state.critic_verdict is not None
                else None
            ),
        ),
    )
```

- [ ] **Step 4: Add `get_compiled_graph` and `get_session_factory_dep` to dependencies**

In `app/api/dependencies.py` add:

```python
from app.graph import build_graph
from app.llm.client import LLMClient
from app.memory.db import get_session_factory
from app.tools.consensus import ConsensusFetcher  # or wherever the project's fetcher lives
from app.tools.embeddings import EmbeddingsClient

_GRAPH = None  # populated on first call


def get_session_factory_dep():
    return get_session_factory()


def get_compiled_graph():
    """Return the compiled graph singleton, built lazily on first request."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(
            edgar=EdgarClient(user_agent=get_settings().edgar_user_agent),
            consensus_fetcher=ConsensusFetcher(),
            embeddings=EmbeddingsClient(),
            llm=LLMClient(),
            session_factory=get_session_factory(),
        )
    return _GRAPH
```

If the project's clients have different constructors, follow whatever the existing test integration uses (read `tests/integration/test_graph.py` to confirm).

- [ ] **Step 5: Wire the router**

In the router-wiring module add `app.include_router(upload.router)`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_upload_api.py -q`
Expected: all upload-related tests pass.

- [ ] **Step 7: Lint, type-check, commit**

```bash
git add app/api/ tests/integration/test_upload_api.py
git commit -m "phase-4a: POST /api/upload runs the pipeline end-to-end on PDFs"
```

---

## Task 14: `POST /api/chat` minimal stub

For 4A this is intentionally a 501 stub so the route shape is locked in for Phase 6. No LLM call, no Anthropic SDK touch — just a clear "not yet" response.

**Files:**
- Create: `app/api/chat.py`
- Modify: router-wiring module
- Test: `tests/integration/test_upload_api.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_upload_api.py`:

```python
@pytest.mark.asyncio
async def test_chat_returns_501_stub(async_client):
    response = await async_client.post(
        "/api/chat", json={"trace_id": "x", "message": "What was revenue?"}
    )
    assert response.status_code == 501
    body = response.json()
    assert "phase 6" in body["detail"].lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/integration/test_upload_api.py::test_chat_returns_501_stub -v`
Expected: 404.

- [ ] **Step 3: Implement `app/api/chat.py`**

```python
"""``POST /api/chat``: stubbed in Phase 4A; the citation-enforced chat agent
lands in Phase 6. We register the route now so the upload UI can be wired
against a stable URL.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    trace_id: str
    message: str


@router.post("/chat")
async def post_chat(_body: ChatRequest) -> None:
    raise HTTPException(
        status_code=501,
        detail=(
            "/api/chat is reserved for the Phase 6 citation-enforced chat agent. "
            "Phase 4A ships only the route shape."
        ),
    )
```

- [ ] **Step 4: Wire the router**

Add `app.include_router(chat.router)`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/integration/test_upload_api.py::test_chat_returns_501_stub -v`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, commit**

```bash
git add app/api/chat.py tests/integration/test_upload_api.py
git commit -m "phase-4a: POST /api/chat 501 stub (Phase 6 reserves the route)"
```

---

## Task 15: Full test sweep + coverage check

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: all green.

- [ ] **Step 2: Run the full integration suite**

Run: `uv run pytest tests/integration -q`
Expected: all green.

- [ ] **Step 3: Coverage gate**

Run: `uv run pytest --cov=app --cov-report=term-missing tests/ -q`
Expected: `TOTAL` line coverage ≥ 85%. Any new module added in Phase 4A must have ≥ 80% line coverage individually.

If any uncovered branch is non-trivial, add a focused test before considering Task 15 done. Trivial uncovered lines (defensive `raise` after exhaustive `match`) are acceptable.

- [ ] **Step 4: Lint + type-check sweep**

Run: `uv run ruff check app/ tests/` then `uv run mypy app/`. Both must be clean.

- [ ] **Step 5: pip-audit**

Run: `uv run pip-audit`
Expected: no known vulnerabilities.

- [ ] **Step 6: Update the Phase 4 status block in `CLAUDE.md`**

Mark Phase 4A complete:

```markdown
**Phase 4A — Upload infrastructure: complete** (commit `<short SHA>`, <YYYY-MM-DD>).
```

Append a Phase-4A "Added in" subsection mirroring the style of Phase 1/2/3:

```markdown
Added in Phase 4A:
- **Document parser** at [`app/tools/documents.py`](app/tools/documents.py) with PDF + plain-text intake, scanned-PDF rejection, and SHA-256 content hashing.
- **EDGAR advisor** at [`app/tools/advisor.py`](app/tools/advisor.py) + agent wrapper at [`app/agents/document_advisor.py`](app/agents/document_advisor.py).
- **Upload intake node** at [`app/agents/upload_intake.py`](app/agents/upload_intake.py) — idempotent on SHA-256, emits FilingEvent.UPLOAD.
- **API routes**: `POST /api/advise`, `POST /api/upload`, `POST /api/chat` (stub).
- **Watcher gated** behind `WATCHER_MODE_ENABLED`; `/health` reports `not_applicable` when off.
- **Migration** `0004_phase4a_uploaded_documents`.
- **Sample fixtures**: small MSFT 8-Ks at [`tests/fixtures/uploaded_pdfs/`](tests/fixtures/uploaded_pdfs/).
```

- [ ] **Step 7: Commit the status update**

```bash
git add CLAUDE.md
git commit -m "phase-4a: status block + Added-in summary"
```

---

## Acceptance criteria recap

Phase 4A is done when:

1. `uv run pytest tests/unit -q` → all green
2. `uv run pytest tests/integration -q` → all green
3. `uv run ruff check app/ tests/` and `uv run mypy app/` → clean
4. `uv run pytest --cov=app tests/` → line coverage ≥ 85%
5. `POST /api/advise` with `{"ticker": "MSFT"}` returns a ranked checklist
6. `POST /api/upload` with one of the small 8-K fixtures returns a structured analysis with a non-null `final_note`
7. `POST /api/chat` returns 501 with a Phase 6 hint
8. `GET /health` reports `edgar_watcher: not_applicable` when `WATCHER_MODE_ENABLED=false`
9. CLAUDE.md status block reflects Phase 4A completion
10. Plan 4B can begin (transcript analyzer + Q&A pair extraction + commitment extractor)
