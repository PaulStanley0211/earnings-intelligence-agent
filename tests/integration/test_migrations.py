"""Integration tests for the Alembic migrations.

Runs ``alembic upgrade head`` against the test database and verifies that
each table defined by :class:`app.memory.models.Base.metadata` is present
with the expected columns. Catching schema drift here keeps the migrations
and the ORM in lockstep without relying on a human to re-run autogenerate.

Phase 2 adds ``consensus_estimates`` and ``comparisons``; both are picked
up by the ``expected <= actual`` assertion since they are part of the
declarative metadata now.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from app.config import get_settings
from app.memory.db import _async_url
from app.memory.models import Base

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sync_url() -> str:
    """Return the sync psycopg DSN derived from the configured async one."""
    return _async_url(get_settings().database_url).replace(
        "postgresql+psycopg://", "postgresql+psycopg://"
    )


@pytest.fixture()
def clean_database() -> Iterator[None]:
    """Drop the public schema before and after each test for a clean slate."""
    engine = create_engine(_sync_url(), future=True)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    yield
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()


def _alembic_config() -> Config:
    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", _sync_url())
    return cfg


def test_alembic_upgrade_creates_all_tables(clean_database: None) -> None:
    command.upgrade(_alembic_config(), "head")

    engine = create_engine(_sync_url(), future=True)
    inspector = inspect(engine)
    expected = {table.name for table in Base.metadata.sorted_tables}
    actual = set(inspector.get_table_names()) - {"alembic_version"}
    assert expected <= actual, f"missing tables: {expected - actual}"

    columns = {col["name"] for col in inspector.get_columns("filings")}
    assert {
        "accession_number",
        "cik",
        "ticker",
        "form",
        "filed_at",
        "source_url",
        "status",
        "processed_at",
    } <= columns

    # Phase 2 columns land on the new tables.
    consensus_cols = {col["name"] for col in inspector.get_columns("consensus_estimates")}
    assert {"ticker", "fiscal_year", "fiscal_period", "metric", "value", "source"} <= consensus_cols
    comparison_cols = {col["name"] for col in inspector.get_columns("comparisons")}
    assert {
        "filing_accession",
        "metric",
        "reported_value",
        "consensus_value",
        "surprise_pct",
        "direction",
    } <= comparison_cols
    engine.dispose()


def test_alembic_downgrade_removes_all_tables(clean_database: None) -> None:
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    engine = create_engine(_sync_url(), future=True)
    inspector = inspect(engine)
    remaining = set(inspector.get_table_names()) - {"alembic_version"}
    assert remaining == set(), f"downgrade left tables behind: {remaining}"
    engine.dispose()


def test_phase3_migration_creates_pgvector_and_tables(clean_database: None) -> None:
    """0003_phase3_schema enables pgvector and adds filing_sections + language_diffs."""
    command.upgrade(_alembic_config(), "head")

    engine = create_engine(_sync_url(), future=True)
    with engine.connect() as conn:
        ext = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        assert ext.scalar_one_or_none() == "vector"

        cols = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'filings' AND column_name = 'primary_document'"
            )
        )
        assert cols.scalar_one_or_none() == "primary_document"

        tables = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name IN ('filing_sections', 'language_diffs') "
                "ORDER BY table_name"
            )
        )
        assert [row[0] for row in tables.all()] == ["filing_sections", "language_diffs"]
    engine.dispose()


def test_migration_0008_creates_notes_table(clean_database: None) -> None:
    """0008_phase5a_notes adds the notes table with the expected columns and index."""
    command.upgrade(_alembic_config(), "head")

    engine = create_engine(_sync_url(), future=True)
    inspector = inspect(engine)

    cols = [c["name"] for c in inspector.get_columns("notes")]
    idx = [i["name"] for i in inspector.get_indexes("notes")]

    assert "id" in cols
    assert "filing_accession" in cols
    assert "markdown_body" in cols
    assert "prompt_template_sha" in cols
    assert "ix_notes_ticker_created" in idx

    engine.dispose()
