"""Integration test for the Phase 1 Alembic migration.

Runs ``alembic upgrade head`` against the test database and verifies that
each table defined by :class:`app.memory.models.Base.metadata` is present
with the expected columns. Catching schema drift here keeps the migration
and the ORM in lockstep without relying on a human to re-run autogenerate.
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


def test_alembic_upgrade_creates_phase1_tables(clean_database: None) -> None:
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
    engine.dispose()


def test_alembic_downgrade_removes_all_phase1_tables(clean_database: None) -> None:
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    engine = create_engine(_sync_url(), future=True)
    inspector = inspect(engine)
    remaining = set(inspector.get_table_names()) - {"alembic_version"}
    assert remaining == set(), f"downgrade left tables behind: {remaining}"
    engine.dispose()
