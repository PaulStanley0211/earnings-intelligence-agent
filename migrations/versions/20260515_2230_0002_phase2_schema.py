"""Phase 2 schema.

Adds the two tables that back the consensus fetcher and the comparator:

- ``consensus_estimates``: analyst consensus value per
  ``(ticker, fiscal_year, fiscal_period, metric, source)``.
- ``comparisons``: one row per filing/metric capturing reported value,
  consensus value (nullable), surprise, and direction.

Hand-written so the constraints and indexes are visible in one place for
review, matching the Phase 1 migration style.

Revision ID: 0002_phase2_schema
Revises: 0001_phase1_schema
Create Date: 2026-05-15 22:30:00+00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_phase2_schema"
down_revision: str | None = "0001_phase1_schema"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Create the Phase 2 consensus and comparison tables."""
    op.create_table(
        "consensus_estimates",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=False),
            primary_key=True,
        ),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("fiscal_period", sa.String(length=8), nullable=False),
        sa.Column("metric", sa.String(length=32), nullable=False),
        sa.Column("value", sa.Numeric(28, 6), nullable=False),
        sa.Column("analyst_count", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "ticker",
            "fiscal_year",
            "fiscal_period",
            "metric",
            "source",
            name="uq_consensus_estimates_ticker_period_metric_source",
        ),
        sa.CheckConstraint(
            "source IN ('finnhub', 'yfinance')",
            name="consensus_estimates_source_valid",
        ),
    )
    op.create_index(
        "ix_consensus_estimates_ticker_period",
        "consensus_estimates",
        ["ticker", "fiscal_year", "fiscal_period"],
    )

    op.create_table(
        "comparisons",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=False),
            primary_key=True,
        ),
        sa.Column(
            "filing_accession",
            sa.String(length=32),
            sa.ForeignKey("filings.accession_number", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metric", sa.String(length=32), nullable=False),
        sa.Column("reported_value", sa.Numeric(28, 6), nullable=False),
        sa.Column("reported_unit", sa.String(length=32), nullable=False),
        sa.Column("consensus_value", sa.Numeric(28, 6), nullable=True),
        sa.Column("consensus_source", sa.String(length=16), nullable=True),
        sa.Column("surprise_abs", sa.Numeric(28, 6), nullable=True),
        sa.Column("surprise_pct", sa.Numeric(10, 4), nullable=True),
        sa.Column("direction", sa.String(length=8), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "filing_accession",
            "metric",
            name="uq_comparisons_filing_metric",
        ),
        sa.CheckConstraint(
            "direction IS NULL OR direction IN ('beat', 'miss', 'in_line')",
            name="comparisons_direction_valid",
        ),
    )
    op.create_index(
        "ix_comparisons_filing_accession",
        "comparisons",
        ["filing_accession"],
    )


def downgrade() -> None:
    """Drop the Phase 2 tables in reverse dependency order."""
    op.drop_index("ix_comparisons_filing_accession", table_name="comparisons")
    op.drop_table("comparisons")
    op.drop_index(
        "ix_consensus_estimates_ticker_period", table_name="consensus_estimates"
    )
    op.drop_table("consensus_estimates")
