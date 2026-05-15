"""Phase 3 schema.

Enables pgvector and adds the two tables backing the language differ:

- ``filing_sections``: one row per parsed paragraph of MD&A / Risk Factors
  for a filing, plus its 1536-dim embedding from
  ``text-embedding-3-small``.
- ``language_diffs``: one row per material change (added / removed /
  modified). Unchanged paragraphs are not persisted.

Also adds ``filings.primary_document`` so the differ does not need to
re-call the submissions API to resolve the HTML filename.

Hand-written and reviewable in one file, matching the Phase 1/2 style.

Revision ID: 0003_phase3_schema
Revises: 0002_phase2_schema
Create Date: 2026-05-15 23:30:00+00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_phase3_schema"
down_revision: str | None = "0002_phase2_schema"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Enable pgvector, extend filings, create filing_sections and language_diffs."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "filings",
        sa.Column("primary_document", sa.Text(), nullable=True),
    )

    op.create_table(
        "filing_sections",
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
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("section_kind", sa.String(length=16), nullable=False),
        sa.Column("paragraph_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_sha", sa.CHAR(length=64), nullable=False),
        sa.Column(
            "embedding",
            sa.dialects.postgresql.ARRAY(sa.Float()),
            nullable=True,
        ),
        sa.Column("embedding_model", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "filing_accession",
            "section_kind",
            "paragraph_index",
            name="uq_filing_sections_filing_section_paragraph",
        ),
        sa.CheckConstraint(
            "section_kind IN ('mda', 'risk_factors')",
            name="filing_sections_section_kind_valid",
        ),
    )
    # Replace the array placeholder with a real pgvector column. Alembic's
    # native sa.dialects.postgresql does not include the vector type, so we
    # alter via raw SQL after the table exists. The table has no rows at this
    # point, so USING NULL discards nothing; it merely satisfies PostgreSQL's
    # requirement for an explicit conversion expression. The column stays
    # NULL-able so a degraded run can persist text without vectors.
    op.execute(
        "ALTER TABLE filing_sections "
        "ALTER COLUMN embedding TYPE vector(1536) USING NULL"
    )
    op.create_index(
        "ix_filing_sections_ticker_section_filing",
        "filing_sections",
        ["ticker", "section_kind", "filing_accession"],
    )
    op.create_index(
        "ix_filing_sections_cik_section",
        "filing_sections",
        ["cik", "section_kind"],
    )

    op.create_table(
        "language_diffs",
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
        sa.Column(
            "prior_filing_accession",
            sa.String(length=32),
            sa.ForeignKey("filings.accession_number", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("section_kind", sa.String(length=16), nullable=False),
        sa.Column("change_type", sa.String(length=16), nullable=False),
        sa.Column(
            "current_section_id",
            sa.BigInteger(),
            sa.ForeignKey("filing_sections.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "prior_section_id",
            sa.BigInteger(),
            sa.ForeignKey("filing_sections.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("similarity", sa.Numeric(6, 4), nullable=True),
        sa.Column("severity", sa.String(length=8), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "filing_accession",
            "section_kind",
            "change_type",
            "current_section_id",
            "prior_section_id",
            name="uq_language_diffs_filing_section_change_pair",
        ),
        sa.CheckConstraint(
            "change_type IN ('added', 'removed', 'modified')",
            name="language_diffs_change_type_valid",
        ),
        sa.CheckConstraint(
            "severity IN ('major', 'minor')",
            name="language_diffs_severity_valid",
        ),
        sa.CheckConstraint(
            "section_kind IN ('mda', 'risk_factors')",
            name="language_diffs_section_kind_valid",
        ),
    )
    op.create_index(
        "ix_language_diffs_filing_section",
        "language_diffs",
        ["filing_accession", "section_kind"],
    )


def downgrade() -> None:
    """Drop the Phase 3 tables and column. The vector extension is preserved."""
    op.drop_index("ix_language_diffs_filing_section", table_name="language_diffs")
    op.drop_table("language_diffs")
    op.drop_index("ix_filing_sections_cik_section", table_name="filing_sections")
    op.drop_index(
        "ix_filing_sections_ticker_section_filing", table_name="filing_sections"
    )
    op.drop_table("filing_sections")
    op.drop_column("filings", "primary_document")
