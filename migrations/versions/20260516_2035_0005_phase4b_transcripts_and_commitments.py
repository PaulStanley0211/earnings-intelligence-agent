"""Phase 4B: qa_pairs and commitments tables for the transcript analyzer.

Append-only except for ``commitments.status`` and its companion ``resolved_*``
and ``updated_at`` columns. The ``(ticker, status)`` index on ``commitments``
supports cross-quarter reconciliation; the ``UNIQUE (filing_accession, ordinal)``
on ``qa_pairs`` makes the analyzer's bulk insert idempotent.

Revision ID: 0005_phase4b_transcripts_and_commitments
Revises: 0004_phase4a_uploaded_documents
Create Date: 2026-05-16 20:35:00+00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_phase4b_transcripts_and_commitments"
down_revision: str | None = "0004_phase4a_uploaded_documents"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Create the ``qa_pairs`` and ``commitments`` tables.

    Also widens ``alembic_version.version_num`` from the Alembic default
    ``VARCHAR(32)`` to ``VARCHAR(64)`` because this migration's revision id
    is 42 characters. Future descriptive revision ids will fit without a
    second ALTER.
    """
    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )

    op.create_table(
        "qa_pairs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "filing_accession",
            sa.String(length=32),
            sa.ForeignKey("filings.accession_number", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("analyst_name", sa.Text, nullable=True),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column("answer_text", sa.Text, nullable=False),
        sa.Column("answer_class", sa.String(length=16), nullable=False),
        sa.Column("sha256_text", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "filing_accession",
            "ordinal",
            name="uq_qa_pairs_filing_accession_ordinal",
        ),
    )
    op.create_index(
        "ix_qa_pairs_filing_accession",
        "qa_pairs",
        ["filing_accession"],
    )

    op.create_table(
        "commitments",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "filing_accession",
            sa.String(length=32),
            sa.ForeignKey("filings.accession_number", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("commitment_text", sa.Text, nullable=False),
        sa.Column("target_period", sa.Text, nullable=True),
        sa.Column("source_quote", sa.Text, nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default="open",
            nullable=False,
        ),
        sa.Column(
            "resolved_filing_accession",
            sa.String(length=32),
            sa.ForeignKey("filings.accession_number", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("resolved_reason", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_commitments_ticker_status",
        "commitments",
        ["ticker", "status"],
    )


def downgrade() -> None:
    """Drop the indexes then the tables (reverse of upgrade).

    Restores ``alembic_version.version_num`` to its original ``VARCHAR(32)``
    width last, after the version row has already been rewound by Alembic
    to the prior (shorter) revision id.
    """
    op.drop_index("ix_commitments_ticker_status", table_name="commitments")
    op.drop_table("commitments")
    op.drop_index("ix_qa_pairs_filing_accession", table_name="qa_pairs")
    op.drop_table("qa_pairs")
    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
