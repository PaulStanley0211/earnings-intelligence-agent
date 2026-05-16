"""Phase 4A: uploaded_documents table for user-supplied filings.

Append-only. One row per accepted upload. SHA-256 is unique to deduplicate
re-uploads of the same content.

Revision ID: 0004_phase4a_uploaded_documents
Revises: 0003_phase3_schema
Create Date: 2026-05-16 12:53:43+00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_phase4a_uploaded_documents"
down_revision: str | None = "0003_phase3_schema"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Create the append-only ``uploaded_documents`` table."""
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
    """Drop the indexes then the table (reverse of upgrade)."""
    op.drop_index("ix_uploaded_documents_ticker", table_name="uploaded_documents")
    op.drop_index("ix_uploaded_documents_content_sha256", table_name="uploaded_documents")
    op.drop_table("uploaded_documents")
