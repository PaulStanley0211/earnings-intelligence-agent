"""Phase 5a: notes table.

Stores the final markdown research note produced by the synthesizer/critic
loop for each filing. One note per filing (enforced by the unique constraint
on ``filing_accession``). The ``prompt_template_sha`` records the content
hash of the prompt used so notes can be invalidated when the prompt changes.

Revision ID: 0008_phase5a_notes
Revises: 0007_widen_filings_accession_number
Create Date: 2026-05-17 11:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0008_phase5a_notes"
down_revision: str | None = "0007_widen_filings_accession_number"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Create the ``notes`` table and its ticker+created_at index."""
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
    """Drop the ``notes`` table and its index."""
    op.drop_index("ix_notes_ticker_created", table_name="notes")
    op.drop_table("notes")
