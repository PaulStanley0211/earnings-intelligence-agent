"""Phase 4B: relax the ``filings.form`` column to include TRANSCRIPT.

Phase 1 created ``filings`` with ``CHECK (form IN ('10-K', '10-Q', '8-K'))``
and the column itself was ``VARCHAR(8)``. Phase 4B introduces
``FilingForm.TRANSCRIPT`` so user-uploaded earnings-call transcripts can
flow through the same pipeline as SEC filings. Two schema changes are
required or every transcript-driven ``record_filing`` would fail before
``transcript_analyzer`` ever runs:

1. Widen ``form`` from ``VARCHAR(8)`` to ``VARCHAR(16)`` because the literal
   ``"TRANSCRIPT"`` is ten characters and would right-truncate.
2. Replace the ``filings_form_supported`` CHECK constraint with one that
   accepts ``TRANSCRIPT`` alongside the three SEC forms.

The CHECK constraint is dropped and recreated under the same name so the
ORM-side declaration in :mod:`app.memory.models` continues to match the
database schema exactly.

Revision ID: 0006_phase4b_relax_filings_form_check
Revises: 0005_phase4b_transcripts_and_commitments
Create Date: 2026-05-16 19:54:00+00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_phase4b_relax_filings_form_check"
down_revision: str | None = "0005_phase4b_transcripts_and_commitments"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Widen ``filings.form`` and relax its CHECK to include TRANSCRIPT."""
    op.alter_column(
        "filings",
        "form",
        existing_type=sa.String(length=8),
        type_=sa.String(length=16),
        existing_nullable=False,
    )
    op.drop_constraint("filings_form_supported", "filings", type_="check")
    op.create_check_constraint(
        "filings_form_supported",
        "filings",
        "form IN ('10-K', '10-Q', '8-K', 'TRANSCRIPT')",
    )


def downgrade() -> None:
    """Restore the original VARCHAR(8) column and the pre-4B CHECK constraint.

    ``downgrade`` assumes no ``TRANSCRIPT`` rows are present -- if any
    exist the column shrink will fail, which is the right behavior
    because Phase 4B introduced the value and rolling back without
    deleting those rows would silently lose data.
    """
    op.drop_constraint("filings_form_supported", "filings", type_="check")
    op.create_check_constraint(
        "filings_form_supported",
        "filings",
        "form IN ('10-K', '10-Q', '8-K')",
    )
    op.alter_column(
        "filings",
        "form",
        existing_type=sa.String(length=16),
        type_=sa.String(length=8),
        existing_nullable=False,
    )
