"""Phase 4B: widen ``filings.accession_number`` (and child FKs) to VARCHAR(64).

Phase 1 sized ``filings.accession_number`` at ``VARCHAR(32)`` because real
SEC accession numbers are 18 characters (e.g. ``0001193125-23-123456``).
Phase 4A introduced :func:`app.agents.upload_intake.intake_upload` which
generates the upload-derived accession ``f"upload-{uuid4().hex}"`` -- a
39-character string that overflows the 32-char column the first time the
intake calls ``Repository.record_filing``. That call is wired up by
Task 11c of the Phase 4B transcript analyzer plan, so this migration
widens the column ahead of the e2e flow.

The widening must also cover every child column with a foreign key onto
``filings.accession_number`` so the FK type matches the parent. Postgres
allows VARCHAR widening without rewriting the table (cheap operation).

The user picked ``VARCHAR(64)`` for headroom: it accommodates the current
39-char upload pattern, ``upload-{uuid4().hex}-{suffix}`` variations, and
any future composite key (e.g. ``upload-{uuid4().hex}-{ordinal}``) without
forcing a third migration.

Revision ID: 0007_widen_filings_accession_number
Revises: 0006_phase4b_relax_filings_form_check
Create Date: 2026-05-16 21:29:00+00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_widen_filings_accession_number"
down_revision: str | None = "0006_phase4b_relax_filings_form_check"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Widen ``filings.accession_number`` and every child FK to VARCHAR(64)."""
    op.alter_column(
        "filings",
        "accession_number",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "financial_facts",
        "filing_accession",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "comparisons",
        "filing_accession",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "filing_sections",
        "filing_accession",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "language_diffs",
        "filing_accession",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "language_diffs",
        "prior_filing_accession",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=True,
    )
    op.alter_column(
        "qa_pairs",
        "filing_accession",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "commitments",
        "filing_accession",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "commitments",
        "resolved_filing_accession",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Shrink every widened column back to VARCHAR(32).

    Child FKs are shrunk first so the parent ``filings.accession_number``
    can be reduced last. ``downgrade`` will fail if any row holds an
    accession longer than 32 characters -- that is the intended behavior,
    because upload-derived accessions (e.g. ``upload-{uuid4().hex}``)
    cannot round-trip through the narrower type. Operators rolling back
    Phase 4B must first purge any upload-source filings (and their child
    rows) before invoking ``alembic downgrade``.
    """
    op.alter_column(
        "commitments",
        "resolved_filing_accession",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=True,
    )
    op.alter_column(
        "commitments",
        "filing_accession",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
    op.alter_column(
        "qa_pairs",
        "filing_accession",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
    op.alter_column(
        "language_diffs",
        "prior_filing_accession",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=True,
    )
    op.alter_column(
        "language_diffs",
        "filing_accession",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
    op.alter_column(
        "filing_sections",
        "filing_accession",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
    op.alter_column(
        "comparisons",
        "filing_accession",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
    op.alter_column(
        "financial_facts",
        "filing_accession",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
    op.alter_column(
        "filings",
        "accession_number",
        existing_type=sa.String(length=64),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
