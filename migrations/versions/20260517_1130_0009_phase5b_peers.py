"""Phase 5b: peers table.

Stores curated peer-ticker relationships used by the peer-comparison critic
node.  The composite primary key ``(ticker, peer_ticker)`` enforces uniqueness,
and the ``peers_no_self_reference`` check constraint prevents a ticker from
being its own peer.

Revision ID: 0009_phase5b_peers
Revises: 0008_phase5a_notes
Create Date: 2026-05-17 11:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0009_phase5b_peers"
down_revision: str | None = "0008_phase5a_notes"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Create the ``peers`` table and its ticker lookup index."""
    op.create_table(
        "peers",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("peer_ticker", sa.String(16), nullable=False),
        sa.Column(
            "source", sa.String(32), nullable=False, server_default="curated"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("ticker", "peer_ticker", name="pk_peers"),
        sa.CheckConstraint(
            "ticker <> peer_ticker", name="peers_no_self_reference"
        ),
        sa.CheckConstraint("source IN ('curated')", name="peers_source_valid"),
    )
    op.create_index("ix_peers_ticker", "peers", ["ticker"])


def downgrade() -> None:
    """Drop the ``peers`` table and its index."""
    op.drop_index("ix_peers_ticker", table_name="peers")
    op.drop_table("peers")
