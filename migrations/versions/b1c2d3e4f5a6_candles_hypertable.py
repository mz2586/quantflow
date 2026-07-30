"""Convert candles to a TimescaleDB hypertable when the extension is available.

Candles are the highest-cardinality table in the system: one row per symbol, per
timeframe, per bar. A hypertable partitions it by ``open_time`` so range scans touch only
the relevant chunks, and lets old chunks be compressed.

The migration degrades gracefully: on a plain Postgres (CI, a developer's local instance,
a managed database without the extension) it becomes a no-op and the table stays a normal
table, which is functionally identical, just slower at scale.

Revision ID: b1c2d3e4f5a6
Revises: e5f8fef9a71b
Create Date: 2026-07-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b1c2d3e4f5a6"
down_revision: str | None = "e5f8fef9a71b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CHUNK_INTERVAL = "7 days"


def _timescale_available(connection: sa.Connection) -> bool:
    """Whether the timescaledb extension can be created in this database."""
    return bool(
        connection.execute(
            sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
        ).scalar()
    )


def upgrade() -> None:
    """Apply the migration."""
    connection = op.get_bind()
    if connection.dialect.name != "postgresql" or not _timescale_available(connection):
        return

    connection.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
    connection.execute(
        sa.text(
            """
            SELECT create_hypertable(
                'candles',
                'open_time',
                chunk_time_interval => CAST(:interval AS INTERVAL),
                migrate_data => TRUE,
                if_not_exists => TRUE
            )
            """
        ).bindparams(interval=CHUNK_INTERVAL)
    )

    # Compress chunks older than 30 days, segmented by series so the compressed form
    # still supports efficient per-symbol range queries.
    connection.execute(
        sa.text(
            """
            ALTER TABLE candles SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'symbol, timeframe',
                timescaledb.compress_orderby = 'open_time DESC'
            )
            """
        )
    )
    connection.execute(
        sa.text("SELECT add_compression_policy('candles', INTERVAL '30 days', if_not_exists => TRUE)")
    )


def downgrade() -> None:
    """Revert the migration.

    A hypertable cannot be converted back to a plain table in place. Removing the policies
    and compression settings restores normal (uncompressed) behaviour, which is what a
    downgrade needs; the partitioning itself is left alone rather than rewriting the table
    and risking data loss.
    """
    connection = op.get_bind()
    if connection.dialect.name != "postgresql":
        return
    is_hypertable = connection.execute(
        sa.text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'hypertable' AND table_schema = '_timescaledb_catalog'"
        )
    ).scalar()
    if not is_hypertable:
        return
    connection.execute(
        sa.text("SELECT remove_compression_policy('candles', if_exists => TRUE)")
    )
    connection.execute(sa.text("ALTER TABLE candles SET (timescaledb.compress = FALSE)"))
