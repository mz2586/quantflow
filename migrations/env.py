"""Alembic environment.

The database URL is read from :class:`quantflow.core.config.Settings`, never from
``alembic.ini`` — one source of truth means a migration cannot be applied to a different
database than the application talks to.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from quantflow.core.config import get_settings
from quantflow.persistence.base import Base

# Import every model module so `Base.metadata` is fully populated before autogenerate.
import quantflow.persistence.models  # noqa: F401  # isort: skip

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database.sync_dsn)

target_metadata = Base.metadata


def include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    """Exclude TimescaleDB's internal chunk tables from autogenerate.

    Without this, a hypertable's chunks look like unmanaged tables and autogenerate emits
    spurious drop statements for them.
    """
    del obj, reflected, compare_to
    if type_ == "table" and name is not None:
        return not name.startswith(("_hyper_", "_timescaledb", "_compressed_hypertable"))
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=settings.database.sync_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
            # Wrap DDL in a transaction so a failed migration leaves no partial schema.
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
