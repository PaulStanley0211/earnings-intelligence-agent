"""Alembic environment.

Wires the ``DATABASE_URL`` from application settings into Alembic so
``uv run alembic upgrade head`` works without duplicating connection
configuration in ``alembic.ini``. Concrete migrations land in Phase 1 onward.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

from app.config import get_settings

# Alembic Config object, access to the values within the .ini file in use.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the live DATABASE_URL so we never check a credential into alembic.ini.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

# Phase 1: bind to the project's declarative metadata so ``alembic revision
# --autogenerate`` can diff against the ORM. The import sits below the
# config wiring so missing env vars surface a Settings error before alembic
# tries to introspect anything.
from app.memory.models import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a live connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
