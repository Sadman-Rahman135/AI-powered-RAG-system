"""Alembic environment: the URL and metadata come from the application, so migrations always match the models."""

from __future__ import annotations

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.config import get_settings
from app.db import models  # noqa: F401 - register models on the metadata
from app.db.base import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))
target_metadata = Base.metadata

# Created by app.db.pg_extras (not mapped on the models): autogenerate must not try to drop them.
PG_ONLY_OBJECTS = frozenset(
    {
        "search_tsv",
        "ix_chunks_search_tsv",
        "ix_chunks_embedding_hnsw",
        "ix_invoices_supplier_name_trgm",
        "ix_invoices_customer_name_trgm",
    }
)


def include_object(_obj: object, name: str | None, _type: str, _reflected: bool, _compare_to: object) -> bool:
    return name not in PG_ONLY_OBJECTS


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
