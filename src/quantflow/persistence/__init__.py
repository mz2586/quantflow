"""Persistence layer: ORM models, repositories and transactional scope."""

from __future__ import annotations

from quantflow.persistence import models as models
from quantflow.persistence.base import Base
from quantflow.persistence.database import Database, UnitOfWork, build_engine, build_session_factory

# `models` is imported for its side effect: importing it registers every table on
# `Base.metadata`. Without it, anything that only imports `Base` — Alembic autogenerate, a
# test fixture calling `create_all` — sees empty metadata and silently produces no schema.
__all__ = [
    "Base",
    "Database",
    "UnitOfWork",
    "build_engine",
    "build_session_factory",
    "models",
]
