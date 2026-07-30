"""Fixtures for tests that need real Postgres and Redis.

The suite creates and drops a dedicated ``quantflow_test`` database so a run can never
touch development data. Every test gets a clean schema.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from quantflow.cache.redis import Cache, build_redis
from quantflow.core.config import DatabaseSettings, RedisSettings
from quantflow.persistence.base import Base
from quantflow.persistence.database import Database

pytestmark = pytest.mark.integration

TEST_DATABASE_NAME = "quantflow_test"


def _database_settings(name: str) -> DatabaseSettings:
    return DatabaseSettings(
        host=os.getenv("QF_TEST_DB_HOST", "localhost"),
        port=int(os.getenv("QF_TEST_DB_PORT", "55432")),
        user=os.getenv("QF_TEST_DB_USER", "quantflow"),
        password=os.getenv("QF_TEST_DB_PASSWORD", "quantflow"),  # type: ignore[arg-type]
        name=name,
    )


def _redis_settings() -> RedisSettings:
    return RedisSettings(
        host=os.getenv("QF_TEST_REDIS_HOST", "localhost"),
        port=int(os.getenv("QF_TEST_REDIS_PORT", "56379")),
        # A dedicated logical database so a stray FLUSH cannot disturb dev data.
        db=int(os.getenv("QF_TEST_REDIS_DB", "15")),
        key_prefix="qftest",
    )


@pytest.fixture(scope="session")
def database_settings() -> DatabaseSettings:
    """Connection settings for the throwaway test database."""
    return _database_settings(TEST_DATABASE_NAME)


@pytest.fixture(scope="session")
def _provision_database(database_settings: DatabaseSettings) -> Iterator[None]:
    """Create the test database once per session, and drop it afterwards.

    Deliberately synchronous: CREATE/DROP DATABASE is one-off DDL, and a session-scoped
    *async* fixture would have to share an event loop with function-scoped tests.
    """
    del database_settings  # the admin connection targets the maintenance database
    admin_dsn = _database_settings("postgres").sync_dsn

    def _drop_and_create(create: bool) -> None:
        engine = sa.create_engine(admin_dsn, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as connection:
                connection.execute(
                    sa.text(f'DROP DATABASE IF EXISTS "{TEST_DATABASE_NAME}" WITH (FORCE)')
                )
                if create:
                    connection.execute(sa.text(f'CREATE DATABASE "{TEST_DATABASE_NAME}"'))
        finally:
            engine.dispose()

    try:
        _drop_and_create(create=True)
    except Exception as exc:
        pytest.skip(f"Postgres is unavailable for integration tests: {exc}")

    yield

    _drop_and_create(create=False)


@pytest.fixture
async def database(
    database_settings: DatabaseSettings,
    _provision_database: None,
) -> AsyncIterator[Database]:
    """A `Database` bound to a freshly created schema, dropped after the test."""
    db = Database(database_settings, use_null_pool=True)
    async with db.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield db
    finally:
        async with db.engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await db.aclose()


@pytest.fixture
async def session(database: Database) -> AsyncIterator[AsyncSession]:
    """A session that rolls back at the end of the test."""
    async with database.session_factory() as db_session:
        yield db_session
        await db_session.rollback()


@pytest.fixture
async def cache() -> AsyncIterator[Cache]:
    """A `Cache` on a dedicated Redis logical database, flushed before and after."""
    settings = _redis_settings()
    client = build_redis(settings)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"Redis is unavailable for integration tests: {exc}")

    instance = Cache(client, prefix=settings.key_prefix)
    await instance.flush_namespace()
    try:
        yield instance
    finally:
        await instance.flush_namespace()
        await instance.aclose()
