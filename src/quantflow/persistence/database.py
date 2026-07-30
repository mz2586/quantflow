"""Async engine, session factory and unit of work."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Self

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import text

from quantflow.core.config import DatabaseSettings, Settings
from quantflow.core.errors import DatabaseError
from quantflow.core.logging import get_logger

logger = get_logger(__name__)


def build_engine(settings: DatabaseSettings, *, use_null_pool: bool = False) -> AsyncEngine:
    """Create the async engine.

    Args:
        settings: Database configuration.
        use_null_pool: Disable pooling. Required for test fixtures that create and drop
            the schema between cases, and for short-lived CLI processes where a pool
            would just add teardown latency.

    """
    connect_args: dict[str, Any] = {
        "server_settings": {
            # Cap runaway queries at the server so a pathological analytics query cannot
            # hold a connection open indefinitely.
            "statement_timeout": str(settings.statement_timeout_ms),
            "application_name": "quantflow",
        }
    }

    kwargs: dict[str, Any] = {
        "echo": settings.echo,
        "future": True,
        "pool_pre_ping": True,
        "connect_args": connect_args,
    }
    if use_null_pool:
        kwargs["poolclass"] = NullPool
    else:
        kwargs.update(
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout_seconds,
            pool_recycle=settings.pool_recycle_seconds,
        )

    logger.debug("database.engine_created", dsn=settings.safe_dsn, pooled=not use_null_pool)
    return create_async_engine(settings.async_dsn, **kwargs)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create the session factory.

    ``expire_on_commit=False`` so that ORM objects stay usable after the transaction
    closes — otherwise every attribute read after a commit triggers a lazy refresh
    against a closed session and raises.
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )


class Database:
    """Owns the engine and session factory for the process lifetime."""

    __slots__ = ("_engine", "_session_factory", "_settings")

    def __init__(self, settings: DatabaseSettings, *, use_null_pool: bool = False) -> None:
        self._settings = settings
        self._engine = build_engine(settings, use_null_pool=use_null_pool)
        self._session_factory = build_session_factory(self._engine)

    @classmethod
    def from_settings(cls, settings: Settings) -> Database:
        """Build from the root settings object."""
        return cls(settings.database)

    @property
    def engine(self) -> AsyncEngine:
        """The underlying async engine."""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """The configured session factory."""
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session that commits on success and rolls back on failure."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except SQLAlchemyError as exc:
                await session.rollback()
                raise DatabaseError(f"database operation failed: {exc}") from exc
            except Exception:
                await session.rollback()
                raise

    @asynccontextmanager
    async def read_session(self) -> AsyncIterator[AsyncSession]:
        """Yield a read-only session that never commits."""
        async with self._session_factory() as session:
            try:
                yield session
            finally:
                await session.rollback()

    def unit_of_work(self) -> UnitOfWork:
        """Create a unit of work bound to this database."""
        return UnitOfWork(self._session_factory)

    async def ping(self) -> bool:
        """Check connectivity. Returns ``False`` rather than raising, for health checks."""
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            logger.warning("database.ping_failed", error=str(exc))
            return False
        return True

    async def has_extension(self, name: str) -> bool:
        """Whether a Postgres extension (e.g. ``timescaledb``) is installed."""
        try:
            async with self._engine.connect() as connection:
                result = await connection.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = :name"), {"name": name}
                )
                return result.scalar() is not None
        except SQLAlchemyError:
            return False

    async def aclose(self) -> None:
        """Dispose of the connection pool."""
        await self._engine.dispose()
        logger.debug("database.engine_disposed")


class UnitOfWork:
    """Transactional scope around a set of repository operations.

    The whole block commits or rolls back together, so a partially-written aggregate
    (an order without its fills, a fill without its position update) is impossible::

        async with database.unit_of_work() as uow:
            await uow.orders.save(order)
            await uow.positions.save(position)
        # committed here, or rolled back if the block raised
    """

    __slots__ = ("_committed", "_session", "_session_factory")

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._committed = False

    @property
    def session(self) -> AsyncSession:
        """The active session.

        Raises:
            DatabaseError: if accessed outside an ``async with`` block.

        """
        if self._session is None:
            raise DatabaseError("unit of work is not active; use 'async with'")
        return self._session

    async def __aenter__(self) -> Self:
        """Open the session and begin the transaction."""
        self._session = self._session_factory()
        self._committed = False
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Commit on clean exit, roll back on exception, then close."""
        session = self._session
        if session is None:  # pragma: no cover — defensive
            return
        try:
            if exc_type is None and not self._committed:
                await session.commit()
            elif exc_type is not None:
                await session.rollback()
        except SQLAlchemyError as commit_error:
            await session.rollback()
            raise DatabaseError(f"transaction failed: {commit_error}") from commit_error
        finally:
            await session.close()
            self._session = None

    async def commit(self) -> None:
        """Commit early, inside the block."""
        await self.session.commit()
        self._committed = True

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        await self.session.rollback()

    async def flush(self) -> None:
        """Flush pending changes without committing (to obtain generated ids)."""
        await self.session.flush()

    # -- Repository accessors ------------------------------------------- #
    # Imported lazily so `persistence.database` stays importable without pulling in
    # every repository (and therefore every model) at module import time.

    @property
    def candles(self) -> Any:
        """Candle repository."""
        from quantflow.persistence.repositories import CandleRepository

        return CandleRepository(self.session)

    @property
    def instruments(self) -> Any:
        """Instrument repository."""
        from quantflow.persistence.repositories import InstrumentRepository

        return InstrumentRepository(self.session)

    @property
    def orders(self) -> Any:
        """Order repository."""
        from quantflow.persistence.repositories import OrderRepository

        return OrderRepository(self.session)

    @property
    def positions(self) -> Any:
        """Position repository."""
        from quantflow.persistence.repositories import PositionRepository

        return PositionRepository(self.session)

    @property
    def trades(self) -> Any:
        """Closed-trade repository."""
        from quantflow.persistence.repositories import ClosedTradeRepository

        return ClosedTradeRepository(self.session)

    @property
    def sessions(self) -> Any:
        """Trading-session repository."""
        from quantflow.persistence.repositories import TradingSessionRepository

        return TradingSessionRepository(self.session)

    @property
    def equity(self) -> Any:
        """Equity-snapshot repository."""
        from quantflow.persistence.repositories import EquityRepository

        return EquityRepository(self.session)

    @property
    def risk_events(self) -> Any:
        """Risk-event repository."""
        from quantflow.persistence.repositories import RiskEventRepository

        return RiskEventRepository(self.session)

    @property
    def backtests(self) -> Any:
        """Backtest-run repository."""
        from quantflow.persistence.repositories import BacktestRepository

        return BacktestRepository(self.session)
