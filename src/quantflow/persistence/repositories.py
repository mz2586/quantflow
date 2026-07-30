"""Repositories: the only place that translates between ORM records and domain objects.

Keeping the mapping here (rather than on the models) means the domain layer never imports
SQLAlchemy, and a schema change shows up as a compile error in exactly one file.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from itertools import pairwise
from typing import Any, cast

from sqlalchemy import CursorResult, Select, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from quantflow.core.clock import utc_now
from quantflow.core.errors import NotFoundError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import (
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RunStatus,
    Timeframe,
    TimeInForce,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, DataIntegrityReport
from quantflow.domain.orders import Fill, Order
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import ClosedTrade, Lot, Position
from quantflow.persistence.models import (
    BacktestRunRecord,
    CandleRecord,
    ClosedTradeRecord,
    EquitySnapshotRecord,
    FillRecord,
    InstrumentRecord,
    KillSwitchRecord,
    OrderRecord,
    PositionRecord,
    RiskEventRecord,
    TradingSessionRecord,
)

logger = get_logger(__name__)

#: Batch size for bulk candle upserts. asyncpg caps a single statement at 32767 bind
#: parameters (a tighter limit than Postgres's own 65535). A candle binds 10 parameters,
#: so 3000 rows (30k parameters) stays inside the limit with headroom.
UPSERT_CHUNK_SIZE = 3_000


class Repository[TRecord]:
    """Base repository holding the session."""

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def session(self) -> AsyncSession:
        """The session this repository writes through."""
        return self._session

    async def _scalars(self, statement: Select[tuple[TRecord]]) -> Sequence[TRecord]:
        result = await self._session.execute(statement)
        return result.scalars().all()


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
class CandleRepository(Repository[CandleRecord]):
    """Persistence for OHLCV bars."""

    async def upsert_many(self, candles: Iterable[Candle]) -> int:
        """Insert or update candles by their natural key.

        Idempotent: re-downloading an overlapping range updates rows in place rather than
        duplicating them, so a resumed backfill is always safe.

        Returns:
            The number of rows written.

        """
        rows = [
            {
                "symbol": candle.symbol.slashed,
                "timeframe": candle.timeframe,
                "open_time": candle.open_time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "quote_volume": candle.quote_volume,
                "trades": candle.trades,
            }
            for candle in candles
        ]
        if not rows:
            return 0

        written = 0
        for start in range(0, len(rows), UPSERT_CHUNK_SIZE):
            chunk = rows[start : start + UPSERT_CHUNK_SIZE]
            statement = pg_insert(CandleRecord).values(chunk)
            statement = statement.on_conflict_do_update(
                index_elements=["symbol", "timeframe", "open_time"],
                set_={
                    "open": statement.excluded.open,
                    "high": statement.excluded.high,
                    "low": statement.excluded.low,
                    "close": statement.excluded.close,
                    "volume": statement.excluded.volume,
                    "quote_volume": statement.excluded.quote_volume,
                    "trades": statement.excluded.trades,
                },
            )
            await self._session.execute(statement)
            written += len(chunk)
        return written

    async def fetch(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[Candle]:
        """Load candles in ``[start, end)``, ordered by open time."""
        statement = select(CandleRecord).where(
            CandleRecord.symbol == symbol.slashed,
            CandleRecord.timeframe == timeframe,
        )
        if start is not None:
            statement = statement.where(CandleRecord.open_time >= start)
        if end is not None:
            statement = statement.where(CandleRecord.open_time < end)
        statement = statement.order_by(
            CandleRecord.open_time.desc() if newest_first else CandleRecord.open_time.asc()
        )
        if limit is not None:
            statement = statement.limit(limit)

        records = await self._scalars(statement)
        candles = [_to_candle(record) for record in records]
        if newest_first:
            candles.reverse()
        return candles

    async def latest_open_time(self, symbol: Symbol, timeframe: Timeframe) -> datetime | None:
        """Open time of the most recent stored bar, if any.

        Used by the downloader to resume a backfill without re-fetching known data.
        """
        statement = select(func.max(CandleRecord.open_time)).where(
            CandleRecord.symbol == symbol.slashed,
            CandleRecord.timeframe == timeframe,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def earliest_open_time(self, symbol: Symbol, timeframe: Timeframe) -> datetime | None:
        """Open time of the oldest stored bar, if any."""
        statement = select(func.min(CandleRecord.open_time)).where(
            CandleRecord.symbol == symbol.slashed,
            CandleRecord.timeframe == timeframe,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def count(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Number of stored bars in the range."""
        statement = (
            select(func.count())
            .select_from(CandleRecord)
            .where(
                CandleRecord.symbol == symbol.slashed,
                CandleRecord.timeframe == timeframe,
            )
        )
        if start is not None:
            statement = statement.where(CandleRecord.open_time >= start)
        if end is not None:
            statement = statement.where(CandleRecord.open_time < end)
        result = await self._session.execute(statement)
        return int(result.scalar_one())

    async def integrity_report(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> DataIntegrityReport:
        """Audit a stored range for gaps before it is used for a backtest.

        Backtesting across an undetected gap produces a plausible-looking equity curve that
        is simply wrong, so this runs before any dataset is trusted.
        """
        candles = await self.fetch(symbol, timeframe, start=start, end=end)
        if not candles:
            return DataIntegrityReport(
                symbol=symbol, timeframe=timeframe, candle_count=0, start=None, end=None
            )

        step = timeframe.delta
        gaps: list[tuple[datetime, datetime]] = []
        duplicates: list[datetime] = []
        anomalies: list[str] = []

        for previous, current in pairwise(candles):
            expected = previous.open_time + step
            if current.open_time == previous.open_time:
                duplicates.append(current.open_time)
            elif current.open_time > expected:
                gaps.append((expected, current.open_time))

        for candle in candles:
            if candle.volume == ZERO and candle.range == ZERO:
                anomalies.append(f"flat zero-volume bar at {candle.open_time.isoformat()}")

        return DataIntegrityReport(
            symbol=symbol,
            timeframe=timeframe,
            candle_count=len(candles),
            start=candles[0].open_time,
            end=candles[-1].open_time,
            gaps=tuple(gaps),
            duplicate_open_times=tuple(duplicates),
            anomalies=tuple(anomalies[:50]),
        )

    async def delete_range(
        self, symbol: Symbol, timeframe: Timeframe, *, start: datetime, end: datetime
    ) -> int:
        """Delete candles in ``[start, end)``. Returns the number of rows removed."""
        statement = delete(CandleRecord).where(
            CandleRecord.symbol == symbol.slashed,
            CandleRecord.timeframe == timeframe,
            CandleRecord.open_time >= start,
            CandleRecord.open_time < end,
        )
        result = cast(CursorResult[Any], await self._session.execute(statement))
        return int(result.rowcount or 0)

    async def available_series(self) -> list[tuple[Symbol, Timeframe, int]]:
        """Every stored ``(symbol, timeframe)`` pair with its bar count."""
        statement = (
            select(CandleRecord.symbol, CandleRecord.timeframe, func.count().label("bars"))
            .group_by(CandleRecord.symbol, CandleRecord.timeframe)
            .order_by(CandleRecord.symbol, CandleRecord.timeframe)
        )
        result = await self._session.execute(statement)
        return [(Symbol.parse(row.symbol), row.timeframe, int(row.bars)) for row in result.all()]


class InstrumentRepository(Repository[InstrumentRecord]):
    """Persistence for cached exchange trading rules."""

    async def upsert(self, instrument: Instrument, *, exchange: str = "binance") -> None:
        """Insert or refresh an instrument's rules."""
        values = {
            "exchange": exchange,
            "symbol": instrument.symbol.slashed,
            "market_type": instrument.market_type.value,
            "base_asset": instrument.symbol.base,
            "quote_asset": instrument.symbol.quote,
            "price_tick": instrument.price_tick,
            "quantity_step": instrument.quantity_step,
            "min_quantity": instrument.min_quantity,
            "max_quantity": instrument.max_quantity,
            "min_notional": instrument.min_notional,
            "max_notional": instrument.max_notional,
            "maker_fee": instrument.maker_fee,
            "taker_fee": instrument.taker_fee,
            "max_leverage": instrument.max_leverage,
            "contract_size": instrument.contract_size,
            "active": instrument.active,
        }
        statement = pg_insert(InstrumentRecord).values(**values)
        mutable = {
            key: value
            for key, value in values.items()
            if key not in {"exchange", "symbol", "market_type"}
        }
        statement = statement.on_conflict_do_update(
            index_elements=["exchange", "symbol", "market_type"], set_=mutable
        )
        await self._session.execute(statement)

    async def get(self, symbol: Symbol, *, exchange: str = "binance") -> Instrument | None:
        """Load an instrument's cached rules, if present."""
        statement = select(InstrumentRecord).where(
            InstrumentRecord.exchange == exchange,
            InstrumentRecord.symbol == symbol.slashed,
        )
        record = (await self._session.execute(statement)).scalars().first()
        return _to_instrument(record) if record is not None else None

    async def list_active(self, *, exchange: str = "binance") -> list[Instrument]:
        """Every active instrument for a venue."""
        statement = select(InstrumentRecord).where(
            InstrumentRecord.exchange == exchange, InstrumentRecord.active.is_(True)
        )
        return [_to_instrument(record) for record in await self._scalars(statement)]


# --------------------------------------------------------------------------- #
# Trading
# --------------------------------------------------------------------------- #
class OrderRepository(Repository[OrderRecord]):
    """Persistence for orders and their fills."""

    async def save(self, order: Order, *, session_id: str | None = None) -> None:
        """Insert or update an order together with any new fills."""
        record = await self._session.get(OrderRecord, order.order_id)
        if record is None:
            record = OrderRecord(id=order.order_id, session_id=session_id)
            self._session.add(record)

        record.client_order_id = order.client_order_id
        record.venue_order_id = order.venue_order_id
        record.symbol = order.symbol.slashed
        record.side = order.side
        record.order_type = order.order_type
        record.status = order.status
        record.time_in_force = order.time_in_force
        record.quantity = order.quantity
        record.price = order.price
        record.trigger_price = order.trigger_price
        record.filled_quantity = order.filled_quantity
        record.average_fill_price = order.average_fill_price
        record.fees_paid = order.fees_paid
        record.stop_loss_price = order.stop_loss_price
        record.take_profit_price = order.take_profit_price
        record.reduce_only = order.reduce_only
        record.strategy_id = order.strategy_id
        record.signal_id = order.signal_id
        record.reject_reason = order.reject_reason
        record.meta = dict(order.metadata)
        if session_id is not None:
            record.session_id = session_id

        await self._session.flush()

        # Query the persisted fill ids directly rather than reading `record.fills`: on a
        # just-inserted record that relationship is unloaded, and touching it would emit
        # IO outside SQLAlchemy's async greenlet context.
        existing = await self._session.execute(
            select(FillRecord.venue_fill_id).where(FillRecord.order_id == order.order_id)
        )
        known = set(existing.scalars().all())
        for fill in order.fills:
            if fill.fill_id in known:
                continue
            self._session.add(
                FillRecord(
                    id=str(uuid.uuid4()),
                    venue_fill_id=fill.fill_id,
                    order_id=order.order_id,
                    symbol=fill.symbol.slashed,
                    side=fill.side,
                    quantity=fill.quantity,
                    price=fill.price,
                    fee=fill.fee,
                    fee_currency=fill.fee_currency,
                    role=fill.role,
                    timestamp=fill.timestamp,
                )
            )

    async def get(self, order_id: str) -> Order | None:
        """Load an order by id."""
        record = await self._session.get(OrderRecord, order_id)
        return _to_order(record) if record is not None else None

    async def require(self, order_id: str) -> Order:
        """Load an order, raising if absent."""
        order = await self.get(order_id)
        if order is None:
            raise NotFoundError(f"order {order_id} not found", order_id=order_id)
        return order

    async def get_by_client_id(self, client_order_id: str) -> Order | None:
        """Load an order by its client id (the key used for venue idempotency)."""
        statement = select(OrderRecord).where(OrderRecord.client_order_id == client_order_id)
        record = (await self._session.execute(statement)).scalars().first()
        return _to_order(record) if record is not None else None

    async def list_open(self, *, session_id: str | None = None) -> list[Order]:
        """Every order that can still receive fills.

        Called on startup so a restarted engine adopts, rather than orphans, live orders.
        """
        from quantflow.domain.enums import OPEN_ORDER_STATUSES

        statement = select(OrderRecord).where(OrderRecord.status.in_(OPEN_ORDER_STATUSES))
        if session_id is not None:
            statement = statement.where(OrderRecord.session_id == session_id)
        statement = statement.order_by(OrderRecord.created_at)
        return [_to_order(record) for record in await self._scalars(statement)]

    async def list_recent(
        self, *, limit: int = 100, symbol: Symbol | None = None, session_id: str | None = None
    ) -> list[Order]:
        """Most recent orders, newest first."""
        statement = select(OrderRecord)
        if symbol is not None:
            statement = statement.where(OrderRecord.symbol == symbol.slashed)
        if session_id is not None:
            statement = statement.where(OrderRecord.session_id == session_id)
        statement = statement.order_by(OrderRecord.created_at.desc()).limit(limit)
        return [_to_order(record) for record in await self._scalars(statement)]

    async def count_since(self, since: datetime, *, session_id: str | None = None) -> int:
        """Orders created since ``since`` — backs the order-rate risk limit."""
        statement = (
            select(func.count()).select_from(OrderRecord).where(OrderRecord.created_at >= since)
        )
        if session_id is not None:
            statement = statement.where(OrderRecord.session_id == session_id)
        result = await self._session.execute(statement)
        return int(result.scalar_one())


class PositionRepository(Repository[PositionRecord]):
    """Persistence for open and historical positions."""

    async def save(self, position: Position, *, session_id: str | None = None) -> None:
        """Insert or update the position for a symbol within a session."""
        statement = select(PositionRecord).where(
            PositionRecord.symbol == position.symbol.slashed,
            PositionRecord.session_id == session_id,
            PositionRecord.closed_at.is_(None),
        )
        record = (await self._session.execute(statement)).scalars().first()

        if record is None:
            if position.is_flat:
                return
            record = PositionRecord(
                id=str(uuid.uuid4()),
                session_id=session_id,
                symbol=position.symbol.slashed,
            )
            self._session.add(record)

        record.side = position.side
        record.quantity = position.quantity
        record.average_entry_price = position.average_entry_price
        record.realized_pnl = position.realized_pnl
        record.fees_paid = position.fees_paid
        record.stop_loss_price = position.stop_loss_price
        record.take_profit_price = position.take_profit_price
        record.strategy_id = position.strategy_id
        record.opened_at = position.opened_at
        record.closed_at = position.updated_at if position.is_flat else None
        record.lots = [
            {
                "quantity": str(lot.quantity),
                "price": str(lot.price),
                "opened_at": lot.opened_at.isoformat(),
                "fee": str(lot.fee),
            }
            for lot in position.lots
        ]

    async def get_open(self, symbol: Symbol, *, session_id: str | None = None) -> Position | None:
        """The open position in ``symbol``, if any."""
        statement = select(PositionRecord).where(
            PositionRecord.symbol == symbol.slashed,
            PositionRecord.session_id == session_id,
            PositionRecord.closed_at.is_(None),
        )
        record = (await self._session.execute(statement)).scalars().first()
        return _to_position(record) if record is not None else None

    async def list_open(self, *, session_id: str | None = None) -> list[Position]:
        """Every open position, used to restore state after a restart."""
        statement = select(PositionRecord).where(
            PositionRecord.session_id == session_id,
            PositionRecord.closed_at.is_(None),
        )
        return [_to_position(record) for record in await self._scalars(statement)]


class ClosedTradeRepository(Repository[ClosedTradeRecord]):
    """Persistence for completed round-trips."""

    async def add_many(
        self, trades: Iterable[ClosedTrade], *, session_id: str | None = None
    ) -> int:
        """Append closed trades. Returns the number written."""
        count = 0
        for trade in trades:
            self._session.add(
                ClosedTradeRecord(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    symbol=trade.symbol.slashed,
                    side=trade.side,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    gross_pnl=trade.gross_pnl,
                    fees=trade.fees,
                    net_pnl=trade.net_pnl,
                    return_pct=trade.return_pct,
                    holding_period_seconds=int(trade.holding_period),
                    strategy_id=trade.strategy_id,
                )
            )
            count += 1
        return count

    async def list_for_session(self, session_id: str, *, limit: int = 1000) -> list[ClosedTrade]:
        """Closed trades for a session, oldest first."""
        statement = (
            select(ClosedTradeRecord)
            .where(ClosedTradeRecord.session_id == session_id)
            .order_by(ClosedTradeRecord.exit_time)
            .limit(limit)
        )
        return [_to_closed_trade(record) for record in await self._scalars(statement)]

    async def list_between(
        self,
        start: datetime,
        end: datetime,
        *,
        strategy_id: str | None = None,
        limit: int = 1000,
    ) -> list[ClosedTrade]:
        """Closed trades whose exit falls in ``[start, end)``."""
        statement = select(ClosedTradeRecord).where(
            ClosedTradeRecord.exit_time >= start, ClosedTradeRecord.exit_time < end
        )
        if strategy_id is not None:
            statement = statement.where(ClosedTradeRecord.strategy_id == strategy_id)
        statement = statement.order_by(ClosedTradeRecord.exit_time).limit(limit)
        return [_to_closed_trade(record) for record in await self._scalars(statement)]

    async def realized_pnl_since(
        self, since: datetime, *, session_id: str | None = None
    ) -> Decimal:
        """Net realised PnL since ``since`` — backs the daily-loss risk limit."""
        statement = select(func.coalesce(func.sum(ClosedTradeRecord.net_pnl), 0)).where(
            ClosedTradeRecord.exit_time >= since
        )
        if session_id is not None:
            statement = statement.where(ClosedTradeRecord.session_id == session_id)
        result = await self._session.execute(statement)
        return Decimal(str(result.scalar_one()))


# --------------------------------------------------------------------------- #
# Sessions, equity, risk
# --------------------------------------------------------------------------- #
class TradingSessionRepository(Repository[TradingSessionRecord]):
    """Persistence for engine runs."""

    async def create(
        self,
        *,
        session_id: str,
        mode: str,
        strategy_id: str,
        symbols: Sequence[Symbol],
        timeframe: Timeframe,
        starting_equity: Decimal,
        base_currency: str = "USDT",
        strategy_params: dict[str, Any] | None = None,
        risk_config: dict[str, Any] | None = None,
    ) -> TradingSessionRecord:
        """Open a new session record."""
        record = TradingSessionRecord(
            id=session_id,
            mode=mode,
            status=RunStatus.RUNNING,
            strategy_id=strategy_id,
            strategy_params={key: str(value) for key, value in (strategy_params or {}).items()},
            symbols=[symbol.slashed for symbol in symbols],
            timeframe=timeframe,
            base_currency=base_currency,
            starting_equity=starting_equity,
            started_at=utc_now(),
            risk_config={key: str(value) for key, value in (risk_config or {}).items()},
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def finish(
        self,
        session_id: str,
        *,
        status: RunStatus,
        final_equity: Decimal | None = None,
        metrics: dict[str, float] | None = None,
        error: str | None = None,
    ) -> None:
        """Close out a session."""
        record = await self._session.get(TradingSessionRecord, session_id)
        if record is None:
            raise NotFoundError(f"trading session {session_id} not found", session_id=session_id)
        record.status = status
        record.final_equity = final_equity
        record.finished_at = utc_now()
        if metrics is not None:
            record.metrics = metrics
        if error is not None:
            record.error = error

    async def get(self, session_id: str) -> TradingSessionRecord | None:
        """Load a session record."""
        return await self._session.get(TradingSessionRecord, session_id)

    async def list_recent(self, *, limit: int = 50) -> list[TradingSessionRecord]:
        """Recent sessions, newest first."""
        statement = (
            select(TradingSessionRecord)
            .order_by(TradingSessionRecord.created_at.desc())
            .limit(limit)
        )
        return list(await self._scalars(statement))


class EquityRepository(Repository[EquitySnapshotRecord]):
    """Persistence for equity-curve samples."""

    async def add(
        self,
        session_id: str,
        point: EquityPoint,
        *,
        gross_exposure: Decimal = ZERO,
    ) -> None:
        """Append an equity sample, replacing any sample at the same instant."""
        statement = pg_insert(EquitySnapshotRecord).values(
            session_id=session_id,
            timestamp=point.timestamp,
            equity=point.equity,
            cash=point.cash,
            position_count=point.position_count,
            gross_exposure=gross_exposure,
            realized_pnl=point.realized_pnl,
            unrealized_pnl=point.unrealized_pnl,
            drawdown_pct=point.drawdown_pct,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["session_id", "timestamp"],
            set_={
                "equity": statement.excluded.equity,
                "cash": statement.excluded.cash,
                "position_count": statement.excluded.position_count,
                "gross_exposure": statement.excluded.gross_exposure,
                "realized_pnl": statement.excluded.realized_pnl,
                "unrealized_pnl": statement.excluded.unrealized_pnl,
                "drawdown_pct": statement.excluded.drawdown_pct,
            },
        )
        await self._session.execute(statement)

    async def curve(self, session_id: str, *, limit: int = 100_000) -> list[EquityPoint]:
        """The session's equity curve, oldest first."""
        statement = (
            select(EquitySnapshotRecord)
            .where(EquitySnapshotRecord.session_id == session_id)
            .order_by(EquitySnapshotRecord.timestamp)
            .limit(limit)
        )
        return [
            EquityPoint(
                timestamp=record.timestamp,
                equity=record.equity,
                cash=record.cash,
                position_count=record.position_count,
                drawdown_pct=record.drawdown_pct,
                realized_pnl=record.realized_pnl,
                unrealized_pnl=record.unrealized_pnl,
            )
            for record in await self._scalars(statement)
        ]

    async def latest(self, session_id: str) -> EquityPoint | None:
        """The most recent equity sample for a session."""
        statement = (
            select(EquitySnapshotRecord)
            .where(EquitySnapshotRecord.session_id == session_id)
            .order_by(EquitySnapshotRecord.timestamp.desc())
            .limit(1)
        )
        record = (await self._session.execute(statement)).scalars().first()
        if record is None:
            return None
        return EquityPoint(
            timestamp=record.timestamp,
            equity=record.equity,
            cash=record.cash,
            position_count=record.position_count,
            drawdown_pct=record.drawdown_pct,
            realized_pnl=record.realized_pnl,
            unrealized_pnl=record.unrealized_pnl,
        )


class RiskEventRepository(Repository[RiskEventRecord]):
    """Persistence for the risk audit trail and kill-switch state."""

    async def record(
        self,
        *,
        rule: str,
        message: str,
        severity: str = "warning",
        symbol: Symbol | None = None,
        observed_value: Decimal | None = None,
        limit_value: Decimal | None = None,
        blocked_order: bool = False,
        halted_trading: bool = False,
        session_id: str | None = None,
        context: dict[str, str] | None = None,
    ) -> None:
        """Append a risk event."""
        self._session.add(
            RiskEventRecord(
                id=str(uuid.uuid4()),
                session_id=session_id,
                rule=rule,
                severity=severity,
                symbol=symbol.slashed if symbol else None,
                message=message,
                observed_value=observed_value,
                limit_value=limit_value,
                blocked_order=blocked_order,
                halted_trading=halted_trading,
                context=context or {},
            )
        )

    async def list_recent(
        self, *, limit: int = 100, session_id: str | None = None
    ) -> list[RiskEventRecord]:
        """Recent risk events, newest first."""
        statement = select(RiskEventRecord)
        if session_id is not None:
            statement = statement.where(RiskEventRecord.session_id == session_id)
        statement = statement.order_by(RiskEventRecord.created_at.desc()).limit(limit)
        return list(await self._scalars(statement))

    async def get_kill_switch(self) -> KillSwitchRecord:
        """Load the kill-switch row, creating it disengaged if absent."""
        record = await self._session.get(KillSwitchRecord, 1)
        if record is None:
            record = KillSwitchRecord(id=1, engaged=False)
            self._session.add(record)
            await self._session.flush()
        return record

    async def set_kill_switch(
        self, *, engaged: bool, reason: str | None = None, actor: str = "system"
    ) -> KillSwitchRecord:
        """Latch or clear the kill switch."""
        record = await self.get_kill_switch()
        record.engaged = engaged
        if engaged:
            record.reason = reason
            record.engaged_at = utc_now()
            record.engaged_by = actor
            record.cleared_at = None
            record.cleared_by = None
        else:
            record.cleared_at = utc_now()
            record.cleared_by = actor
        await self._session.flush()
        return record


class BacktestRepository(Repository[BacktestRunRecord]):
    """Persistence for backtest runs."""

    async def save(self, record: BacktestRunRecord) -> BacktestRunRecord:
        """Insert or update a backtest run."""
        merged = await self._session.merge(record)
        await self._session.flush()
        return merged

    async def get(self, run_id: str) -> BacktestRunRecord | None:
        """Load a backtest run."""
        return await self._session.get(BacktestRunRecord, run_id)

    async def list_recent(
        self, *, limit: int = 50, strategy_id: str | None = None
    ) -> list[BacktestRunRecord]:
        """Recent backtest runs, newest first."""
        statement = select(BacktestRunRecord)
        if strategy_id is not None:
            statement = statement.where(BacktestRunRecord.strategy_id == strategy_id)
        statement = statement.order_by(BacktestRunRecord.created_at.desc()).limit(limit)
        return list(await self._scalars(statement))


# --------------------------------------------------------------------------- #
# Record -> domain mapping
# --------------------------------------------------------------------------- #
def _to_candle(record: CandleRecord) -> Candle:
    return Candle(
        symbol=Symbol.parse(record.symbol),
        timeframe=record.timeframe,
        open_time=record.open_time,
        open=record.open,
        high=record.high,
        low=record.low,
        close=record.close,
        volume=record.volume,
        quote_volume=record.quote_volume,
        trades=record.trades,
    )


def _to_instrument(record: InstrumentRecord) -> Instrument:
    from quantflow.core.config import MarketType

    return Instrument(
        symbol=Symbol.parse(record.symbol),
        market_type=MarketType(record.market_type),
        price_tick=record.price_tick,
        quantity_step=record.quantity_step,
        min_quantity=record.min_quantity,
        max_quantity=record.max_quantity,
        min_notional=record.min_notional,
        max_notional=record.max_notional,
        maker_fee=record.maker_fee,
        taker_fee=record.taker_fee,
        max_leverage=record.max_leverage,
        contract_size=record.contract_size,
        active=record.active,
    )


def _to_order(record: OrderRecord) -> Order:
    symbol = Symbol.parse(record.symbol)
    fills = tuple(
        Fill(
            fill_id=fill.venue_fill_id,
            order_id=record.id,
            symbol=symbol,
            side=OrderSide(fill.side),
            quantity=fill.quantity,
            price=fill.price,
            fee=fill.fee,
            fee_currency=fill.fee_currency,
            timestamp=fill.timestamp,
            role=LiquidityRole(fill.role),
        )
        for fill in sorted(record.fills, key=lambda item: item.timestamp)
    )
    return Order(
        order_id=record.id,
        client_order_id=record.client_order_id,
        symbol=symbol,
        side=OrderSide(record.side),
        order_type=OrderType(record.order_type),
        quantity=record.quantity,
        status=OrderStatus(record.status),
        created_at=record.created_at,
        updated_at=record.updated_at,
        price=record.price,
        trigger_price=record.trigger_price,
        time_in_force=TimeInForce(record.time_in_force),
        filled_quantity=record.filled_quantity,
        average_fill_price=record.average_fill_price,
        fees_paid=record.fees_paid,
        fills=fills,
        venue_order_id=record.venue_order_id,
        stop_loss_price=record.stop_loss_price,
        take_profit_price=record.take_profit_price,
        reduce_only=record.reduce_only,
        strategy_id=record.strategy_id,
        signal_id=record.signal_id,
        reject_reason=record.reject_reason,
        metadata=dict(record.meta),
    )


def _to_position(record: PositionRecord) -> Position:
    lots = tuple(
        Lot(
            quantity=Decimal(raw["quantity"]),
            price=Decimal(raw["price"]),
            opened_at=datetime.fromisoformat(raw["opened_at"]),
            fee=Decimal(raw.get("fee", "0")),
        )
        for raw in record.lots
    )
    return Position(
        symbol=Symbol.parse(record.symbol),
        quantity=record.quantity,
        lots=lots,
        realized_pnl=record.realized_pnl,
        fees_paid=record.fees_paid,
        opened_at=record.opened_at,
        updated_at=record.updated_at,
        strategy_id=record.strategy_id,
        stop_loss_price=record.stop_loss_price,
        take_profit_price=record.take_profit_price,
    )


def _to_closed_trade(record: ClosedTradeRecord) -> ClosedTrade:
    return ClosedTrade(
        symbol=Symbol.parse(record.symbol),
        side=PositionSide(record.side),
        quantity=record.quantity,
        entry_price=record.entry_price,
        exit_price=record.exit_price,
        entry_time=record.entry_time,
        exit_time=record.exit_time,
        gross_pnl=record.gross_pnl,
        fees=record.fees,
        strategy_id=record.strategy_id,
    )
