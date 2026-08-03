"""Paper-trading engine.

Live market data, simulated fills. The point of paper trading is to expose everything the
backtest cannot: real bar timing, real gaps, real reconnects, real latency between a signal
and the price moving on.

It shares the **same** strategy, risk engine, portfolio manager and fill model as the
backtester and as live trading. The only substitution is where the fills come from, which
is what makes paper results meaningful evidence about live behaviour rather than a second,
differently-wrong simulation.

State is persisted after every fill, so a restart resumes rather than silently starting
flat with positions it believes it does not hold.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.cache.redis import EventBus
from quantflow.core.clock import Clock, SystemClock
from quantflow.core.config import RiskSettings, TradingMode
from quantflow.core.errors import MarketDataError, ValidationError
from quantflow.core.logging import get_logger, log_context
from quantflow.core.precision import ZERO
from quantflow.domain.enums import MarketRegime, PositionSide, RunStatus, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.domain.orders import Fill, Order
from quantflow.domain.signals import Signal
from quantflow.exchange.base import MarketDataGateway
from quantflow.exchange.simulator import (
    FeeModel,
    SimulatedBroker,
    SlippageModel,
    VolumeShareSlippage,
)
from quantflow.persistence.database import Database
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine, assert_protected
from quantflow.strategy.base import Strategy, StrategyContext
from quantflow.strategy.indicators import atr

logger = get_logger(__name__)

#: How many closed bars to preload per symbol before trading starts. A strategy that has
#: not warmed up produces nothing, so seeding history is what makes the engine useful from
#: the first live bar rather than hours later.
DEFAULT_HISTORY_BARS = 500

#: Cap on retained in-memory history per symbol.
MAX_HISTORY_BARS = 5_000

#: ATR window used for volatility-scaled sizing.
SIZING_ATR_PERIOD = 14


@dataclass(frozen=True, slots=True)
class PaperConfig:
    """Configuration for a paper-trading session."""

    symbols: tuple[Symbol, ...]
    timeframe: Timeframe
    starting_equity: Decimal = Decimal("10000")
    base_currency: str = "USDT"
    risk: RiskSettings = field(default_factory=RiskSettings)
    slippage: SlippageModel = field(default_factory=VolumeShareSlippage)
    fees: FeeModel = field(default_factory=FeeModel)
    history_bars: int = DEFAULT_HISTORY_BARS
    persist: bool = True
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(slots=True)
class PaperSessionState:
    """Live counters for one session."""

    session_id: str
    status: RunStatus = RunStatus.PENDING
    bars_seen: int = 0
    signals: int = 0
    orders: int = 0
    fills: int = 0
    rejections: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API and dashboard."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "bars_seen": self.bars_seen,
            "signals": self.signals,
            "orders": self.orders,
            "fills": self.fills,
            "rejections": self.rejections,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
        }


class PaperTradingEngine:
    """Runs a strategy against a live bar feed with simulated fills."""

    __slots__ = (
        "_broker",
        "_bus",
        "_clock",
        "_config",
        "_database",
        "_history",
        "_instruments",
        "_orders",
        "_pending_protection",
        "_portfolio",
        "_risk",
        "_state",
        "_stopping",
        "_strategy",
    )

    def __init__(
        self,
        strategy: Strategy,
        config: PaperConfig,
        *,
        instruments: dict[Symbol, Instrument],
        database: Database | None = None,
        event_bus: EventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._instruments = instruments
        self._database = database if config.persist else None
        self._bus = event_bus
        self._clock = clock or SystemClock()
        self._portfolio = PortfolioManager(
            base_currency=config.base_currency,
            starting_equity=config.starting_equity,
            clock=self._clock,
        )
        self._risk = RiskEngine(
            config.risk,
            clock=self._clock,
            database=self._database,
            session_id=config.session_id,
        )
        self._broker = SimulatedBroker(
            instruments=instruments, slippage=config.slippage, fees=config.fees
        )
        self._history: dict[Symbol, list[Candle]] = {symbol: [] for symbol in config.symbols}
        self._orders: dict[str, Order] = {}
        self._pending_protection: dict[str, tuple[Decimal | None, Decimal | None]] = {}
        self._state = PaperSessionState(session_id=config.session_id)
        self._stopping = False

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> PaperSessionState:
        """Live session counters."""
        return self._state

    @property
    def portfolio(self) -> PortfolioManager:
        """The simulated portfolio."""
        return self._portfolio

    @property
    def risk(self) -> RiskEngine:
        """The risk engine guarding this session."""
        return self._risk

    @property
    def mode(self) -> TradingMode:
        """Always ``PAPER``."""
        return TradingMode.PAPER

    def snapshot(self) -> dict[str, Any]:
        """Everything the dashboard needs in one call."""
        return {
            "session": self._state.to_dict(),
            "portfolio": self._portfolio.summary(),
            "positions": [
                {
                    "symbol": str(position.symbol),
                    "side": position.side.value,
                    "quantity": str(position.quantity),
                    "entry_price": str(position.average_entry_price),
                    "unrealized_pnl": str(
                        position.unrealized_pnl(self._portfolio.mark_price(position.symbol) or ZERO)
                    ),
                    "stop_loss": (
                        str(position.stop_loss_price) if position.stop_loss_price else None
                    ),
                }
                for position in self._portfolio.positions
            ],
            "risk": self._risk.describe(),
        }

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def prepare(self, gateway: MarketDataGateway) -> None:
        """Load history and restore any prior state.

        Seeding history matters: without it the strategy spends its entire warm-up period
        producing nothing, which for a 200-bar hourly strategy is over a week of silence.
        """
        await self._risk.start()
        await self._restore()

        for symbol in self._config.symbols:
            if symbol not in self._instruments:
                raise ValidationError(f"no instrument metadata for {symbol}", symbol=str(symbol))
            candles = await gateway.fetch_candles(
                symbol, self._config.timeframe, limit=self._config.history_bars
            )
            closed = [candle for candle in candles if candle.is_closed(self._clock.now())]
            if not closed:
                raise MarketDataError(f"no closed history for {symbol}", symbol=str(symbol))
            self._history[symbol] = closed
            self._portfolio.update_mark_price(symbol, closed[-1].close)
            logger.info(
                "paper.history_loaded",
                symbol=str(symbol),
                bars=len(closed),
                from_time=closed[0].open_time.isoformat(),
            )

        self._strategy.on_start(self._config.symbols)

        if self._database is not None:
            async with self._database.unit_of_work() as uow:
                await uow.sessions.create(
                    session_id=self._config.session_id,
                    mode=TradingMode.PAPER.value,
                    strategy_id=self._strategy.strategy_id,
                    symbols=self._config.symbols,
                    timeframe=self._config.timeframe,
                    starting_equity=self._config.starting_equity,
                    base_currency=self._config.base_currency,
                    strategy_params=self._strategy.params.to_dict(),
                )

        self._state.status = RunStatus.RUNNING
        self._state.started_at = self._clock.now()

    async def run(self, feed: AsyncIterator[Candle]) -> PaperSessionState:
        """Consume a live feed of **closed** bars until it ends or ``stop`` is called.

        The feed must yield closed bars only; acting on a forming bar means acting on a
        price that can still move within the same bar, which nothing in the backtest ever
        reproduces.
        """
        with log_context(
            session_id=self._config.session_id, strategy_id=self._strategy.strategy_id
        ):
            logger.info(
                "paper.started",
                symbols=[str(symbol) for symbol in self._config.symbols],
                timeframe=self._config.timeframe.value,
                starting_equity=str(self._config.starting_equity),
            )
            try:
                async for candle in feed:
                    if self._stopping:
                        break
                    await self.on_candle(candle)
            except asyncio.CancelledError:
                logger.info("paper.cancelled")
                raise
            except Exception as exc:
                self._state.status = RunStatus.FAILED
                self._state.error = str(exc)
                logger.exception("paper.failed", error=str(exc))
            else:
                self._state.status = RunStatus.COMPLETED
            finally:
                await self._finish()

        return self._state

    async def on_candle(self, candle: Candle) -> None:
        """Process one closed bar for one symbol.

        Mirrors the backtest loop exactly: match resting orders, apply protective exits,
        mark, sample equity, then decide.
        """
        symbol = candle.symbol
        if symbol not in self._history:
            return

        # 1. Orders placed on the previous bar match against this one.
        for order, fill in self._broker.process_candle(candle):
            self._orders[order.order_id] = order
            if order.status.is_terminal and not order.fills:
                continue
            self._portfolio.apply_fill(fill, strategy_id=order.strategy_id)
            self._attach_protection(order)
            self._state.fills += 1
            await self._publish_fill(fill)

        # 2. Protective exits, checked against the bar's range.
        await self._check_protective_exits(symbol, candle)

        # 3. Mark and sample.
        self._portfolio.update_mark_price(symbol, candle.close)
        history = self._history[symbol]
        if history and candle.open_time <= history[-1].open_time:
            logger.warning(
                "paper.out_of_order_bar",
                symbol=str(symbol),
                received=candle.open_time.isoformat(),
                last=history[-1].open_time.isoformat(),
            )
            return
        history.append(candle)
        if len(history) > MAX_HISTORY_BARS:
            del history[: len(history) - MAX_HISTORY_BARS]

        self._state.bars_seen += 1
        point = self._portfolio.record_equity(candle.close_time)
        await self._persist_equity(point)

        # 4. Decide, for execution on the next bar.
        if len(history) < self._strategy.warmup_bars:
            return
        await self._decide(symbol, history, candle)

    async def stop(self, *, flatten: bool = False) -> None:
        """Request a graceful stop, optionally flattening every position first."""
        self._stopping = True
        if flatten:
            await self.flatten_all(reason="session stopping")
        logger.info("paper.stop_requested", flatten=flatten)

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #
    async def _decide(self, symbol: Symbol, history: list[Candle], candle: Candle) -> None:
        context = StrategyContext(
            symbol=symbol,
            timeframe=self._config.timeframe,
            history=CandleSeries(history[-MAX_HISTORY_BARS:]),
            now=candle.close_time,
            portfolio=self._portfolio.snapshot(candle.close_time),
            position=self._portfolio.position_for(symbol),
            regime=MarketRegime.UNKNOWN,
        )

        signal = self._strategy.evaluate(context)
        if not signal.is_actionable:
            return
        self._state.signals += 1
        await self._publish_signal(signal)

        window = history[-MAX_HISTORY_BARS:]
        volatility = atr(window, SIZING_ATR_PERIOD)[-1] if len(window) > SIZING_ATR_PERIOD else None
        decision = await self._risk.evaluate_signal(
            signal,
            portfolio=self._portfolio.snapshot(candle.close_time),
            instrument=self._instruments[symbol],
            reference_price=candle.close,
            volatility=volatility,
        )
        if not decision.approved or decision.request is None:
            self._state.rejections += 1
            logger.info(
                "paper.signal_rejected",
                symbol=str(symbol),
                direction=signal.direction.value,
                reason=decision.reason,
            )
            return

        request = decision.request
        assert_protected(request, self._config.risk)
        order = self._broker.submit(request, now=self._clock.now(), reference_price=candle.close)
        self._orders[order.order_id] = order
        self._risk.record_order()
        self._state.orders += 1
        if request.stop_loss_price is not None or request.take_profit_price is not None:
            self._pending_protection[order.order_id] = (
                request.stop_loss_price,
                request.take_profit_price,
            )

        logger.info(
            "paper.order_placed",
            order_id=order.order_id,
            symbol=str(symbol),
            side=request.side.value,
            quantity=str(request.quantity),
            reason=signal.reason[:120],
        )
        await self._persist_order(order)

    def _attach_protection(self, order: Order) -> None:
        """Apply protective levels once the order's fill has created the position."""
        levels = self._pending_protection.get(order.order_id)
        if levels is None:
            return
        if self._portfolio.position_for(order.symbol) is None:
            return
        stop, target = levels
        self._portfolio.set_protection(order.symbol, stop_loss_price=stop, take_profit_price=target)
        if order.is_terminal:
            del self._pending_protection[order.order_id]

    async def _check_protective_exits(self, symbol: Symbol, candle: Candle) -> None:
        """Close a position whose stop or target was reached inside this bar.

        As in the backtest, when both levels fall inside one bar the **stop wins**: the bar
        does not record which came first, and assuming the favourable one inflates results.
        """
        position = self._portfolio.position_for(symbol)
        if position is None:
            return

        is_long = position.side is PositionSide.LONG
        stop_hit = position.is_stop_breached(candle.low if is_long else candle.high)
        target_hit = position.is_target_reached(candle.high if is_long else candle.low)
        if not stop_hit and not target_hit:
            return

        exit_price = position.stop_loss_price if stop_hit else position.take_profit_price
        if exit_price is None:  # pragma: no cover — guarded above
            return

        instrument = self._instruments[symbol]
        closing_side = position.closing_side()
        assert closing_side is not None
        quantity = instrument.normalize_quantity(position.absolute_quantity)
        if quantity <= ZERO:
            return

        from quantflow.domain.enums import LiquidityRole

        fill = Fill(
            fill_id=f"protective-{symbol.concatenated}-{candle.open_time.isoformat()}",
            order_id=f"protective-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=closing_side,
            quantity=quantity,
            price=exit_price,
            fee=self._config.fees.compute(
                instrument, quantity=quantity, price=exit_price, role=LiquidityRole.TAKER
            ),
            fee_currency=symbol.quote,
            timestamp=candle.close_time,
        )
        self._portfolio.apply_fill(fill, strategy_id=position.strategy_id)
        self._state.fills += 1
        logger.info(
            "paper.protective_exit",
            symbol=str(symbol),
            kind="stop" if stop_hit else "target",
            price=str(exit_price),
        )
        await self._publish_fill(fill)

    async def flatten_all(self, *, reason: str = "flatten") -> list[Fill]:
        """Close every open position at the last known mark price."""
        from quantflow.domain.enums import LiquidityRole

        fills: list[Fill] = []
        for position in self._portfolio.positions:
            price = self._portfolio.mark_price(position.symbol)
            if price is None:
                logger.warning("paper.flatten_skipped_no_price", symbol=str(position.symbol))
                continue
            instrument = self._instruments[position.symbol]
            closing_side = position.closing_side()
            assert closing_side is not None
            quantity = instrument.normalize_quantity(position.absolute_quantity)
            if quantity <= ZERO:
                continue
            fill = Fill(
                fill_id=f"flatten-{uuid.uuid4().hex[:12]}",
                order_id=f"flatten-{uuid.uuid4().hex[:8]}",
                symbol=position.symbol,
                side=closing_side,
                quantity=quantity,
                price=price,
                fee=self._config.fees.compute(
                    instrument, quantity=quantity, price=price, role=LiquidityRole.TAKER
                ),
                fee_currency=position.symbol.quote,
                timestamp=self._clock.now(),
            )
            self._portfolio.apply_fill(fill)
            fills.append(fill)
            logger.info("paper.flattened", symbol=str(position.symbol), reason=reason)
        return fills

    # ------------------------------------------------------------------ #
    # Persistence and events
    # ------------------------------------------------------------------ #
    async def _restore(self) -> None:
        """Rebuild portfolio state from the database, if a prior session exists.

        Without this a restart would resume flat while the recorded positions still exist,
        and every subsequent risk check would be measured against a fiction.
        """
        if self._database is None:
            return
        try:
            async with self._database.read_session() as session:
                from quantflow.persistence.repositories import PositionRepository

                positions = await PositionRepository(session).list_open(
                    session_id=self._config.session_id
                )
        except Exception as exc:
            logger.debug("paper.restore_skipped", reason=str(exc))
            return

        if not positions:
            return
        self._portfolio.restore(
            cash=self._config.starting_equity,
            positions=positions,
            peak_equity=self._config.starting_equity,
        )
        logger.info("paper.state_restored", positions=len(positions))

    async def _persist_order(self, order: Order) -> None:
        if self._database is None:
            return
        try:
            async with self._database.unit_of_work() as uow:
                await uow.orders.save(order, session_id=self._config.session_id)
        except Exception as exc:
            logger.exception("paper.persist_order_failed", error=str(exc))

    async def _persist_equity(self, point: Any) -> None:
        if self._database is None:
            return
        try:
            async with self._database.unit_of_work() as uow:
                await uow.equity.add(self._config.session_id, point)
        except Exception as exc:
            logger.exception("paper.persist_equity_failed", error=str(exc))

    async def _publish_signal(self, signal: Signal) -> None:
        if self._bus is None:
            return
        with contextlib.suppress(Exception):
            await self._bus.publish_signal(
                {
                    "session_id": self._config.session_id,
                    "symbol": str(signal.symbol),
                    "direction": signal.direction.value,
                    "reason": signal.reason,
                    "timestamp": signal.timestamp.isoformat(),
                }
            )

    async def _publish_fill(self, fill: Fill) -> None:
        if self._bus is None:
            return
        with contextlib.suppress(Exception):
            await self._bus.publish_fill(
                {
                    "session_id": self._config.session_id,
                    "symbol": str(fill.symbol),
                    "side": fill.side.value,
                    "quantity": str(fill.quantity),
                    "price": str(fill.price),
                    "timestamp": fill.timestamp.isoformat(),
                }
            )

    async def _finish(self) -> None:
        """Close out the session record and flush final state."""
        self._state.finished_at = self._clock.now()
        self._strategy.on_finish()

        if self._database is None:
            return
        try:
            async with self._database.unit_of_work() as uow:
                await uow.trades.add_many(
                    self._portfolio.closed_trades, session_id=self._config.session_id
                )
                await uow.sessions.finish(
                    self._config.session_id,
                    status=self._state.status,
                    final_equity=self._portfolio.equity(),
                    metrics={"bars": self._state.bars_seen, "fills": self._state.fills},
                    error=self._state.error,
                )
        except Exception as exc:
            logger.exception("paper.finish_persist_failed", error=str(exc))

        logger.info(
            "paper.finished",
            status=self._state.status.value,
            bars=self._state.bars_seen,
            orders=self._state.orders,
            fills=self._state.fills,
            final_equity=str(self._portfolio.equity()),
        )


async def candle_feed(
    stream: Any, symbols: Sequence[Symbol], timeframe: Timeframe
) -> AsyncIterator[Candle]:
    """Adapt a :class:`~quantflow.exchange.binance.ws.BinanceStream` into a bar feed.

    ``closed_only=True`` is not optional: a forming bar's close is still moving, and a
    strategy acting on it is trading information the backtest never had.
    """
    async for candle in stream.watch_many_candles(list(symbols), timeframe, closed_only=True):
        yield candle
