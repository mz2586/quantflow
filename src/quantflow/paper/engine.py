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
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.cache.redis import EventBus
from quantflow.core.clock import Clock, SystemClock
from quantflow.core.config import MarketType, RiskSettings, TradingMode
from quantflow.core.errors import MarketDataError, ValidationError
from quantflow.core.logging import get_logger, log_context
from quantflow.core.precision import ZERO
from quantflow.domain.enums import MarketRegime, OrderSide, PositionSide, RunStatus, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.domain.orders import Fill, Order
from quantflow.domain.positions import ClosedTrade, Position
from quantflow.domain.signals import Signal
from quantflow.exchange.base import MarketDataGateway
from quantflow.exchange.simulator import (
    FeeModel,
    SimulatedBroker,
    SlippageModel,
    VolumeShareSlippage,
)
from quantflow.execution.router import OrderRouter, SimulatedOrderRouter
from quantflow.intelligence.snapshot import portfolio_correlations
from quantflow.persistence.database import Database
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine, assert_protected
from quantflow.risk.monitor import LossMonitor
from quantflow.strategy.base import Strategy, StrategyContext
from quantflow.strategy.indicators import atr

logger = get_logger(__name__)

#: Bars a symbol needs before it contributes to the correlation estimate. Below this the
#: estimate is noise, and a spurious correlation would block legitimate trades.
MIN_CORRELATION_BARS = 60

#: A correlation needs two series. Below this there is nothing to compare.
MIN_SYMBOLS_FOR_CORRELATION = 2

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
    #: Drives the portfolio's accounting. Paper must use the SAME margin math as live, or
    #: paper results describe an account that does not exist on the venue.
    market_type: MarketType = MarketType.SPOT
    leverage: Decimal = Decimal("1")
    #: Funding rate lookup, ``(symbol, settled_at) -> rate or None``. Left unset, no funding
    #: is charged - correct for spot, and honest for a perp session with no rate source
    #: rather than inventing one.
    funding_rate_for: Callable[[Symbol, datetime], Decimal | None] | None = None
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
        "_loss_monitor",
        "_orders",
        "_pending_protection",
        "_persisted_trades",
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
        router: OrderRouter | None = None,
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
            market_type=config.market_type,
            leverage=config.leverage,
        )
        self._risk = RiskEngine(
            config.risk,
            clock=self._clock,
            database=self._database,
            session_id=config.session_id,
        )
        # Injectable so a LIVE session can route to the venue instead. Defaulting to the
        # simulator keeps backtest and paper unchanged; the point of the seam is that a live
        # session can no longer silently end up here.
        self._broker: OrderRouter = router or SimulatedOrderRouter(
            SimulatedBroker(instruments=instruments, slippage=config.slippage, fees=config.fees)
        )
        self._history: dict[Symbol, list[Candle]] = {symbol: [] for symbol in config.symbols}
        self._orders: dict[str, Order] = {}
        self._pending_protection: dict[str, tuple[Decimal | None, Decimal | None]] = {}
        self._state = PaperSessionState(session_id=config.session_id)
        self._stopping = False
        #: Closed trades already written to the database. Trades are flushed as they close
        #: so a *running* session's dashboard reflects them; without this counter the
        #: end-of-session flush would write every trade a second time.
        self._persisted_trades = 0
        # Runs on every equity sample. Without it the loss limits were only consulted when a
        # new order was proposed, so an open loser with no fresh signal had no backstop.
        self._loss_monitor = LossMonitor(self._risk, config.risk, flatten=self._flatten_for_breach)

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
    def router(self) -> OrderRouter:
        """Where approved orders are sent.

        Exposed so the live runner can verify a LIVE session is not routing to a simulator
        before it declares itself armed.
        """
        return self._broker

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
    def _refresh_correlations(self) -> None:
        """Hand the risk engine a fresh correlation estimate.

        Without this the correlation rule is inert: it sees an empty matrix, finds nothing
        correlated with anything, and silently permits a book of ten positions that are
        really one bet ten times over. On a single symbol that never mattered; across a
        basket of crypto pairs correlated 0.7-0.9 it is the whole point of the rule.

        Aligned on shared timestamps inside `portfolio_correlations`, so symbols with
        different history lengths are compared over the same window rather than by
        position.
        """
        usable = {
            symbol: bars
            for symbol, bars in self._history.items()
            if len(bars) > MIN_CORRELATION_BARS
        }
        if len(usable) < MIN_SYMBOLS_FOR_CORRELATION:
            return
        self._risk.set_correlations(portfolio_correlations(usable))

    async def _align_venue_leverage(self, gateway: MarketDataGateway) -> None:
        """Set each symbol's venue leverage to the value this engine assumes.

        The accounting reserves ``config.leverage`` per position. If the venue disagrees it
        reserves a different amount, and free margin, exposure and every equity-derived
        limit are then measured against a reservation that does not exist. Setting it here
        makes the venue match the assumption instead of the bot hoping they coincide.

        Best-effort: a gateway with no ``set_leverage`` (the simulator, or spot) is simply
        skipped, and a refusal is logged rather than raised - the reconciler reads the
        venue's actual value regardless, so a failure here degrades to "reconcile to truth".
        """
        if self._config.market_type is not MarketType.FUTURE:
            return
        setter = getattr(gateway, "set_leverage", None)
        if not callable(setter):
            return
        for symbol in self._config.symbols:
            try:
                await setter(symbol, self._config.leverage)
            except Exception as exc:  # pragma: no cover - defensive, never blocks startup
                logger.warning(
                    "paper.set_leverage_failed",
                    symbol=str(symbol),
                    leverage=str(self._config.leverage),
                    error=str(exc)[:160],
                )

    async def prepare(self, gateway: MarketDataGateway) -> None:
        """Load history and restore any prior state.

        Seeding history matters: without it the strategy spends its entire warm-up period
        producing nothing, which for a 200-bar hourly strategy is over a week of silence.
        """
        await self._risk.start()
        await self._restore()
        await self._align_venue_leverage(gateway)

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

        self._refresh_correlations()
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
                # Rejected without a fill - most often because the order was larger than
                # the bar's liquidity. It still has to be written through: the row was
                # saved as NEW at submission, so skipping it leaves the database claiming
                # a live order that the engine has already given up on.
                await self._persist_order(order)
                continue
            position, closed = self._portfolio.apply_fill(fill, strategy_id=order.strategy_id)
            self._record_trade_results(closed)
            self._attach_protection(order)
            self._state.fills += 1
            await self._publish_fill(fill)
            # The order row was written at submission with status NEW and no fill. Re-saving
            # it here is what makes the persisted order agree with reality; without it a
            # filled order reads as unfilled for the life of the session.
            await self._persist_order(order)
            await self._persist_position(position)
            await self._persist_trades(closed)

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
        # Funding settles before the equity sample so the recorded curve includes it - a
        # cost charged after the sample would be invisible on the chart it belongs to.
        if self._config.funding_rate_for is not None:
            self._portfolio.settle_funding(
                candle.close_time, rate_for=self._config.funding_rate_for
            )
        point = self._portfolio.record_equity(candle.close_time)
        await self._persist_equity(point)
        await self._loss_monitor.check(self._portfolio.snapshot(candle.close_time))

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
            # A held bar used to vanish here without trace, which made "the engine is not
            # trading" indistinguishable from "the engine is broken" - the strategy's own
            # reason for standing aside was discarded at the one point it mattered.
            logger.info(
                "paper.signal_hold",
                symbol=str(symbol),
                strategy=self._strategy.strategy_id,
                reason=signal.reason,
                bar=candle.close_time.isoformat(),
            )
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
        order = await self._broker.submit(
            request, now=self._clock.now(), reference_price=candle.close
        )
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

    def _record_trade_results(self, trades: Sequence[ClosedTrade]) -> None:
        """Feed closed trades to the risk engine's loss-streak tracker.

        The cooldown rule is only as good as what it is told: without this the streak
        counter never advances and the rule silently never fires.
        """
        for trade in trades:
            self._risk.record_trade_result(
                trade.net_pnl, closed_at=trade.exit_time, symbol=trade.symbol
            )
            # A composite strategy scores its members partly on their realised record; a
            # plain strategy ignores this.
            self._strategy.on_trade_closed(trade)

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

    def _protective_exit_price(
        self,
        level: Decimal,
        *,
        candle: Candle,
        side: OrderSide,
        quantity: Decimal,
        is_stop: bool,
    ) -> Decimal:
        """Fill price for a protective exit, with gap and slippage applied.

        These used to fill at the exact stop or target, which quietly assumes the venue
        always gives you your level. It does not: a bar that gaps straight through a stop
        fills at the open, and the difference is pure loss the backtest never charged.

        A stop fills at the *worse* of its level and the bar open - the same rule the
        simulator already applies to explicit STOP orders. A target is not gapped in the
        trader's favour: assuming a better-than-asked fill is the optimism this exists to
        remove, so it fills at the level and then pays slippage like anything else.
        """
        if is_stop:
            # Closing a long sells: a gap down opens below the stop and that is where it
            # fills. Closing a short buys: a gap up fills higher.
            worst = min(level, candle.open) if side is OrderSide.SELL else max(level, candle.open)
        else:
            worst = level
        return self._config.slippage.apply(
            reference_price=worst, side=side, quantity=quantity, candle=candle
        )

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

        level = position.stop_loss_price if stop_hit else position.take_profit_price
        if level is None:  # pragma: no cover — guarded above
            return

        instrument = self._instruments[symbol]
        closing_side = position.closing_side()
        assert closing_side is not None
        quantity = instrument.normalize_quantity(position.absolute_quantity)
        if quantity <= ZERO:
            return

        from quantflow.domain.enums import LiquidityRole

        exit_price = self._protective_exit_price(
            level, candle=candle, side=closing_side, quantity=quantity, is_stop=stop_hit
        )

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
        exited, closed = self._portfolio.apply_fill(fill, strategy_id=position.strategy_id)
        self._record_trade_results(closed)
        self._state.fills += 1
        logger.info(
            "paper.protective_exit",
            symbol=str(symbol),
            kind="stop" if stop_hit else "target",
            price=str(exit_price),
        )
        await self._publish_fill(fill)
        await self._persist_position(exited)
        await self._persist_trades(closed)

    async def _flatten_for_breach(self, reason: str) -> list[Fill]:
        """Close everything after a loss-limit breach."""
        return await self.flatten_all(reason=reason)

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
                from quantflow.persistence.repositories import (
                    ClosedTradeRepository,
                    EquityRepository,
                    PositionRepository,
                )

                positions = await PositionRepository(session).list_open(
                    session_id=self._config.session_id
                )
                curve = await EquityRepository(session).curve(self._config.session_id)
                closed = await ClosedTradeRepository(session).list_for_session(
                    self._config.session_id
                )
        except Exception as exc:
            logger.debug("paper.restore_skipped", reason=str(exc))
            return

        # A flat session still has state worth restoring: cash that has drifted from the
        # opening balance, realized PnL and fees. Returning early on "no positions" left
        # all three at their constructor defaults.
        if not positions and not curve:
            return

        last = curve[-1] if curve else None
        # Cash, not starting equity. Restoring the opening balance *and* the positions that
        # balance was spent on counts the same money twice: equity comes back inflated by
        # the deployed notional, and nothing downstream disagrees, so it goes unnoticed.
        cash = last.cash if last is not None else self._config.starting_equity
        realized = last.realized_pnl if last is not None else ZERO
        # Fees are not carried on the equity curve, so rebuild them: what each completed
        # round-trip was charged, plus the entry fees still sitting in open positions.
        fees = sum((trade.fees for trade in closed), ZERO) + sum(
            (position.fees_paid for position in positions), ZERO
        )
        # Peak drives the drawdown rule. Seeded from the opening balance it would forget
        # every high the session reached and under-report drawdown after a restart.
        peak = max((point.equity for point in curve), default=cash)

        self._portfolio.restore(
            cash=cash,
            positions=positions,
            peak_equity=peak,
            realized_pnl=realized,
            fees_paid=fees,
        )
        # Hand the restored positions to the strategy so a composite one can re-adopt the
        # member that opened each. Without it a restart leaves every open trade ownerless.
        self._strategy.on_restore(positions)
        logger.info(
            "paper.state_restored",
            positions=len(positions),
            cash=str(cash),
            realized_pnl=str(realized),
            fees_paid=str(fees),
            peak_equity=str(peak),
        )

    async def _persist_order(self, order: Order) -> None:
        if self._database is None:
            return
        try:
            async with self._database.unit_of_work() as uow:
                await uow.orders.save(order, session_id=self._config.session_id)
        except Exception as exc:
            logger.exception("paper.persist_order_failed", error=str(exc))

    async def _persist_position(self, position: Position) -> None:
        """Write a position through, open or just-closed.

        Nothing else in this engine writes the positions table, so without this an open
        position exists only in memory: the dashboard shows none, and ``_restore`` has
        nothing to restore after a restart. ``save`` stamps ``closed_at`` when the position
        is flat, so the same call serves both opening and closing.
        """
        if self._database is None:
            return
        # `_attach_protection` may have set stop/target after the fill was applied, so
        # prefer the portfolio's current view and fall back to the post-fill object, which
        # is all that remains once the position has gone flat.
        current = self._portfolio.position_for(position.symbol) or position
        try:
            async with self._database.unit_of_work() as uow:
                await uow.positions.save(current, session_id=self._config.session_id)
        except Exception as exc:
            logger.exception("paper.persist_position_failed", error=str(exc))

    async def _persist_trades(self, trades: Sequence[ClosedTrade]) -> None:
        """Flush closed trades as they close rather than at session end."""
        if self._database is None or not trades:
            return
        try:
            async with self._database.unit_of_work() as uow:
                await uow.trades.add_many(trades, session_id=self._config.session_id)
        except Exception as exc:
            logger.exception("paper.persist_trades_failed", error=str(exc))
            return
        self._persisted_trades += len(trades)

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
                # Only the tail: everything before this was already flushed as it closed,
                # and add_many mints a fresh id per call, so re-sending them would
                # duplicate every trade of the session.
                await uow.trades.add_many(
                    self._portfolio.closed_trades[self._persisted_trades :],
                    session_id=self._config.session_id,
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
    """Adapt a :class:`~quantflow.exchange.bybit.ws.BybitStream` into a bar feed.

    ``closed_only=True`` is not optional: a forming bar's close is still moving, and a
    strategy acting on it is trading information the backtest never had.
    """
    async for candle in stream.watch_many_candles(list(symbols), timeframe, closed_only=True):
        yield candle
