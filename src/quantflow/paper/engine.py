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
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:  # pragma: no cover - import cycle: `quantflow.live` imports this module.
    from quantflow.live.reconcile import LiveReconciler

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
    #: What this session actually is. The runner drives this engine for live sessions too —
    #: one code path so a live-only branch cannot rot — so the engine must be told, or it
    #: files real orders against a real venue under the label ``paper``.
    #: Defaults to PAPER: nothing becomes live by omission.
    mode: TradingMode = TradingMode.PAPER
    #: Whether ``starting_equity`` was read from the venue rather than configured. Only an
    #: authoritative figure may override the cash restored from a session's own history:
    #: a fallback constant that happened to be wrong is not evidence about the account.
    #: Defaults to False, so nothing overrides persisted state by omission.
    equity_is_authoritative: bool = False


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
        "_reconciler",
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
        #: Set in `prepare` for a live session. A simulated router invents its own fills and
        #: hands them back through `process_candle`; a real venue does not, so on a live
        #: session this is the *only* thing that returns a fill to the portfolio.
        self._reconciler: LiveReconciler | None = None

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
        """The mode this session is actually running in, not an assumed one."""
        return self._config.mode

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

    @property
    def is_live(self) -> bool:
        """Whether orders from this session reach a real venue.

        Both halves matter. The mode says what the operator asked for; the router says where
        an order actually goes. Fills are only ever reconciled from a venue when both agree,
        so a paper session can never start polling an exchange for executions it never sent.
        """
        return self._config.mode is TradingMode.LIVE and not self._broker.is_simulated

    async def prepare(self, gateway: MarketDataGateway) -> None:
        """Load history and restore any prior state.

        Seeding history matters: without it the strategy spends its entire warm-up period
        producing nothing, which for a 200-bar hourly strategy is over a week of silence.
        """
        await self._risk.start()
        await self._restore()
        await self._align_venue_leverage(gateway)
        if self.is_live:
            from quantflow.live.reconcile import LiveReconciler

            self._reconciler = LiveReconciler(
                gateway,
                self._portfolio,
                symbols=self._config.symbols,
                clock=self._clock.now,
                quote=self._config.base_currency,
            )
            self._reconciler.register_venue_ids(self._orders.values())

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
                    mode=self._config.mode.value,
                    strategy_id=self._strategy.strategy_id,
                    symbols=self._config.symbols,
                    timeframe=self._config.timeframe,
                    starting_equity=self._config.starting_equity,
                    base_currency=self._config.base_currency,
                    strategy_params=self._strategy.params.to_dict(),
                )

        # Last, and after the session row exists so the positions and trades this writes
        # have somewhere to hang. A live session must not evaluate its first signal against
        # a book it has not checked: the venue may hold positions this process opened before
        # it died, and sizing around them starts with knowing they are there.
        await self._reconcile_live(initial=True)

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

        # 1b. On a live venue the previous loop yields nothing — the exchange reports its own
        # fills, and this is where they are collected. Without it a live session's portfolio
        # stays permanently flat while the venue holds real positions, and every risk limit
        # is then measured against an empty book.
        await self._reconcile_live()

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
        # A market order on a real venue is already filled by the time the acknowledgement
        # comes back. Reconciling now rather than waiting for the next bar is the difference
        # between a position appearing in the book in seconds and appearing a quarter of an
        # hour later — during which the risk engine would size the next signal against a
        # book that is missing the trade just made.
        #
        # Narrowed to this symbol on purpose. A bar can produce an order on every traded
        # symbol, and a full pass per submission would multiply one bar's reconciliation
        # into a burst of requests the venue's rate limiter would start refusing.
        await self._reconcile_live(only=(symbol,))

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

        Simulated sessions only. On a live venue the stop and the target are held by the
        exchange, and it is the exchange that fills them; synthesising a second, local exit
        from the bar's range would book a close that never happened and leave the local book
        flat against a position the venue is still holding — the exact inversion of the
        defect this engine's reconciliation exists to fix.
        """
        if self.is_live:
            return
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
        """Close every open position.

        Live sessions send real reduce-only orders; simulated ones book a fill at the last
        mark. The split is not cosmetic: a live session that "flattened" by inventing local
        fills would report itself flat while the venue still held every position, which is
        the worst possible state to be in immediately after a loss limit has tripped.
        """
        if self.is_live:
            return await self._flatten_at_venue(reason=reason)

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

    async def _flatten_at_venue(self, *, reason: str) -> list[Fill]:
        """Close every open position with real reduce-only market orders.

        ``reduce_only`` is what makes this safe to issue against a book that may already
        have moved: an order for more than the venue is holding closes what is there instead
        of opening the opposite position on the remainder.

        The returned fills come from reconciliation afterwards, not from the submission —
        an order is not a fill, and reporting one as the other is how a local book comes to
        disagree with a venue in the first place.
        """
        from quantflow.domain.enums import OrderType, TimeInForce
        from quantflow.domain.orders import OrderRequest

        for position in self._portfolio.positions:
            closing_side = position.closing_side()
            if closing_side is None:
                continue
            instrument = self._instruments.get(position.symbol)
            quantity = (
                instrument.normalize_quantity(position.absolute_quantity)
                if instrument is not None
                else position.absolute_quantity
            )
            if quantity <= ZERO:
                continue
            try:
                order = await self._broker.submit(
                    OrderRequest(
                        symbol=position.symbol,
                        side=closing_side,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        time_in_force=TimeInForce.GTC,
                        reduce_only=True,
                        strategy_id=position.strategy_id,
                        metadata={"reason": reason[:200], "source": "flatten"},
                    ),
                    now=self._clock.now(),
                )
            except Exception as exc:
                logger.exception(
                    "paper.venue_flatten_failed", symbol=str(position.symbol), error=str(exc)
                )
                continue
            self._orders[order.order_id] = order
            self._state.orders += 1
            await self._persist_order(order)
            logger.critical(
                "paper.venue_flatten_submitted",
                symbol=str(position.symbol),
                quantity=str(quantity),
                reason=reason,
            )

        await self._reconcile_live()
        return []

    async def _reconcile_live(
        self, *, initial: bool = False, only: Sequence[Symbol] | None = None
    ) -> None:
        """Pull the venue's orders, executions and positions into the local book.

        Everything downstream of a fill happens here for a live session: the portfolio move,
        the closed-trade record, the loss-streak counter, persistence and the event bus. It
        is the same set of consequences the simulated path applies in ``on_candle`` — routed
        through one method so the two cannot drift apart.

        Never raises. A venue read that fails leaves the book where it was and the next pass
        tries again; taking the trading loop down because one HTTP call timed out would turn
        a transient failure into an unmanaged position.
        """
        if self._reconciler is None:
            return
        try:
            outcome = await self._reconciler.reconcile(
                list(self._orders.values()), initial=initial, only=only
            )
        except Exception as exc:
            logger.exception("paper.live_reconcile_failed", error=str(exc))
            return

        for order in outcome.orders:
            self._orders[order.order_id] = order
            self._attach_protection(order)
            await self._persist_order(order)

        for fill in outcome.fills:
            self._state.fills += 1
            await self._publish_fill(fill)

        for position in outcome.positions:
            await self._persist_position(position)

        if outcome.closed_trades:
            self._record_trade_results(outcome.closed_trades)
            await self._persist_trades(outcome.closed_trades)

        # The wallet is the account. Reconstructing cash from a bounded window of executions
        # is a model of it, and a model is re-anchored the moment the venue has been asked
        # directly - at startup, and whenever the book had to be repaired from the venue's
        # own statement rather than from fills.
        if outcome.venue_cash is not None:
            cash = outcome.venue_cash
            if self._config.market_type is not MarketType.FUTURE:
                # Spot counts a held asset inside equity, so the capital already spent on
                # open positions comes out of cash or it is counted twice.
                cash -= sum((position.cost_basis for position in self._portfolio.positions), ZERO)
            self._portfolio.anchor_cash(
                cash,
                reason="startup reconciliation" if initial else "venue state repair",
            )

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
                    OrderRepository,
                    PositionRepository,
                )

                positions = await PositionRepository(session).list_open(
                    session_id=self._config.session_id
                )
                curve = await EquityRepository(session).curve(self._config.session_id)
                closed = await ClosedTradeRepository(session).list_for_session(
                    self._config.session_id
                )
                order_repository = OrderRepository(session)
                open_orders = await order_repository.list_open(session_id=self._config.session_id)
                # Which venue executions have already been folded in. Without this a restart
                # replays the whole execution window and applies every fill a second time:
                # the positions double, the closed trades duplicate, and the cash moves twice
                # for money that only moved once.
                known_fills = await order_repository.venue_fill_ids(
                    session_id=self._config.session_id
                )
        except Exception as exc:
            logger.debug("paper.restore_skipped", reason=str(exc))
            return

        # Adopt orders that were still working when the process died, so reconciliation asks
        # the venue what became of them instead of orphaning them.
        for order in open_orders:
            self._orders[order.order_id] = order

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

        cash = self._anchor_cash_to_venue(cash, positions)
        # A re-anchor may raise equity above every high the curve remembers. Leaving the old
        # peak in place would be harmless, but leaving a *higher* stale peak would report a
        # drawdown that never happened, so the peak tracks the restored equity either way.
        peak = max(peak, cash)

        self._portfolio.restore(
            cash=cash,
            positions=positions,
            peak_equity=peak,
            realized_pnl=realized,
            fees_paid=fees,
            applied_fill_ids=known_fills,
        )
        # Seed a mark for anything restored that this session does not stream. `equity()`
        # refuses to value a position it has no price for, and that refusal propagates out
        # of `record_equity` and kills the bar loop. A symbol the account still holds but
        # the universe no longer selects — a meme market that dropped out of the eligible
        # set — would do exactly that. Its own entry price is the honest stand-in until the
        # venue supplies a better one, which reconciliation does moments later.
        for position in positions:
            if not position.is_flat and self._portfolio.mark_price(position.symbol) is None:
                self._portfolio.update_mark_price(position.symbol, position.average_entry_price)

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

    def _anchor_cash_to_venue(self, restored: Decimal, positions: Sequence[Position]) -> Decimal:
        """Replace restored cash with the venue's balance when the venue is authoritative.

        A live session's money lives on the venue, not in our equity curve. Every fill this
        session made was a real fill, so the wallet balance already carries its PnL and its
        fees — which makes the balance the *current* truth about the account, not a starting
        figure to be superseded by our own bookkeeping.

        The bug this closes: the demo session's curve opened at a hardcoded 10,000 from
        before equity was read from the venue at all. Restore then reinstated that 10,000 on
        every restart, silently overriding the ~49,940 the runner had just read, so
        ``max_position_pct`` of 5% meant 500 USDT against an account holding fifty thousand.
        BTC's lot minimum alone is worth more than 500, so every BTC entry was rejected as
        "below_venue_min_quantity" — a genuine-looking rejection produced by a fictional
        account size.

        Nothing is reset and no history is discarded: the session id, its trades and its
        curve are untouched. Only the number the account is *currently* worth is corrected,
        and only when it was read from the venue.
        """
        if self._config.mode is not TradingMode.LIVE or not self._config.equity_is_authoritative:
            return restored

        authoritative = self._config.starting_equity
        if self._config.market_type is MarketType.FUTURE:
            # Futures: the wallet balance IS cash. An open perp is not an asset bought with
            # it, so nothing is deducted.
            anchored = authoritative
        else:
            # Spot accounting counts a held asset inside equity, so the capital already
            # spent on open positions must come out of cash or it is counted twice.
            deployed = sum((position.cost_basis for position in positions), ZERO)
            anchored = authoritative - deployed

        if anchored <= ZERO:
            logger.warning(
                "paper.venue_anchor_rejected",
                reason="venue balance minus deployed capital is not positive; keeping "
                "the restored cash rather than sizing against a non-positive account",
                venue_equity=str(authoritative),
                restored_cash=str(restored),
            )
            return restored

        if anchored != restored:
            logger.warning(
                "paper.cash_reanchored_to_venue",
                restored_cash=str(restored),
                venue_equity=str(authoritative),
                anchored_cash=str(anchored),
                open_positions=len(positions),
                reason="the venue balance is the account's current truth; the persisted "
                "curve had drifted from it",
            )
        return anchored

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
