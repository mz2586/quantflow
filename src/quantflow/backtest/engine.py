"""Event-driven backtester.

Two properties define this engine, and both are enforced structurally rather than by
convention:

**No look-ahead.** On bar *i*, the strategy sees `history[0..i]` and nothing else. Orders
generated on bar *i* are matched against bar *i+1*, because a decision made from bar *i*'s
close cannot be executed until the next bar opens. A backtester that fills at the same
bar's close manufactures returns that do not exist.

**The same code path as live.** Signals go through the identical `RiskEngine` and
`PortfolioManager` that live trading uses. If a backtest and a live session disagree, the
difference is the venue, not the machinery — which is the only way a backtest result is
evidence about anything.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.core.clock import FrozenClock
from quantflow.core.config import MarketType, RiskSettings, TradingMode
from quantflow.core.errors import BacktestError, InsufficientDataError
from quantflow.core.logging import get_logger, log_context
from quantflow.core.precision import ZERO
from quantflow.domain.enums import (
    LiquidityRole,
    MarketRegime,
    OrderSide,
    PositionSide,
    RunStatus,
    SignalDirection,
    Timeframe,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import ClosedTrade
from quantflow.domain.signals import Signal
from quantflow.exchange.simulator import (
    FeeModel,
    SimulatedBroker,
    SlippageModel,
    VolumeShareSlippage,
)
from quantflow.portfolio.funding import FundingSchedule
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine, assert_protected
from quantflow.strategy.base import Strategy, StrategyContext
from quantflow.strategy.indicators import atr

logger = get_logger(__name__)

#: Cap on the trailing history handed to a strategy. Without it, a multi-year 1m backtest
#: passes a growing list on every bar and degrades to O(n^2).
MAX_HISTORY_BARS = 5_000

#: ATR window used to feed volatility-scaled position sizing.
SIZING_ATR_PERIOD = 14


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Everything needed to run a backtest."""

    symbols: tuple[Symbol, ...]
    timeframe: Timeframe
    starting_equity: Decimal = Decimal("10000")
    base_currency: str = "USDT"
    risk: RiskSettings = field(default_factory=RiskSettings)
    slippage: SlippageModel = field(default_factory=VolumeShareSlippage)
    fees: FeeModel = field(default_factory=FeeModel)
    warmup_bars: int | None = None
    """Override the strategy's own warm-up. Rarely needed."""
    max_history_bars: int = MAX_HISTORY_BARS
    reject_oversized_orders: bool = True
    #: Same accounting as paper and live, so a backtest cannot describe a spot account while
    #: the strategy trades perps.
    market_type: MarketType = MarketType.SPOT
    #: Pinned to 1x so the backtest cannot quietly assume leverage the live account does not
    #: use. Live reconciles to the venue's value; a backtest has no venue to ask.
    leverage: Decimal = Decimal("1")
    #: Historical funding rates per symbol, keyed by settlement time.
    funding: dict[Symbol, FundingSchedule] = field(default_factory=dict)
    risk_free_rate: float = 0.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The outcome of one backtest run."""

    run_id: str
    status: RunStatus
    config: BacktestConfig
    strategy_id: str
    strategy_params: dict[str, Any]
    equity_curve: tuple[EquityPoint, ...] = field(default_factory=tuple)
    closed_trades: tuple[ClosedTrade, ...] = field(default_factory=tuple)
    orders: tuple[Order, ...] = field(default_factory=tuple)
    signals: tuple[Signal, ...] = field(default_factory=tuple)
    rejected_signals: tuple[tuple[Signal, str], ...] = field(default_factory=tuple)
    bars_processed: int = 0
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the run completed."""
        return self.status is RunStatus.COMPLETED

    @property
    def final_equity(self) -> Decimal:
        """Equity at the end of the run."""
        return self.equity_curve[-1].equity if self.equity_curve else self.config.starting_equity

    def metrics(self) -> Any:
        """Compute the performance metrics for this run."""
        from quantflow.backtest.metrics import compute_metrics

        fees = sum((trade.fees for trade in self.closed_trades), ZERO)
        return compute_metrics(
            curve=self.equity_curve,
            trades=self.closed_trades,
            starting_equity=self.config.starting_equity,
            timeframe=self.config.timeframe,
            total_fees=fees,
            risk_free_rate=self.config.risk_free_rate,
        )


class BacktestEngine:
    """Replays historical bars through the live strategy → risk → execution path."""

    __slots__ = (
        "_broker",
        "_clock",
        "_config",
        "_instruments",
        "_orders",
        "_pending_protection",
        "_portfolio",
        "_rejected",
        "_risk",
        "_signals",
        "_strategy",
    )

    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig,
        instruments: dict[Symbol, Instrument],
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._instruments = instruments
        # Virtual time: the engine advances the clock bar by bar, so every timestamp in
        # the result is historical rather than wall-clock.
        self._clock = FrozenClock()
        self._portfolio = PortfolioManager(
            market_type=config.market_type,
            leverage=config.leverage,
            base_currency=config.base_currency,
            starting_equity=config.starting_equity,
            clock=self._clock,
        )
        self._risk = RiskEngine(config.risk, clock=self._clock)
        self._broker = SimulatedBroker(
            instruments=instruments,
            slippage=config.slippage,
            fees=config.fees,
            reject_oversized=config.reject_oversized_orders,
        )
        self._orders: list[Order] = []
        #: Stop/target levels keyed by order id, applied once that order actually fills.
        #: They cannot be attached at placement time — the position does not exist yet.
        self._pending_protection: dict[str, tuple[Decimal | None, Decimal | None]] = {}
        self._signals: list[Signal] = []
        self._rejected: list[tuple[Signal, str]] = []

    @property
    def portfolio(self) -> PortfolioManager:
        """The portfolio being simulated."""
        return self._portfolio

    def _funding_rate_for(self, symbol: Symbol, settled_at: datetime) -> Decimal | None:
        """The historical rate that applied at a settlement, or ``None`` if unknown.

        Unknown means no charge. A backtest that guessed a rate would be inventing a cost,
        which is exactly as dishonest as ignoring a real one.
        """
        schedule = self._config.funding.get(symbol)
        return schedule.rate_at(settled_at) if schedule is not None else None

    async def run(self, data: dict[Symbol, Sequence[Candle]]) -> BacktestResult:
        """Replay the supplied bars.

        Args:
            data: Closed candles per symbol, oldest first.

        Raises:
            BacktestError: if the data is unusable (empty, misaligned, or too short).

        """
        started = time.perf_counter()
        self._validate(data)

        timeline = self._build_timeline(data)
        warmup = self._config.warmup_bars or self._strategy.warmup_bars
        by_symbol = {symbol: list(candles) for symbol, candles in data.items()}
        indexes: dict[Symbol, int] = dict.fromkeys(by_symbol, 0)

        self._strategy.on_start(tuple(by_symbol))
        self._clock.set(timeline[0])

        with log_context(run_id=self._config.run_id, strategy_id=self._strategy.strategy_id):
            logger.info(
                "backtest.started",
                symbols=[str(symbol) for symbol in by_symbol],
                timeframe=self._config.timeframe.value,
                bars=len(timeline),
                warmup_bars=warmup,
                starting_equity=str(self._config.starting_equity),
            )
            try:
                bars = await self._replay(timeline, by_symbol, indexes, warmup)
            except Exception as exc:
                logger.exception("backtest.failed", error=str(exc))
                return BacktestResult(
                    run_id=self._config.run_id,
                    status=RunStatus.FAILED,
                    config=self._config,
                    strategy_id=self._strategy.strategy_id,
                    strategy_params=self._strategy.params.to_dict(),
                    equity_curve=self._portfolio.equity_curve,
                    closed_trades=self._portfolio.closed_trades,
                    orders=tuple(self._orders),
                    duration_seconds=time.perf_counter() - started,
                    error=str(exc),
                )

            self._strategy.on_finish()
            duration = time.perf_counter() - started
            logger.info(
                "backtest.finished",
                bars=bars,
                trades=len(self._portfolio.closed_trades),
                final_equity=str(self._portfolio.equity()),
                duration_seconds=round(duration, 3),
            )

        return BacktestResult(
            run_id=self._config.run_id,
            status=RunStatus.COMPLETED,
            config=self._config,
            strategy_id=self._strategy.strategy_id,
            strategy_params=self._strategy.params.to_dict(),
            equity_curve=self._portfolio.equity_curve,
            closed_trades=self._portfolio.closed_trades,
            orders=tuple(self._orders),
            signals=tuple(self._signals),
            rejected_signals=tuple(self._rejected),
            bars_processed=bars,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------ #
    # The replay loop
    # ------------------------------------------------------------------ #
    async def _replay(
        self,
        timeline: list[datetime],
        by_symbol: dict[Symbol, list[Candle]],
        indexes: dict[Symbol, int],
        warmup: int,
    ) -> int:
        """Walk the timeline one bar at a time.

        Per bar, in this order:
          1. Match orders resting from the previous bar against **this** bar.
          2. Apply the resulting fills to the portfolio.
          3. Check stops and targets against this bar's range.
          4. Mark to this bar's close, sample equity.
          5. Ask the strategy for a decision using history up to and including this bar.
          6. Risk-check and place the resulting order — to be matched on the *next* bar.
        """
        processed = 0

        for moment in timeline:
            self._clock.set(moment)
            current: dict[Symbol, Candle] = {}

            for symbol, candles in by_symbol.items():
                index = indexes[symbol]
                if index >= len(candles) or candles[index].open_time != moment:
                    continue
                current[symbol] = candles[index]
                indexes[symbol] = index + 1

            if not current:
                continue

            # 1-2. Orders placed on the previous bar are matched against this one.
            for candle in current.values():
                for order, fill in self._broker.process_candle(candle):
                    self._record_order(order)
                    if order.status.is_terminal and not order.fills:
                        continue  # a rejection carries a placeholder fill
                    _, closed = self._portfolio.apply_fill(fill, strategy_id=order.strategy_id)
                    self._record_trade_results(closed)
                    self._attach_protection(order)

            # 3. Protective exits, evaluated against the bar's range rather than its close.
            for symbol, candle in current.items():
                await self._check_protective_exits(symbol, candle)

            # 4. Mark and sample.
            for symbol, candle in current.items():
                self._portfolio.update_mark_price(symbol, candle.close)
            # Funding settles before the equity sample so the curve reflects it.
            if self._config.market_type is MarketType.FUTURE and self._config.funding:
                self._portfolio.settle_funding(moment, rate_for=self._funding_rate_for)
            self._portfolio.record_equity(moment)
            processed += 1

            # 5-6. Decide, then place for the *next* bar.
            for symbol, candle in current.items():
                history = by_symbol[symbol][: indexes[symbol]]
                if len(history) < warmup:
                    continue
                await self._decide(symbol, history, candle, moment)

        return processed

    async def _decide(
        self,
        symbol: Symbol,
        history: list[Candle],
        candle: Candle,
        moment: datetime,
    ) -> None:
        """Ask the strategy for a decision and route any signal through risk."""
        window = history[-self._config.max_history_bars :]
        context = StrategyContext(
            symbol=symbol,
            timeframe=self._config.timeframe,
            history=CandleSeries(window),
            # The decision is made as of the bar's *close*, which is when the information
            # became available.
            now=candle.close_time,
            portfolio=self._portfolio.snapshot(moment),
            position=self._portfolio.position_for(symbol),
            regime=MarketRegime.UNKNOWN,
        )

        signal = self._strategy.evaluate(context)
        if not signal.is_actionable:
            return
        self._signals.append(signal)

        volatility = atr(window, SIZING_ATR_PERIOD)[-1] if len(window) > SIZING_ATR_PERIOD else None
        decision = await self._risk.evaluate_signal(
            signal,
            portfolio=self._portfolio.snapshot(moment),
            instrument=self._instruments[symbol],
            reference_price=candle.close,
            volatility=volatility,
        )
        if not decision.approved or decision.request is None:
            self._rejected.append((signal, decision.reason))
            return

        self._place(decision.request, candle.close)

    def _place(self, request: OrderRequest, reference_price: Decimal) -> None:
        """Submit to the simulated venue, after the same protection assert live uses."""
        assert_protected(request, self._config.risk)
        order = self._broker.submit(request, now=self._clock.now(), reference_price=reference_price)
        self._record_order(order)
        self._risk.record_order()
        if request.stop_loss_price is not None or request.take_profit_price is not None:
            self._pending_protection[order.order_id] = (
                request.stop_loss_price,
                request.take_profit_price,
            )

    def _record_trade_results(self, trades: Sequence[ClosedTrade]) -> None:
        """Feed closed trades to the risk engine's loss-streak tracker.

        The cooldown rule is only as good as what it is told: without this the streak
        counter never advances and the rule silently never fires.
        """
        for trade in trades:
            self._risk.record_trade_result(
                trade.net_pnl, closed_at=trade.exit_time, symbol=trade.symbol
            )

    def _attach_protection(self, order: Order) -> None:
        """Apply an order's protective levels to the position it just opened.

        Deferred until the fill lands: at placement time the position does not exist, and
        setting protection on a flat symbol is a no-op — which is exactly how a position
        ends up running without the stop its own order specified.
        """
        levels = self._pending_protection.get(order.order_id)
        if levels is None:
            return
        stop, target = levels
        if self._portfolio.position_for(order.symbol) is None:
            return
        self._portfolio.set_protection(order.symbol, stop_loss_price=stop, take_profit_price=target)
        if order.is_terminal:
            del self._pending_protection[order.order_id]

    async def _check_protective_exits(self, symbol: Symbol, candle: Candle) -> None:
        """Close a position whose stop or target was reached inside this bar.

        Uses the bar's low and high, not its close: a stop is triggered intrabar, and
        checking only the close would let a strategy sail through a violent spike in a
        backtest while being stopped out in reality.

        When both the stop and the target are inside the same bar, the **stop wins**. The
        bar does not record which came first, and assuming the favourable one is how a
        backtest quietly inflates its win rate.
        """
        position = self._portfolio.position_for(symbol)
        if position is None:
            return

        stop_hit = position.is_stop_breached(
            candle.low if position.side is PositionSide.LONG else candle.high
        )
        target_hit = position.is_target_reached(
            candle.high if position.side is PositionSide.LONG else candle.low
        )
        if not stop_hit and not target_hit:
            return

        exit_price = position.stop_loss_price if stop_hit else position.take_profit_price
        if exit_price is None:  # pragma: no cover — guarded by the checks above
            return

        instrument = self._instruments[symbol]
        closing_side = position.closing_side()
        assert closing_side is not None
        quantity = instrument.normalize_quantity(position.absolute_quantity)
        if quantity <= ZERO:
            return

        role_fee = self._config.fees.compute(
            instrument,
            quantity=quantity,
            price=exit_price,
            role=LiquidityRole.TAKER,
        )
        fill = Fill(
            fill_id=f"protective-{symbol.concatenated}-{candle.open_time.isoformat()}",
            order_id=f"protective-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=closing_side,
            quantity=quantity,
            price=exit_price,
            fee=role_fee,
            fee_currency=symbol.quote,
            timestamp=candle.close_time,
        )
        _, closed = self._portfolio.apply_fill(fill, strategy_id=position.strategy_id)
        self._record_trade_results(closed)
        logger.debug(
            "backtest.protective_exit",
            symbol=str(symbol),
            kind="stop" if stop_hit else "target",
            price=str(exit_price),
        )

    def _record_order(self, order: Order) -> None:
        """Track an order, replacing any earlier state for the same id."""
        for index, existing in enumerate(self._orders):
            if existing.order_id == order.order_id:
                self._orders[index] = order
                return
        self._orders.append(order)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate(self, data: dict[Symbol, Sequence[Candle]]) -> None:
        """Reject unusable data before wasting a run on it."""
        if not data:
            raise BacktestError("no market data supplied")

        missing = [symbol for symbol in data if symbol not in self._instruments]
        if missing:
            raise BacktestError(f"no instrument metadata for: {', '.join(str(s) for s in missing)}")

        warmup = self._config.warmup_bars or self._strategy.warmup_bars
        for symbol, candles in data.items():
            if not candles:
                raise BacktestError(f"no candles for {symbol}", symbol=str(symbol))
            if len(candles) <= warmup:
                raise InsufficientDataError(
                    f"{symbol} has {len(candles)} bars but the strategy needs "
                    f"more than {warmup} to warm up",
                    symbol=str(symbol),
                    available=len(candles),
                    required=warmup + 1,
                )
            wrong = [c for c in candles if c.timeframe is not self._config.timeframe]
            if wrong:
                raise BacktestError(
                    f"{symbol} contains {len(wrong)} bars that are not "
                    f"{self._config.timeframe.value}",
                    symbol=str(symbol),
                )
            # Constructing the series validates ordering and rejects duplicates.
            CandleSeries(candles)

    def _build_timeline(self, data: dict[Symbol, Sequence[Candle]]) -> list[datetime]:
        """The sorted union of every symbol's bar times.

        A union rather than an intersection: symbols legitimately have gaps, and dropping a
        bar because one unrelated symbol lacks it would silently change the other's result.
        """
        moments: set[datetime] = set()
        for candles in data.values():
            moments.update(candle.open_time for candle in candles)
        return sorted(moments)


async def run_backtest(
    strategy: Strategy,
    data: dict[Symbol, Sequence[Candle]],
    *,
    instruments: dict[Symbol, Instrument],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Convenience wrapper: build an engine and run it."""
    effective = config or BacktestConfig(
        symbols=tuple(data), timeframe=next(iter(data.values()))[0].timeframe
    )
    return await BacktestEngine(strategy, effective, instruments).run(data)


def signal_summary(result: BacktestResult) -> dict[str, int]:
    """Counts of signals produced, acted on and rejected.

    A run with hundreds of signals and zero orders means the risk configuration is
    blocking everything — a common and otherwise invisible misconfiguration.
    """
    directions: dict[str, int] = {}
    for signal in result.signals:
        directions[signal.direction.value] = directions.get(signal.direction.value, 0) + 1
    return {
        "signals": len(result.signals),
        "orders": len(result.orders),
        "rejected": len(result.rejected_signals),
        **{f"signal_{key}": value for key, value in directions.items()},
    }


def rejection_reasons(result: BacktestResult) -> dict[str, int]:
    """Histogram of why signals were refused, for diagnosing a dead run."""
    reasons: dict[str, int] = {}
    for _, reason in result.rejected_signals:
        key = reason.split(";")[0].strip()[:80]
        reasons[key] = reasons.get(key, 0) + 1
    return dict(sorted(reasons.items(), key=lambda item: -item[1]))


def entry_and_exit_counts(result: BacktestResult) -> tuple[int, int]:
    """How many orders opened versus closed exposure."""
    entries = sum(1 for order in result.orders if not order.reduce_only)
    exits = sum(1 for order in result.orders if order.reduce_only)
    return entries, exits


def realised_side_breakdown(result: BacktestResult) -> dict[str, int]:
    """Trade counts by direction."""
    counts = {"long": 0, "short": 0}
    for trade in result.closed_trades:
        if trade.side is PositionSide.LONG:
            counts["long"] += 1
        elif trade.side is PositionSide.SHORT:
            counts["short"] += 1
    return counts


def order_side_counts(result: BacktestResult) -> dict[str, int]:
    """Order counts by side."""
    counts = {OrderSide.BUY.value: 0, OrderSide.SELL.value: 0}
    for order in result.orders:
        counts[order.side.value] += 1
    return counts


def assert_no_lookahead(result: BacktestResult) -> None:
    """Assert every order was placed after the bar that justified it.

    A regression guard: if a refactor ever lets a fill land on the same bar as its signal,
    this fails loudly rather than silently improving every reported return.

    Raises:
        BacktestError: if any fill precedes the signal that produced it.

    """
    signal_times = {signal.signal_id: signal.timestamp for signal in result.signals}
    for order in result.orders:
        if order.signal_id is None:
            continue
        signal_time = signal_times.get(order.signal_id)
        if signal_time is None:
            continue
        for fill in order.fills:
            if fill.timestamp < signal_time:
                raise BacktestError(
                    f"look-ahead detected: fill at {fill.timestamp.isoformat()} precedes "
                    f"its signal at {signal_time.isoformat()}",
                    order_id=order.order_id,
                )


def mode_for_backtest() -> TradingMode:
    """The trading mode a backtest engine reports."""
    return TradingMode.BACKTEST


def direction_of(signal: Signal) -> SignalDirection:
    """Convenience accessor used by report templates."""
    return signal.direction
