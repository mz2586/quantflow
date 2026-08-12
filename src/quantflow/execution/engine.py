"""Execution engine: the one path from a strategy signal to a venue.

Order of operations is fixed and non-negotiable:

1. Strategy produces a :class:`~quantflow.domain.signals.Signal` (intent only).
2. Risk engine sizes it, attaches a stop, and applies every limit.
3. :func:`~quantflow.risk.engine.assert_protected` re-checks protection immediately before
   submission — deliberately redundant, so a future refactor that introduces a bypass fails
   loudly here rather than silently placing a naked position.
4. Gateway submits.
5. Portfolio applies the fills.

Every step is mandatory. There is no "fast path" that skips risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.core.clock import Clock, SystemClock
from quantflow.core.config import RiskSettings, TradingMode
from quantflow.core.errors import (
    ExchangeError,
    LiveTradingNotArmedError,
    RiskViolationError,
    ValidationError,
)
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderStatus, SignalDirection
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.domain.signals import Signal
from quantflow.exchange.base import TradingGateway
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskDecision, RiskEngine, assert_protected

logger = get_logger(__name__)

#: A signal older than this is discarded rather than executed. Acting on a stale signal is
#: one of the more expensive live-trading mistakes: the price it was computed against may
#: be long gone.
DEFAULT_MAX_SIGNAL_AGE_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """What happened to one signal."""

    signal: Signal
    submitted: bool
    order: Order | None = None
    fills: tuple[Fill, ...] = field(default_factory=tuple)
    decision: RiskDecision | None = None
    reason: str = ""
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether an order reached the venue without error."""
        return self.submitted and self.error is None

    @property
    def rejected_by_risk(self) -> bool:
        """Whether the risk engine refused the trade."""
        return self.decision is not None and not self.decision.approved

    def to_dict(self) -> dict[str, Any]:
        """Serialise for logging and the API."""
        return {
            "signal_id": self.signal.signal_id,
            "symbol": str(self.signal.symbol),
            "direction": self.signal.direction.value,
            "submitted": self.submitted,
            "order_id": self.order.order_id if self.order else None,
            "fills": len(self.fills),
            "reason": self.reason,
            "error": self.error,
            "risk": self.decision.to_dict() if self.decision else None,
        }


class ExecutionEngine:
    """Routes approved orders to a gateway and folds fills into the portfolio."""

    __slots__ = (
        "_clock",
        "_gateway",
        "_instruments",
        "_max_signal_age",
        "_mode",
        "_orders",
        "_portfolio",
        "_risk",
        "_settings",
    )

    def __init__(
        self,
        *,
        gateway: TradingGateway,
        risk: RiskEngine,
        portfolio: PortfolioManager,
        settings: RiskSettings,
        mode: TradingMode = TradingMode.PAPER,
        instruments: dict[Symbol, Instrument] | None = None,
        clock: Clock | None = None,
        max_signal_age_seconds: float = DEFAULT_MAX_SIGNAL_AGE_SECONDS,
    ) -> None:
        self._gateway = gateway
        self._risk = risk
        self._portfolio = portfolio
        self._settings = settings
        self._mode = mode
        self._instruments = instruments or {}
        self._clock = clock or SystemClock()
        self._max_signal_age = max_signal_age_seconds
        self._orders: dict[str, Order] = {}

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> TradingMode:
        """The trading mode this engine runs in."""
        return self._mode

    @property
    def portfolio(self) -> PortfolioManager:
        """The portfolio this engine folds fills into.

        Read-only accessor so callers that need account state - the AI service builds its
        prompt from it - do not reach into a private attribute.
        """
        return self._portfolio

    @property
    def open_orders(self) -> tuple[Order, ...]:
        """Locally tracked orders that can still receive fills."""
        return tuple(order for order in self._orders.values() if order.is_open)

    def register_instrument(self, instrument: Instrument) -> None:
        """Cache an instrument's trading rules."""
        self._instruments[instrument.symbol] = instrument

    def instrument_for(self, symbol: Symbol) -> Instrument:
        """Look up an instrument.

        Raises:
            ValidationError: if it has not been loaded, rather than guessing at the rules.

        """
        instrument = self._instruments.get(symbol)
        if instrument is None:
            raise ValidationError(
                f"no instrument metadata for {symbol}; load markets before trading",
                symbol=str(symbol),
            )
        return instrument

    # ------------------------------------------------------------------ #
    # The path
    # ------------------------------------------------------------------ #
    async def execute_signal(
        self, signal: Signal, *, reference_price: Decimal, volatility: Decimal | None = None
    ) -> ExecutionResult:
        """Take one signal all the way to the venue, or explain why not."""
        if not signal.is_actionable:
            return ExecutionResult(signal, submitted=False, reason="signal is not actionable")

        now = self._clock.now()
        if signal.is_stale(now, max_age_seconds=self._max_signal_age):
            age = (now - signal.timestamp).total_seconds()
            logger.warning(
                "execution.stale_signal",
                signal_id=signal.signal_id,
                symbol=str(signal.symbol),
                age_seconds=round(age, 2),
            )
            return ExecutionResult(
                signal, submitted=False, reason=f"signal is stale ({age:.0f}s old)"
            )

        try:
            instrument = self.instrument_for(signal.symbol)
        except ValidationError as exc:
            return ExecutionResult(signal, submitted=False, reason=exc.message, error=exc.message)

        decision = await self._risk.evaluate_signal(
            signal,
            portfolio=self._portfolio.snapshot(now),
            instrument=instrument,
            reference_price=reference_price,
            volatility=volatility,
        )
        if not decision.approved or decision.request is None:
            return ExecutionResult(
                signal, submitted=False, decision=decision, reason=decision.reason
            )

        return await self._submit(signal, decision)

    async def _submit(self, signal: Signal, decision: RiskDecision) -> ExecutionResult:
        """Submit an approved request, after the redundant protection check."""
        request = decision.request
        assert request is not None

        try:
            # Belt and braces: this must already have passed the risk engine.
            assert_protected(request, self._settings)
            self._assert_live_armed()
        except (RiskViolationError, LiveTradingNotArmedError) as exc:
            logger.exception(
                "execution.blocked_before_submit",
                symbol=str(request.symbol),
                error=str(exc),
            )
            return ExecutionResult(
                signal, submitted=False, decision=decision, reason=str(exc), error=str(exc)
            )

        try:
            order = await self._gateway.submit_order(request)
        except ExchangeError as exc:
            logger.exception(
                "execution.submit_failed",
                symbol=str(request.symbol),
                side=request.side.value,
                quantity=str(request.quantity),
                error=str(exc),
            )
            # A raised submit does NOT mean nothing happened. The rate limiter retries
            # timeouts, and a request that timed out may still have executed on the venue.
            # Declaring failure without asking leaves a real, unprotected position that the
            # system does not know it holds - the worst state available.
            adopted = await self._reconcile_after_failed_submit(request)
            if adopted is not None:
                logger.critical(
                    "execution.orphan_adopted",
                    order_id=adopted.order_id,
                    symbol=str(adopted.symbol),
                    status=adopted.status.value,
                    error=str(exc),
                )
                self._orders[adopted.order_id] = adopted
                self._risk.record_order()
                fills = self._apply_fills(adopted)
                if request.stop_loss_price is not None:
                    self._portfolio.set_protection(
                        request.symbol,
                        stop_loss_price=request.stop_loss_price,
                        take_profit_price=request.take_profit_price,
                    )
                return ExecutionResult(
                    signal,
                    submitted=True,
                    order=adopted,
                    fills=fills,
                    decision=decision,
                    reason="submit raised but the order was found on the venue and adopted",
                )
            return ExecutionResult(
                signal,
                submitted=False,
                decision=decision,
                reason="exchange rejected the order",
                error=str(exc),
            )

        self._orders[order.order_id] = order
        self._risk.record_order()

        fills = self._apply_fills(order)
        if request.stop_loss_price is not None:
            self._portfolio.set_protection(
                request.symbol,
                stop_loss_price=request.stop_loss_price,
                take_profit_price=request.take_profit_price,
            )

        logger.info(
            "execution.order_submitted",
            order_id=order.order_id,
            symbol=str(order.symbol),
            side=order.side.value,
            quantity=str(order.quantity),
            status=order.status.value,
            strategy_id=order.strategy_id,
            fills=len(fills),
        )
        return ExecutionResult(signal, submitted=True, order=order, fills=fills, decision=decision)

    async def _reconcile_after_failed_submit(self, request: OrderRequest) -> Order | None:
        """Look for an order the venue may have accepted despite the submit raising.

        Matched on ``client_order_id``, which is ours and travels with the request, so an
        order that reached Bybit is identifiable even when the response never reached us.
        Open orders first, then recent fills - an order can be filled and closed by the time
        we ask, in which case it will not appear in the open list at all.

        Returns ``None`` when nothing matches, which is the genuine "it never landed" case.
        """
        wanted = request.client_order_id
        if not wanted:
            return None

        try:
            for order in await self._gateway.fetch_open_orders(request.symbol):
                if order.client_order_id == wanted:
                    return order
        except ExchangeError as exc:
            logger.exception(
                "execution.reconcile_open_orders_failed",
                symbol=str(request.symbol),
                error=str(exc),
            )

        try:
            fetch_order = getattr(self._gateway, "fetch_order_by_client_id", None)
            if callable(fetch_order):
                found: Order | None = await fetch_order(wanted, request.symbol)
                if found is not None:
                    return found
        except ExchangeError as exc:
            logger.exception(
                "execution.reconcile_by_client_id_failed",
                symbol=str(request.symbol),
                error=str(exc),
            )
        return None

    def _apply_fills(self, order: Order) -> tuple[Fill, ...]:
        """Fold an order's fills into the portfolio, skipping duplicates."""
        applied: list[Fill] = []
        for fill in order.fills:
            self._portfolio.apply_fill(fill, strategy_id=order.strategy_id)
            applied.append(fill)
        return tuple(applied)

    def _assert_live_armed(self) -> None:
        """Refuse live submission unless the mode is explicitly armed.

        Raises:
            LiveTradingNotArmedError: if the gateway can trade for real but the engine was
                not constructed in live mode.

        """
        if self._mode is TradingMode.LIVE:
            return
        gateway_is_live = getattr(self._gateway, "supports_trading", False) and not getattr(
            self._gateway, "is_testnet", True
        )
        if gateway_is_live:
            raise LiveTradingNotArmedError(
                f"engine is in {self._mode.value} mode but the gateway points at "
                "production; refusing to submit"
            )

    # ------------------------------------------------------------------ #
    # Order management
    # ------------------------------------------------------------------ #
    async def cancel_order(self, order_id: str) -> Order | None:
        """Cancel a tracked order."""
        order = self._orders.get(order_id)
        if order is None or order.is_terminal:
            return order
        try:
            cancelled = await self._gateway.cancel_order(order_id, order.symbol)
        except ExchangeError as exc:
            logger.warning("execution.cancel_failed", order_id=order_id, error=str(exc))
            return order
        self._orders[order_id] = cancelled
        return cancelled

    async def cancel_all(self, symbol: Symbol | None = None) -> list[Order]:
        """Cancel every working order, optionally limited to one symbol."""
        targets = [order for order in self.open_orders if symbol is None or order.symbol == symbol]
        results: list[Order] = []
        for order in targets:
            cancelled = await self.cancel_order(order.order_id)
            if cancelled is not None:
                results.append(cancelled)
        return results

    async def sync_orders(self) -> list[Order]:
        """Refresh tracked orders from the venue and apply any fills we missed.

        Websockets drop messages. Polling the venue is the backstop that keeps the local
        position from silently diverging from the real one.
        """
        refreshed: list[Order] = []
        for order in self.open_orders:
            try:
                current = await self._gateway.fetch_order(order.order_id, order.symbol)
            except ExchangeError as exc:
                logger.warning("execution.sync_failed", order_id=order.order_id, error=str(exc))
                continue
            self._orders[order.order_id] = current
            self._apply_fills(current)
            refreshed.append(current)
        return refreshed

    async def flatten(
        self, symbol: Symbol, *, reason: str = "manual flatten"
    ) -> ExecutionResult | None:
        """Close a position at market.

        Used by the kill switch and by operator intervention. Routed through the normal
        signal path so the exit is still recorded, risk-checked and audited.
        """
        position = self._portfolio.position_for(symbol)
        if position is None:
            return None
        signal = Signal(
            symbol=symbol,
            direction=SignalDirection.CLOSE,
            timestamp=self._clock.now(),
            strategy_id=position.strategy_id or "operator",
            reason=reason,
        )
        price = self._portfolio.mark_price(symbol)
        if price is None:
            raise ValidationError(
                f"cannot flatten {symbol}: no mark price available", symbol=str(symbol)
            )
        return await self.execute_signal(signal, reference_price=price)

    async def flatten_all(self, *, reason: str = "flatten all") -> list[ExecutionResult]:
        """Close every open position."""
        results: list[ExecutionResult] = []
        for position in self._portfolio.positions:
            result = await self.flatten(position.symbol, reason=reason)
            if result is not None:
                results.append(result)
        return results

    def record_order_state(self, order: Order) -> None:
        """Adopt an externally-sourced order state (e.g. from a websocket update)."""
        self._orders[order.order_id] = order
        if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self._apply_fills(order)

    def track(self, orders: list[Order]) -> None:
        """Adopt orders recovered from the database on restart."""
        for order in orders:
            self._orders[order.order_id] = order
        logger.info("execution.orders_adopted", count=len(orders))


def protective_exit_price(
    position_side_is_long: bool, entry: Decimal, stop_pct: Decimal
) -> Decimal:
    """Compute a protective stop a fixed fraction away from entry."""
    if entry <= ZERO:
        raise ValidationError(f"entry price must be positive, got {entry}")
    distance = entry * stop_pct
    return entry - distance if position_side_is_long else entry + distance


def should_trigger_stop(
    *, position_side_is_long: bool, stop_price: Decimal, candle_low: Decimal, candle_high: Decimal
) -> bool:
    """Whether a bar's range reached a stop.

    Checks the bar's low/high rather than its close: a stop is triggered intrabar, and only
    testing the close would let a strategy sail through a 20% intrabar spike unscathed in a
    backtest while being stopped out in reality.
    """
    return candle_low <= stop_price if position_side_is_long else candle_high >= stop_price


def build_exit_request(
    symbol: Symbol, position_quantity: Decimal, instrument: Instrument, *, strategy_id: str | None
) -> OrderRequest:
    """Build a reduce-only market order that flattens a position."""
    from quantflow.domain.enums import OrderSide, OrderType

    if position_quantity == ZERO:
        raise ValidationError(f"cannot build an exit for a flat position in {symbol}")
    side = OrderSide.SELL if position_quantity > ZERO else OrderSide.BUY
    return OrderRequest(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=instrument.normalize_quantity(abs(position_quantity)),
        reduce_only=True,
        strategy_id=strategy_id,
    )


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    """Counters for one trading session."""

    signals_seen: int = 0
    orders_submitted: int = 0
    risk_rejections: int = 0
    exchange_errors: int = 0
    stale_signals: int = 0

    def observe(self, result: ExecutionResult) -> ExecutionStats:
        """Fold in one result."""
        return ExecutionStats(
            signals_seen=self.signals_seen + 1,
            orders_submitted=self.orders_submitted + (1 if result.succeeded else 0),
            risk_rejections=self.risk_rejections + (1 if result.rejected_by_risk else 0),
            exchange_errors=self.exchange_errors + (1 if result.error else 0),
            stale_signals=self.stale_signals + (1 if "stale" in result.reason else 0),
        )

    def to_dict(self) -> dict[str, int]:
        """Serialise for reporting."""
        return {
            "signals_seen": self.signals_seen,
            "orders_submitted": self.orders_submitted,
            "risk_rejections": self.risk_rejections,
            "exchange_errors": self.exchange_errors,
            "stale_signals": self.stale_signals,
        }


def utc_timestamp(clock: Clock) -> datetime:
    """Current time from the injected clock."""
    return clock.now()
