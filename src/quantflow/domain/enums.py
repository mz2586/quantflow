"""Domain enumerations.

These are exchange-agnostic. Venue-specific spellings are translated in
``quantflow.exchange.binance.mapping``.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from typing import Final


class OrderSide(StrEnum):
    """Direction of an order."""

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> OrderSide:
        """The side that closes a position opened on this side."""
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY

    @property
    def sign(self) -> int:
        """``+1`` for buys, ``-1`` for sells — used in signed-quantity maths."""
        return 1 if self is OrderSide.BUY else -1


class PositionSide(StrEnum):
    """Direction of an open position."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    @property
    def sign(self) -> int:
        """``+1`` long, ``-1`` short, ``0`` flat."""
        if self is PositionSide.LONG:
            return 1
        if self is PositionSide.SHORT:
            return -1
        return 0

    @classmethod
    def from_signed_quantity(cls, quantity: object) -> PositionSide:
        """Derive the side from a signed quantity."""
        value = quantity  # Decimal or int; comparison is all we need
        if value > 0:  # type: ignore[operator]
            return cls.LONG
        if value < 0:  # type: ignore[operator]
            return cls.SHORT
        return cls.FLAT

    @property
    def entry_side(self) -> OrderSide:
        """The order side that opens this position."""
        if self is PositionSide.SHORT:
            return OrderSide.SELL
        return OrderSide.BUY

    @property
    def exit_side(self) -> OrderSide:
        """The order side that closes this position."""
        return self.entry_side.opposite


class OrderType(StrEnum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"

    @property
    def requires_price(self) -> bool:
        """Whether a limit price is mandatory."""
        return self in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT)

    @property
    def requires_trigger_price(self) -> bool:
        """Whether a stop/trigger price is mandatory."""
        return self in (
            OrderType.STOP_MARKET,
            OrderType.STOP_LIMIT,
            OrderType.TAKE_PROFIT_MARKET,
            OrderType.TAKE_PROFIT_LIMIT,
        )

    @property
    def is_market(self) -> bool:
        """Whether the order executes immediately at the prevailing price."""
        return self in (
            OrderType.MARKET,
            OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT_MARKET,
        )


class TimeInForce(StrEnum):
    """Order lifetime instruction."""

    GTC = "gtc"
    """Good till cancelled."""
    IOC = "ioc"
    """Immediate or cancel."""
    FOK = "fok"
    """Fill or kill."""
    GTD = "gtd"
    """Good till date."""


class OrderStatus(StrEnum):
    """Lifecycle state of an order in the OMS."""

    PENDING_NEW = "pending_new"
    """Created locally, not yet acknowledged by the venue."""
    NEW = "new"
    """Acknowledged and live on the venue."""
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        """Whether no further transitions are possible."""
        return self in TERMINAL_ORDER_STATUSES

    @property
    def is_open(self) -> bool:
        """Whether the order can still receive fills."""
        return self in OPEN_ORDER_STATUSES


TERMINAL_ORDER_STATUSES: Final[frozenset[OrderStatus]] = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
)

OPEN_ORDER_STATUSES: Final[frozenset[OrderStatus]] = frozenset(
    {
        OrderStatus.PENDING_NEW,
        OrderStatus.NEW,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
    }
)


class SignalDirection(StrEnum):
    """What a strategy wants to happen."""

    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"

    @property
    def is_actionable(self) -> bool:
        """Whether the signal should produce an order."""
        return self is not SignalDirection.HOLD


class Timeframe(StrEnum):
    """Candle interval. Values match Binance/CCXT interval strings."""

    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"

    @property
    def delta(self) -> timedelta:
        """Interval as a :class:`datetime.timedelta`."""
        return _TIMEFRAME_DELTAS[self]

    @property
    def seconds(self) -> int:
        """Interval length in whole seconds."""
        return int(self.delta.total_seconds())

    @property
    def milliseconds(self) -> int:
        """Interval length in milliseconds (the unit Binance uses)."""
        return self.seconds * 1000

    @property
    def periods_per_year(self) -> float:
        """Number of bars in a 365-day crypto trading year.

        Crypto markets trade continuously, so there is no 252-day convention here.
        Used to annualise Sharpe and return metrics.
        """
        return 365.0 * 24 * 3600 / self.seconds

    @classmethod
    def parse(cls, value: str) -> Timeframe:
        """Parse a timeframe string, raising a domain error on unknown values."""
        from quantflow.core.errors import ValidationError

        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValidationError(
                f"unsupported timeframe {value!r}; supported: {supported}"
            ) from exc


_TIMEFRAME_DELTAS: Final[dict[Timeframe, timedelta]] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M3: timedelta(minutes=3),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.M30: timedelta(minutes=30),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H2: timedelta(hours=2),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.H6: timedelta(hours=6),
    Timeframe.H8: timedelta(hours=8),
    Timeframe.H12: timedelta(hours=12),
    Timeframe.D1: timedelta(days=1),
    Timeframe.D3: timedelta(days=3),
    Timeframe.W1: timedelta(weeks=1),
}


class LiquidityRole(StrEnum):
    """Whether a fill added or removed liquidity — determines the fee tier."""

    MAKER = "maker"
    TAKER = "taker"


class RunStatus(StrEnum):
    """Lifecycle of a backtest, optimisation or trading session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether the run has finished."""
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


class MarketRegime(StrEnum):
    """Coarse market state produced by the AI regime detector."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"
