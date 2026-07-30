"""Market-data value objects: candles, trades, tickers, order books."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Self

from quantflow.core.clock import from_epoch_ms, to_epoch_ms
from quantflow.core.errors import MarketDataError, ValidationError
from quantflow.core.precision import ZERO, safe_divide, to_decimal
from quantflow.domain.enums import OrderSide, Timeframe
from quantflow.domain.instruments import Symbol

#: CCXT returns OHLCV rows as [timestamp_ms, open, high, low, close, volume].
CCXT_OHLCV_COLUMNS = 6


@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV bar.

    ``open_time`` is the bar's opening instant; the bar covers
    ``[open_time, open_time + timeframe)``. A candle is only safe for a strategy to act on
    once :meth:`is_closed` is true — acting on a forming bar is look-ahead bias.
    """

    symbol: Symbol
    timeframe: Timeframe
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal = ZERO
    trades: int = 0

    def __post_init__(self) -> None:
        """Validate OHLC ordering and non-negativity."""
        if self.open_time.tzinfo is None:
            raise ValidationError("candle open_time must be timezone-aware UTC")
        if self.high < self.low:
            raise ValidationError(f"candle high {self.high} below low {self.low} for {self.symbol}")
        for name in ("open", "close"):
            price: Decimal = getattr(self, name)
            if not (self.low <= price <= self.high):
                raise ValidationError(
                    f"candle {name} {price} outside [low={self.low}, high={self.high}] "
                    f"for {self.symbol} at {self.open_time.isoformat()}"
                )
        if self.low < ZERO:
            raise ValidationError(f"negative price in candle for {self.symbol}")
        if self.volume < ZERO or self.quote_volume < ZERO:
            raise ValidationError(f"negative volume in candle for {self.symbol}")
        if self.trades < 0:
            raise ValidationError(f"negative trade count in candle for {self.symbol}")

    @property
    def close_time(self) -> datetime:
        """The instant at which the bar closes (exclusive upper bound)."""
        return self.open_time + self.timeframe.delta

    def is_closed(self, now: datetime) -> bool:
        """Whether the bar has completed as of ``now``."""
        return now >= self.close_time

    @property
    def typical_price(self) -> Decimal:
        """``(high + low + close) / 3`` — the HLC3 reference price."""
        return (self.high + self.low + self.close) / Decimal(3)

    @property
    def median_price(self) -> Decimal:
        """``(high + low) / 2``."""
        return (self.high + self.low) / Decimal(2)

    @property
    def range(self) -> Decimal:
        """High-minus-low."""
        return self.high - self.low

    @property
    def body(self) -> Decimal:
        """Signed close-minus-open."""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        """Whether the bar closed above its open."""
        return self.close > self.open

    @property
    def vwap(self) -> Decimal:
        """Volume-weighted average price, falling back to the typical price."""
        if ZERO in (self.volume, self.quote_volume):
            return self.typical_price
        return self.quote_volume / self.volume

    @property
    def return_pct(self) -> Decimal:
        """Fractional bar return, open to close."""
        return safe_divide(self.close - self.open, self.open)

    @classmethod
    def from_ccxt(cls, symbol: Symbol, timeframe: Timeframe, row: Sequence[object]) -> Self:
        """Build a candle from a CCXT ``fetch_ohlcv`` row.

        CCXT rows are ``[timestamp_ms, open, high, low, close, volume]`` with float values.
        """
        if len(row) < CCXT_OHLCV_COLUMNS:
            raise MarketDataError(f"malformed OHLCV row for {symbol}: {row!r}")
        timestamp = row[0]
        if not isinstance(timestamp, (int, float)):
            raise MarketDataError(f"non-numeric OHLCV timestamp for {symbol}: {timestamp!r}")
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            open_time=from_epoch_ms(int(timestamp)),
            open=to_decimal(row[1]),  # type: ignore[arg-type]
            high=to_decimal(row[2]),  # type: ignore[arg-type]
            low=to_decimal(row[3]),  # type: ignore[arg-type]
            close=to_decimal(row[4]),  # type: ignore[arg-type]
            volume=to_decimal(row[5]),  # type: ignore[arg-type]
        )

    def to_row(self) -> tuple[int, str, str, str, str, str]:
        """Serialise to a compact, lossless row for Parquet/JSON transport."""
        return (
            to_epoch_ms(self.open_time),
            str(self.open),
            str(self.high),
            str(self.low),
            str(self.close),
            str(self.volume),
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """A single executed public trade from the exchange tape."""

    symbol: Symbol
    trade_id: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    side: OrderSide

    def __post_init__(self) -> None:
        """Validate the trade."""
        if self.timestamp.tzinfo is None:
            raise ValidationError("trade timestamp must be timezone-aware UTC")
        if self.price <= ZERO:
            raise ValidationError(f"trade price must be positive for {self.symbol}")
        if self.quantity <= ZERO:
            raise ValidationError(f"trade quantity must be positive for {self.symbol}")

    @property
    def notional(self) -> Decimal:
        """Quote-currency value of the trade."""
        return self.price * self.quantity


@dataclass(frozen=True, slots=True)
class Ticker:
    """Best bid/ask plus last price for a symbol."""

    symbol: Symbol
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal
    bid_volume: Decimal = ZERO
    ask_volume: Decimal = ZERO

    def __post_init__(self) -> None:
        """Validate the quote."""
        if self.timestamp.tzinfo is None:
            raise ValidationError("ticker timestamp must be timezone-aware UTC")
        if self.bid <= ZERO or self.ask <= ZERO or self.last <= ZERO:
            raise ValidationError(f"ticker prices must be positive for {self.symbol}")
        if self.ask < self.bid:
            raise ValidationError(
                f"crossed ticker for {self.symbol}: bid {self.bid} > ask {self.ask}"
            )

    @property
    def mid(self) -> Decimal:
        """Mid price."""
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread(self) -> Decimal:
        """Absolute spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> Decimal:
        """Spread as a fraction of the mid price."""
        return safe_divide(self.spread, self.mid)

    def price_for(self, side: OrderSide) -> Decimal:
        """The price a taker would receive on ``side``."""
        return self.ask if side is OrderSide.BUY else self.bid


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """A single price level."""

    price: Decimal
    quantity: Decimal


@dataclass(frozen=True, slots=True)
class OrderBook:
    """L2 order-book snapshot, best-first on both sides."""

    symbol: Symbol
    timestamp: datetime
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]

    def __post_init__(self) -> None:
        """Validate level ordering."""
        if self.timestamp.tzinfo is None:
            raise ValidationError("order book timestamp must be timezone-aware UTC")
        if any(a.price <= b.price for a, b in zip(self.bids, self.bids[1:], strict=False)):
            raise ValidationError(f"bids must descend for {self.symbol}")
        if any(a.price >= b.price for a, b in zip(self.asks, self.asks[1:], strict=False)):
            raise ValidationError(f"asks must ascend for {self.symbol}")

    @property
    def best_bid(self) -> Decimal | None:
        """Highest bid price, if any."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Lowest ask price, if any."""
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        """Mid price, if both sides are populated."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / Decimal(2)

    def sweep_cost(self, side: OrderSide, quantity: Decimal) -> tuple[Decimal, Decimal]:
        """Simulate a market order walking the book.

        Returns:
            ``(average_price, filled_quantity)``. ``filled_quantity`` is below ``quantity``
            when the visible book is too thin, which callers must treat as a partial fill
            rather than silently assuming completion.

        """
        levels = self.asks if side is OrderSide.BUY else self.bids
        remaining = quantity
        cost = ZERO
        for level in levels:
            if remaining <= ZERO:
                break
            take = min(remaining, level.quantity)
            cost += take * level.price
            remaining -= take
        filled = quantity - remaining
        if filled == ZERO:
            return ZERO, ZERO
        return cost / filled, filled


class CandleSeries:
    """An immutable, gap-checked, chronologically sorted sequence of candles.

    Provides the read-only window a strategy sees. Construction validates that every candle
    shares the same symbol and timeframe and that timestamps strictly increase, which turns
    a whole class of silent data bugs into loud failures at the boundary.
    """

    __slots__ = ("_candles", "_open_times", "_symbol", "_timeframe")

    def __init__(self, candles: Iterable[Candle]) -> None:
        ordered = tuple(candles)
        if not ordered:
            raise MarketDataError("cannot build a CandleSeries from an empty sequence")

        first = ordered[0]
        self._symbol = first.symbol
        self._timeframe = first.timeframe

        previous: datetime | None = None
        for candle in ordered:
            if candle.symbol != self._symbol:
                raise MarketDataError(
                    f"mixed symbols in series: {self._symbol} and {candle.symbol}"
                )
            if candle.timeframe != self._timeframe:
                raise MarketDataError(
                    f"mixed timeframes in series: {self._timeframe} and {candle.timeframe}"
                )
            if previous is not None and candle.open_time <= previous:
                raise MarketDataError(
                    f"non-monotonic candles for {self._symbol}: "
                    f"{candle.open_time.isoformat()} follows {previous.isoformat()}"
                )
            previous = candle.open_time

        self._candles = ordered
        self._open_times = tuple(candle.open_time for candle in ordered)

    @property
    def symbol(self) -> Symbol:
        """The series symbol."""
        return self._symbol

    @property
    def timeframe(self) -> Timeframe:
        """The series timeframe."""
        return self._timeframe

    @property
    def start(self) -> datetime:
        """Open time of the first candle."""
        return self._open_times[0]

    @property
    def end(self) -> datetime:
        """Open time of the last candle."""
        return self._open_times[-1]

    def __len__(self) -> int:
        return len(self._candles)

    def __iter__(self) -> Iterator[Candle]:
        return iter(self._candles)

    def __getitem__(self, index: int) -> Candle:
        return self._candles[index]

    @property
    def candles(self) -> tuple[Candle, ...]:
        """The underlying tuple of candles."""
        return self._candles

    def closes(self) -> tuple[Decimal, ...]:
        """Close prices."""
        return tuple(candle.close for candle in self._candles)

    def highs(self) -> tuple[Decimal, ...]:
        """High prices."""
        return tuple(candle.high for candle in self._candles)

    def lows(self) -> tuple[Decimal, ...]:
        """Low prices."""
        return tuple(candle.low for candle in self._candles)

    def volumes(self) -> tuple[Decimal, ...]:
        """Base-asset volumes."""
        return tuple(candle.volume for candle in self._candles)

    def window(self, size: int) -> CandleSeries:
        """The trailing ``size`` candles."""
        if size <= 0:
            raise ValidationError(f"window size must be positive, got {size}")
        return CandleSeries(self._candles[-size:])

    def slice(self, start: datetime, end: datetime) -> CandleSeries:
        """Candles with ``start <= open_time < end``."""
        lo = bisect_left(self._open_times, start)
        hi = bisect_left(self._open_times, end)
        return CandleSeries(self._candles[lo:hi])

    def missing_intervals(self) -> tuple[tuple[datetime, datetime], ...]:
        """Gaps in the series, as ``(gap_start, gap_end)`` open-time pairs.

        Binance omits bars for periods with no trades on illiquid pairs, so a gap is not
        automatically a bug — but it must be visible before it is used for a backtest.
        """
        step: timedelta = self._timeframe.delta
        gaps: list[tuple[datetime, datetime]] = []
        for previous, current in zip(self._open_times, self._open_times[1:], strict=False):
            expected = previous + step
            if current > expected:
                gaps.append((expected, current))
        return tuple(gaps)

    @property
    def is_contiguous(self) -> bool:
        """Whether the series has no gaps."""
        return not self.missing_intervals()


@dataclass(frozen=True, slots=True)
class DataIntegrityReport:
    """Outcome of validating a stored candle dataset."""

    symbol: Symbol
    timeframe: Timeframe
    candle_count: int
    start: datetime | None
    end: datetime | None
    gaps: tuple[tuple[datetime, datetime], ...] = field(default_factory=tuple)
    duplicate_open_times: tuple[datetime, ...] = field(default_factory=tuple)
    anomalies: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_clean(self) -> bool:
        """Whether the dataset is safe to backtest against."""
        return not (self.gaps or self.duplicate_open_times or self.anomalies)

    @property
    def missing_bar_count(self) -> int:
        """Total number of absent bars implied by the gap list."""
        step = self.timeframe.delta
        return sum(int((end - start) / step) for start, end in self.gaps)
