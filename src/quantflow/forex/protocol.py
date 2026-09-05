"""The transport-agnostic Forex broker interface.

This is the only surface QuantFlow core is allowed to see. MT5, OANDA v20 and any future
venue implement :class:`ForexBroker`; nothing above this line knows which one is wired in,
and nothing in this module imports a venue SDK.

Two things live here besides the interface itself:

* the **DTOs** every transport normalises into — always :class:`~decimal.Decimal` money,
  always timezone-aware UTC instants, always validated on construction, so a malformed
  venue payload fails at the adapter boundary rather than three layers up; and
* the **pure helpers** that need no connection — :func:`ensure_fresh` and
  :func:`reconcile_positions` — which are the reconciliation and staleness hooks the live
  loop calls, and which are fully testable without a venue.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Final, Protocol

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide
from quantflow.forex.errors import StaleMarketDataError
from quantflow.forex.instruments import ForexInstrument
from quantflow.forex.sessions import require_utc

#: How old a quote may be before it is refused for sizing or execution. FX quotes update
#: several times a second in liquid hours, so ten seconds already means something is wrong.
DEFAULT_MAX_TICK_AGE: Final = timedelta(seconds=10)

#: Default price tolerance on a market order, in points.
DEFAULT_DEVIATION_POINTS: Final = Decimal("10")


class ForexTimeframe(StrEnum):
    """Bar granularities, with the per-venue spellings kept next to the canonical name."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

    @property
    def mt5_constant(self) -> str:
        """Name of the matching ``MetaTrader5.TIMEFRAME_*`` constant."""
        return f"TIMEFRAME_{self.name}"

    @property
    def oanda_granularity(self) -> str:
        """Matching OANDA v20 ``granularity`` code."""
        return _OANDA_GRANULARITIES[self]

    @property
    def duration(self) -> timedelta:
        """Nominal length of one bar. Calendar months are approximated as 30 days."""
        return _TIMEFRAME_DURATIONS[self]


_OANDA_GRANULARITIES: Final[dict[ForexTimeframe, str]] = {
    ForexTimeframe.M1: "M1",
    ForexTimeframe.M5: "M5",
    ForexTimeframe.M15: "M15",
    ForexTimeframe.M30: "M30",
    ForexTimeframe.H1: "H1",
    ForexTimeframe.H4: "H4",
    ForexTimeframe.D1: "D",
    ForexTimeframe.W1: "W",
    ForexTimeframe.MN1: "M",
}

_TIMEFRAME_DURATIONS: Final[dict[ForexTimeframe, timedelta]] = {
    ForexTimeframe.M1: timedelta(minutes=1),
    ForexTimeframe.M5: timedelta(minutes=5),
    ForexTimeframe.M15: timedelta(minutes=15),
    ForexTimeframe.M30: timedelta(minutes=30),
    ForexTimeframe.H1: timedelta(hours=1),
    ForexTimeframe.H4: timedelta(hours=4),
    ForexTimeframe.D1: timedelta(days=1),
    ForexTimeframe.W1: timedelta(weeks=1),
    ForexTimeframe.MN1: timedelta(days=30),
}


class ForexOrderType(StrEnum):
    """Order types every FX venue in scope supports."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ForexOrderStatus(StrEnum):
    """Lifecycle state of an order at the venue."""

    PENDING = "pending"
    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ForexTimeInForce(StrEnum):
    """How long a resting order stays live."""

    GTC = "gtc"
    GTD = "gtd"
    DAY = "day"
    IOC = "ioc"
    FOK = "fok"


@dataclass(frozen=True, slots=True)
class AccountInfo:
    """Snapshot of the trading account."""

    login: int
    server: str
    currency: str
    balance: Decimal
    equity: Decimal
    margin_used: Decimal
    margin_free: Decimal
    margin_level: Decimal
    leverage: int
    trade_allowed: bool
    is_demo: bool
    name: str = ""

    def __post_init__(self) -> None:
        """Reject an account payload with no currency."""
        if not self.currency.strip():
            raise ValidationError("account currency must not be blank")


@dataclass(frozen=True, slots=True)
class ForexTick:
    """A top-of-book quote."""

    symbol: str
    bid: Decimal
    ask: Decimal
    timestamp: datetime
    last: Decimal | None = None
    volume: Decimal = ZERO

    def __post_init__(self) -> None:
        """Reject non-positive or crossed quotes."""
        if self.bid <= ZERO or self.ask <= ZERO:
            raise ValidationError("bid and ask must be positive", symbol=self.symbol)
        if self.ask < self.bid:
            raise ValidationError(
                "crossed quote: ask below bid",
                symbol=self.symbol,
                bid=str(self.bid),
                ask=str(self.ask),
            )
        object.__setattr__(self, "timestamp", require_utc(self.timestamp, field="timestamp"))

    @property
    def mid(self) -> Decimal:
        """Mid price."""
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread(self) -> Decimal:
        """Spread as a price distance."""
        return self.ask - self.bid

    def spread_points(self, instrument: ForexInstrument) -> Decimal:
        """Spread expressed in the instrument's points."""
        return instrument.price_to_points(self.spread)

    def age(self, now: datetime) -> timedelta:
        """How long ago this quote was stamped. Negative if it is from the future."""
        return require_utc(now, field="now") - self.timestamp


@dataclass(frozen=True, slots=True)
class ForexBar:
    """One OHLC bar."""

    symbol: str
    timeframe: ForexTimeframe
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    tick_volume: int = 0
    spread_points: Decimal = ZERO
    real_volume: Decimal = ZERO

    def __post_init__(self) -> None:
        """Reject a bar whose extremes do not contain its body."""
        if self.high < self.low:
            raise ValidationError("bar high is below its low", symbol=self.symbol)
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValidationError("bar high/low does not contain open/close", symbol=self.symbol)
        object.__setattr__(self, "open_time", require_utc(self.open_time, field="open_time"))

    @property
    def close_time(self) -> datetime:
        """When this bar's period ends."""
        return self.open_time + self.timeframe.duration


@dataclass(frozen=True, slots=True)
class ForexOrderRequest:
    """An instruction to open or place an order, expressed in lots."""

    symbol: str
    side: OrderSide
    lots: Decimal
    order_type: ForexOrderType = ForexOrderType.MARKET
    price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    time_in_force: ForexTimeInForce = ForexTimeInForce.GTC
    expiry: datetime | None = None
    deviation_points: Decimal = DEFAULT_DEVIATION_POINTS
    comment: str = ""
    client_tag: str = ""
    magic: int = 0

    def __post_init__(self) -> None:
        """Reject an order that could not be routed as written."""
        if self.lots <= ZERO:
            raise ValidationError("lots must be positive", symbol=self.symbol, lots=str(self.lots))
        if self.order_type is not ForexOrderType.MARKET and self.price is None:
            raise ValidationError(
                f"{self.order_type.value} orders need a price", symbol=self.symbol
            )
        if self.time_in_force is ForexTimeInForce.GTD and self.expiry is None:
            raise ValidationError("GTD orders need an expiry", symbol=self.symbol)


@dataclass(frozen=True, slots=True)
class OrderAck:
    """What the venue said in response to an order instruction."""

    accepted: bool
    status: ForexOrderStatus
    ticket: int | None = None
    filled_lots: Decimal = ZERO
    average_price: Decimal | None = None
    message: str = ""
    venue_code: str = ""

    def __bool__(self) -> bool:
        """Truthy when the venue accepted the instruction."""
        return self.accepted


@dataclass(frozen=True, slots=True)
class ForexOrder:
    """A resting or historical order."""

    ticket: int
    symbol: str
    side: OrderSide
    order_type: ForexOrderType
    status: ForexOrderStatus
    lots: Decimal
    filled_lots: Decimal = ZERO
    price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    created_at: datetime | None = None
    magic: int = 0
    comment: str = ""


@dataclass(frozen=True, slots=True)
class ForexPosition:
    """An open position at the venue."""

    ticket: int
    symbol: str
    side: OrderSide
    lots: Decimal
    entry_price: Decimal
    current_price: Decimal
    opened_at: datetime
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    swap: Decimal = ZERO
    commission: Decimal = ZERO
    profit: Decimal = ZERO
    magic: int = 0
    comment: str = ""

    def __post_init__(self) -> None:
        """Reject a position with a non-positive size."""
        if self.lots <= ZERO:
            raise ValidationError("position lots must be positive", symbol=self.symbol)
        object.__setattr__(self, "opened_at", require_utc(self.opened_at, field="opened_at"))

    @property
    def is_long(self) -> bool:
        """Whether this is a long position."""
        return self.side is OrderSide.BUY

    @property
    def signed_lots(self) -> Decimal:
        """Lots signed by direction — positive long, negative short."""
        return self.lots if self.is_long else -self.lots


@dataclass(frozen=True, slots=True)
class ForexFill:
    """A single execution (MT5 calls these deals, OANDA calls them transactions)."""

    ticket: int
    order_ticket: int
    symbol: str
    side: OrderSide
    lots: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal = ZERO
    swap: Decimal = ZERO
    profit: Decimal = ZERO
    is_entry: bool = True
    magic: int = 0
    comment: str = ""

    def __post_init__(self) -> None:
        """Normalise the execution timestamp to UTC."""
        object.__setattr__(self, "timestamp", require_utc(self.timestamp, field="timestamp"))


@dataclass(frozen=True, slots=True)
class PositionDelta:
    """A position whose size differs between our books and the venue's."""

    ticket: int
    symbol: str
    expected_lots: Decimal
    actual_lots: Decimal


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """The result of comparing our position book against the venue's."""

    checked_at: datetime
    matched: tuple[int, ...] = ()
    only_at_broker: tuple[int, ...] = ()
    only_local: tuple[int, ...] = ()
    lot_mismatches: tuple[PositionDelta, ...] = ()
    side_mismatches: tuple[int, ...] = ()

    @property
    def is_clean(self) -> bool:
        """Whether the two books agree exactly."""
        return not (
            self.only_at_broker or self.only_local or self.lot_mismatches or self.side_mismatches
        )

    @property
    def discrepancy_count(self) -> int:
        """How many individual disagreements were found."""
        return (
            len(self.only_at_broker)
            + len(self.only_local)
            + len(self.lot_mismatches)
            + len(self.side_mismatches)
        )


def ensure_fresh(tick: ForexTick, now: datetime, max_age: timedelta = DEFAULT_MAX_TICK_AGE) -> None:
    """Raise unless ``tick`` is recent enough to act on.

    A quote from the *future* is rejected too: it means the venue clock and ours disagree,
    and an unnoticed clock skew silently disables every other staleness check.

    Raises:
        StaleMarketDataError: if the quote is older than ``max_age`` or ahead of ``now``.

    """
    age = tick.age(now)
    if age > max_age:
        raise StaleMarketDataError(
            f"quote for {tick.symbol} is {age} old (budget {max_age})",
            symbol=tick.symbol,
            age_seconds=age.total_seconds(),
        )
    if age < timedelta(0):
        raise StaleMarketDataError(
            f"quote for {tick.symbol} is stamped {-age} in the future; check clock skew",
            symbol=tick.symbol,
            age_seconds=age.total_seconds(),
        )


def reconcile_positions(
    expected: Sequence[ForexPosition],
    actual: Sequence[ForexPosition],
    *,
    now: datetime,
) -> ReconciliationReport:
    """Compare our position book against the venue's, keyed by ticket.

    Purely a diff — it never mutates and never places an order. The live loop decides what
    to do about a discrepancy; this only makes sure the discrepancy cannot go unseen.
    """
    expected_by_ticket = {position.ticket: position for position in expected}
    actual_by_ticket = {position.ticket: position for position in actual}

    matched: list[int] = []
    lot_mismatches: list[PositionDelta] = []
    side_mismatches: list[int] = []

    for ticket in sorted(expected_by_ticket.keys() & actual_by_ticket.keys()):
        ours = expected_by_ticket[ticket]
        theirs = actual_by_ticket[ticket]
        disagrees = False
        if ours.side is not theirs.side:
            side_mismatches.append(ticket)
            disagrees = True
        if ours.lots != theirs.lots:
            lot_mismatches.append(
                PositionDelta(
                    ticket=ticket,
                    symbol=theirs.symbol,
                    expected_lots=ours.lots,
                    actual_lots=theirs.lots,
                )
            )
            disagrees = True
        if not disagrees:
            matched.append(ticket)

    return ReconciliationReport(
        checked_at=require_utc(now, field="now"),
        matched=tuple(matched),
        only_at_broker=tuple(sorted(actual_by_ticket.keys() - expected_by_ticket.keys())),
        only_local=tuple(sorted(expected_by_ticket.keys() - actual_by_ticket.keys())),
        lot_mismatches=tuple(lot_mismatches),
        side_mismatches=tuple(side_mismatches),
    )


class ForexBroker(Protocol):
    """Everything QuantFlow needs from an FX venue, and nothing venue-specific.

    Implementations are expected to be synchronous and to raise
    :mod:`quantflow.forex.errors` types on failure. Connection management deliberately sits
    outside this interface — see :class:`ForexConnection` — so that a transport which needs
    no session (a stateless REST client) is not forced to fake one.
    """

    def get_account(self) -> AccountInfo:
        """Current account balance, equity and margin."""
        ...

    def get_symbols(self, symbols: Sequence[str] | None = None) -> tuple[ForexInstrument, ...]:
        """Discover tradable instruments, optionally narrowed to ``symbols``."""
        ...

    def subscribe_ticks(self, symbols: Sequence[str]) -> Iterator[ForexTick]:
        """Yield quotes for ``symbols`` as they arrive, until the caller stops consuming."""
        ...

    def get_bars(
        self,
        symbol: str,
        timeframe: ForexTimeframe,
        count: int,
        end: datetime | None = None,
    ) -> tuple[ForexBar, ...]:
        """Fetch the most recent ``count`` bars at or before ``end``, oldest first."""
        ...

    def submit_order(self, request: ForexOrderRequest) -> OrderAck:
        """Send an order to the venue."""
        ...

    def modify_stop(
        self,
        ticket: int,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> OrderAck:
        """Attach or move the protective stop and/or take-profit on an open position."""
        ...

    def close_position(self, ticket: int, lots: Decimal | None = None) -> OrderAck:
        """Close a position — fully, or partially when ``lots`` is given."""
        ...

    def get_orders(self, symbol: str | None = None) -> tuple[ForexOrder, ...]:
        """List working (pending) orders."""
        ...

    def get_positions(self, symbol: str | None = None) -> tuple[ForexPosition, ...]:
        """List open positions."""
        ...

    def get_fills(
        self,
        since: datetime,
        until: datetime | None = None,
        symbol: str | None = None,
    ) -> tuple[ForexFill, ...]:
        """List executions in a time range, oldest first."""
        ...


class ForexConnection(Protocol):
    """Optional session management for transports that hold one."""

    def connect(self) -> AccountInfo:
        """Establish the session and return the account it resolved to."""
        ...

    def disconnect(self) -> None:
        """Tear the session down. Safe to call when not connected."""
        ...


def latest_tick(broker: ForexBroker, symbol: str) -> ForexTick:
    """Pull a single current quote off the broker's tick stream.

    A convenience over :meth:`ForexBroker.subscribe_ticks` so that snapshot callers do not
    each reimplement "take one and close the generator".

    Raises:
        StaleMarketDataError: if the stream ends without producing a quote.

    """
    stream = broker.subscribe_ticks([symbol])
    try:
        for tick in stream:
            return tick
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    raise StaleMarketDataError(f"no quote available for {symbol}", symbol=symbol)


@dataclass(frozen=True, slots=True)
class BrokerDescription:
    """Static self-description a transport publishes so operators can see what is wired in."""

    venue: str
    transport: str
    supports_streaming: bool = True
    supports_partial_close: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)
