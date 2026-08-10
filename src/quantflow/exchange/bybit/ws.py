"""Bybit V5 public websocket streams.

Bybit V5 is not Binance-shaped and this module reflects that rather than pretending
otherwise.

* **Subscription is a message, not a URL.** Binance encodes streams in the path; V5 opens
  one socket per category and sends ``{"op": "subscribe", "args": [...]}``. That means the
  subscribe has to be re-sent after every reconnect, which is the single easiest thing to
  get wrong here — the socket comes back, no error is raised, and no data ever arrives.
* **Topics are category-scoped.** ``wss://stream.bybit.com/v5/public/spot`` and
  ``.../linear`` are different endpoints, and subscribing to a linear symbol on the spot
  socket fails silently rather than erroring.
* **Intervals are Bybit's own vocabulary.** Minutes are bare numbers ("60" is one hour),
  and days, weeks and months are letters. ``1h`` sent verbatim is rejected.
* **Every frame carries a list.** ``data`` is an array even for a single kline, and the
  ``confirm`` flag — not Binance's ``x`` — marks a closed bar.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from quantflow.core.clock import Clock, SystemClock, from_epoch_ms
from quantflow.core.config import ExchangeSettings, MarketType
from quantflow.core.errors import MarketDataError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO, to_decimal
from quantflow.domain.enums import OrderSide, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, Ticker, Trade
from quantflow.exchange.ratelimit import backoff_delay

logger = get_logger(__name__)

PUBLIC_WS_URL: Final = "wss://stream.bybit.com/v5/public"
PUBLIC_TESTNET_WS_URL: Final = "wss://stream-testnet.bybit.com/v5/public"

#: Bybit expects a ping every 20 seconds and closes the socket after 30 without one.
#: That is far tighter than Binance's 3-minute allowance, and using the old value here
#: would produce a connection that drops every 30 seconds for no visible reason.
PING_INTERVAL_SECONDS: Final = 20.0
PING_TIMEOUT_SECONDS: Final = 10.0

#: If no message arrives within this window, assume the connection is silently dead and
#: reconnect. A half-open TCP connection produces no error at all — it simply stops
#: delivering data, which for a trading loop is the worst failure mode.
STALE_MESSAGE_TIMEOUT_SECONDS: Final = 60.0

#: QuantFlow timeframes to Bybit V5 kline intervals. Bybit counts minutes as bare numbers
#: and switches to letters at daily, so no arithmetic shortcut covers both halves.
TIMEFRAME_TO_INTERVAL: Final[dict[Timeframe, str]] = {
    Timeframe.M1: "1",
    Timeframe.M3: "3",
    Timeframe.M5: "5",
    Timeframe.M15: "15",
    Timeframe.M30: "30",
    Timeframe.H1: "60",
    Timeframe.H2: "120",
    Timeframe.H4: "240",
    Timeframe.H6: "360",
    Timeframe.H12: "720",
    Timeframe.D1: "D",
    Timeframe.W1: "W",
}


def bybit_interval(timeframe: Timeframe) -> str:
    """Bybit V5's interval code for a timeframe.

    Raises:
        MarketDataError: for a timeframe Bybit does not publish klines for. Failing here
            is deliberate: silently substituting a neighbouring interval would feed a
            strategy bars it never asked for.

    """
    interval = TIMEFRAME_TO_INTERVAL.get(timeframe)
    if interval is None:
        raise MarketDataError(f"Bybit V5 publishes no kline interval for {timeframe.value}")
    return interval


def stream_url(settings: ExchangeSettings) -> str:
    """Public websocket URL for the configured category and network."""
    base = PUBLIC_TESTNET_WS_URL if settings.testnet else PUBLIC_WS_URL
    category = "linear" if settings.market_type is MarketType.FUTURE else "spot"
    return f"{base}/{category}"


class BybitStream:
    """Auto-reconnecting websocket client for Bybit V5 public topics."""

    __slots__ = ("_clock", "_settings", "_url")

    def __init__(self, settings: ExchangeSettings, *, clock: Clock | None = None) -> None:
        self._settings = settings
        self._clock = clock or SystemClock()
        self._url = stream_url(settings)

    @property
    def url(self) -> str:
        """The base stream URL in use."""
        return self._url

    async def _messages(self, topics: list[str]) -> AsyncIterator[dict[str, Any]]:
        """Yield decoded topic frames, reconnecting with backoff on failure.

        The generator never terminates on a network error; only the caller's cancellation
        ends it. A trading loop must survive a venue restart without intervention.
        """
        attempt = 0

        while True:
            try:
                async with websockets.connect(
                    self._url,
                    ping_interval=PING_INTERVAL_SECONDS,
                    ping_timeout=PING_TIMEOUT_SECONDS,
                    close_timeout=5.0,
                    max_queue=1024,
                ) as connection:
                    # Re-subscribing on every connection is mandatory, not defensive:
                    # V5 keeps no subscription state across a reconnect, so skipping this
                    # yields a healthy socket that never delivers a single frame.
                    await connection.send(json.dumps({"op": "subscribe", "args": topics}))

                    if attempt:
                        logger.info("stream.reconnected", topics=topics, attempts=attempt)
                    else:
                        logger.info("stream.connected", topics=topics)
                    attempt = 0

                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                connection.recv(), timeout=STALE_MESSAGE_TIMEOUT_SECONDS
                            )
                        except TimeoutError:
                            logger.warning("stream.stale", topics=topics)
                            break  # force a reconnect
                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning("stream.malformed_message", topics=topics)
                            continue
                        if not isinstance(message, dict):
                            continue

                        # Subscription acks and pongs carry no topic; a failed ack is
                        # worth surfacing because it is otherwise indistinguishable from
                        # a quiet market.
                        if "topic" not in message:
                            if message.get("op") == "subscribe" and not message.get(
                                "success", True
                            ):
                                logger.error(
                                    "stream.subscribe_rejected",
                                    topics=topics,
                                    error=message.get("ret_msg"),
                                )
                            continue
                        yield message

            except asyncio.CancelledError:
                logger.debug("stream.cancelled", topics=topics)
                raise
            except (ConnectionClosed, WebSocketException, OSError) as exc:
                delay = backoff_delay(
                    attempt, base=1.0, cap=self._settings.ws_reconnect_max_seconds
                )
                attempt += 1
                logger.warning(
                    "stream.disconnected",
                    topics=topics,
                    error=str(exc),
                    attempt=attempt,
                    retry_in_seconds=round(delay, 2),
                )
                await self._clock.sleep(delay)

    # ------------------------------------------------------------------ #
    # Typed streams
    # ------------------------------------------------------------------ #
    async def watch_candles(
        self, symbol: Symbol, timeframe: Timeframe, *, closed_only: bool = False
    ) -> AsyncIterator[Candle]:
        """Stream kline updates for a symbol.

        Args:
            symbol: The pair to watch.
            timeframe: Bar interval.
            closed_only: Emit only completed bars. Strategies must use this — acting on a
                forming bar means acting on a price that can still move against you within
                the same bar, which no backtest will ever reproduce.

        """
        topic = f"kline.{bybit_interval(timeframe)}.{symbol.concatenated}"
        async for message in self._messages([topic]):
            for entry in _rows(message):
                if closed_only and not entry.get("confirm"):
                    continue
                try:
                    yield _parse_kline(symbol, timeframe, entry)
                except (MarketDataError, ValueError) as exc:
                    logger.warning("stream.bad_kline", symbol=str(symbol), error=str(exc))

    async def watch_ticker(self, symbol: Symbol) -> AsyncIterator[Ticker]:
        """Stream best bid/ask updates."""
        topic = f"tickers.{symbol.concatenated}"
        async for message in self._messages([topic]):
            payload = message.get("data")
            # The tickers topic sends an object, not a list, unlike every other topic.
            if not isinstance(payload, dict):
                continue
            try:
                yield _parse_ticker(symbol, payload, message, self._clock.now())
            except (MarketDataError, ValueError) as exc:
                logger.warning("stream.bad_ticker", symbol=str(symbol), error=str(exc))

    async def watch_trades(self, symbol: Symbol) -> AsyncIterator[Trade]:
        """Stream the public trade tape."""
        topic = f"publicTrade.{symbol.concatenated}"
        async for message in self._messages([topic]):
            for entry in _rows(message):
                try:
                    yield _parse_trade(symbol, entry)
                except (MarketDataError, ValueError) as exc:
                    logger.warning("stream.bad_trade", symbol=str(symbol), error=str(exc))

    async def watch_many_candles(
        self, symbols: list[Symbol], timeframe: Timeframe, *, closed_only: bool = True
    ) -> AsyncIterator[Candle]:
        """Stream klines for several symbols over one multiplexed connection.

        One socket for N symbols rather than N sockets: Bybit caps args per subscription
        and connections per IP, and a single socket also gives a single reconnection story.
        """
        interval = bybit_interval(timeframe)
        by_topic = {f"kline.{interval}.{symbol.concatenated}": symbol for symbol in symbols}
        async for message in self._messages(list(by_topic)):
            # The frame identifies its symbol only through the topic, so it is resolved
            # from the subscription map rather than from the payload.
            symbol = by_topic.get(str(message.get("topic", "")))
            if symbol is None:
                continue
            for entry in _rows(message):
                if closed_only and not entry.get("confirm"):
                    continue
                try:
                    yield _parse_kline(symbol, timeframe, entry)
                except (MarketDataError, ValueError) as exc:
                    logger.warning("stream.bad_kline", symbol=str(symbol), error=str(exc))


class CandleGapDetector:
    """Flags missing bars in a live stream.

    A websocket reconnect silently skips whatever was published while disconnected. The
    engine must know, because a strategy that computes an EMA across a hole produces a
    signal it would never have produced on complete data.
    """

    __slots__ = ("_last_open_time", "_on_gap", "_timeframe")

    def __init__(
        self,
        timeframe: Timeframe,
        *,
        on_gap: Callable[[Symbol, datetime, datetime, int], None] | None = None,
    ) -> None:
        self._timeframe = timeframe
        self._last_open_time: dict[Symbol, datetime] = {}
        self._on_gap = on_gap

    def observe(self, candle: Candle) -> int:
        """Record a bar and return the number of bars missing before it."""
        previous = self._last_open_time.get(candle.symbol)
        self._last_open_time[candle.symbol] = candle.open_time
        if previous is None or candle.open_time <= previous:
            return 0
        step = self._timeframe.delta
        missing = int((candle.open_time - previous) / step) - 1
        if missing > 0:
            expected = previous + step
            logger.warning(
                "stream.candle_gap",
                symbol=str(candle.symbol),
                missing_bars=missing,
                gap_start=expected.isoformat(),
                gap_end=candle.open_time.isoformat(),
            )
            if self._on_gap is not None:
                self._on_gap(candle.symbol, expected, candle.open_time, missing)
        return max(0, missing)

    def reset(self, symbol: Symbol | None = None) -> None:
        """Forget the last-seen bar for one symbol, or for all of them."""
        if symbol is None:
            self._last_open_time.clear()
        else:
            self._last_open_time.pop(symbol, None)


def _rows(message: dict[str, Any]) -> list[dict[str, Any]]:
    """The data rows in a V5 frame. Always a list, even for one kline."""
    data = message.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _decimal(payload: dict[str, Any], key: str, default: Decimal = ZERO) -> Decimal:
    value = payload.get(key)
    if value is None or value == "":
        return default
    return to_decimal(value)


def _parse_kline(symbol: Symbol, timeframe: Timeframe, payload: dict[str, Any]) -> Candle:
    """Parse a V5 kline row.

    Bybit names the quote-currency volume ``turnover``; reading ``volume`` for both would
    silently report base volume as quote volume and skew every liquidity measure.
    """
    open_time = payload.get("start")
    if open_time is None:
        raise MarketDataError(f"kline for {symbol} has no start time")
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open_time=from_epoch_ms(int(open_time)),
        open=_decimal(payload, "open"),
        high=_decimal(payload, "high"),
        low=_decimal(payload, "low"),
        close=_decimal(payload, "close"),
        volume=_decimal(payload, "volume"),
        quote_volume=_decimal(payload, "turnover"),
        # V5 klines carry no trade count.
        trades=0,
    )


def _parse_ticker(
    symbol: Symbol, payload: dict[str, Any], envelope: dict[str, Any], now: datetime
) -> Ticker:
    """Parse a V5 tickers frame.

    Spot and linear name their fields differently — ``bid1Price`` on both, but linear
    tickers are delta frames that may omit a side entirely — so a missing quote is an
    error rather than a zero.
    """
    bid = _decimal(payload, "bid1Price")
    ask = _decimal(payload, "ask1Price")
    if bid <= ZERO or ask <= ZERO:
        raise MarketDataError(f"ticker for {symbol} has no usable prices")
    event_time = envelope.get("ts")
    last = _decimal(payload, "lastPrice")
    return Ticker(
        symbol=symbol,
        timestamp=from_epoch_ms(int(event_time)) if event_time else now,
        bid=bid,
        ask=ask,
        last=last if last > ZERO else (bid + ask) / Decimal(2),
        bid_volume=_decimal(payload, "bid1Size"),
        ask_volume=_decimal(payload, "ask1Size"),
    )


def _parse_trade(symbol: Symbol, payload: dict[str, Any]) -> Trade:
    """Parse a V5 publicTrade row.

    Bybit reports the *aggressor's* side directly in ``S``, which is the opposite
    convention to Binance's maker flag. Copying the Binance logic here would invert every
    trade's direction.
    """
    timestamp = payload.get("T")
    if timestamp is None:
        raise MarketDataError(f"trade for {symbol} has no timestamp")
    side = str(payload.get("S") or "").lower()
    return Trade(
        symbol=symbol,
        trade_id=str(payload.get("i") or ""),
        timestamp=from_epoch_ms(int(timestamp)),
        price=_decimal(payload, "p"),
        quantity=_decimal(payload, "v"),
        side=OrderSide.SELL if side == "sell" else OrderSide.BUY,
    )
