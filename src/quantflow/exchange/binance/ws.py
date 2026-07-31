"""Binance websocket market-data client.

Written directly against Binance's stream protocol rather than through CCXT Pro (which is
commercial). The protocol is small and stable, and owning it gives us explicit control over
reconnection, gap detection and backpressure — all of which matter more for a trading loop
than the convenience of a wrapper.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from quantflow.core.clock import Clock, SystemClock, from_epoch_ms
from quantflow.core.config import ExchangeSettings, MarketType
from quantflow.core.errors import ExchangeConnectionError, MarketDataError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO, to_decimal
from quantflow.domain.enums import OrderSide, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, Ticker, Trade
from quantflow.exchange.ratelimit import backoff_delay

logger = get_logger(__name__)

SPOT_WS_URL: Final = "wss://stream.binance.com:9443/ws"
SPOT_TESTNET_WS_URL: Final = "wss://stream.testnet.binance.vision/ws"
FUTURES_WS_URL: Final = "wss://fstream.binance.com/ws"
FUTURES_TESTNET_WS_URL: Final = "wss://stream.binancefuture.com/ws"

#: Binance closes an idle connection after 24 hours and expects a pong within 10 minutes
#: of each ping. A 3-minute application-level ping keeps us comfortably inside that.
PING_INTERVAL_SECONDS: Final = 180.0
PING_TIMEOUT_SECONDS: Final = 20.0

#: If no message arrives within this window on a subscribed stream, assume the connection
#: is silently dead and reconnect. A half-open TCP connection produces no error at all —
#: it simply stops delivering data, which for a trading loop is the worst failure mode.
STALE_MESSAGE_TIMEOUT_SECONDS: Final = 60.0


def stream_url(settings: ExchangeSettings) -> str:
    """Base websocket URL for the configured market type and network."""
    if settings.market_type is MarketType.FUTURE:
        return FUTURES_TESTNET_WS_URL if settings.testnet else FUTURES_WS_URL
    return SPOT_TESTNET_WS_URL if settings.testnet else SPOT_WS_URL


class BinanceStream:
    """Auto-reconnecting websocket client for Binance public streams."""

    __slots__ = ("_clock", "_settings", "_url")

    def __init__(self, settings: ExchangeSettings, *, clock: Clock | None = None) -> None:
        self._settings = settings
        self._clock = clock or SystemClock()
        self._url = stream_url(settings)

    @property
    def url(self) -> str:
        """The base stream URL in use."""
        return self._url

    async def _messages(self, streams: list[str]) -> AsyncIterator[dict[str, Any]]:
        """Yield decoded messages, reconnecting with backoff on failure.

        The generator never terminates on a network error; it is the caller's cancellation
        that ends it. That is deliberate — a trading loop must survive a venue restart
        without the operator intervening.
        """
        endpoint = f"{self._url}/{'/'.join(streams)}"
        attempt = 0

        while True:
            try:
                async with websockets.connect(
                    endpoint,
                    ping_interval=PING_INTERVAL_SECONDS,
                    ping_timeout=PING_TIMEOUT_SECONDS,
                    close_timeout=5.0,
                    max_queue=1024,
                ) as connection:
                    if attempt:
                        logger.info("stream.reconnected", streams=streams, attempts=attempt)
                    else:
                        logger.info("stream.connected", streams=streams)
                    attempt = 0

                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                connection.recv(), timeout=STALE_MESSAGE_TIMEOUT_SECONDS
                            )
                        except TimeoutError:
                            logger.warning("stream.stale", streams=streams)
                            break  # force a reconnect
                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning("stream.malformed_message", streams=streams)
                            continue
                        if isinstance(message, dict):
                            yield message

            except asyncio.CancelledError:
                logger.debug("stream.cancelled", streams=streams)
                raise
            except (ConnectionClosed, WebSocketException, OSError) as exc:
                delay = backoff_delay(
                    attempt, base=1.0, cap=self._settings.ws_reconnect_max_seconds
                )
                attempt += 1
                logger.warning(
                    "stream.disconnected",
                    streams=streams,
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
        stream = f"{symbol.concatenated.lower()}@kline_{timeframe.value}"
        async for message in self._messages([stream]):
            payload = message.get("k")
            if not isinstance(payload, dict):
                continue
            is_closed = bool(payload.get("x"))
            if closed_only and not is_closed:
                continue
            try:
                yield _parse_kline(symbol, timeframe, payload)
            except (MarketDataError, ValueError) as exc:
                logger.warning("stream.bad_kline", symbol=str(symbol), error=str(exc))

    async def watch_ticker(self, symbol: Symbol) -> AsyncIterator[Ticker]:
        """Stream best bid/ask updates."""
        stream = f"{symbol.concatenated.lower()}@bookTicker"
        async for message in self._messages([stream]):
            try:
                yield _parse_book_ticker(symbol, message, self._clock.now())
            except (MarketDataError, ValueError) as exc:
                logger.warning("stream.bad_ticker", symbol=str(symbol), error=str(exc))

    async def watch_trades(self, symbol: Symbol) -> AsyncIterator[Trade]:
        """Stream the public trade tape."""
        stream = f"{symbol.concatenated.lower()}@trade"
        async for message in self._messages([stream]):
            try:
                yield _parse_trade(symbol, message)
            except (MarketDataError, ValueError) as exc:
                logger.warning("stream.bad_trade", symbol=str(symbol), error=str(exc))

    async def watch_many_candles(
        self, symbols: list[Symbol], timeframe: Timeframe, *, closed_only: bool = True
    ) -> AsyncIterator[Candle]:
        """Stream klines for several symbols over a single multiplexed connection.

        One connection for N symbols rather than N connections: Binance limits concurrent
        connections per IP, and a single socket also gives a single reconnection story.
        """
        by_stream = {
            f"{symbol.concatenated.lower()}@kline_{timeframe.value}": symbol for symbol in symbols
        }
        async for message in self._messages(list(by_stream)):
            payload = message.get("k")
            if not isinstance(payload, dict):
                continue
            if closed_only and not payload.get("x"):
                continue
            raw_symbol = str(message.get("s") or payload.get("s") or "")
            symbol = _match_symbol(raw_symbol, symbols)
            if symbol is None:
                continue
            try:
                yield _parse_kline(symbol, timeframe, payload)
            except (MarketDataError, ValueError) as exc:
                logger.warning("stream.bad_kline", symbol=raw_symbol, error=str(exc))


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


# --------------------------------------------------------------------------- #
# Payload parsing
# --------------------------------------------------------------------------- #
def _decimal(payload: dict[str, Any], key: str, default: Decimal = ZERO) -> Decimal:
    value = payload.get(key)
    if value is None or value == "":
        return default
    return to_decimal(value)


def _parse_kline(symbol: Symbol, timeframe: Timeframe, payload: dict[str, Any]) -> Candle:
    open_time = payload.get("t")
    if open_time is None:
        raise MarketDataError(f"kline for {symbol} has no open time")
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open_time=from_epoch_ms(int(open_time)),
        open=_decimal(payload, "o"),
        high=_decimal(payload, "h"),
        low=_decimal(payload, "l"),
        close=_decimal(payload, "c"),
        volume=_decimal(payload, "v"),
        quote_volume=_decimal(payload, "q"),
        trades=int(payload.get("n") or 0),
    )


def _parse_book_ticker(symbol: Symbol, payload: dict[str, Any], now: datetime) -> Ticker:
    bid = _decimal(payload, "b")
    ask = _decimal(payload, "a")
    if bid <= ZERO or ask <= ZERO:
        raise MarketDataError(f"book ticker for {symbol} has no usable prices")
    event_time = payload.get("E")
    return Ticker(
        symbol=symbol,
        timestamp=from_epoch_ms(int(event_time)) if event_time else now,
        bid=bid,
        ask=ask,
        last=(bid + ask) / Decimal(2),
        bid_volume=_decimal(payload, "B"),
        ask_volume=_decimal(payload, "A"),
    )


def _parse_trade(symbol: Symbol, payload: dict[str, Any]) -> Trade:
    timestamp = payload.get("T") or payload.get("E")
    if timestamp is None:
        raise MarketDataError(f"trade for {symbol} has no timestamp")
    # Binance's `m` flag means "the buyer is the market maker", i.e. the aggressor sold.
    aggressor_sold = bool(payload.get("m"))
    return Trade(
        symbol=symbol,
        trade_id=str(payload.get("t") or ""),
        timestamp=from_epoch_ms(int(timestamp)),
        price=_decimal(payload, "p"),
        quantity=_decimal(payload, "q"),
        side=OrderSide.SELL if aggressor_sold else OrderSide.BUY,
    )


def _match_symbol(raw: str, candidates: list[Symbol]) -> Symbol | None:
    target = raw.upper()
    for symbol in candidates:
        if symbol.concatenated == target:
            return symbol
    return None


async def open_stream(settings: ExchangeSettings, *, clock: Clock | None = None) -> BinanceStream:
    """Build a stream client, verifying the endpoint is reachable.

    Raises:
        ExchangeConnectionError: if the venue cannot be reached.

    """
    stream = BinanceStream(settings, clock=clock)
    try:
        async with asyncio.timeout(10):
            connection = await websockets.connect(stream.url, close_timeout=2.0)
            with contextlib.suppress(Exception):
                await connection.close()
    except (TimeoutError, OSError, WebSocketException) as exc:
        raise ExchangeConnectionError(f"cannot reach {stream.url}: {exc}", url=stream.url) from exc
    return stream
