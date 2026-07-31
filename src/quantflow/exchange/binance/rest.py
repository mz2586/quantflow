"""Binance REST gateway, implemented over CCXT.

CCXT rather than a hand-rolled client: it already tracks Binance's endpoint changes,
signing rules and market metadata. Everything venue-specific that CCXT does *not*
normalise — error semantics, precision handling, symbol forms — is handled here and in
``mapping``, so nothing above this module ever sees a CCXT type.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt

from quantflow.core.clock import Clock, SystemClock, to_epoch_ms
from quantflow.core.config import ExchangeSettings, MarketType
from quantflow.core.errors import (
    ExchangeError,
    InvalidSymbolError,
    NotFoundError,
    ValidationError,
)
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, OrderBook, OrderBookLevel, Ticker, Trade
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.domain.portfolio import Balance
from quantflow.exchange.base import MAX_CANDLES_PER_REQUEST, InstrumentCache, normalize_order
from quantflow.exchange.binance.mapping import (
    ORDER_TYPE_TO_CCXT,
    TIME_IN_FORCE_TO_CCXT,
    from_ccxt_symbol,
    parse_fill,
    parse_instrument,
    parse_order,
    to_ccxt_symbol,
    translate_exception,
)
from quantflow.exchange.ratelimit import RateLimiter, retry_async

logger = get_logger(__name__)

#: Binance request weights, used to size the token-bucket cost of each call. Sourced from
#: the published API limits; deep order books are dramatically more expensive than tickers.
ENDPOINT_WEIGHTS: dict[str, float] = {
    "load_markets": 10.0,
    "fetch_candles": 2.0,
    "fetch_ticker": 2.0,
    "fetch_order_book": 5.0,
    "fetch_trades": 2.0,
    "submit_order": 1.0,
    "cancel_order": 1.0,
    "fetch_order": 2.0,
    "fetch_open_orders": 6.0,
    "fetch_my_trades": 10.0,
    "fetch_balances": 10.0,
    "server_time": 1.0,
}

#: Maximum tolerated difference between our clock and Binance's. Binance rejects a signed
#: request whose timestamp is outside its recv window, so drift beyond this is a hard fault
#: rather than something to retry through.
MAX_CLOCK_DRIFT_SECONDS = 5.0

#: An order-book level is [price, quantity].
ORDER_BOOK_LEVEL_WIDTH = 2


class BinanceGateway:
    """Binance spot / USDⓈ-M futures gateway.

    Implements :class:`~quantflow.exchange.base.ExchangeGateway`.
    """

    __slots__ = (
        "_client",
        "_clock",
        "_connected",
        "_instruments",
        "_limiter",
        "_local_by_venue_id",
        "_settings",
    )

    def __init__(self, settings: ExchangeSettings, *, clock: Clock | None = None) -> None:
        self._settings = settings
        self._clock = clock or SystemClock()
        self._instruments = InstrumentCache()
        self._limiter = RateLimiter(
            settings.rate_limit_per_second, settings.rate_limit_burst, clock=self._clock
        )
        self._client = self._build_client(settings)
        self._connected = False
        #: Maps a venue order id back to our own order id, so a fetched order reconciles.
        self._local_by_venue_id: dict[str, str] = {}

    @staticmethod
    def _build_client(settings: ExchangeSettings) -> ccxt.binance:
        client = ccxt.binance(
            {
                "apiKey": settings.api_key.get_secret_value() if settings.api_key else None,
                "secret": settings.api_secret.get_secret_value() if settings.api_secret else None,
                "timeout": int(settings.request_timeout_seconds * 1000),
                # CCXT's own throttle stays on as a second line of defence behind our
                # token bucket; belt and braces is cheap here and an IP ban is not.
                "enableRateLimit": True,
                "options": {
                    "defaultType": (
                        "future" if settings.market_type is MarketType.FUTURE else "spot"
                    ),
                    "recvWindow": settings.recv_window_ms,
                    "adjustForTimeDifference": True,
                },
            }
        )
        if settings.testnet:
            client.set_sandbox_mode(True)
        return client

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Venue identifier."""
        return "binance"

    @property
    def is_testnet(self) -> bool:
        """Whether this gateway points at the sandbox."""
        return self._settings.testnet

    @property
    def supports_trading(self) -> bool:
        """Whether credentials are configured for order placement."""
        return self._settings.has_credentials

    @property
    def instruments(self) -> InstrumentCache:
        """The loaded instrument cache."""
        return self._instruments

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def connect(self) -> None:
        """Load markets and verify clock synchronisation."""
        if self._connected:
            return
        await self.load_instruments()
        await self._assert_clock_sync()
        self._connected = True
        logger.info(
            "exchange.connected",
            exchange=self.name,
            testnet=self.is_testnet,
            market_type=self._settings.market_type.value,
            instruments=len(self._instruments),
            trading_enabled=self.supports_trading,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP session."""
        await self._client.close()
        self._connected = False
        logger.debug("exchange.closed", exchange=self.name)

    async def _assert_clock_sync(self) -> None:
        """Fail fast on clock drift.

        A signed Binance request whose timestamp falls outside ``recvWindow`` is rejected.
        Detecting that at startup beats discovering it when an exit order is rejected.
        """
        if not self.supports_trading:
            return
        venue_time = await self.server_time()
        drift = abs((venue_time - self._clock.now()).total_seconds())
        if drift > MAX_CLOCK_DRIFT_SECONDS:
            raise ExchangeError(
                f"local clock differs from {self.name} by {drift:.2f}s "
                f"(limit {MAX_CLOCK_DRIFT_SECONDS}s); synchronise the system clock",
                drift_seconds=round(drift, 3),
            )
        logger.debug("exchange.clock_checked", drift_seconds=round(drift, 3))

    # ------------------------------------------------------------------ #
    # Call plumbing
    # ------------------------------------------------------------------ #
    async def _call(self, endpoint: str, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Rate-limit, execute and retry a CCXT call, translating its errors."""
        weight = ENDPOINT_WEIGHTS.get(endpoint, 1.0)

        async def invoke() -> Any:
            await self._limiter.acquire(weight)
            try:
                return await operation(*args, **kwargs)
            except Exception as exc:
                raise translate_exception(exc) from exc

        return await retry_async(
            invoke,
            max_retries=self._settings.max_retries,
            base_delay=self._settings.retry_backoff_seconds,
            clock=self._clock,
            description=endpoint,
        )

    def _ccxt_symbol(self, symbol: Symbol) -> str:
        return to_ccxt_symbol(symbol, self._settings.market_type)

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #
    async def load_instruments(self) -> dict[Symbol, Instrument]:
        """Load every tradable instrument's rules."""
        markets = await self._call("load_markets", self._client.load_markets, True)
        loaded: dict[Symbol, Instrument] = {}
        for market in (markets or {}).values():
            instrument = parse_instrument(market)
            if instrument is None:
                continue
            if instrument.market_type is not self._settings.market_type:
                continue
            loaded[instrument.symbol] = instrument
        self._instruments.put_many(list(loaded.values()))
        logger.debug("exchange.instruments_loaded", count=len(loaded))
        return loaded

    async def get_instrument(self, symbol: Symbol) -> Instrument:
        """Fetch one instrument's rules, loading markets on a cache miss."""
        cached = self._instruments.get(symbol)
        if cached is not None:
            return cached
        await self.load_instruments()
        instrument = self._instruments.get(symbol)
        if instrument is None:
            raise InvalidSymbolError(
                f"{symbol} is not listed on {self.name} " f"({self._settings.market_type.value})",
                symbol=str(symbol),
            )
        return instrument

    async def fetch_candles(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        since: datetime | None = None,
        limit: int = MAX_CANDLES_PER_REQUEST,
    ) -> list[Candle]:
        """Fetch OHLCV bars, oldest first."""
        capped = min(limit, MAX_CANDLES_PER_REQUEST)
        rows = await self._call(
            "fetch_candles",
            self._client.fetch_ohlcv,
            self._ccxt_symbol(symbol),
            timeframe.value,
            to_epoch_ms(since) if since else None,
            capped,
        )
        return [Candle.from_ccxt(symbol, timeframe, row) for row in rows or []]

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:
        """Fetch the current best bid/ask and last price."""
        raw = await self._call("fetch_ticker", self._client.fetch_ticker, self._ccxt_symbol(symbol))
        return _parse_ticker(symbol, raw, self._clock.now())

    async def fetch_order_book(self, symbol: Symbol, *, depth: int = 20) -> OrderBook:
        """Fetch an L2 order-book snapshot."""
        raw = await self._call(
            "fetch_order_book", self._client.fetch_order_book, self._ccxt_symbol(symbol), depth
        )
        return _parse_order_book(symbol, raw, self._clock.now())

    async def fetch_recent_trades(self, symbol: Symbol, *, limit: int = 100) -> list[Trade]:
        """Fetch recent public trades."""
        rows = await self._call(
            "fetch_trades", self._client.fetch_trades, self._ccxt_symbol(symbol), None, limit
        )
        return [_parse_public_trade(symbol, row) for row in rows or []]

    async def server_time(self) -> datetime:
        """Fetch the venue's clock."""
        from quantflow.core.clock import from_epoch_ms

        milliseconds = await self._call("server_time", self._client.fetch_time)
        return from_epoch_ms(int(milliseconds))

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    async def submit_order(self, request: OrderRequest) -> Order:
        """Submit an order to the venue.

        The request is normalised onto the venue grids and validated against its rules
        *before* it leaves the process, so a precision violation surfaces as a local
        :class:`ValidationError` rather than a venue rejection.
        """
        self._require_trading()
        instrument = await self.get_instrument(request.symbol)
        normalised = normalize_order(request, instrument)

        reference = normalised.price or normalised.trigger_price
        if reference is None:
            ticker = await self.fetch_ticker(normalised.symbol)
            reference = ticker.price_for(normalised.side)
        instrument.validate_order(normalised.quantity, reference)

        params: dict[str, Any] = {
            "clientOrderId": normalised.client_order_id,
            "newOrderRespType": "RESULT",
        }
        if not normalised.order_type.is_market:
            params["timeInForce"] = TIME_IN_FORCE_TO_CCXT[normalised.time_in_force]
        if normalised.trigger_price is not None:
            params["stopPrice"] = float(normalised.trigger_price)
        if normalised.reduce_only:
            params["reduceOnly"] = True

        raw = await self._call(
            "submit_order",
            self._client.create_order,
            self._ccxt_symbol(normalised.symbol),
            ORDER_TYPE_TO_CCXT[normalised.order_type],
            normalised.side.value,
            float(normalised.quantity),
            float(normalised.price) if normalised.price is not None else None,
            params,
        )

        order = Order.from_request(normalised, now=self._clock.now())
        acknowledged = parse_order(
            raw,
            local_order_id=order.order_id,
            strategy_id=normalised.strategy_id,
            stop_loss_price=normalised.stop_loss_price,
            take_profit_price=normalised.take_profit_price,
        )
        if acknowledged.venue_order_id:
            self._local_by_venue_id[acknowledged.venue_order_id] = order.order_id

        logger.info(
            "exchange.order_submitted",
            order_id=order.order_id,
            venue_order_id=acknowledged.venue_order_id,
            symbol=str(normalised.symbol),
            side=normalised.side.value,
            order_type=normalised.order_type.value,
            quantity=str(normalised.quantity),
            price=str(normalised.price) if normalised.price else None,
            status=acknowledged.status.value,
        )
        return acknowledged

    async def cancel_order(self, order_id: str, symbol: Symbol) -> Order:
        """Cancel a working order."""
        self._require_trading()
        venue_id = self._venue_id_for(order_id)
        raw = await self._call(
            "cancel_order", self._client.cancel_order, venue_id, self._ccxt_symbol(symbol)
        )
        cancelled = parse_order(raw, local_order_id=order_id)
        logger.info(
            "exchange.order_cancelled",
            order_id=order_id,
            venue_order_id=venue_id,
            status=cancelled.status.value,
        )
        return cancelled

    async def fetch_order(self, order_id: str, symbol: Symbol) -> Order:
        """Fetch the current state of one order."""
        self._require_trading()
        venue_id = self._venue_id_for(order_id)
        try:
            raw = await self._call(
                "fetch_order", self._client.fetch_order, venue_id, self._ccxt_symbol(symbol)
            )
        except ExchangeError as exc:
            if "not found" in str(exc).lower():
                raise NotFoundError(
                    f"order {order_id} not found on {self.name}", order_id=order_id
                ) from exc
            raise
        return parse_order(raw, local_order_id=order_id)

    async def fetch_open_orders(self, symbol: Symbol | None = None) -> list[Order]:
        """Fetch every order still working on the venue."""
        self._require_trading()
        rows = await self._call(
            "fetch_open_orders",
            self._client.fetch_open_orders,
            self._ccxt_symbol(symbol) if symbol else None,
        )
        orders: list[Order] = []
        for raw in rows or []:
            venue_id = str(raw.get("id") or "")
            orders.append(parse_order(raw, local_order_id=self._local_by_venue_id.get(venue_id)))
        return orders

    async def fetch_my_trades(
        self, symbol: Symbol, *, since: datetime | None = None, limit: int = 100
    ) -> list[Fill]:
        """Fetch our own executions — the source of truth for reconciliation."""
        self._require_trading()
        rows = await self._call(
            "fetch_my_trades",
            self._client.fetch_my_trades,
            self._ccxt_symbol(symbol),
            to_epoch_ms(since) if since else None,
            limit,
        )
        fills: list[Fill] = []
        for raw in rows or []:
            venue_order_id = str(raw.get("order") or "")
            fills.append(
                parse_fill(
                    raw,
                    order_id=self._local_by_venue_id.get(venue_order_id, venue_order_id),
                    symbol=symbol,
                )
            )
        return fills

    async def fetch_balances(self) -> dict[str, Balance]:
        """Fetch account balances keyed by asset."""
        self._require_trading()
        raw = await self._call("fetch_balances", self._client.fetch_balance)
        balances: dict[str, Balance] = {}
        for asset, entry in (raw or {}).items():
            if not isinstance(entry, dict) or "free" not in entry:
                continue
            free = _safe_decimal(entry.get("free"))
            locked = _safe_decimal(entry.get("used"))
            if free == ZERO and locked == ZERO:
                continue
            balances[str(asset)] = Balance(asset=str(asset), free=free, locked=locked)
        return balances

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _require_trading(self) -> None:
        if not self.supports_trading:
            raise ExchangeError(
                "trading requires API credentials; "
                "set QF_EXCHANGE__API_KEY and QF_EXCHANGE__API_SECRET"
            )

    def _venue_id_for(self, order_id: str) -> str:
        """Translate our order id to the venue's, falling back to the id itself."""
        for venue_id, local_id in self._local_by_venue_id.items():
            if local_id == order_id:
                return venue_id
        return order_id

    def register_venue_id(self, order_id: str, venue_order_id: str) -> None:
        """Record a local-to-venue id mapping recovered from the database on restart."""
        self._local_by_venue_id[venue_order_id] = order_id


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _safe_decimal(value: Any) -> Decimal:
    from quantflow.core.precision import to_decimal

    if value is None or value == "":
        return ZERO
    return to_decimal(value)


def _parse_ticker(symbol: Symbol, raw: dict[str, Any], now: datetime) -> Ticker:
    """Build a :class:`Ticker`, falling back to ``last`` when a side is missing.

    Some Binance pairs briefly report a null bid or ask; using ``last`` keeps the mark
    price usable instead of failing an otherwise healthy trading loop.
    """
    from quantflow.core.clock import from_epoch_ms

    last = _safe_decimal(raw.get("last") or raw.get("close"))
    bid = _safe_decimal(raw.get("bid")) or last
    ask = _safe_decimal(raw.get("ask")) or last
    if last == ZERO:
        raise ValidationError(f"ticker for {symbol} has no usable price", symbol=str(symbol))
    timestamp = raw.get("timestamp")
    return Ticker(
        symbol=symbol,
        timestamp=from_epoch_ms(int(timestamp)) if timestamp else now,
        bid=min(bid, ask),
        ask=max(bid, ask),
        last=last,
        bid_volume=_safe_decimal(raw.get("bidVolume")),
        ask_volume=_safe_decimal(raw.get("askVolume")),
    )


def _parse_order_book(symbol: Symbol, raw: dict[str, Any], now: datetime) -> OrderBook:
    from quantflow.core.clock import from_epoch_ms

    def levels(rows: Any) -> tuple[OrderBookLevel, ...]:
        parsed: list[OrderBookLevel] = []
        for row in rows or []:
            if len(row) < ORDER_BOOK_LEVEL_WIDTH:
                continue
            price, quantity = _safe_decimal(row[0]), _safe_decimal(row[1])
            if price > ZERO and quantity > ZERO:
                parsed.append(OrderBookLevel(price=price, quantity=quantity))
        return tuple(parsed)

    timestamp = raw.get("timestamp")
    return OrderBook(
        symbol=symbol,
        timestamp=from_epoch_ms(int(timestamp)) if timestamp else now,
        bids=levels(raw.get("bids")),
        asks=levels(raw.get("asks")),
    )


def _parse_public_trade(symbol: Symbol, raw: dict[str, Any]) -> Trade:
    from quantflow.core.clock import from_epoch_ms, utc_now

    timestamp = raw.get("timestamp")
    return Trade(
        symbol=symbol,
        trade_id=str(raw.get("id") or ""),
        timestamp=from_epoch_ms(int(timestamp)) if timestamp else utc_now(),
        price=_safe_decimal(raw.get("price")),
        quantity=_safe_decimal(raw.get("amount")),
        side=OrderSide.SELL if raw.get("side") == "sell" else OrderSide.BUY,
    )


def parse_ccxt_symbol(raw: str) -> Symbol:
    """Public re-export of the CCXT symbol parser."""
    return from_ccxt_symbol(raw)
