"""Bybit V5 REST gateway, implemented over CCXT.

CCXT rather than a hand-rolled V5 client: its Bybit implementation already targets the V5
endpoints, tracks their signing rules and keeps up with the category routing that V5
requires. Everything venue-specific that CCXT does *not* normalise — error semantics,
precision handling, symbol forms, category selection — is handled here and in ``mapping``,
so nothing above this module ever sees a CCXT type.

The gateway satisfies the same `ExchangeGateway` protocol as every other venue, which is
why swapping Binance for Bybit touched no strategy, risk, execution or backtest code.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt

from quantflow.core.clock import Clock, SystemClock, to_epoch_ms
from quantflow.core.config import EXCHANGE_HOSTS, ExchangeEnv, ExchangeSettings, MarketType
from quantflow.core.errors import (
    ConfigurationError,
    ExchangeError,
    InvalidSymbolError,
    NotFoundError,
    ValidationError,
)
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO, to_decimal
from quantflow.domain.enums import OrderSide, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, OrderBook, OrderBookLevel, Ticker, Trade
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.domain.portfolio import Balance
from quantflow.exchange.base import MAX_CANDLES_PER_REQUEST, InstrumentCache, normalize_order
from quantflow.exchange.bybit.mapping import (
    ORDER_TYPE_TO_CCXT,
    TIME_IN_FORCE_TO_CCXT,
    bybit_category,
    from_ccxt_symbol,
    parse_fill,
    parse_instrument,
    parse_order,
    to_ccxt_symbol,
    translate_exception,
)
from quantflow.exchange.ratelimit import RateLimiter, retry_async

logger = get_logger(__name__)

#: Which price Bybit compares a protective trigger against. LastPrice is the conservative
#: choice: mark price can diverge from traded price during a squeeze and trigger a stop the
#: market never actually printed.
STOP_TRIGGER_BY = "LastPrice"

#: Relative request costs used to size the token-bucket spend of each call.
#:
#: Bybit V5 rate-limits by requests-per-second per endpoint group rather than by Binance's
#: weight system, so these are not venue-published weights - they are a local ordering
#: that keeps expensive calls from crowding out cheap ones. Kept because the limiter needs
#: *some* relative cost, and a flat 1.0 everywhere would let a burst of order-book reads
#: starve order placement.
ENDPOINT_WEIGHTS: dict[str, float] = {
    "load_markets": 10.0,
    "fetch_candles": 2.0,
    "fetch_ticker": 2.0,
    "fetch_tickers": 10.0,
    "fetch_order_book": 5.0,
    "fetch_trades": 2.0,
    "submit_order": 1.0,
    "cancel_order": 1.0,
    "fetch_order": 2.0,
    "fetch_closed_order": 2.0,
    "fetch_open_orders": 6.0,
    "fetch_my_trades": 10.0,
    "fetch_balances": 10.0,
    "server_time": 1.0,
}

#: Params for ``fetchOrder``. CCXT refuses the call outright unless the caller acknowledges
#: that Bybit only serves the last 500 orders of any status — it raises ArgumentsRequired
#: rather than returning nothing. Without this every post-submit enrichment logged a warning
#: and fell back to the ack-derived record, so a freshly placed order was never read back
#: from the venue that had just accepted it.
FETCH_ORDER_PARAMS: dict[str, Any] = {"acknowledged": True}

#: Extra params tried, in order, when an order has left the realtime feed and has to be
#: looked up in order *history* instead.
#:
#: Bybit files a conditional order under a different history filter from a plain one, and
#: CCXT only sets that filter when told to. A protective stop is precisely the order most
#: likely to end without our having cancelled it — it triggers, or the venue deactivates it
#: when the position goes — so leaving it unfindable would mean the one order class we never
#: submit an exit for is also the one whose outcome we can never read.
FINISHED_ORDER_FILTERS: tuple[dict[str, Any], ...] = ({}, {"trigger": True})

#: Maximum tolerated difference between our clock and Bybit's. V5 rejects a signed request
#: whose timestamp falls outside ``recvWindow`` (error 10002), so drift beyond this is a
#: hard fault rather than something to retry through.
MAX_CLOCK_DRIFT_SECONDS = 5.0

#: Venue codes and CCXT class names meaning "your request timestamp is unacceptable".
#:
#: Matched on the retCode where possible: 10002 is Bybit's signed-request timestamp
#: rejection, and CCXT raises it as ``InvalidNonce``.
_CLOCK_SKEW_CODE = "10002"
_CLOCK_SKEW_CLASS = "InvalidNonce"


def _is_clock_skew(exc: BaseException) -> bool:
    """Whether this failure is the venue refusing our request timestamp."""
    if type(exc).__name__ == _CLOCK_SKEW_CLASS:
        return True
    message = str(exc)
    return f'"retCode":{_CLOCK_SKEW_CODE}' in message.replace(" ", "")


#: An order-book level is [price, quantity].
ORDER_BOOK_LEVEL_WIDTH = 2


def stop_confirmation_is_due(order: Order) -> bool:
    """Whether an accepted entry already has exposure that must be proven protected.

    Protection is confirmed by finding the stop on an **open position**, which only exists
    once the entry has filled. A post-only limit rests until the market reaches it, and
    while it rests the account holds nothing — so there is no position to inspect, and
    demanding one concludes that the stop failed when in truth no stop was needed yet.

    That is not theoretical: twelve minutes after maker-first went live on 2026-08-15 the
    session died on *"stop failed to attach on the venue for SOL/USDT"* with no SOL
    position in existence.

    The invariant is unchanged for anything holding risk. Any fill at all — partial
    included — is a live position and must be proven protected before this returns. Only a
    completely unfilled order defers, and its ``stopLoss`` still travelled with the order,
    so the venue attaches it the instant it fills.

    Exposure is measured by fills and by nothing else. A terminal status is not a proxy for
    it: Bybit *rejects* a post-only order that would cross rather than filling it as taker,
    which is the entire purpose of post-only and a perfectly normal outcome. That order
    comes back terminal with zero fills. Reading terminal as "has a position" made the
    normal case fatal — the code demanded a stop for an entry that never opened, then tried
    to flatten it, and the venue answered "current position is zero, cannot fix reduce-only
    order qty". It killed the session twice on 2026-08-15.
    """
    return order.filled_quantity > ZERO


def bybit_order_params(request: OrderRequest) -> dict[str, Any]:
    """Build the V5 parameters for one order.

    Extracted from ``submit_order`` so the mapping can be asserted without a venue. It
    silently dropping a field is not a hypothetical: ``post_only`` was set correctly by the
    risk engine, carried by ``OrderRequest`` and ``Order``, and then never read here. The
    order left as a plain GTC limit priced at the touch, crossed immediately, and was
    charged the 0.06% taker rate — making maker-first a no-op on the fee bill it exists to
    reduce. Measured live on 2026-08-15: the first "maker" entry filled at 0.0550%,
    identical to every taker fill before it.

    Post-only is expressed through ``timeInForce`` rather than a separate flag because that
    is Bybit's own vocabulary: ``PostOnly`` is a time-in-force value on V5, not an order
    attribute.
    """
    params: dict[str, Any] = {
        "clientOrderId": request.client_order_id,
        "newOrderRespType": "RESULT",
    }
    if not request.order_type.is_market:
        # Post-only supersedes the requested time-in-force: an order that must not take
        # liquidity cannot also be immediate-or-cancel, and the venue would reject the
        # combination rather than quietly pick one.
        params["timeInForce"] = (
            "PostOnly" if request.post_only else TIME_IN_FORCE_TO_CCXT[request.time_in_force]
        )
    if request.trigger_price is not None:
        params["stopPrice"] = float(request.trigger_price)
    if request.reduce_only:
        params["reduceOnly"] = True

    # Sent as strings: Bybit accepts them, and a Decimal formatted as str keeps the exact
    # price the risk engine computed instead of whatever the nearest float happens to be.
    if request.stop_loss_price is not None:
        params["stopLoss"] = str(request.stop_loss_price)
        params["slTriggerBy"] = STOP_TRIGGER_BY
    if request.take_profit_price is not None:
        params["takeProfit"] = str(request.take_profit_price)
        params["tpTriggerBy"] = STOP_TRIGGER_BY
        # The target rests as a limit and earns the maker rate; the stop keeps market
        # execution. Exits are roughly half of all fills, so leaving them aggressive threw
        # away half the saving maker-first exists for.
        #
        # The route was found the hard way, and both dead ends cost a live session:
        #
        #   tpLimitPrice, no mode           -> 10001 "tpLimitPrice can not have a value
        #                                      when tpSlMode is empty"
        #   tpslMode=Full + tpOrderType=Limit -> 10001 "tpOrderType only support Market
        #                                      when tpSlMode is Full"
        #
        # Partial is the mode that accepts a limit target. It was verified against the demo
        # venue before shipping rather than discovered in production a third time.
        #
        # Partial sizes each leg independently, which is the one real hazard here: a stop
        # sized below the position would leave the remainder naked while every local record
        # claimed it was covered. Both legs are therefore stated explicitly at the full
        # quantity rather than left to a venue default. The stop stays Market because it
        # exists for the case where price is running away, and a stop that waits for a
        # better price is not protection.
        quantity = str(request.quantity)
        params["tpslMode"] = "Partial"
        params["tpOrderType"] = "Limit"
        params["tpLimitPrice"] = str(request.take_profit_price)
        params["tpSize"] = quantity
        params["slOrderType"] = "Market"
        params["slSize"] = quantity
    return params


class BybitGateway:
    """Bybit V5 spot / linear-perpetual gateway.

    Implements :class:`~quantflow.exchange.base.ExchangeGateway`.
    """

    __slots__ = (
        "_client",
        "_clock",
        "_connected",
        "_data_client",
        "_instruments",
        "_limiter",
        "_local_by_venue_id",
        "_quantity_by_order_id",
        "_settings",
        "_strategy_by_order_id",
        "_venue_ids_by_symbol",
    )

    def __init__(self, settings: ExchangeSettings, *, clock: Clock | None = None) -> None:
        self._settings = settings
        self._clock = clock or SystemClock()
        self._instruments = InstrumentCache()
        self._limiter = RateLimiter(
            settings.rate_limit_per_second, settings.rate_limit_burst, clock=self._clock
        )
        self._client = self._build_client(settings)
        # Bybit's testnet carries thin history and synthetic prices, so public data is
        # read from production unless explicitly disabled. This client carries no
        # credentials — public endpoints do not need them.
        self._data_client = (
            self._build_client(
                settings.model_copy(
                    update={
                        "env": ExchangeEnv.MAINNET,
                        "testnet": False,
                        "api_key": None,
                        "api_secret": None,
                        "demo_api_key": None,
                        "demo_api_secret": None,
                    }
                )
            )
            if settings.use_production_market_data
            else self._client
        )
        self._connected = False
        #: Maps a venue order id back to our own order id, so a fetched order reconciles.
        self._local_by_venue_id: dict[str, str] = {}
        #: Quantity we submitted, per order id. Bybit's cancel and fetch acknowledgements
        #: omit the amount, so this is the only source for it when parsing them back.
        self._quantity_by_order_id: dict[str, Decimal] = {}
        #: Order id -> the strategy that produced it, under both local and venue ids.
        #: The venue cannot supply this, so it has to survive locally or attribution dies
        #: on the first re-read.
        self._strategy_by_order_id: dict[str, str] = {}
        #: Venue order ids this gateway has submitted, per symbol - including the conditional
        #: stops Bybit creates alongside an entry. Cleanup and reconciliation need to target
        #: those by venue id; without the registry they are orders we can see but not name.
        self._venue_ids_by_symbol: dict[Symbol, list[str]] = {}

    @staticmethod
    def _build_client(settings: ExchangeSettings) -> ccxt.bybit:
        # Credentials come from the resolved environment, so a demo key can never be sent
        # to production and a mainnet key can never be sent to demo.
        key = settings.active_api_key
        secret = settings.active_api_secret
        client = ccxt.bybit(
            {
                "apiKey": key.get_secret_value() if key else None,
                "secret": secret.get_secret_value() if secret else None,
                "timeout": int(settings.request_timeout_seconds * 1000),
                # CCXT's own throttle stays on as a second line of defence behind our
                # token bucket; belt and braces is cheap here and an IP ban is not.
                "enableRateLimit": True,
                "options": {
                    # Bybit V5 has no "future" category. USDT-margined perpetuals are
                    # "linear"; sending "future" makes CCXT build a request the venue
                    # rejects outright.
                    "defaultType": bybit_category(settings.market_type),
                    "recvWindow": settings.recv_window_ms,
                    "adjustForTimeDifference": True,
                },
            }
        )
        env = settings.resolved_env
        if env is ExchangeEnv.TESTNET:
            client.set_sandbox_mode(True)
        elif env is ExchangeEnv.DEMO:
            # Demo trading is its own host with its own keys - not the sandbox. CCXT models
            # it as a distinct mode; where that is unavailable the host is set directly so
            # the behaviour does not silently fall back to production.
            enable_demo = getattr(client, "enable_demo_trading", None)
            if callable(enable_demo):
                enable_demo(True)
            else:  # pragma: no cover - depends on the installed ccxt version
                client.urls["api"] = dict.fromkeys(
                    client.urls["api"], EXCHANGE_HOSTS[ExchangeEnv.DEMO]
                )

        # Belt and braces: whatever ccxt did above, a non-mainnet configuration must not be
        # left pointing at production.
        if not env.is_mainnet:
            resolved = str(client.urls.get("api", ""))
            if "api.bybit.com" in resolved and "api-demo" not in resolved:
                raise ConfigurationError(
                    f"exchange env is {env.value} but the client resolved to production "
                    f"({resolved}); refusing to build a gateway that would trade real money"
                )
        return client

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Venue identifier."""
        return "bybit"

    @property
    def is_testnet(self) -> bool:
        """Whether this gateway points at the sandbox."""
        return self._settings.testnet

    @property
    def network(self) -> str:
        """The environment this gateway is actually connected to.

        Not derivable from :attr:`is_testnet`: demo is neither testnet nor mainnet, so a
        two-way flag reported a demo session as ``mainnet`` — beside a real balance and
        real open positions, on the one field an operator reads to decide whether the
        money is real.
        """
        return str(self._settings.resolved_env.value)

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
        """Close the underlying HTTP sessions."""
        await self._client.close()
        if self._data_client is not self._client:
            await self._data_client.close()
        self._connected = False
        logger.debug("exchange.closed", exchange=self.name)

    async def _assert_clock_sync(self) -> None:
        """Fail fast on clock drift.

        A signed Bybit V5 request whose timestamp falls outside ``recvWindow`` is rejected
        with error 10002.
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
                if not _is_clock_skew(exc):
                    raise translate_exception(exc) from exc
                # Drift that appeared after connect. CCXT measures the venue offset once
                # and caches it, so a host that suspends and wakes several seconds ahead
                # invalidates that offset with nothing to correct it — every signed request
                # is then rejected, indefinitely. Measured live on 2026-08-15: 2h40m of
                # total venue blackout from a 4.3s skew.
                #
                # Reloaded and retried exactly once. A second failure is a clock that is
                # genuinely wrong rather than merely drifted, and must surface rather than
                # become a silent retry loop.
                logger.warning(
                    "exchange.clock_skew_detected",
                    exchange=self.name,
                    action="reloading the venue time offset and retrying once",
                    error=str(exc)[:200],
                )
                try:
                    await self._client.load_time_difference()
                except Exception as reload_error:
                    logger.warning("exchange.clock_resync_failed", error=str(reload_error)[:160])
                    raise translate_exception(exc) from exc
                try:
                    result = await operation(*args, **kwargs)
                except Exception as retry_error:
                    raise translate_exception(retry_error) from retry_error
                logger.info("exchange.clock_resynced", exchange=self.name)
                return result

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
        markets = await self._call("load_markets", self._data_client.load_markets, True)
        loaded: dict[Symbol, Instrument] = {}
        for market in (markets or {}).values():
            instrument = parse_instrument(market)
            if instrument is None:
                continue
            if instrument.market_type is not self._settings.market_type:
                continue
            existing = loaded.get(instrument.symbol)
            if existing is not None:
                # Two venue markets can normalise onto one Symbol. Whichever arrived first
                # is kept, because silently replacing it would make the lot step and the
                # minimum notional depend on dict ordering - and an order sized against the
                # wrong grid is rejected by the venue with no clue as to why.
                if existing != instrument:
                    logger.warning(
                        "exchange.instrument_symbol_collision",
                        symbol=str(instrument.symbol),
                        kept_min_quantity=str(existing.min_quantity),
                        discarded_min_quantity=str(instrument.min_quantity),
                        discarded_market=str(market.get("symbol")),
                    )
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
                f"{symbol} is not listed on {self.name} ({self._settings.market_type.value})",
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
            self._data_client.fetch_ohlcv,
            self._ccxt_symbol(symbol),
            timeframe.value,
            to_epoch_ms(since) if since else None,
            capped,
        )
        return [Candle.from_ccxt(symbol, timeframe, row) for row in rows or []]

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:
        """Fetch the current best bid/ask and last price."""
        raw = await self._call(
            "fetch_ticker", self._data_client.fetch_ticker, self._ccxt_symbol(symbol)
        )
        return _parse_ticker(symbol, raw, self._clock.now())

    async def fetch_quote_turnover_24h(self) -> dict[Symbol, Decimal]:
        """Rolling 24h traded value per symbol, in the quote asset, for the whole category.

        One request for the entire universe. The per-symbol alternative is a request each,
        and ranking ~800 linear markets by liquidity at startup that way would take longer
        than the bar it is trying to start inside — quite apart from what it would do to
        the rate limiter.

        ``turnover24h`` is read from the venue's own payload rather than reconstructed by
        summing candles. A 96-bar sum is an approximation that silently degrades whenever a
        bar is missing, and it answers a slightly different question: the venue's figure is
        a true rolling window, while the sum is however many bars happened to come back.

        Symbols the venue reports without a usable turnover are omitted rather than
        recorded as zero. Absent means "not measured"; zero means "nothing traded", and a
        liquidity filter that confuses the two rejects markets for having no data.
        """
        raw = await self._call("fetch_tickers", self._data_client.fetch_tickers)
        turnover: dict[Symbol, Decimal] = {}
        for ccxt_symbol, ticker in (raw or {}).items():
            info = ticker.get("info") if isinstance(ticker, dict) else None
            value = info.get("turnover24h") if isinstance(info, dict) else None
            if value in (None, ""):
                continue
            try:
                amount = to_decimal(value)
                symbol = from_ccxt_symbol(str(ccxt_symbol))
            except Exception as exc:
                # One malformed row must not lose the other 800. Logged rather than passed
                # over in silence: a symbol that never appears in the ranking would
                # otherwise look like a market the venue does not list.
                logger.debug(
                    "exchange.turnover_row_skipped",
                    symbol=str(ccxt_symbol),
                    error=str(exc)[:120],
                )
                continue
            if amount >= ZERO:
                turnover[symbol] = amount
        return turnover

    async def fetch_order_book(self, symbol: Symbol, *, depth: int = 20) -> OrderBook:
        """Fetch an L2 order-book snapshot."""
        raw = await self._call(
            "fetch_order_book", self._data_client.fetch_order_book, self._ccxt_symbol(symbol), depth
        )
        return _parse_order_book(symbol, raw, self._clock.now())

    async def fetch_recent_trades(self, symbol: Symbol, *, limit: int = 100) -> list[Trade]:
        """Fetch recent public trades."""
        rows = await self._call(
            "fetch_trades", self._data_client.fetch_trades, self._ccxt_symbol(symbol), None, limit
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
            # A MARKET order carries no price of its own, so the ticker stands in purely as
            # a reference for the tick and notional checks. It is a quote, not an order
            # price, and nothing guarantees it sits on the venue's grid — the last trade or
            # a derived mid need not. Validating it unsnapped is what killed every market
            # entry on ETH/USDT with "price 1893.93 is not a multiple of tick 0.1".
            ticker = await self.fetch_ticker(normalised.symbol)
            reference = instrument.normalize_price(
                ticker.price_for(normalised.side),
                side_is_buy=normalised.side is OrderSide.BUY,
            )
        instrument.validate_order(normalised.quantity, reference)

        # An entry with no stop must never reach the venue. The in-memory portfolio used to
        # be the only thing that believed a position was protected; the venue knew nothing,
        # so a fill sat naked through any disconnect, restart or crash.
        if not normalised.reduce_only and normalised.stop_loss_price is None:
            raise ValidationError(
                f"refusing to submit an unprotected entry for {normalised.symbol}: "
                "no stop_loss_price. A protective stop must be attached atomically with "
                "the entry, not applied afterwards.",
                symbol=str(normalised.symbol),
            )

        params = bybit_order_params(normalised)

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

        # An ack carrying an order id means the venue ACCEPTED the order. Bybit V5 returns
        # only orderId/orderLinkId there - no amount, no price, no fees - so parsing it as
        # though it were a complete order used to raise on the positive-quantity invariant.
        # Raising on a well-formed ack reports failure for an order that is live on the
        # exchange: an orphan, unprotected, invisible to the local book. Acceptance is
        # decided by the id; the numbers are then fetched from the authoritative source.
        if not self._ack_has_order_id(raw):
            raise ExchangeError(
                f"venue acknowledged the order for {normalised.symbol} without an order id; "
                "cannot confirm it was accepted",
                symbol=str(normalised.symbol),
            )

        acknowledged = parse_order(
            raw,
            local_order_id=order.order_id,
            strategy_id=normalised.strategy_id,
            stop_loss_price=normalised.stop_loss_price,
            take_profit_price=normalised.take_profit_price,
            fallback_quantity=normalised.quantity,
        )
        self._quantity_by_order_id[order.order_id] = normalised.quantity
        # Recorded before any venue read can blank it. Everything downstream — the enrich
        # below, reconciliation, fill attribution — re-parses from venue payloads that have
        # never heard of a strategy.
        self.remember_strategy(
            order.order_id, normalised.strategy_id, venue_order_id=acknowledged.venue_order_id
        )
        if acknowledged.venue_order_id:
            self._local_by_venue_id[acknowledged.venue_order_id] = order.order_id
            self._venue_ids_by_symbol.setdefault(normalised.symbol, []).append(
                acknowledged.venue_order_id
            )

        acknowledged = await self._enrich_from_venue(acknowledged, normalised)

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

        if (
            not normalised.reduce_only
            and normalised.stop_loss_price is not None
            and stop_confirmation_is_due(acknowledged)
        ):
            await self._require_stop_on_venue(normalised, acknowledged, raw)
        elif not normalised.reduce_only and normalised.stop_loss_price is not None:
            logger.info(
                "exchange.stop_confirmation_deferred",
                symbol=str(normalised.symbol),
                order_id=acknowledged.order_id,
                reason=(
                    "the entry is resting unfilled, so no position exists to protect; the "
                    "venue holds the stop in the order and attaches it on fill"
                ),
            )
        return acknowledged

    def submitted_venue_ids(self, symbol: Symbol | None = None) -> list[str]:
        """Venue order ids submitted through this gateway."""
        if symbol is not None:
            return list(self._venue_ids_by_symbol.get(symbol, []))
        return [i for ids in self._venue_ids_by_symbol.values() for i in ids]

    def track_venue_order(self, symbol: Symbol, venue_order_id: str, quantity: Decimal) -> None:
        """Register an order discovered on the venue so cancel/cleanup can target it.

        Conditional stops are created by the exchange alongside an entry, so they never pass
        through ``submit_order`` and have no local record. Registering them on discovery is
        what lets cleanup cancel them by venue id with a usable quantity.
        """
        self._venue_ids_by_symbol.setdefault(symbol, []).append(venue_order_id)
        self._quantity_by_order_id[venue_order_id] = quantity

    @staticmethod
    def _ack_has_order_id(raw: Any) -> bool:
        """Whether the acknowledgement identifies an accepted order.

        The presence of an id is the acceptance signal. Amount, price and fees are absent
        from a V5 create_order ack and say nothing about whether it landed.
        """
        if not isinstance(raw, dict):
            return False
        if raw.get("id"):
            return True
        info = raw.get("info")
        if isinstance(info, dict):
            result = info.get("result")
            if isinstance(result, dict) and (result.get("orderId") or result.get("orderLinkId")):
                return True
            if info.get("orderId") or info.get("orderLinkId"):
                return True
        return bool(raw.get("clientOrderId"))

    async def _enrich_from_venue(self, acknowledged: Order, request: OrderRequest) -> Order:
        """Replace ack placeholders with the venue's authoritative view.

        Best-effort by design: the order is already accepted, so a failure to read it back
        must not turn a live order into a reported failure. The ack-derived record stands in
        until the next reconciliation pass if this cannot complete.
        """
        venue_id = acknowledged.venue_order_id
        if not venue_id:
            return acknowledged
        try:
            authoritative = await self.fetch_order(acknowledged.order_id, request.symbol)
        except Exception as exc:
            # Deliberately broad. The order is ALREADY ACCEPTED by the venue; letting any
            # failure here propagate would report failure for a live order - precisely the
            # orphan this method exists to prevent. Enrichment is best-effort by contract.
            logger.warning(
                "exchange.order_enrich_failed",
                order_id=acknowledged.order_id,
                venue_order_id=venue_id,
                error=str(exc)[:160],
            )
            return acknowledged
        logger.info(
            "exchange.order_enriched",
            order_id=authoritative.order_id,
            status=authoritative.status.value,
            filled=str(authoritative.filled_quantity),
            average=str(authoritative.average_fill_price),
        )
        return authoritative

    async def _require_stop_on_venue(self, request: OrderRequest, order: Order, raw: Any) -> None:
        """Confirm the venue is holding a stop, and close the entry if it is not.

        "Protected" has to mean the exchange says so. An accepted entry whose ``stopLoss``
        silently failed to attach leaves a real position with no server-side protection,
        and the local record would keep claiming it was covered. If confirmation fails the
        entry is closed reduce-only immediately: an unprotected position is a worse outcome
        than a flat one, and no amount of local bookkeeping substitutes for the venue.

        Raises:
            ExchangeError: if no stop can be confirmed, after attempting to close.

        """
        if self._stop_confirmed_in_response(raw):
            return
        try:
            positions = await self.fetch_positions()
        except ExchangeError as exc:
            logger.exception(
                "exchange.stop_confirmation_unavailable",
                symbol=str(request.symbol),
                error=str(exc),
            )
            positions = []

        target = self._ccxt_symbol(request.symbol)
        for position in positions:
            if str(position.get("symbol", "")).replace("/", "").replace(":USDT", "") not in (
                target.replace("/", "").replace(":USDT", ""),
            ):
                continue
            info = position.get("info", {}) if isinstance(position, dict) else {}
            stop = info.get("stopLoss") or position.get("stopLossPrice")
            if stop not in (None, "", "0", 0):
                logger.info(
                    "exchange.stop_confirmed",
                    symbol=str(request.symbol),
                    stop_loss=str(stop),
                )
                return

        logger.critical(
            "exchange.stop_attach_failed",
            symbol=str(request.symbol),
            order_id=order.order_id,
            requested_stop=str(request.stop_loss_price),
        )
        await self._emergency_flatten(request)
        raise ExchangeError(
            f"stop failed to attach on the venue for {request.symbol}; the entry was closed "
            "reduce-only. Refusing to hold an unprotected position.",
            symbol=str(request.symbol),
        )

    @staticmethod
    def _stop_confirmed_in_response(raw: Any) -> bool:
        """Whether the venue's own response evidences a live stop.

        Reads the raw payload deliberately. The parsed local order carries the stop we
        *asked* for - `parse_order` is handed `normalised.stop_loss_price` - so trusting it
        would confirm protection from our own request and reproduce the exact defect this
        method exists to catch.
        """
        if not isinstance(raw, dict):
            return False
        info = raw.get("info")
        stop = info.get("stopLoss") if isinstance(info, dict) else None
        if stop in (None, "", "0", 0):
            return False
        try:
            return Decimal(str(stop)) > ZERO
        except (ArithmeticError, ValueError):
            return False

    async def _emergency_flatten(self, request: OrderRequest) -> None:
        """Close an entry that could not be protected, reduce-only.

        Best-effort and never raises: the caller is already raising, and masking that with a
        secondary failure would hide why the position was being closed.
        """
        from quantflow.domain.enums import OrderType

        closing = OrderSide.SELL if request.side is OrderSide.BUY else OrderSide.BUY
        try:
            await self._call(
                "submit_order",
                self._client.create_order,
                self._ccxt_symbol(request.symbol),
                ORDER_TYPE_TO_CCXT[OrderType.MARKET],
                closing.value,
                float(request.quantity),
                None,
                {"reduceOnly": True, "newOrderRespType": "RESULT"},
            )
            logger.critical(
                "exchange.unprotected_entry_closed",
                symbol=str(request.symbol),
                quantity=str(request.quantity),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "exchange.emergency_close_failed",
                symbol=str(request.symbol),
                error=str(exc),
            )

    async def cancel_order(
        self, order_id: str, symbol: Symbol, *, quantity: Decimal | None = None
    ) -> Order:
        """Cancel a working order.

        ``quantity`` covers orders this process did not submit - a conditional stop the
        venue created alongside an entry, for instance. Bybit's cancel acknowledgement omits
        the amount, and without a fallback the parsed Order fails its positive-quantity
        invariant on an order that was cancelled perfectly well.
        """
        self._require_trading()
        venue_id = self._venue_id_for(order_id)
        raw = await self._call(
            "cancel_order", self._client.cancel_order, venue_id, self._ccxt_symbol(symbol)
        )
        cancelled = parse_order(
            raw,
            local_order_id=order_id,
            fallback_quantity=quantity or self._quantity_by_order_id.get(order_id),
        )
        logger.info(
            "exchange.order_cancelled",
            order_id=order_id,
            venue_order_id=venue_id,
            status=cancelled.status.value,
        )
        return cancelled

    async def fetch_order(self, order_id: str, symbol: Symbol) -> Order:
        """Fetch the current state of one order, whether it is still working or finished.

        Two venue reads, not one. CCXT's ``fetchOrder`` serves Bybit's *realtime* order
        endpoint, which carries open orders plus a brief tail of recently-finished ones;
        past that window it raises ``OrderNotFound``. Treating that as the answer meant the
        only orders whose status could ever be read were the ones still working — so an
        order that had filled, been cancelled or been rejected an hour ago was
        indistinguishable from one the venue had never heard of, and the OMS had no way to
        learn any terminal state except by seeing an execution. A rejection and a cancel
        produce no execution at all.

        Order history answers for the rest, and only a miss in *both* is a genuine miss.

        Raises:
            NotFoundError: if neither the realtime feed nor order history knows the id.

        """
        self._require_trading()
        venue_id = self._venue_id_for(order_id)
        ccxt_symbol = self._ccxt_symbol(symbol)

        try:
            raw = await self._call(
                "fetch_order",
                self._client.fetch_order,
                venue_id,
                ccxt_symbol,
                # CCXT's signature here is ``(id, symbol, params)`` — three arguments, not
                # the ``(id, symbol, since, params)`` shape several other endpoints use.
                # An extra positional made every call raise TypeError before it reached the
                # network, so no order in a live session was ever read back from the venue:
                # a rejection or a cancel, neither of which produces an execution, was
                # invisible, and the order sat at NEW for the life of the session.
                FETCH_ORDER_PARAMS,
            )
        except ExchangeError as exc:
            if not _is_order_not_found(exc):
                raise
            raw = await self._fetch_finished_order(venue_id, ccxt_symbol, order_id=order_id)
        return parse_order(raw, local_order_id=order_id, strategy_id=self.strategy_for(order_id))

    async def _fetch_finished_order(
        self, venue_id: str, ccxt_symbol: str, *, order_id: str
    ) -> dict[str, Any]:
        """Look an order up in Bybit's order history, plain first then conditional."""
        for extra in FINISHED_ORDER_FILTERS:
            try:
                raw = await self._call(
                    "fetch_closed_order",
                    self._client.fetch_closed_order,
                    venue_id,
                    ccxt_symbol,
                    dict(extra),
                )
            except ExchangeError as exc:
                if not _is_order_not_found(exc):
                    raise
                continue
            if isinstance(raw, dict):
                return raw
        raise NotFoundError(f"order {order_id} not found on {self.name}", order_id=order_id)

    async def set_trading_stop(
        self, symbol: Symbol, *, stop_loss: Decimal | None = None
    ) -> Decimal | None:
        """Amend the venue-side protective stop on an open position.

        Bybit carries the stop on the *position*, not as a separate order, so moving it is
        a position amendment rather than a cancel-and-replace — which matters, because a
        cancel-and-replace leaves a window with no protection at all.

        The new level is read back from the venue before it is returned. An amendment that
        was silently rejected would otherwise leave the caller believing a position is
        protected at a level the exchange never accepted, which is worse than not having
        moved it: the risk is unchanged but the reported risk is not.

        Returns:
            The stop the venue confirms it is holding, or ``None`` if it reports none.

        Raises:
            ExchangeError: if the venue rejects the amendment.

        """
        self._require_trading()
        instrument = await self.get_instrument(symbol)
        params: dict[str, Any] = {
            "category": bybit_category(self._settings.market_type),
            "symbol": to_ccxt_symbol(symbol).replace("/", "").replace(":USDT", ""),
            "positionIdx": 0,
        }
        if stop_loss is not None:
            snapped = instrument.normalize_stop_price(stop_loss, position_is_long=True)
            params["stopLoss"] = str(snapped)
            params["slTriggerBy"] = STOP_TRIGGER_BY

        await self._call(
            "set_trading_stop",
            self._client.private_post_v5_position_trading_stop,
            params,
        )

        for position in await self.fetch_positions():
            if str(position.get("symbol")) != to_ccxt_symbol(symbol):
                continue
            info = position.get("info") or {}
            confirmed = info.get("stopLoss")
            return Decimal(str(confirmed)) if confirmed else None
        return None

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
            local_id = self._local_by_venue_id.get(venue_id)
            orders.append(
                parse_order(
                    raw,
                    local_order_id=local_id,
                    strategy_id=self.strategy_for(local_id or venue_id),
                )
            )
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
    async def raw_account_info(self) -> dict[str, Any]:
        """Bybit V5 account configuration: unified status and margin mode.

        V5 returns *different balance structures* for UNIFIED and CLASSIC accounts, so a
        balance figure cannot be trusted until the account type is known. There is no
        CCXT-unified call for this, so the V5 endpoint is invoked directly — one of the
        few places a venue-specific route is unavoidable.
        """
        self._require_trading()
        raw = await self._call("account_info", self._client.privateGetV5AccountInfo)
        result = (raw or {}).get("result")
        return result if isinstance(result, dict) else {}

    async def fetch_funding_history(
        self, symbol: Symbol, *, since: datetime | None = None, limit: int = 200
    ) -> list[tuple[datetime, Decimal]]:
        """Historical 8h funding rates, oldest first.

        Read from the public data client: funding history is public, and a backtest should
        not need credentials to price a cost the venue publishes. Rates come back as
        ``Decimal`` via ``str`` so a float literal never reaches money arithmetic.
        """
        rows = await self._call(
            "fetch_candles",
            self._data_client.fetch_funding_rate_history,
            self._ccxt_symbol(symbol),
            to_epoch_ms(since) if since is not None else None,
            limit,
        )
        out: list[tuple[datetime, Decimal]] = []
        for row in rows or []:
            stamp = row.get("timestamp")
            rate = row.get("fundingRate")
            if stamp is None or rate is None:
                continue
            out.append(
                (
                    datetime.fromtimestamp(int(stamp) / 1000, tz=UTC),
                    Decimal(str(rate)),
                )
            )
        out.sort(key=lambda item: item[0])
        return out

    async def set_leverage(self, symbol: Symbol, leverage: Decimal) -> bool:
        """Set a symbol's leverage on the venue before trading it.

        The bot assumes 1x. Assuming is not enough: if Bybit has the symbol at 10x it
        reserves a tenth of the margin the bot believes, so free margin, exposure and every
        equity-derived limit are computed against a reservation that does not exist. Setting
        it explicitly makes the venue agree with the assumption rather than the other way
        round.

        Returns whether the venue accepted the change. Bybit rejects a no-op change with
        "leverage not modified" (110043), which is success for our purposes - the symbol is
        already where we want it.
        """
        if self._settings.market_type is not MarketType.FUTURE:
            return False
        try:
            await self._call(
                "set_leverage",
                self._client.set_leverage,
                float(leverage),
                self._ccxt_symbol(symbol),
            )
        except ExchangeError as exc:
            message = str(exc).lower()
            if "not modified" in message or "110043" in message:
                logger.debug(
                    "exchange.leverage_already_set", symbol=str(symbol), leverage=str(leverage)
                )
                return True
            logger.warning(
                "exchange.set_leverage_failed",
                symbol=str(symbol),
                leverage=str(leverage),
                error=str(exc)[:160],
            )
            return False
        logger.info("exchange.leverage_set", symbol=str(symbol), leverage=str(leverage))
        return True

    async def fetch_positions(self) -> list[dict[str, Any]]:
        """Open derivative positions.

        Spot accounts have no positions endpoint on V5; callers get an empty list rather
        than an error, because "no positions" is the correct answer for a spot account
        and raising here would make a spot verification look like a failure.
        """
        self._require_trading()
        if self._settings.market_type is not MarketType.FUTURE:
            return []
        rows = await self._call("fetch_positions", self._client.fetch_positions)
        return [row for row in (rows or []) if isinstance(row, dict)]

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

    def remember_strategy(
        self, order_id: str, strategy_id: str | None, *, venue_order_id: str | None = None
    ) -> None:
        """Record which strategy produced an order, under both of its identities.

        The venue never returns this. Every read of an order from Bybit reconstructs it
        from a payload that has no notion of a strategy, so without a local memory the
        attribution is lost the first time the order is re-read — which happens
        immediately, because ``submit_order`` enriches from the venue for authoritative
        fill data.
        """
        if not strategy_id:
            return
        self._strategy_by_order_id[order_id] = strategy_id
        if venue_order_id:
            self._strategy_by_order_id[venue_order_id] = strategy_id

    def strategy_for(self, order_id: str) -> str | None:
        """The strategy behind an order, or ``None`` if this gateway never saw it."""
        return self._strategy_by_order_id.get(order_id)

    def register_venue_id(self, order_id: str, venue_order_id: str) -> None:
        """Record a local-to-venue id mapping recovered from the database on restart."""
        self._local_by_venue_id[venue_order_id] = order_id


def _is_order_not_found(exc: ExchangeError) -> bool:
    """Whether the venue's answer was "no such order" rather than a failure to ask.

    CCXT reports it as ``OrderNotFound``, which ``translate_exception`` folds into the same
    ``OrderRejectedError`` as a genuine rejection — so the class alone cannot separate "the
    venue has never heard of this" from "the venue refused it". The venue error name is
    checked first because it is the exact signal; the message is a fallback for the paths
    that lose it.
    """
    if exc.details.get("venue_error") == "OrderNotFound":
        return True
    return "not found" in str(exc).lower()


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _safe_decimal(value: Any) -> Decimal:
    if value is None or value == "":
        return ZERO
    return to_decimal(value)


def _parse_ticker(symbol: Symbol, raw: dict[str, Any], now: datetime) -> Ticker:
    """Build a :class:`Ticker`, falling back to ``last`` when a side is missing.

    Some Bybit pairs briefly report a null bid or ask; using ``last`` keeps the mark
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
