"""CCXT/Bybit translation: symbols, instruments, orders, fills and errors."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import SecretStr

from quantflow.core.config import ExchangeSettings, MarketType
from quantflow.core.errors import (
    ExchangeAuthenticationError,
    ExchangeConnectionError,
    ExchangeError,
    ExchangeTimeoutError,
    InsufficientFundsError,
    InvalidSymbolError,
    OrderRejectedError,
    ProductAgreementRequiredError,
    RateLimitError,
)
from quantflow.domain.enums import (
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    Timeframe,
    TimeInForce,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import OrderRequest
from quantflow.exchange.base import estimate_fee, normalize_order
from quantflow.exchange.bybit.mapping import (
    from_ccxt_symbol,
    parse_fill,
    parse_instrument,
    parse_order,
    parse_order_status,
    parse_order_type,
    parse_side,
    to_ccxt_symbol,
    translate_exception,
)
from quantflow.exchange.bybit.rest import BybitGateway
from quantflow.exchange.bybit.ws import (
    CandleGapDetector,
    TickerBook,
    _parse_kline,
    _parse_ticker,
    _parse_trade,
    bybit_interval,
    stream_url,
)
from tests.conftest import REFERENCE_TIME


def spot_market(**overrides: Any) -> dict[str, Any]:
    market: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "spot": True,
        "linear": False,
        "active": True,
        "maker": 0.001,
        "taker": 0.001,
        "precision": {"price": 2, "amount": 5},
        "limits": {
            "amount": {"min": 0.00001, "max": 9000.0},
            "cost": {"min": 5.0, "max": None},
        },
    }
    market.update(overrides)
    return market


class TestSymbolTranslation:
    def test_spot_symbol(self, btc: Symbol) -> None:
        assert to_ccxt_symbol(btc) == "BTC/USDT"

    def test_futures_symbol_carries_the_settlement_suffix(self, btc: Symbol) -> None:
        assert to_ccxt_symbol(btc, MarketType.FUTURE) == "BTC/USDT:USDT"

    @pytest.mark.parametrize("raw", ["BTC/USDT", "BTC/USDT:USDT"])
    def test_round_trip(self, raw: str, btc: Symbol) -> None:
        assert from_ccxt_symbol(raw) == btc


class TestEnumParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("NEW", OrderStatus.NEW),
            ("open", OrderStatus.NEW),
            ("PARTIALLY_FILLED", OrderStatus.PARTIALLY_FILLED),
            ("FILLED", OrderStatus.FILLED),
            ("closed", OrderStatus.FILLED),
            ("CANCELED", OrderStatus.CANCELLED),
            ("REJECTED", OrderStatus.REJECTED),
            ("EXPIRED", OrderStatus.EXPIRED),
        ],
    )
    def test_status(self, raw: str, expected: OrderStatus) -> None:
        assert parse_order_status(raw) is expected

    def test_unknown_status_defaults_to_new(self) -> None:
        # Dropping an unrecognised status would orphan a live order on the venue.
        assert parse_order_status("SOMETHING_NEW") is OrderStatus.NEW
        assert parse_order_status(None) is OrderStatus.NEW

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("market", OrderType.MARKET),
            ("LIMIT", OrderType.LIMIT),
            ("stop_loss_limit", OrderType.STOP_LIMIT),
            ("take_profit", OrderType.TAKE_PROFIT_MARKET),
            ("limit_maker", OrderType.LIMIT),
        ],
    )
    def test_order_type(self, raw: str, expected: OrderType) -> None:
        assert parse_order_type(raw) is expected

    def test_side(self) -> None:
        assert parse_side("SELL") is OrderSide.SELL
        assert parse_side("buy") is OrderSide.BUY
        assert parse_side(None) is OrderSide.BUY


class TestInstrumentParsing:
    def test_parses_a_spot_market(self, btc: Symbol) -> None:
        instrument = parse_instrument(spot_market())
        assert instrument is not None
        assert instrument.symbol == btc
        assert instrument.price_tick == Decimal("0.01")
        assert instrument.quantity_step == Decimal("0.00001")
        assert instrument.min_notional == Decimal("5")
        assert instrument.market_type is MarketType.SPOT

    def test_accepts_tick_size_style_precision(self) -> None:
        # CCXT reports precision either as a place count or as a literal tick size.
        instrument = parse_instrument(spot_market(precision={"price": 0.01, "amount": 0.001}))
        assert instrument is not None
        assert instrument.price_tick == Decimal("0.01")
        assert instrument.quantity_step == Decimal("0.001")

    def test_skips_markets_without_precision(self) -> None:
        assert parse_instrument(spot_market(precision={})) is None

    def test_skips_non_spot_non_linear_markets(self) -> None:
        assert parse_instrument(spot_market(spot=False, linear=False)) is None

    def test_parses_a_linear_future(self) -> None:
        instrument = parse_instrument(
            spot_market(
                symbol="BTC/USDT:USDT",
                spot=False,
                linear=True,
                contractSize=1,
                limits={
                    "amount": {"min": 0.001},
                    "cost": {"min": 5},
                    "leverage": {"max": 20},
                },
            )
        )
        assert instrument is not None
        assert instrument.market_type is MarketType.FUTURE
        assert instrument.max_leverage == Decimal("20")

    def test_min_quantity_never_below_the_step(self) -> None:
        instrument = parse_instrument(
            spot_market(
                precision={"price": 2, "amount": 3},
                limits={"amount": {"min": 0.0000001}, "cost": {"min": 5}},
            )
        )
        assert instrument is not None
        assert instrument.min_quantity >= instrument.quantity_step

    def test_inactive_market_is_marked_inactive(self) -> None:
        instrument = parse_instrument(spot_market(active=False))
        assert instrument is not None
        assert not instrument.active


class TestInstrumentSymbolCollisions:
    """Two venue markets can normalise onto one Symbol; only one may survive.

    Which one is not a matter of taste. The lot step and the minimum notional come from
    whichever wins, so resolving the tie by dict ordering means an order is sized against a
    grid chosen at random — and the venue rejects it without saying why.
    """

    @staticmethod
    def _gateway(markets: dict[str, Any]) -> BybitGateway:
        gateway = BybitGateway(
            ExchangeSettings(
                name="bybit",
                api_key=SecretStr("k" * 18),
                api_secret=SecretStr("s" * 36),
                testnet=True,
                market_type=MarketType.SPOT,
            )
        )

        async def load_markets(reload: bool = False) -> dict[str, Any]:
            return markets

        gateway._data_client = SimpleNamespace(load_markets=load_markets)
        return gateway

    async def test_the_first_market_wins(self) -> None:
        gateway = self._gateway(
            {
                "a": spot_market(precision={"price": 2, "amount": 3}),
                "b": spot_market(symbol="BTC/USDT:USDT", precision={"price": 2, "amount": 1}),
            }
        )

        loaded = await gateway.load_instruments()

        assert loaded[Symbol.parse("BTC/USDT")].quantity_step == Decimal("0.001")

    async def test_only_one_instrument_survives_the_collision(self) -> None:
        gateway = self._gateway(
            {
                "a": spot_market(precision={"price": 2, "amount": 3}),
                "b": spot_market(symbol="BTC/USDT:USDT", precision={"price": 2, "amount": 1}),
            }
        )

        loaded = await gateway.load_instruments()

        assert len(loaded) == 1

    async def test_markets_of_another_type_are_not_loaded(self) -> None:
        """A spot gateway must not adopt a perpetual's rules, or vice versa."""
        gateway = self._gateway({"a": spot_market(symbol="ETH/USDT:USDT", spot=False, linear=True)})

        assert await gateway.load_instruments() == {}


class TestOrderParsing:
    def _raw(self, **overrides: Any) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "id": "venue-123",
            "clientOrderId": "qf-abc",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "status": "open",
            "amount": 1.5,
            "filled": 0.5,
            "price": 50000.0,
            "average": 50010.0,
            "timeInForce": "GTC",
            "timestamp": 1767225600000,
            "fee": {"cost": 0.25, "currency": "USDT"},
        }
        raw.update(overrides)
        return raw

    def test_parses_a_working_order(self, btc: Symbol) -> None:
        order = parse_order(self._raw())
        assert order.symbol == btc
        assert order.side is OrderSide.BUY
        assert order.order_type is OrderType.LIMIT
        assert order.status is OrderStatus.NEW
        assert order.quantity == Decimal("1.5")
        assert order.filled_quantity == Decimal("0.5")
        assert order.average_fill_price == Decimal("50010")
        assert order.venue_order_id == "venue-123"
        assert order.time_in_force is TimeInForce.GTC
        assert order.created_at == datetime(2026, 1, 1, tzinfo=UTC)

    def test_local_order_id_is_preserved(self) -> None:
        order = parse_order(self._raw(), local_order_id="our-id")
        assert order.order_id == "our-id"
        assert order.venue_order_id == "venue-123"

    def test_overreported_fill_is_clamped(self) -> None:
        # A payload claiming more filled than ordered would violate the Order invariant
        # and crash a reconciliation pass.
        order = parse_order(self._raw(amount=1.0, filled=1.2))
        assert order.filled_quantity == Decimal("1")

    def test_average_price_is_derived_from_fills_when_absent(self, btc: Symbol) -> None:
        order = parse_order(
            self._raw(
                average=None,
                filled=2.0,
                amount=2.0,
                status="closed",
                trades=[
                    {"id": "t1", "side": "buy", "amount": 1.0, "price": 100.0, "timestamp": 1},
                    {"id": "t2", "side": "buy", "amount": 1.0, "price": 200.0, "timestamp": 2},
                ],
            )
        )
        assert order.average_fill_price == Decimal("150")
        assert len(order.fills) == 2

    def test_protective_levels_are_attached(self) -> None:
        order = parse_order(
            self._raw(), stop_loss_price=Decimal("49000"), take_profit_price=Decimal("52000")
        )
        assert order.stop_loss_price == Decimal("49000")
        assert order.take_profit_price == Decimal("52000")


class TestFillParsing:
    def test_parses_a_maker_fill(self, btc: Symbol) -> None:
        fill = parse_fill(
            {
                "id": "t-1",
                "side": "sell",
                "amount": 0.5,
                "price": 50000.0,
                "fee": {"cost": 12.5, "currency": "USDT"},
                "timestamp": 1767225600000,
                "takerOrMaker": "maker",
            },
            order_id="o-1",
            symbol=btc,
        )
        assert fill.side is OrderSide.SELL
        assert fill.quantity == Decimal("0.5")
        assert fill.fee == Decimal("12.5")
        assert fill.role is LiquidityRole.MAKER
        assert fill.notional == Decimal("25000")

    def test_defaults_to_taker(self, btc: Symbol) -> None:
        fill = parse_fill(
            {"id": "t", "side": "buy", "amount": 1, "price": 1, "timestamp": 1},
            order_id="o",
            symbol=btc,
        )
        assert fill.role is LiquidityRole.TAKER
        assert fill.fee == Decimal("0")


class TestErrorTranslation:
    @pytest.mark.parametrize(
        ("ccxt_name", "expected"),
        [
            ("AuthenticationError", ExchangeAuthenticationError),
            ("PermissionDenied", ExchangeAuthenticationError),
            ("RateLimitExceeded", RateLimitError),
            ("DDoSProtection", RateLimitError),
            ("RequestTimeout", ExchangeTimeoutError),
            ("NetworkError", ExchangeConnectionError),
            ("ExchangeNotAvailable", ExchangeConnectionError),
            ("InsufficientFunds", InsufficientFundsError),
            ("BadSymbol", InvalidSymbolError),
            ("InvalidOrder", OrderRejectedError),
        ],
    )
    def test_known_errors_map_to_typed_exceptions(
        self, ccxt_name: str, expected: type[ExchangeError]
    ) -> None:
        fake = type(ccxt_name, (Exception,), {})("boom")
        assert isinstance(translate_exception(fake), expected)

    def test_unknown_error_degrades_to_generic(self) -> None:
        # A CCXT upgrade that renames an exception must not crash the translator.
        fake = type("SomeBrandNewCcxtError", (Exception,), {})("boom")
        translated = translate_exception(fake)
        assert type(translated) is ExchangeError
        assert "SomeBrandNewCcxtError" in str(translated)

    @pytest.mark.parametrize(
        ("ret_code", "ret_msg"),
        [
            # Verified live against api-demo.bybit.com on 2026-08-14. Three separate
            # agreements, three separate codes - signing one does not clear the others.
            (110123, "You must agree to the Trading Terms before trading this contract."),
            (110125, "You must agree to the Crude Oil Trading Terms before trading this contract."),
            (110126, "You must sign the required agreement before trading this contract."),
        ],
    )
    def test_unsigned_product_agreement_is_its_own_error(self, ret_code: int, ret_msg: str) -> None:
        """An unsigned agreement is an account state, not a bad order.

        It must be distinguishable from every other rejection, because the correct
        response is unique: stop trading that asset class and tell the operator to sign,
        rather than retrying, resizing, or failing the session.
        """
        fake = type("ExchangeError", (Exception,), {})(
            f'bybit {{"retCode":{ret_code},"retMsg":"{ret_msg}","result":{{}}}}'
        )
        translated = translate_exception(fake)
        assert isinstance(translated, ProductAgreementRequiredError)
        # Still an order rejection, so existing handlers keep working.
        assert isinstance(translated, OrderRejectedError)
        assert translated.venue_error == str(ret_code)

    def test_ordinary_rejection_is_not_mistaken_for_an_agreement(self) -> None:
        fake = type("InvalidOrder", (Exception,), {})(
            'bybit {"retCode":110007,"retMsg":"ab not enough for new order"}'
        )
        translated = translate_exception(fake)
        assert isinstance(translated, OrderRejectedError)
        assert not isinstance(translated, ProductAgreementRequiredError)


class TestClockDriftRecovery:
    """Drift that appears *after* startup must be corrected without intervention.

    The gateway measures drift once when it connects and CCXT caches the offset. That is
    fine until the host suspends: the container clock wakes several seconds ahead, the
    cached offset is now wrong, and every signed request is rejected with

        retCode 10002 — invalid request, please check your server timestamp

    Observed live on 2026-08-15: the API container ran 11 hours, drifted ~4.3s ahead, and
    every Bybit read failed for 2h40m. Nothing retried the offset, so it never recovered.
    A startup-only check cannot catch this by construction — the clock was correct at
    startup.
    """

    @staticmethod
    def _gateway() -> BybitGateway:
        return BybitGateway(
            ExchangeSettings(
                name="bybit",
                api_key=SecretStr("k" * 18),
                api_secret=SecretStr("s" * 36),
                testnet=True,
                market_type=MarketType.SPOT,
                max_retries=0,
            )
        )

    async def test_an_invalid_nonce_reloads_the_offset_and_retries(self) -> None:
        gateway = self._gateway()
        reloads: list[bool] = []
        attempts: list[int] = []

        async def load_time_difference() -> int:
            reloads.append(True)
            return 0

        async def operation() -> str:
            attempts.append(1)
            if len(attempts) == 1:
                raise type("InvalidNonce", (Exception,), {})(
                    'bybit {"retCode":10002,"retMsg":"invalid request, please check your '
                    'server timestamp or recv_window param"}'
                )
            return "ok"

        gateway._client.load_time_difference = load_time_difference

        result = await gateway._call("test", operation)

        assert result == "ok"
        assert reloads, "the cached time offset must be reloaded before retrying"
        assert len(attempts) == 2, "the call must be retried once the offset is corrected"

    async def test_a_persistent_nonce_failure_still_surfaces(self) -> None:
        # Recovery must not become an infinite retry that hides a genuinely wrong clock.
        gateway = self._gateway()

        async def load_time_difference() -> int:
            return 0

        async def operation() -> str:
            raise type("InvalidNonce", (Exception,), {})(
                'bybit {"retCode":10002,"retMsg":"invalid request, please check your '
                'server timestamp"}'
            )

        gateway._client.load_time_difference = load_time_difference

        with pytest.raises(ExchangeError):
            await gateway._call("test", operation)


class TestOrderNormalisation:
    def test_snaps_quantity_and_price_to_the_venue_grid(
        self, btc: Symbol, btc_instrument: Instrument
    ) -> None:
        request = OrderRequest(
            symbol=btc,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),
            price=Decimal("50000.567"),
        )
        normalised = normalize_order(request, btc_instrument)
        assert normalised.quantity == Decimal("0.12345")
        assert normalised.price == Decimal("50000.56")  # buy rounds down

    def test_sell_price_rounds_up(self, btc: Symbol, btc_instrument: Instrument) -> None:
        request = OrderRequest(
            symbol=btc,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("50000.561"),
        )
        assert normalize_order(request, btc_instrument).price == Decimal("50000.57")

    def test_stop_loss_rounds_away_from_the_position(
        self, btc: Symbol, btc_instrument: Instrument
    ) -> None:
        """A long's stop rounds DOWN — away from entry.

        This reverses the convention this test previously asserted ("rounds up, so it
        triggers no later than intended"). Rounding up moves a long's stop *toward* its own
        entry, which tightens the risk the engine sized for and, on a wide tick, can put
        the stop at or above the trigger price it is supposed to sit below — a venue
        rejection, which leaves the position unprotected entirely.

        The cost of the new rule is that a realised loss can exceed the intended one by up
        to a single tick. That is a bounded, known error; a rejected stop is not.
        """
        request = OrderRequest(
            symbol=btc,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("50000.00"),
            stop_loss_price=Decimal("49000.001"),
        )
        assert normalize_order(request, btc_instrument).stop_loss_price == Decimal("49000.00")

    def test_short_stop_loss_rounds_up_away_from_the_position(
        self, btc: Symbol, btc_instrument: Instrument
    ) -> None:
        """The mirror case: a short's stop sits above entry, so it rounds up."""
        request = OrderRequest(
            symbol=btc,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("50000.00"),
            stop_loss_price=Decimal("51000.001"),
        )
        assert normalize_order(request, btc_instrument).stop_loss_price == Decimal("51000.01")

    def test_normalisation_preserves_metadata(
        self, btc: Symbol, btc_instrument: Instrument
    ) -> None:
        request = OrderRequest(
            symbol=btc,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            strategy_id="ema_cross",
            client_order_id="fixed-id",
        )
        normalised = normalize_order(request, btc_instrument)
        assert normalised.strategy_id == "ema_cross"
        assert normalised.client_order_id == "fixed-id"


def test_estimate_fee(btc_instrument: Instrument) -> None:
    assert estimate_fee(btc_instrument, Decimal("2"), Decimal("50000"), is_maker=False) == Decimal(
        "100"
    )


class TestStreamUrls:
    def test_selects_by_category_and_network(self) -> None:
        from quantflow.core.config import ExchangeSettings

        # V5 opens one socket per category, so the category is part of the URL.
        assert stream_url(ExchangeSettings(testnet=False)) == (
            "wss://stream.bybit.com/v5/public/spot"
        )
        assert stream_url(ExchangeSettings(testnet=True)) == (
            "wss://stream-testnet.bybit.com/v5/public/spot"
        )
        assert (
            stream_url(ExchangeSettings(testnet=False, market_type=MarketType.FUTURE))
            == "wss://stream.bybit.com/v5/public/linear"
        )


class TestIntervalMapping:
    """Bybit counts minutes as bare numbers and switches to letters at daily."""

    def test_hours_are_expressed_in_minutes(self) -> None:
        assert bybit_interval(Timeframe.H1) == "60"
        assert bybit_interval(Timeframe.H4) == "240"

    def test_minutes_pass_through_as_numbers(self) -> None:
        assert bybit_interval(Timeframe.M15) == "15"

    def test_daily_and_weekly_are_letters(self) -> None:
        assert bybit_interval(Timeframe.D1) == "D"
        assert bybit_interval(Timeframe.W1) == "W"

    def test_an_unsupported_interval_is_refused(self) -> None:
        # Substituting a neighbouring interval would feed a strategy bars it never asked
        # for, which is worse than failing.
        from quantflow.core.errors import MarketDataError

        with pytest.raises(MarketDataError, match="no kline interval"):
            bybit_interval(Timeframe.D3)


class TestStreamPayloadParsing:
    """V5 payloads share no field names with Binance's."""

    def test_kline(self, btc: Symbol) -> None:
        candle = _parse_kline(
            btc,
            Timeframe.H1,
            {
                "start": 1767225600000,
                "end": 1767229199999,
                "interval": "60",
                "open": "50000.00",
                "high": "50500.00",
                "low": "49800.00",
                "close": "50200.00",
                "volume": "123.456",
                "turnover": "6200000.00",
                "confirm": True,
            },
        )
        assert candle.open_time == datetime(2026, 1, 1, tzinfo=UTC)
        assert candle.close == Decimal("50200.00")
        # Bybit names quote volume "turnover"; reading "volume" for both would report
        # base volume as quote volume and skew every liquidity measure.
        assert candle.quote_volume == Decimal("6200000.00")
        # V5 klines carry no trade count.
        assert candle.trades == 0

    def test_kline_without_a_start_is_rejected(self, btc: Symbol) -> None:
        from quantflow.core.errors import MarketDataError

        with pytest.raises(MarketDataError, match="no start time"):
            _parse_kline(btc, Timeframe.H1, {"open": "1", "close": "1"})

    def test_ticker(self, btc: Symbol) -> None:
        ticker = _parse_ticker(
            btc,
            {
                "bid1Price": "49999.00",
                "ask1Price": "50001.00",
                "bid1Size": "2",
                "ask1Size": "3",
                "lastPrice": "50000.50",
            },
            {"ts": 1767225600000},
            REFERENCE_TIME,
        )
        assert ticker.bid == Decimal("49999.00")
        assert ticker.mid == Decimal("50000.00")
        assert ticker.last == Decimal("50000.50")

    def test_a_ticker_without_prices_and_no_prior_state_is_rejected(self, btc: Symbol) -> None:
        # A frame carrying no quote, with nothing before it to merge into, really has no
        # usable price. Inventing a zero here would put a zero mark into the risk engine.
        from quantflow.core.errors import MarketDataError

        with pytest.raises(MarketDataError, match="usable prices"):
            _parse_ticker(btc, {"bid1Price": "0", "ask1Price": "0"}, {}, REFERENCE_TIME)


class TestLinearTickerDeltas:
    """Bybit V5 ``tickers`` on the linear category is a snapshot-plus-delta stream.

    Captured live on 2026-08-14: every symbol sends exactly ONE ``snapshot`` frame with
    the full field set, then hundreds of ``delta`` frames carrying only what changed. Of
    ~215 XRP frames in 30 seconds only 90 held both sides; of ~134 XAU frames only 15 did.

    Treating each frame as standalone therefore discarded 58% of XRP ticks and 89% of XAU
    ticks as "no usable prices" — which is what left the intrabar manager blind while the
    socket was in fact perfectly healthy.
    """

    #: A real frame, trimmed. Note `bid1Price` present, `ask1Price` absent.
    BID_ONLY_DELTA: ClassVar[dict[str, str]] = {
        "symbol": "BTCUSDT",
        "tickDirection": "ZeroPlusTick",
        "bid1Price": "62985.50",
        "bid1Size": "4.553",
    }

    SNAPSHOT: ClassVar[dict[str, str]] = {
        "symbol": "BTCUSDT",
        "lastPrice": "62975.90",
        "markPrice": "62975.80",
        "indexPrice": "63002.96",
        "bid1Price": "62975.80",
        "bid1Size": "0.822",
        "ask1Price": "62975.90",
        "ask1Size": "1.5",
    }

    #: The real frame that preceded it, carrying both sides.
    BOTH_SIDES_DELTA: ClassVar[dict[str, str]] = {
        "symbol": "BTCUSDT",
        "bid1Price": "62985.50",
        "bid1Size": "5.128",
        "ask1Price": "62985.60",
        "ask1Size": "9.575",
    }

    def test_a_delta_merges_into_the_snapshot_instead_of_being_discarded(self, btc: Symbol) -> None:
        # The exact three-frame sequence captured from the venue.
        book = TickerBook()
        book.merge(btc, self.SNAPSHOT, "snapshot")
        book.merge(btc, self.BOTH_SIDES_DELTA, "delta")

        merged = book.merge(btc, self.BID_ONLY_DELTA, "delta")
        ticker = _parse_ticker(btc, merged, {"ts": 1767225600000}, REFERENCE_TIME)

        # The bid-only frame is usable because the ask is carried forward. Before this it
        # was discarded outright as "no usable prices".
        assert ticker.bid == Decimal("62985.50")
        assert ticker.ask == Decimal("62985.60")

    def test_a_merge_that_would_cross_the_book_is_refused_not_published(self, btc: Symbol) -> None:
        """A one-sided delta can momentarily cross a stale opposite side.

        The venue corrects it on the next frame, so the right response is to skip this
        reading — never to publish a crossed quote, and never to let the exception escape
        and tear down the stream.
        """
        from quantflow.core.errors import ValidationError

        book = TickerBook()
        book.merge(btc, self.SNAPSHOT, "snapshot")
        merged = book.merge(btc, self.BID_ONLY_DELTA, "delta")

        with pytest.raises(ValidationError, match="crossed ticker"):
            _parse_ticker(btc, merged, {}, REFERENCE_TIME)

    def test_a_delta_with_no_price_fields_still_yields_the_last_known_quote(
        self, btc: Symbol
    ) -> None:
        # Frames carrying only funding or open-interest updates arrive constantly. They
        # are not an error and they must not blank the book.
        book = TickerBook()
        book.merge(btc, self.SNAPSHOT, "snapshot")

        merged = book.merge(btc, {"symbol": "BTCUSDT", "openInterest": "66642.128"}, "delta")
        ticker = _parse_ticker(btc, merged, {}, REFERENCE_TIME)

        assert ticker.bid == Decimal("62975.80")
        assert ticker.ask == Decimal("62975.90")

    def test_a_snapshot_replaces_rather_than_merges(self, btc: Symbol) -> None:
        # A reconnect re-snapshots. Merging into stale state would keep a dead quote alive.
        book = TickerBook()
        book.merge(btc, self.SNAPSHOT, "snapshot")
        replacement = {"symbol": "BTCUSDT", "bid1Price": "100.0", "ask1Price": "101.0"}

        merged = book.merge(btc, replacement, "snapshot")

        assert merged.get("lastPrice") is None
        assert merged["bid1Price"] == "100.0"

    def test_symbols_do_not_bleed_into_one_another(self, btc: Symbol, eth: Symbol) -> None:
        book = TickerBook()
        book.merge(btc, self.SNAPSHOT, "snapshot")

        merged = book.merge(eth, {"symbol": "ETHUSDT", "bid1Price": "1876.0"}, "delta")

        # ETH has never snapshotted, so BTC's ask must not be borrowed for it.
        assert merged.get("ask1Price") is None

    def test_a_delta_before_any_snapshot_is_not_invented(self, btc: Symbol) -> None:
        book = TickerBook()
        merged = book.merge(btc, self.BID_ONLY_DELTA, "delta")

        from quantflow.core.errors import MarketDataError

        with pytest.raises(MarketDataError, match="usable prices"):
            _parse_ticker(btc, merged, {}, REFERENCE_TIME)

    def test_trade_side_is_the_aggressor_directly(self, btc: Symbol) -> None:
        # Bybit reports the aggressor's side in `S`, the OPPOSITE convention to Binance's
        # maker flag. Carrying the Binance logic across would invert every trade.
        sell = _parse_trade(btc, {"i": "1", "T": 1767225600000, "p": "1", "v": "1", "S": "Sell"})
        buy = _parse_trade(btc, {"i": "2", "T": 1767225600000, "p": "1", "v": "1", "S": "Buy"})
        assert sell.side is OrderSide.SELL
        assert buy.side is OrderSide.BUY

    def test_a_trade_without_a_timestamp_is_rejected(self, btc: Symbol) -> None:
        from quantflow.core.errors import MarketDataError

        with pytest.raises(MarketDataError, match="no timestamp"):
            _parse_trade(btc, {"i": "1", "p": "1", "v": "1", "S": "Buy"})


class TestCandleGapDetector:
    def _candle(self, btc: Symbol, index: int):
        from tests.conftest import make_candle

        return make_candle(btc, open_time=REFERENCE_TIME + Timeframe.H1.delta * index, close=100)

    def test_contiguous_stream_reports_no_gaps(self, btc: Symbol) -> None:
        detector = CandleGapDetector(Timeframe.H1)
        assert detector.observe(self._candle(btc, 0)) == 0
        assert detector.observe(self._candle(btc, 1)) == 0

    def test_reports_missing_bars_after_a_reconnect(self, btc: Symbol) -> None:
        seen: list[tuple[str, int]] = []
        detector = CandleGapDetector(
            Timeframe.H1,
            on_gap=lambda symbol, _start, _end, missing: seen.append((str(symbol), missing)),
        )
        detector.observe(self._candle(btc, 0))
        assert detector.observe(self._candle(btc, 4)) == 3
        assert seen == [("BTC/USDT", 3)]

    def test_repeated_bar_is_not_a_gap(self, btc: Symbol) -> None:
        detector = CandleGapDetector(Timeframe.H1)
        detector.observe(self._candle(btc, 3))
        assert detector.observe(self._candle(btc, 3)) == 0

    def test_symbols_are_tracked_independently(self, btc: Symbol, eth: Symbol) -> None:
        detector = CandleGapDetector(Timeframe.H1)
        detector.observe(self._candle(btc, 0))
        assert detector.observe(self._candle(eth, 5)) == 0

    def test_reset(self, btc: Symbol) -> None:
        detector = CandleGapDetector(Timeframe.H1)
        detector.observe(self._candle(btc, 0))
        detector.reset()
        assert detector.observe(self._candle(btc, 10)) == 0


class TestVenueReadsPreserveLocalAttribution:
    """Re-reading an order from the venue must not erase what only we know.

    The venue knows an order's price, size and status. It has never heard of the strategy
    that produced it. ``submit_order`` parses the acknowledgement *with* the strategy, then
    ``_enrich_from_venue`` immediately re-fetches the order for authoritative fill data —
    and that second parse passed no ``strategy_id`` at all. The enriched order replaced the
    attributed one and was persisted with nothing.

    The effect on the live session: 81 orders, **zero** with a strategy, and therefore 25
    closed trades that cannot be attributed to anything. Every per-strategy question — which
    of the 43 carry edge, which bleed — was unanswerable, and the loss could only be
    localised as far as "XRP and BNB".

    The gateway already tracks the mapping needed to restore it, so the fix is to consult
    what we recorded at submission rather than expecting the venue to tell us.
    """

    @staticmethod
    def _gateway() -> BybitGateway:
        return BybitGateway(
            ExchangeSettings(
                name="bybit",
                api_key=SecretStr("k" * 18),
                api_secret=SecretStr("s" * 36),
                testnet=True,
                market_type=MarketType.SPOT,
            )
        )

    def test_a_remembered_strategy_is_reattached_to_a_venue_read(self) -> None:
        gateway = self._gateway()
        gateway.remember_strategy("local-1", "macd_trend")

        assert gateway.strategy_for("local-1") == "macd_trend"

    def test_an_unknown_order_yields_none_rather_than_a_guess(self) -> None:
        gateway = self._gateway()

        assert gateway.strategy_for("never-seen") is None

    def test_the_venue_order_id_also_resolves(self) -> None:
        # Executions come back under the venue's id, so both identities must resolve.
        gateway = self._gateway()
        gateway.remember_strategy("local-2", "rsi_reversion", venue_order_id="venue-2")

        assert gateway.strategy_for("venue-2") == "rsi_reversion"
