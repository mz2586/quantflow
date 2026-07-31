"""CCXT/Binance translation: symbols, instruments, orders, fills and errors."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from quantflow.core.config import MarketType
from quantflow.core.errors import (
    ExchangeAuthenticationError,
    ExchangeConnectionError,
    ExchangeError,
    ExchangeTimeoutError,
    InsufficientFundsError,
    InvalidSymbolError,
    OrderRejectedError,
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
from quantflow.exchange.binance.mapping import (
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
from quantflow.exchange.binance.ws import (
    CandleGapDetector,
    _parse_book_ticker,
    _parse_kline,
    _parse_trade,
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

    def test_stop_loss_rounds_conservatively_for_the_holder(
        self, btc: Symbol, btc_instrument: Instrument
    ) -> None:
        # A long's stop rounds up, so it triggers no later than intended.
        request = OrderRequest(
            symbol=btc,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("50000.00"),
            stop_loss_price=Decimal("49000.001"),
        )
        assert normalize_order(request, btc_instrument).stop_loss_price == Decimal("49000.01")

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
    def test_selects_by_market_type_and_network(self) -> None:
        from quantflow.core.config import ExchangeSettings

        assert "testnet" in stream_url(ExchangeSettings(testnet=True))
        assert stream_url(ExchangeSettings(testnet=False)).startswith("wss://stream.binance.com")
        assert "fstream" in stream_url(
            ExchangeSettings(testnet=False, market_type=MarketType.FUTURE)
        )


class TestStreamPayloadParsing:
    def test_kline(self, btc: Symbol) -> None:
        candle = _parse_kline(
            btc,
            Timeframe.H1,
            {
                "t": 1767225600000,
                "o": "50000.00",
                "h": "50500.00",
                "l": "49800.00",
                "c": "50200.00",
                "v": "123.456",
                "q": "6200000.00",
                "n": 4321,
                "x": True,
            },
        )
        assert candle.open_time == datetime(2026, 1, 1, tzinfo=UTC)
        assert candle.close == Decimal("50200.00")
        assert candle.trades == 4321

    def test_book_ticker(self, btc: Symbol) -> None:
        ticker = _parse_book_ticker(
            btc, {"b": "49999.00", "a": "50001.00", "E": 1767225600000}, REFERENCE_TIME
        )
        assert ticker.bid == Decimal("49999.00")
        assert ticker.mid == Decimal("50000.00")

    def test_book_ticker_without_prices_is_rejected(self, btc: Symbol) -> None:
        from quantflow.core.errors import MarketDataError

        with pytest.raises(MarketDataError, match="usable prices"):
            _parse_book_ticker(btc, {"b": "0", "a": "0"}, REFERENCE_TIME)

    def test_trade_side_follows_the_aggressor(self, btc: Symbol) -> None:
        # Binance's `m` flag means the buyer was the maker, i.e. the aggressor sold.
        sell = _parse_trade(btc, {"t": 1, "T": 1767225600000, "p": "1", "q": "1", "m": True})
        buy = _parse_trade(btc, {"t": 2, "T": 1767225600000, "p": "1", "q": "1", "m": False})
        assert sell.side is OrderSide.SELL
        assert buy.side is OrderSide.BUY


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
