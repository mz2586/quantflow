"""Symbols, instruments and market-data value objects."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.core.errors import MarketDataError, ValidationError
from quantflow.domain.enums import OrderSide, Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import (
    Candle,
    CandleSeries,
    OrderBook,
    OrderBookLevel,
    Ticker,
    Trade,
)
from tests.conftest import REFERENCE_TIME, make_candle, make_candles


class TestSymbol:
    def test_normalises_case_and_whitespace(self) -> None:
        assert Symbol(base=" btc ", quote="usdt").slashed == "BTC/USDT"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("BTC/USDT", "BTC/USDT"),
            ("btc-usdt", "BTC/USDT"),
            ("ETH_USDT", "ETH/USDT"),
            ("BTCUSDT", "BTC/USDT"),
            ("ETHBTC", "ETH/BTC"),
            ("SOLUSDC", "SOL/USDC"),
            ("BTC:USDT", "BTC/USDT"),
        ],
    )
    def test_parse(self, raw: str, expected: str) -> None:
        assert str(Symbol.parse(raw)) == expected

    def test_parse_is_idempotent_on_symbol(self, btc: Symbol) -> None:
        assert Symbol.parse(btc) is btc

    def test_parse_rejects_unknown_concatenation(self) -> None:
        with pytest.raises(ValidationError, match="cannot parse symbol"):
            Symbol.parse("SOMETHINGWEIRD")

    def test_rejects_identical_base_and_quote(self) -> None:
        with pytest.raises(ValidationError, match="must differ"):
            Symbol(base="BTC", quote="BTC")

    def test_rejects_empty_components(self) -> None:
        with pytest.raises(ValidationError, match="requires base and quote"):
            Symbol(base="", quote="USDT")

    def test_rejects_non_alphanumeric(self) -> None:
        with pytest.raises(ValidationError, match="alphanumeric"):
            Symbol(base="BT!C", quote="USDT")

    def test_concatenated_form(self, btc: Symbol) -> None:
        assert btc.concatenated == "BTCUSDT"

    def test_hashable_and_ordered(self, btc: Symbol, eth: Symbol) -> None:
        assert len({btc, Symbol(base="BTC", quote="USDT")}) == 1
        assert sorted([eth, btc])[0] == btc


class TestInstrument:
    def test_normalises_quantity_downward(self, btc_instrument: Instrument) -> None:
        assert btc_instrument.normalize_quantity(Decimal("0.123456789")) == Decimal("0.12345")

    def test_normalises_price_per_side(self, btc_instrument: Instrument) -> None:
        assert btc_instrument.normalize_price(Decimal("100.567"), side_is_buy=True) == Decimal(
            "100.56"
        )
        assert btc_instrument.normalize_price(Decimal("100.561"), side_is_buy=False) == Decimal(
            "100.57"
        )

    def test_notional(self, btc_instrument: Instrument) -> None:
        assert btc_instrument.notional(Decimal("2"), Decimal("100")) == Decimal("200")

    def test_notional_is_unsigned(self, btc_instrument: Instrument) -> None:
        assert btc_instrument.notional(Decimal("-2"), Decimal("100")) == Decimal("200")

    def test_validate_order_accepts_valid(self, btc_instrument: Instrument) -> None:
        btc_instrument.validate_order(Decimal("0.001"), Decimal("50000.00"))

    def test_rejects_below_min_quantity(self, btc: Symbol) -> None:
        instrument = Instrument(symbol=btc, min_quantity=Decimal("1"), quantity_step=Decimal("1"))
        with pytest.raises(ValidationError, match="below minimum"):
            instrument.validate_order(Decimal("0"), Decimal("100"))

    def test_rejects_above_max_quantity(self, btc: Symbol) -> None:
        instrument = Instrument(
            symbol=btc,
            quantity_step=Decimal("1"),
            min_quantity=Decimal("1"),
            max_quantity=Decimal("5"),
        )
        with pytest.raises(ValidationError, match="above maximum"):
            instrument.validate_order(Decimal("6"), Decimal("100"))

    def test_rejects_off_step_quantity(self, btc_instrument: Instrument) -> None:
        with pytest.raises(ValidationError, match="multiple of step"):
            btc_instrument.validate_order(Decimal("0.0000123456"), Decimal("50000.00"))

    def test_rejects_off_tick_price(self, btc_instrument: Instrument) -> None:
        with pytest.raises(ValidationError, match="multiple of tick"):
            btc_instrument.validate_order(Decimal("0.001"), Decimal("50000.005"))

    def test_rejects_below_min_notional(self, btc_instrument: Instrument) -> None:
        with pytest.raises(ValidationError, match="below minimum"):
            btc_instrument.validate_order(Decimal("0.00001"), Decimal("1.00"))

    def test_rejects_inactive_instrument(self, btc: Symbol) -> None:
        instrument = Instrument(symbol=btc, active=False)
        with pytest.raises(ValidationError, match="not tradable"):
            instrument.validate_order(Decimal("1"), Decimal("100"))

    def test_rejects_invalid_rules(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="price_tick"):
            Instrument(symbol=btc, price_tick=Decimal("0"))
        with pytest.raises(ValidationError, match="quantity_step"):
            Instrument(symbol=btc, quantity_step=Decimal("-1"))
        with pytest.raises(ValidationError, match="max_leverage"):
            Instrument(symbol=btc, max_leverage=Decimal("0.5"))

    def test_fee_rate_by_role(self, btc_instrument: Instrument) -> None:
        assert btc_instrument.fee_rate(is_maker=True) == btc_instrument.maker_fee
        assert btc_instrument.fee_rate(is_maker=False) == btc_instrument.taker_fee


class TestCandle:
    def test_derived_properties(self, btc: Symbol) -> None:
        candle = Candle(
            symbol=btc,
            timeframe=Timeframe.H1,
            open_time=REFERENCE_TIME,
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("10"),
            quote_volume=Decimal("1030"),
        )
        assert candle.close_time == REFERENCE_TIME + timedelta(hours=1)
        assert candle.range == Decimal("15")
        assert candle.body == Decimal("5")
        assert candle.median_price == Decimal("102.5")
        assert candle.typical_price == (Decimal("110") + Decimal("95") + Decimal("105")) / 3
        assert candle.is_bullish
        assert candle.vwap == Decimal("103")
        assert candle.return_pct == Decimal("0.05")

    def test_vwap_falls_back_to_typical_price(self, btc: Symbol) -> None:
        candle = make_candle(btc, open_time=REFERENCE_TIME, close=100, volume=0)
        assert candle.vwap == candle.typical_price

    def test_is_closed(self, btc: Symbol) -> None:
        candle = make_candle(btc, open_time=REFERENCE_TIME, close=100)
        assert not candle.is_closed(REFERENCE_TIME + timedelta(minutes=59))
        assert candle.is_closed(REFERENCE_TIME + timedelta(hours=1))

    def test_rejects_high_below_low(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="below low"):
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME,
                open=Decimal("100"),
                high=Decimal("90"),
                low=Decimal("95"),
                close=Decimal("95"),
                volume=Decimal("1"),
            )

    def test_rejects_close_outside_range(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="outside"):
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME,
                open=Decimal("100"),
                high=Decimal("110"),
                low=Decimal("95"),
                close=Decimal("120"),
                volume=Decimal("1"),
            )

    def test_rejects_naive_timestamp(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=datetime(2026, 1, 1),  # noqa: DTZ001
                open=Decimal("1"),
                high=Decimal("1"),
                low=Decimal("1"),
                close=Decimal("1"),
                volume=Decimal("1"),
            )

    def test_rejects_negative_volume(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="negative volume"):
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME,
                open=Decimal("1"),
                high=Decimal("1"),
                low=Decimal("1"),
                close=Decimal("1"),
                volume=Decimal("-1"),
            )

    def test_from_ccxt_row(self, btc: Symbol) -> None:
        candle = Candle.from_ccxt(
            btc, Timeframe.H1, [1767225600000, 100.5, 101.0, 99.5, 100.0, 12.5]
        )
        assert candle.open_time == datetime(2026, 1, 1, tzinfo=UTC)
        assert candle.open == Decimal("100.5")
        assert candle.volume == Decimal("12.5")

    def test_from_ccxt_rejects_short_row(self, btc: Symbol) -> None:
        with pytest.raises(MarketDataError, match="malformed"):
            Candle.from_ccxt(btc, Timeframe.H1, [1, 2, 3])

    def test_from_ccxt_rejects_non_numeric_timestamp(self, btc: Symbol) -> None:
        with pytest.raises(MarketDataError, match="non-numeric"):
            Candle.from_ccxt(btc, Timeframe.H1, ["nope", 1, 1, 1, 1, 1])


class TestCandleSeries:
    def test_construction_and_accessors(self, btc: Symbol) -> None:
        series = CandleSeries(make_candles(btc, [100, 101, 102]))
        assert len(series) == 3
        assert series.symbol == btc
        assert series.timeframe is Timeframe.H1
        assert series.closes() == (Decimal("100"), Decimal("101"), Decimal("102"))
        assert series.start == REFERENCE_TIME
        assert series.end == REFERENCE_TIME + timedelta(hours=2)
        assert series.is_contiguous

    def test_rejects_empty(self) -> None:
        with pytest.raises(MarketDataError, match="empty sequence"):
            CandleSeries([])

    def test_rejects_mixed_symbols(self, btc: Symbol, eth: Symbol) -> None:
        candles = [
            make_candle(btc, open_time=REFERENCE_TIME, close=1),
            make_candle(eth, open_time=REFERENCE_TIME + timedelta(hours=1), close=1),
        ]
        with pytest.raises(MarketDataError, match="mixed symbols"):
            CandleSeries(candles)

    def test_rejects_mixed_timeframes(self, btc: Symbol) -> None:
        candles = [
            make_candle(btc, open_time=REFERENCE_TIME, close=1, timeframe=Timeframe.H1),
            make_candle(
                btc, open_time=REFERENCE_TIME + timedelta(hours=1), close=1, timeframe=Timeframe.M5
            ),
        ]
        with pytest.raises(MarketDataError, match="mixed timeframes"):
            CandleSeries(candles)

    def test_rejects_non_monotonic(self, btc: Symbol) -> None:
        candles = [
            make_candle(btc, open_time=REFERENCE_TIME + timedelta(hours=1), close=1),
            make_candle(btc, open_time=REFERENCE_TIME, close=1),
        ]
        with pytest.raises(MarketDataError, match="non-monotonic"):
            CandleSeries(candles)

    def test_rejects_duplicate_open_times(self, btc: Symbol) -> None:
        candle = make_candle(btc, open_time=REFERENCE_TIME, close=1)
        with pytest.raises(MarketDataError, match="non-monotonic"):
            CandleSeries([candle, candle])

    def test_detects_gaps(self, btc: Symbol) -> None:
        candles = [
            make_candle(btc, open_time=REFERENCE_TIME, close=1),
            make_candle(btc, open_time=REFERENCE_TIME + timedelta(hours=4), close=1),
        ]
        series = CandleSeries(candles)
        assert not series.is_contiguous
        assert series.missing_intervals() == (
            (REFERENCE_TIME + timedelta(hours=1), REFERENCE_TIME + timedelta(hours=4)),
        )

    def test_window(self, btc: Symbol) -> None:
        series = CandleSeries(make_candles(btc, list(range(1, 11))))
        assert series.window(3).closes() == (Decimal("8"), Decimal("9"), Decimal("10"))

    def test_window_rejects_non_positive(self, btc: Symbol) -> None:
        series = CandleSeries(make_candles(btc, [1, 2, 3]))
        with pytest.raises(ValidationError, match="window size"):
            series.window(0)

    def test_slice_is_half_open(self, btc: Symbol) -> None:
        series = CandleSeries(make_candles(btc, [1, 2, 3, 4, 5]))
        sliced = series.slice(
            REFERENCE_TIME + timedelta(hours=1), REFERENCE_TIME + timedelta(hours=3)
        )
        assert sliced.closes() == (Decimal("2"), Decimal("3"))


class TestTicker:
    def test_derived_properties(self, btc: Symbol) -> None:
        ticker = Ticker(
            symbol=btc,
            timestamp=REFERENCE_TIME,
            bid=Decimal("99"),
            ask=Decimal("101"),
            last=Decimal("100"),
        )
        assert ticker.mid == Decimal("100")
        assert ticker.spread == Decimal("2")
        assert ticker.spread_pct == Decimal("0.02")
        assert ticker.price_for(OrderSide.BUY) == Decimal("101")
        assert ticker.price_for(OrderSide.SELL) == Decimal("99")

    def test_rejects_crossed_book(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="crossed"):
            Ticker(
                symbol=btc,
                timestamp=REFERENCE_TIME,
                bid=Decimal("101"),
                ask=Decimal("99"),
                last=Decimal("100"),
            )

    def test_rejects_non_positive_prices(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="positive"):
            Ticker(
                symbol=btc,
                timestamp=REFERENCE_TIME,
                bid=Decimal("0"),
                ask=Decimal("1"),
                last=Decimal("1"),
            )


class TestTrade:
    def test_notional(self, btc: Symbol) -> None:
        trade = Trade(
            symbol=btc,
            trade_id="1",
            timestamp=REFERENCE_TIME,
            price=Decimal("100"),
            quantity=Decimal("2"),
            side=OrderSide.BUY,
        )
        assert trade.notional == Decimal("200")

    def test_rejects_non_positive_quantity(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="quantity must be positive"):
            Trade(
                symbol=btc,
                trade_id="1",
                timestamp=REFERENCE_TIME,
                price=Decimal("100"),
                quantity=Decimal("0"),
                side=OrderSide.BUY,
            )


class TestOrderBook:
    def _book(self, symbol: Symbol) -> OrderBook:
        return OrderBook(
            symbol=symbol,
            timestamp=REFERENCE_TIME,
            bids=(
                OrderBookLevel(price=Decimal("99"), quantity=Decimal("1")),
                OrderBookLevel(price=Decimal("98"), quantity=Decimal("2")),
            ),
            asks=(
                OrderBookLevel(price=Decimal("101"), quantity=Decimal("1")),
                OrderBookLevel(price=Decimal("102"), quantity=Decimal("2")),
            ),
        )

    def test_best_levels(self, btc: Symbol) -> None:
        book = self._book(btc)
        assert book.best_bid == Decimal("99")
        assert book.best_ask == Decimal("101")
        assert book.mid == Decimal("100")

    def test_empty_book_has_no_mid(self, btc: Symbol) -> None:
        book = OrderBook(symbol=btc, timestamp=REFERENCE_TIME, bids=(), asks=())
        assert book.mid is None
        assert book.best_bid is None

    def test_rejects_unsorted_levels(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="bids must descend"):
            OrderBook(
                symbol=btc,
                timestamp=REFERENCE_TIME,
                bids=(
                    OrderBookLevel(price=Decimal("98"), quantity=Decimal("1")),
                    OrderBookLevel(price=Decimal("99"), quantity=Decimal("1")),
                ),
                asks=(),
            )

    def test_sweep_within_top_level(self, btc: Symbol) -> None:
        average, filled = self._book(btc).sweep_cost(OrderSide.BUY, Decimal("0.5"))
        assert average == Decimal("101")
        assert filled == Decimal("0.5")

    def test_sweep_across_levels(self, btc: Symbol) -> None:
        average, filled = self._book(btc).sweep_cost(OrderSide.BUY, Decimal("2"))
        assert filled == Decimal("2")
        assert average == Decimal("101.5")  # 1@101 + 1@102

    def test_sweep_reports_partial_fill_on_thin_book(self, btc: Symbol) -> None:
        average, filled = self._book(btc).sweep_cost(OrderSide.BUY, Decimal("10"))
        assert filled == Decimal("3")
        assert average > Decimal("101")

    def test_sweep_of_empty_book_fills_nothing(self, btc: Symbol) -> None:
        book = OrderBook(symbol=btc, timestamp=REFERENCE_TIME, bids=(), asks=())
        assert book.sweep_cost(OrderSide.BUY, Decimal("1")) == (Decimal("0"), Decimal("0"))


class TestTimeframe:
    def test_delta_and_seconds(self) -> None:
        assert Timeframe.H4.delta == timedelta(hours=4)
        assert Timeframe.M15.seconds == 900
        assert Timeframe.M1.milliseconds == 60_000

    def test_periods_per_year_uses_365_day_crypto_year(self) -> None:
        assert Timeframe.D1.periods_per_year == pytest.approx(365.0)
        assert Timeframe.H1.periods_per_year == pytest.approx(8760.0)

    def test_parse_normalises(self) -> None:
        assert Timeframe.parse(" 1H ") is Timeframe.H1

    def test_parse_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError, match="unsupported timeframe"):
            Timeframe.parse("7s")

    def test_every_member_has_a_delta(self) -> None:
        for timeframe in Timeframe:
            assert timeframe.delta > timedelta(0)
