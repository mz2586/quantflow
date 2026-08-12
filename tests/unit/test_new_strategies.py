"""The strategies added for the orchestrator: VWAP, stochastic and ATR expansion."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from quantflow.domain.enums import SignalDirection, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import CandleSeries
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.strategy.base import StrategyContext
from quantflow.strategy.indicators import rolling_vwap, stochastic
from quantflow.strategy.library import (
    AtrExpansionStrategy,
    StochasticReversionParams,
    StochasticReversionStrategy,
    VwapReversionParams,
    VwapReversionStrategy,
)
from tests.conftest import REFERENCE_TIME, make_candle
from tests.unit.test_strategies import open_long

SYMBOL = Symbol.parse("BTC/USDT")


def context_from(candles: list, position=None) -> StrategyContext:  # type: ignore[no-untyped-def]
    series = CandleSeries(candles)
    return StrategyContext(
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
        history=series,
        now=series.end + Timeframe.H1.delta,
        portfolio=PortfolioSnapshot(
            timestamp=series.end,
            base_currency="USDT",
            cash=Decimal("10000"),
            positions=(position,) if position else (),
            mark_prices={SYMBOL: candles[-1].close},
        ),
        position=position,
    )


def bars(specs: list[tuple[str, str, str, str, str]]) -> list:  # type: ignore[no-untyped-def]
    """Build candles from (open, high, low, close, volume) tuples."""
    return [
        make_candle(
            SYMBOL,
            open_time=REFERENCE_TIME + Timeframe.H1.delta * index,
            open_price=spec[0],
            high=spec[1],
            low=spec[2],
            close=spec[3],
            volume=spec[4],
        )
        for index, spec in enumerate(specs)
    ]


def plus(candles: list, spec: tuple[str, str, str, str, str]) -> list:  # type: ignore[no-untyped-def]
    """Append one bar, continuing the existing series' timestamps."""
    return [
        *candles,
        make_candle(
            SYMBOL,
            open_time=candles[-1].open_time + Timeframe.H1.delta,
            open_price=spec[0],
            high=spec[1],
            low=spec[2],
            close=spec[3],
            volume=spec[4],
        ),
    ]


class TestIndicators:
    def test_vwap_weights_by_volume(self) -> None:
        """A high-volume bar must pull VWAP toward its own price."""
        candles = bars([("100", "100", "100", "100", "1"), ("200", "200", "200", "200", "99")])
        value = rolling_vwap(candles, 2)[-1]
        assert value is not None
        assert value > Decimal("190")

    def test_vwap_is_none_when_the_window_has_no_volume(self) -> None:
        candles = bars([("100", "100", "100", "100", "0")] * 3)
        assert rolling_vwap(candles, 3)[-1] is None

    def test_stochastic_reads_100_at_the_top_of_the_range(self) -> None:
        candles = bars(
            [
                ("10", "20", "10", "12", "5"),
                ("12", "20", "10", "20", "5"),
                ("20", "20", "10", "20", "5"),
            ]
        )
        k_line, _ = stochastic(candles, 3, 1)
        assert k_line[-1] == Decimal("100")

    def test_stochastic_is_none_on_a_flat_range(self) -> None:
        candles = bars([("10", "10", "10", "10", "5")] * 5)
        k_line, _ = stochastic(candles, 3, 1)
        assert k_line[-1] is None

    def test_no_lookahead_earlier_values_ignore_later_bars(self) -> None:
        """Truncating the series must not change any value that was already defined."""
        candles = bars(
            [(str(90 + i), str(95 + i), str(85 + i), str(92 + i), "10") for i in range(40)]
        )
        full, _ = stochastic(candles, 14, 3)
        partial, _ = stochastic(candles[:30], 14, 3)
        assert full[:30] == partial
        assert rolling_vwap(candles, 20)[:30] == rolling_vwap(candles[:30], 20)


class TestVwapReversion:
    def test_rejects_an_exit_threshold_at_or_above_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            VwapReversionParams(entry_atr_distance=Decimal("1"), exit_atr_distance=Decimal("1"))

    def test_holds_while_price_sits_on_vwap(self) -> None:
        candles = bars([("100", "101", "99", "100", "10")] * 40)
        signal = VwapReversionStrategy().evaluate(context_from(candles))
        assert not signal.is_actionable

    def test_goes_long_when_price_is_stretched_below_vwap(self) -> None:
        candles = plus(
            bars([("100", "101", "99", "100", "10")] * 39), ("100", "100", "80", "82", "10")
        )
        signal = VwapReversionStrategy().evaluate(context_from(candles))
        assert signal.direction is SignalDirection.LONG
        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None

    def test_conviction_rises_with_distance(self) -> None:
        base = bars([("100", "101", "99", "100", "10")] * 39)
        near = VwapReversionStrategy().evaluate(
            context_from(plus(base, ("100", "100", "88", "89", "10")))
        )
        far = VwapReversionStrategy().evaluate(
            context_from(plus(base, ("100", "100", "70", "72", "10")))
        )
        if near.is_actionable and far.is_actionable:
            assert far.conviction >= near.conviction

    def test_conviction_never_leaves_the_unit_interval(self) -> None:
        candles = plus(
            bars([("100", "101", "99", "100", "10")] * 39), ("100", "100", "1", "2", "10")
        )
        signal = VwapReversionStrategy().evaluate(context_from(candles))
        assert Decimal("0") <= signal.conviction <= Decimal("1")

    def test_shorts_are_disabled_by_default(self) -> None:
        candles = plus(
            bars([("100", "101", "99", "100", "10")] * 39), ("100", "140", "100", "138", "10")
        )
        signal = VwapReversionStrategy().evaluate(context_from(candles))
        assert not signal.is_actionable


class TestStochasticReversion:
    def test_rejects_inverted_levels(self) -> None:
        with pytest.raises(PydanticValidationError):
            StochasticReversionParams(oversold=Decimal("40"), overbought=Decimal("30"))

    def test_holds_without_a_cross_out_of_an_extreme(self) -> None:
        candles = bars(
            [
                (str(100 + i % 3), str(102 + i % 3), str(98 + i % 3), str(100 + i % 3), "10")
                for i in range(60)
            ]
        )
        signal = StochasticReversionStrategy().evaluate(context_from(candles))
        assert not signal.is_actionable

    def test_exits_a_long_once_the_oscillator_recovers(self) -> None:
        candles = bars(
            [(str(100 - i), str(101 - i), str(99 - i), str(100 - i), "10") for i in range(40)]
        )
        candles = plus(candles, ("61", "80", "60", "79", "10"))
        signal = StochasticReversionStrategy().evaluate(
            context_from(candles, position=open_long(SYMBOL))
        )
        assert signal.direction in (SignalDirection.CLOSE, SignalDirection.HOLD)

    def test_conviction_within_bounds(self) -> None:
        candles = bars(
            [(str(100 - i), str(101 - i), str(99 - i), str(100 - i), "10") for i in range(60)]
        )
        signal = StochasticReversionStrategy().evaluate(context_from(candles))
        assert Decimal("0") <= signal.conviction <= Decimal("1")


class TestAtrExpansion:
    def test_holds_while_volatility_is_normal(self) -> None:
        candles = bars([("100", "101", "99", "100", "10")] * 80)
        signal = AtrExpansionStrategy().evaluate(context_from(candles))
        assert not signal.is_actionable

    def test_enters_long_on_an_expanding_decisive_up_bar(self) -> None:
        candles = plus(
            bars([("100", "101", "99", "100", "10")] * 79), ("100", "121", "99", "120", "10")
        )
        signal = AtrExpansionStrategy().evaluate(context_from(candles))
        assert signal.direction is SignalDirection.LONG
        assert signal.stop_loss_price is not None

    def test_rejects_a_wide_bar_with_no_body(self) -> None:
        """A wide range that closes where it opened is indecision, not a breakout."""
        candles = plus(
            bars([("100", "101", "99", "100", "10")] * 79), ("100", "130", "70", "100.2", "10")
        )
        signal = AtrExpansionStrategy().evaluate(context_from(candles))
        assert not signal.is_actionable

    def test_conviction_within_bounds(self) -> None:
        candles = plus(
            bars([("100", "101", "99", "100", "10")] * 79), ("100", "160", "99", "158", "10")
        )
        signal = AtrExpansionStrategy().evaluate(context_from(candles))
        assert Decimal("0") <= signal.conviction <= Decimal("1")

    def test_shorts_disabled_by_default(self) -> None:
        candles = plus(
            bars([("100", "101", "99", "100", "10")] * 79), ("100", "101", "79", "80", "10")
        )
        signal = AtrExpansionStrategy().evaluate(context_from(candles))
        assert not signal.is_actionable
