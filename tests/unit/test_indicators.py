"""Indicators, checked against hand-computed golden values.

Expected values are derived by hand or from the published formula rather than from a
previous run of this code, so a regression cannot silently redefine "correct".
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.core.errors import InsufficientDataError, ValidationError
from quantflow.domain.instruments import Symbol
from quantflow.strategy.indicators import (
    Series,
    atr,
    bollinger_bands,
    crossed_above,
    crossed_below,
    ema,
    last_value,
    macd,
    normalized_atr,
    require_value,
    rolling_max,
    rolling_min,
    rsi,
    slope,
    sma,
    stdev,
    true_range,
    wilder_smoothing,
)
from tests.conftest import make_candles


def decimals(*values: str | int | float) -> list[Decimal]:
    return [Decimal(str(value)) for value in values]


class TestAlignment:
    """Every indicator returns a series the same length as its input.

    Misaligned output forces callers into offset arithmetic, and an off-by-one there is a
    look-ahead bug that no test catches by accident.
    """

    @pytest.mark.parametrize("period", [2, 5, 10])
    def test_moving_averages_are_aligned(self, period: int) -> None:
        values = decimals(*range(1, 21))
        assert len(sma(values, period)) == len(values)
        assert len(ema(values, period)) == len(values)
        assert len(rolling_max(values, period)) == len(values)
        assert len(rolling_min(values, period)) == len(values)

    def test_warmup_is_none_not_zero(self) -> None:
        # Zero would be a *valid* indicator value and would silently trade on nonsense.
        result = sma(decimals(1, 2, 3, 4, 5), 3)
        assert result[:2] == (None, None)
        assert result[2] is not None

    def test_empty_input(self) -> None:
        assert sma([], 5) == ()
        assert ema([], 5) == ()
        assert true_range([]) == ()

    def test_input_shorter_than_period(self) -> None:
        assert ema(decimals(1, 2), 5) == (None, None)
        assert sma(decimals(1, 2), 5) == (None, None)

    @pytest.mark.parametrize("period", [0, -1])
    def test_invalid_period_is_rejected(self, period: int) -> None:
        with pytest.raises(ValidationError, match="period must be"):
            sma(decimals(1, 2, 3), period)


class TestSma:
    def test_golden_values(self) -> None:
        result = sma(decimals(1, 2, 3, 4, 5, 6), 3)
        # (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4, (4+5+6)/3=5
        assert result == (None, None, Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5"))

    def test_constant_series(self) -> None:
        result = sma(decimals(*[7] * 10), 4)
        assert all(value == Decimal("7") for value in result[3:])


class TestEma:
    def test_is_seeded_with_the_sma(self) -> None:
        # Every charting package seeds this way; seeding with the first value instead
        # produces a curve that disagrees on screen for hundreds of bars.
        values = decimals(1, 2, 3, 4, 5)
        result = ema(values, 3)
        assert result[2] == Decimal("2")  # SMA of 1,2,3

    def test_golden_progression(self) -> None:
        values = decimals(1, 2, 3, 4, 5)
        result = ema(values, 3)
        # multiplier = 2/4 = 0.5; next = (4 - 2) * 0.5 + 2 = 3
        assert result[3] == Decimal("3")
        # next = (5 - 3) * 0.5 + 3 = 4
        assert result[4] == Decimal("4")

    def test_reacts_faster_than_sma(self) -> None:
        values = decimals(*([10] * 20 + [20] * 5))
        fast = ema(values, 10)[-1]
        slow = sma(values, 10)[-1]
        assert fast is not None
        assert slow is not None
        assert fast > slow


class TestWilderSmoothing:
    def test_differs_from_ema(self) -> None:
        # Wilder's uses 1/period where an EMA uses 2/(period+1). Substituting one for the
        # other silently changes every RSI and ATR reading.
        values = decimals(*range(1, 30))
        assert wilder_smoothing(values, 14)[-1] != ema(values, 14)[-1]

    def test_seeded_with_the_sma(self) -> None:
        values = decimals(2, 4, 6, 8)
        assert wilder_smoothing(values, 3)[2] == Decimal("4")


class TestRsi:
    def test_unbroken_gains_reach_100(self) -> None:
        # Division by zero is not the answer here; RSI is 100 by definition.
        result = rsi(decimals(*range(1, 30)), 14)
        assert result[-1] == Decimal("100")

    def test_unbroken_losses_reach_0(self) -> None:
        result = rsi(decimals(*range(30, 1, -1)), 14)
        assert result[-1] == Decimal("0")

    def test_flat_series_is_neutral(self) -> None:
        result = rsi(decimals(*[100] * 30), 14)
        assert result[-1] == Decimal("50")

    def test_stays_within_bounds(self) -> None:
        values = decimals(
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
            46.03,
            46.41,
            46.22,
            45.64,
        )
        for value in rsi(values, 14):
            if value is not None:
                assert Decimal("0") <= value <= Decimal("100")

    def test_warmup_length(self) -> None:
        result = rsi(decimals(*range(1, 20)), 14)
        assert all(value is None for value in result[:14])
        assert result[14] is not None

    def test_too_short_input(self) -> None:
        assert all(value is None for value in rsi(decimals(1, 2, 3), 14))

    def test_period_below_two_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="period must be at least 2"):
            rsi(decimals(1, 2, 3), 1)


class TestTrueRangeAndAtr:
    def test_true_range_uses_the_prior_close(self, btc: Symbol) -> None:
        candles = make_candles(btc, [100, 110])
        # Both bars are flat OHLC, so the second bar's range comes from the gap.
        ranges = true_range(candles)
        assert ranges[0] == Decimal("0")
        assert ranges[1] == Decimal("10")

    def test_atr_is_positive_for_a_moving_market(self, btc: Symbol) -> None:
        candles = make_candles(btc, [100 + (index % 5) * 3 for index in range(40)])
        value = atr(candles, 14)[-1]
        assert value is not None
        assert value > Decimal("0")

    def test_atr_of_a_flat_market_is_zero(self, btc: Symbol) -> None:
        candles = make_candles(btc, [100] * 40)
        assert atr(candles, 14)[-1] == Decimal("0")

    def test_atr_too_short(self, btc: Symbol) -> None:
        assert all(value is None for value in atr(make_candles(btc, [1, 2, 3]), 14))

    def test_normalized_atr_is_a_fraction_of_price(self, btc: Symbol) -> None:
        candles = make_candles(btc, [100 + (index % 5) * 3 for index in range(40)])
        value = normalized_atr(candles, 14)[-1]
        assert value is not None
        assert Decimal("0") < value < Decimal("1")


class TestRollingExtremes:
    def test_rolling_max(self) -> None:
        result = rolling_max(decimals(1, 5, 3, 2, 8, 4), 3)
        assert result == (None, None, Decimal("5"), Decimal("5"), Decimal("8"), Decimal("8"))

    def test_rolling_min(self) -> None:
        result = rolling_min(decimals(1, 5, 3, 2, 8, 4), 3)
        assert result == (None, None, Decimal("1"), Decimal("2"), Decimal("2"), Decimal("2"))


class TestStdevAndBands:
    def test_stdev_of_a_constant_series_is_zero(self) -> None:
        assert stdev(decimals(*[5] * 10), 5)[-1] == Decimal("0")

    def test_stdev_golden_value(self) -> None:
        # Population stdev of 2,4,4,4,5,5,7,9 is exactly 2.
        value = stdev(decimals(2, 4, 4, 4, 5, 5, 7, 9), 8)[-1]
        assert value is not None
        assert value == pytest.approx(Decimal("2"), abs=Decimal("0.0000001"))

    def test_bollinger_bands_bracket_the_middle(self) -> None:
        values = decimals(*[100 + (index % 7) for index in range(40)])
        upper, middle, lower = bollinger_bands(values, 20)
        assert upper[-1] is not None
        assert middle[-1] is not None
        assert lower[-1] is not None
        assert lower[-1] < middle[-1] < upper[-1]

    def test_bollinger_bands_are_aligned(self) -> None:
        values = decimals(*range(1, 41))
        for series in bollinger_bands(values, 20):
            assert len(series) == 40


class TestMacd:
    def test_all_three_series_are_aligned(self) -> None:
        values = decimals(*[100 + index for index in range(80)])
        for series in macd(values):
            assert len(series) == 80

    def test_histogram_is_the_difference(self) -> None:
        values = decimals(*[100 + (index % 11) for index in range(90)])
        line, signal, histogram = macd(values)
        for index in range(len(values)):
            if line[index] is not None and signal[index] is not None:
                assert histogram[index] == line[index] - signal[index]  # type: ignore[operator]

    def test_rejects_inverted_periods(self) -> None:
        with pytest.raises(ValidationError, match="must be below"):
            macd(decimals(*range(1, 50)), fast=26, slow=12)


class TestCrossings:
    """A crossing is a *change* in relationship, not a comparison.

    Testing ``fast > slow`` alone re-fires an entry on every bar of a trend, which turns
    one intended position into hundreds of round trips and a fee bill that destroys the
    strategy.
    """

    def _series(self, values: list[float | None]) -> Series:
        return tuple(None if value is None else Decimal(str(value)) for value in values)

    def test_detects_the_crossing_bar_only(self) -> None:
        fast = self._series([1, 2, 4, 5, 6])
        slow = self._series([3, 3, 3, 3, 3])
        assert not crossed_above(fast, slow, 1)
        assert crossed_above(fast, slow, 2)  # crossed here
        assert not crossed_above(fast, slow, 3)  # already above
        assert not crossed_above(fast, slow, 4)

    def test_crossed_below_is_the_mirror(self) -> None:
        fast = self._series([6, 5, 2, 1])
        slow = self._series([3, 3, 3, 3])
        assert crossed_below(fast, slow, 2)
        assert not crossed_below(fast, slow, 3)

    def test_touching_then_rising_counts(self) -> None:
        # Equal on the prior bar then strictly above: this is a cross.
        fast = self._series([3, 4])
        slow = self._series([3, 3])
        assert crossed_above(fast, slow, 1)

    def test_none_values_never_cross(self) -> None:
        fast = self._series([None, 4])
        slow = self._series([3, 3])
        assert not crossed_above(fast, slow, 1)

    def test_index_bounds(self) -> None:
        fast = self._series([1, 2])
        slow = self._series([3, 3])
        assert not crossed_above(fast, slow, 0)
        assert not crossed_above(fast, slow, 99)


class TestHelpers:
    def test_slope(self) -> None:
        series = tuple(Decimal(str(value)) for value in (1, 3, 6, 10))
        assert slope(series, 3) == Decimal("4")
        assert slope(series, 3, lookback=3) == Decimal("9")
        assert slope(series, 0) is None

    def test_last_value_skips_trailing_none(self) -> None:
        assert last_value((Decimal("1"), Decimal("2"), None)) == Decimal("2")
        assert last_value((None, None)) is None
        assert last_value(()) is None

    def test_require_value_returns_the_value(self) -> None:
        assert require_value((Decimal("5"),), 0, "test") == Decimal("5")

    def test_require_value_raises_during_warmup(self) -> None:
        with pytest.raises(InsufficientDataError, match="warm-up"):
            require_value((None, None), 1, "sma")

    def test_require_value_raises_out_of_range(self) -> None:
        with pytest.raises(InsufficientDataError):
            require_value((Decimal("1"),), 5, "sma")
