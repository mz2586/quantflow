"""The six trend strategies added alongside the structural indicators they are built on.

Three properties are checked for every one of them, because each is a way the whole family
can be wrong at once rather than a quirk of any single member: they must hold rather than
raise before their warm-up is complete, they must survive degenerate input (a flat market,
zero volume, a single bar), and — the one that actually matters — a decision made at a bar
must not change when the bars after it are taken away.

That last property is enforced twice over. The indicators are checked directly for prefix
invariance (``f(candles)[:k] == f(candles[:k])``), which is where a displaced cloud or a
swing pivot read at the wrong index would show up. The strategies are then checked to
produce identical decisions whether or not the instance has already walked the rest of the
series, which is where hidden state between bars would show up. Neither check is
satisfiable by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import OrderSide, SignalDirection, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.domain.orders import Fill
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.positions import Position
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext
from quantflow.strategy.indicators import (
    higher_timeframe_closes,
    ichimoku,
    parabolic_sar,
    supertrend,
    swing_pivots,
)
from quantflow.strategy.library.ichimoku_trend import (
    IchimokuTrendParams,
    IchimokuTrendStrategy,
)
from quantflow.strategy.library.mtf_trend import MtfTrendParams, MtfTrendStrategy
from quantflow.strategy.library.parabolic_sar import ParabolicSarParams, ParabolicSarStrategy
from quantflow.strategy.library.pullback_continuation import (
    PullbackContinuationParams,
    PullbackContinuationStrategy,
)
from quantflow.strategy.library.supertrend import SupertrendParams, SupertrendStrategy
from quantflow.strategy.library.swing_structure import (
    SwingStructureParams,
    SwingStructureStrategy,
)
from tests.conftest import REFERENCE_TIME

SYMBOL = Symbol.parse("BTC/USDT")

StrategyFactory = Callable[[], Strategy]

#: Every strategy under test, built with its defaults.
FACTORIES: list[tuple[str, StrategyFactory]] = [
    ("supertrend", SupertrendStrategy),
    ("ichimoku_trend", IchimokuTrendStrategy),
    ("parabolic_sar", ParabolicSarStrategy),
    ("mtf_trend", MtfTrendStrategy),
    ("swing_structure", SwingStructureStrategy),
    ("pullback_continuation", PullbackContinuationStrategy),
]

#: Long enough for the slowest warm-up in the family (mtf_trend's higher timeframe) plus
#: several full cycles of the reference series, so every member actually trades on it.
REFERENCE_SERIES_BARS = 300


# --------------------------------------------------------------------------- #
# Fixtures and builders
# --------------------------------------------------------------------------- #
def candle(index: int, open_: str, high: str, low: str, close: str, volume: str = "10") -> Candle:
    """One bar, positioned by index on the hourly grid."""
    close_value = Decimal(close)
    volume_value = Decimal(volume)
    return Candle(
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
        open_time=REFERENCE_TIME + Timeframe.H1.delta * index,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=close_value,
        volume=volume_value,
        quote_volume=volume_value * close_value,
    )


def series_from(
    closes: Sequence[Decimal], spread: Decimal = Decimal("1"), volume: str = "10"
) -> list[Candle]:
    """Bars whose body runs from the previous close to this one, with a fixed wick."""
    bars: list[Candle] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        bars.append(
            candle(
                index,
                str(previous),
                str(max(previous, close) + spread),
                str(min(previous, close) - spread),
                str(close),
                volume,
            )
        )
        previous = close
    return bars


def centred_series(
    closes: Sequence[Decimal], spread: Decimal = Decimal("1"), volume: str = "10"
) -> list[Candle]:
    """Bars centred on their own close.

    ``series_from`` runs each bar's body from the previous close, which makes the bar after
    a local peak carry the *same* high as the peak itself. That is realistic enough for
    band and crossover tests but it silently suppresses every swing pivot, since a pivot is
    defined by a strict extreme. Anything that needs pivots uses this shape instead.
    """
    return [
        candle(index, str(close), str(close + spread), str(close - spread), str(close), volume)
        for index, close in enumerate(closes)
    ]


def context_from(bars: Sequence[Candle], position: Position | None = None) -> StrategyContext:
    """A decision context whose visible history ends at the last supplied bar."""
    history = CandleSeries(bars)
    return StrategyContext(
        symbol=SYMBOL,
        timeframe=Timeframe.H1,
        history=history,
        now=history.end + Timeframe.H1.delta,
        portfolio=PortfolioSnapshot(
            timestamp=history.end,
            base_currency="USDT",
            cash=Decimal("10000"),
            positions=(position,) if position else (),
            mark_prices={SYMBOL: bars[-1].close},
        ),
        position=position,
    )


def position_in(side: OrderSide, price: str = "100") -> Position:
    """An open position on the given side."""
    position, _ = Position(symbol=SYMBOL).apply_fill(
        Fill(
            fill_id="f",
            order_id="o",
            symbol=SYMBOL,
            side=side,
            quantity=Decimal("1"),
            price=Decimal(price),
            fee=Decimal("0"),
            fee_currency="USDT",
            timestamp=REFERENCE_TIME,
        )
    )
    return position


def fingerprint(signal: Signal) -> tuple[object, ...]:
    """Everything about a signal that a decision determines, minus its random id."""
    return (
        signal.direction,
        signal.conviction,
        signal.reference_price,
        signal.stop_loss_price,
        signal.take_profit_price,
        signal.reason,
    )


def assert_clean_hold(signal: Signal, strategy_id: str) -> None:
    """A hold that came from a decision, not from a swallowed exception.

    ``Strategy.evaluate`` converts any exception into a hold so one broken strategy cannot
    take the engine down. That makes a bare "it returned a signal" assertion worthless for
    a crash test: the crash would pass it. The reason string is the only evidence.
    """
    assert signal.direction is SignalDirection.HOLD
    assert signal.strategy_id == strategy_id
    assert not signal.reason.startswith("strategy error"), signal.reason
    assert not signal.reason.startswith("insufficient data"), signal.reason


def wavy_closes(count: int) -> list[Decimal]:
    """A deterministic drifting series with cycles and noise, entirely in ``Decimal``.

    Built from an integer generator rather than ``random`` so the look-ahead tests compare
    the same numbers on every run, and from a triangle wave rather than a sine so no float
    ever touches a price. The shape is deliberately tuned — a 60-bar cycle riding an upward
    drift, with noise a fraction of the swing — so that all six strategies find real trades
    on it. A series nothing trades on would make the look-ahead test vacuous.
    """
    closes: list[Decimal] = []
    seed = 12345
    for index in range(count):
        seed = (seed * 1103515245 + 12345) % 2147483648
        shock = Decimal(seed % 21 - 10) / 10
        phase = index % 60
        swing = Decimal(phase if phase < 30 else 60 - phase)
        closes.append(Decimal(100) + Decimal(index) * Decimal("0.6") + swing + shock)
    return closes


def reference_bars(volume: str = "10") -> list[Candle]:
    """The shared series every cross-cutting property is checked against."""
    return centred_series(wavy_closes(REFERENCE_SERIES_BARS), Decimal("1"), volume)


# --------------------------------------------------------------------------- #
# Scenario builders — one per strategy, shared by the behaviour and conviction tests
# --------------------------------------------------------------------------- #
def supertrend_flip_up(final_close: str) -> list[Candle]:
    """A long decline, then a single bar that closes back through the upper band."""
    closes = [Decimal(100 - index) for index in range(45)]
    closes.append(Decimal(final_close))
    return series_from(closes, Decimal("2"))


def ichimoku_breakout(slope: str) -> list[Candle]:
    """A slow climb, a dip that uncrosses tenkan/kijun, then a recovery above the cloud."""
    step = Decimal(slope)
    closes = [Decimal(100) + step * index for index in range(90)]
    closes += [closes[-1] - step * 3 * index for index in range(1, 11)]
    closes += [closes[-1] + step * 4 * index for index in range(1, 13)]
    return series_from(closes, Decimal("3"))


def sar_reversal(jump: int) -> list[Candle]:
    """A steady decline that reverses hard enough to break the accelerating stop."""
    closes = [Decimal(100 - index * 2) for index in range(30)]
    closes += [Decimal(42 + index * jump) for index in range(10)]
    return series_from(closes, Decimal("1"))


def mtf_confirmed_cross(slope: str) -> list[Candle]:
    """A long uptrend, a dip that uncrosses the fast EMAs, then a resumption."""
    step = Decimal(slope)
    base = [Decimal(100) + step * index for index in range(120)]
    dip = [base[-1] - step * 2 * index for index in range(1, 9)]
    up = [dip[-1] + step * 3 * index for index in range(1, 13)]
    return series_from(base + dip + up, Decimal("4"))


def rising_zigzag(step: int) -> list[Candle]:
    """A staircase of higher highs and higher lows."""
    closes: list[Decimal] = []
    level = Decimal(100)
    for _ in range(8):
        up = [level + Decimal(step) * leg for leg in range(1, 7)]
        closes += up
        level = up[-1]
        down = [level - Decimal(step) * leg for leg in range(1, 4)]
        closes += down
        level = down[-1]
    return centred_series(closes, Decimal("2"))


def falling_zigzag(step: int) -> list[Candle]:
    """The mirror image: lower highs and lower lows."""
    closes: list[Decimal] = []
    level = Decimal(400)
    for _ in range(8):
        down = [level - Decimal(step) * leg for leg in range(1, 7)]
        closes += down
        level = down[-1]
        up = [level + Decimal(step) * leg for leg in range(1, 4)]
        closes += up
        level = up[-1]
    return centred_series(closes, Decimal("2"))


def trend_pullback(depth: int) -> list[Candle]:
    """A firm uptrend, a retracement of ``depth``, then a close back through the fast EMA."""
    closes = [Decimal(100 + index * 2) for index in range(120)]
    top = closes[-1]
    closes += [top - depth, top - depth - 2, top - depth + 6, top - depth + 16, top - depth + 26]
    return series_from(closes, Decimal("2"))


def first_entry(strategy: Strategy, bars: Sequence[Candle]) -> tuple[int, Signal]:
    """Walk the series bar by bar and return the first entry signal, with its index.

    Walking rather than evaluating the last bar is the point: it proves the strategy fires
    somewhere specific, and every evaluation sees only the bars up to that point.
    """
    for cut in range(strategy.warmup_bars, len(bars) + 1):
        signal = strategy.evaluate(context_from(bars[:cut]))
        if signal.is_entry:
            return cut - 1, signal
    raise AssertionError(f"{strategy.strategy_id} produced no entry over {len(bars)} bars")


def entry_conviction(strategy: Strategy, bars: Sequence[Candle]) -> Decimal:
    """Conviction of the first entry a strategy takes on a series."""
    return first_entry(strategy, bars)[1].conviction


IDS = [name for name, _ in FACTORIES]
FACTORY_PARAMS = [factory for _, factory in FACTORIES]


# --------------------------------------------------------------------------- #
# Cross-cutting properties
# --------------------------------------------------------------------------- #
class TestWarmUp:
    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_holds_before_the_warm_up_is_complete(self, factory: StrategyFactory) -> None:
        strategy = factory()
        bars = series_from(wavy_closes(strategy.warmup_bars - 1))
        assert_clean_hold(strategy.evaluate(context_from(bars)), strategy.strategy_id)

    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_survives_a_single_bar(self, factory: StrategyFactory) -> None:
        strategy = factory()
        bars = series_from([Decimal("100")])
        assert_clean_hold(strategy.evaluate(context_from(bars)), strategy.strategy_id)

    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_generate_does_not_raise_at_the_first_permitted_bar(
        self, factory: StrategyFactory
    ) -> None:
        """``generate`` is called directly, so an exception is not swallowed by ``evaluate``."""
        strategy = factory()
        bars = series_from(wavy_closes(strategy.warmup_bars))
        signal = strategy.generate(context_from(bars))
        assert signal.strategy_id == strategy.strategy_id

    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_warmup_is_a_sane_positive_number(self, factory: StrategyFactory) -> None:
        strategy = factory()
        assert strategy.warmup_bars > 1
        assert strategy.warmup_bars <= REFERENCE_SERIES_BARS


class TestNoLookAhead:
    @pytest.mark.parametrize(
        "indicator",
        [
            lambda bars: supertrend(bars, 10, Decimal("3"))[0],
            lambda bars: supertrend(bars, 10, Decimal("3"))[1],
            lambda bars: ichimoku(bars)[2],
            lambda bars: ichimoku(bars)[3],
            lambda bars: parabolic_sar(bars)[0],
            lambda bars: parabolic_sar(bars)[1],
            lambda bars: swing_pivots(bars)[0],
            lambda bars: swing_pivots(bars)[1],
        ],
        ids=[
            "supertrend_line",
            "supertrend_direction",
            "senkou_a",
            "senkou_b",
            "sar",
            "sar_direction",
            "pivot_highs",
            "pivot_lows",
        ],
    )
    def test_indicator_values_do_not_change_when_later_bars_are_removed(
        self, indicator: Callable[[Sequence[Candle]], tuple[Decimal | None, ...]]
    ) -> None:
        bars = reference_bars()
        full = indicator(bars)
        for cut in (80, 120, 199):
            assert indicator(bars[:cut]) == full[:cut], f"values changed at cut {cut}"

    def test_higher_timeframe_buckets_do_not_change_when_later_bars_are_removed(self) -> None:
        bars = reference_bars()
        full = higher_timeframe_closes(bars, 4)
        for cut in (81, 122, 199):
            partial = higher_timeframe_closes(bars[:cut], 4)
            assert partial == full[: len(partial)]

    def test_the_partial_bucket_is_never_published(self) -> None:
        """Nine bars with a factor of four yield two buckets, not three."""
        bars = series_from([Decimal(100 + index) for index in range(9)])
        assert higher_timeframe_closes(bars, 4) == (bars[3].close, bars[7].close)

    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_signal_at_a_bar_is_identical_when_later_bars_are_removed(
        self, factory: StrategyFactory
    ) -> None:
        """Two things at once: no look-ahead, and no state carried between bars.

        The reference signals come from one instance walked across the whole series. The
        comparison signals come from fresh instances that have seen only the bars up to the
        cut. If a strategy read a future bar, or remembered anything from one call to the
        next, the two would diverge.
        """
        bars = reference_bars()
        walked = factory()
        cuts = list(range(walked.warmup_bars, len(bars) + 1))
        reference = {cut: fingerprint(walked.evaluate(context_from(bars[:cut]))) for cut in cuts}

        for cut in cuts[::7]:
            fresh = factory()
            assert (
                fingerprint(fresh.evaluate(context_from(bars[:cut]))) == reference[cut]
            ), f"{walked.strategy_id} decided differently at bar {cut - 1}"


class TestDegenerateInput:
    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_a_perfectly_flat_market_is_held_through(self, factory: StrategyFactory) -> None:
        """Zero ATR, zero range, zero swing: every denominator in the family is zero."""
        strategy = factory()
        bars = [
            candle(index, "100", "100", "100", "100") for index in range(strategy.warmup_bars + 20)
        ]
        assert_clean_hold(strategy.evaluate(context_from(bars)), strategy.strategy_id)

    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_zero_volume_bars_are_handled(self, factory: StrategyFactory) -> None:
        """None of these strategies reads volume, so a dead tape must change nothing."""
        strategy = factory()
        with_volume = strategy.evaluate(context_from(reference_bars()))
        without_volume = strategy.evaluate(context_from(reference_bars(volume="0")))
        assert fingerprint(with_volume) == fingerprint(without_volume)

    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_a_monotonic_ramp_does_not_raise(self, factory: StrategyFactory) -> None:
        """No pullback, no pivot, no reversal — every "find the last X" path returns nothing."""
        strategy = factory()
        bars = series_from([Decimal(100 + index) for index in range(strategy.warmup_bars + 20)])
        signal = strategy.evaluate(context_from(bars))
        assert not signal.reason.startswith("strategy error"), signal.reason

    @pytest.mark.parametrize("factory", FACTORY_PARAMS, ids=IDS)
    def test_every_signal_over_a_full_walk_is_well_formed(self, factory: StrategyFactory) -> None:
        """One walk, three properties.

        Conviction must stay inside ``[0, 1]`` on every bar; every entry must carry both
        protective levels on the correct sides of the entry with the target further away
        than the stop; and the strategy must actually enter at least once, without which
        the other two assertions are satisfied by a strategy that never trades.
        """
        strategy = factory()
        bars = reference_bars()
        entries = 0
        for cut in range(strategy.warmup_bars, len(bars) + 1):
            signal = strategy.evaluate(context_from(bars[:cut]))
            assert Decimal("0") <= signal.conviction <= Decimal("1")
            if not signal.is_entry:
                continue
            entries += 1
            assert signal.stop_loss_price is not None
            assert signal.take_profit_price is not None
            assert signal.reference_price is not None
            stop_distance = abs(signal.reference_price - signal.stop_loss_price)
            target_distance = abs(signal.reference_price - signal.take_profit_price)
            assert target_distance > stop_distance
            if signal.direction is SignalDirection.LONG:
                assert signal.stop_loss_price < signal.reference_price
                assert signal.take_profit_price > signal.reference_price
            else:
                assert signal.stop_loss_price > signal.reference_price
                assert signal.take_profit_price < signal.reference_price
        assert entries, f"{strategy.strategy_id} never entered"


# --------------------------------------------------------------------------- #
# The new indicators
# --------------------------------------------------------------------------- #
class TestIndicators:
    def test_supertrend_flips_up_when_price_closes_through_the_upper_band(self) -> None:
        bars = supertrend_flip_up("84")
        line, direction = supertrend(bars, 10, Decimal("3"))
        assert direction[-2] == Decimal("-1")
        assert direction[-1] == Decimal("1")
        # Having flipped, the line is now the *lower* band, below price.
        assert line[-1] is not None
        assert line[-1] < bars[-1].close

    def test_supertrend_rejects_a_non_positive_multiplier(self) -> None:
        bars = series_from(wavy_closes(40))
        with pytest.raises(ValidationError):
            supertrend(bars, 10, Decimal("0"))

    def test_the_cloud_at_a_bar_comes_from_the_displaced_past(self) -> None:
        """Senkou B at bar ``i`` must equal the raw 52-bar midpoint computed at ``i - 26``."""
        bars = series_from(wavy_closes(200))
        _, _, _, span_b = ichimoku(bars)
        index = 150
        window = bars[index - 26 - 51 : index - 26 + 1]
        expected = (max(bar.high for bar in window) + min(bar.low for bar in window)) / 2
        assert span_b[index] == expected

    def test_the_cloud_is_undefined_until_the_displacement_has_elapsed(self) -> None:
        bars = series_from(wavy_closes(200))
        _, _, span_a, span_b = ichimoku(bars)
        assert all(value is None for value in span_a[:26])
        assert all(value is None for value in span_b[:77])

    def test_sar_sits_below_price_in_an_uptrend_and_above_it_in_a_downtrend(self) -> None:
        rising = series_from([Decimal(100 + index * 2) for index in range(40)])
        sar, direction = parabolic_sar(rising)
        assert direction[-1] == Decimal("1")
        assert sar[-1] is not None
        assert sar[-1] < rising[-1].close

        falling = series_from([Decimal(200 - index * 2) for index in range(40)])
        sar, direction = parabolic_sar(falling)
        assert direction[-1] == Decimal("-1")
        assert sar[-1] is not None
        assert sar[-1] > falling[-1].close

    def test_sar_rejects_a_maximum_below_the_step(self) -> None:
        bars = series_from(wavy_closes(40))
        with pytest.raises(ValidationError):
            parabolic_sar(bars, Decimal("0.2"), Decimal("0.02"))

    def test_a_pivot_is_published_at_the_bar_that_confirms_it(self) -> None:
        """A peak at index 5 with ``right=3`` appears at index 8, and nowhere earlier."""
        closes = [Decimal(x) for x in (10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10, 9)]
        bars = centred_series(closes, Decimal("1"))
        highs, _ = swing_pivots(bars, 3, 3)
        assert highs[5] is None
        assert highs[8] == Decimal("21")
        assert sum(1 for value in highs if value is not None) == 1

    def test_a_flat_plateau_produces_no_pivot(self) -> None:
        bars = centred_series([Decimal("100")] * 30, Decimal("1"))
        highs, lows = swing_pivots(bars, 3, 3)
        assert all(value is None for value in highs)
        assert all(value is None for value in lows)

    def test_higher_timeframe_closes_rejects_a_factor_of_one(self) -> None:
        bars = series_from(wavy_closes(20))
        with pytest.raises(ValidationError):
            higher_timeframe_closes(bars, 1)


# --------------------------------------------------------------------------- #
# Per-strategy behaviour
# --------------------------------------------------------------------------- #
class TestSupertrend:
    def test_goes_long_when_the_band_flips_up(self) -> None:
        signal = SupertrendStrategy().evaluate(context_from(supertrend_flip_up("84")))
        assert signal.direction is SignalDirection.LONG
        assert "supertrend band" in signal.reason

    def test_a_decisive_break_carries_more_conviction_than_a_marginal_one(self) -> None:
        marginal = SupertrendStrategy().evaluate(context_from(supertrend_flip_up("78")))
        decisive = SupertrendStrategy().evaluate(context_from(supertrend_flip_up("84")))
        assert marginal.direction is SignalDirection.LONG
        assert marginal.conviction < Decimal("1")
        assert decisive.conviction > marginal.conviction

    def test_shorts_are_refused_unless_enabled(self) -> None:
        bars = series_from(
            [Decimal(100 + index) for index in range(45)] + [Decimal("70")], Decimal("2")
        )
        default = SupertrendStrategy().evaluate(context_from(bars))
        assert default.direction is SignalDirection.HOLD
        assert "short entries disabled" in default.reason

        enabled = SupertrendStrategy(SupertrendParams(allow_short=True)).evaluate(
            context_from(bars)
        )
        assert enabled.direction is SignalDirection.SHORT

    def test_a_long_is_closed_when_the_band_flips_down(self) -> None:
        bars = series_from(
            [Decimal(100 + index) for index in range(45)] + [Decimal("70")], Decimal("2")
        )
        signal = SupertrendStrategy().evaluate(
            context_from(bars, position_in(OrderSide.BUY, "140"))
        )
        assert signal.direction is SignalDirection.CLOSE

    def test_target_must_exceed_stop(self) -> None:
        with pytest.raises(PydanticValidationError, match="must exceed"):
            SupertrendParams(atr_stop_multiple=Decimal("3"), atr_target_multiple=Decimal("2"))


class TestIchimokuTrend:
    def test_goes_long_on_a_cross_above_the_cloud(self) -> None:
        strategy = IchimokuTrendStrategy()
        _, signal = first_entry(strategy, ichimoku_breakout("0.3"))
        assert signal.direction is SignalDirection.LONG
        assert "above the cloud" in signal.reason

    def test_a_cross_inside_the_cloud_is_refused(self) -> None:
        """The same cross, but with price parked on the cloud rather than clear of it."""
        step = Decimal("0.3")
        closes = [Decimal(100) + step * index for index in range(90)]
        closes += [closes[-1] - step * 3 * index for index in range(1, 11)]
        # Recover just enough to cross tenkan over kijun, but not out of the cloud.
        closes += [closes[-1] + step * index for index in range(1, 13)]
        bars = series_from(closes, Decimal("3"))
        strategy = IchimokuTrendStrategy()
        reasons = {
            strategy.evaluate(context_from(bars[:cut])).reason
            for cut in range(strategy.warmup_bars, len(bars) + 1)
        }
        assert not any("above the cloud" in reason for reason in reasons)

    def test_conviction_rises_with_clearance_from_the_cloud(self) -> None:
        gentle = entry_conviction(IchimokuTrendStrategy(), ichimoku_breakout("0.1"))
        firm = entry_conviction(IchimokuTrendStrategy(), ichimoku_breakout("0.3"))
        assert gentle < Decimal("1")
        assert firm > gentle

    def test_periods_must_increase(self) -> None:
        with pytest.raises(PydanticValidationError, match="must increase"):
            IchimokuTrendParams(tenkan_period=30, kijun_period=26)

    def test_warmup_covers_the_span_and_its_displacement(self) -> None:
        strategy = IchimokuTrendStrategy()
        assert strategy.warmup_bars >= 52 + 26


class TestParabolicSar:
    def test_goes_long_when_the_stop_reverses_up(self) -> None:
        strategy = ParabolicSarStrategy()
        _, signal = first_entry(strategy, sar_reversal(5))
        assert signal.direction is SignalDirection.LONG
        assert "sar reversed up" in signal.reason

    def test_a_bigger_reversal_carries_more_conviction(self) -> None:
        small = entry_conviction(ParabolicSarStrategy(), sar_reversal(2))
        large = entry_conviction(ParabolicSarStrategy(), sar_reversal(3))
        assert small < Decimal("1")
        assert large > small

    def test_a_long_is_closed_when_the_stop_reverses_down(self) -> None:
        closes = [Decimal(100 + index * 2) for index in range(30)]
        closes += [Decimal(158 - index * 6) for index in range(1, 8)]
        bars = series_from(closes, Decimal("1"))
        strategy = ParabolicSarStrategy()
        closes_seen = [
            strategy.evaluate(context_from(bars[:cut], position_in(OrderSide.BUY, "150")))
            for cut in range(strategy.warmup_bars, len(bars) + 1)
        ]
        assert any(signal.direction is SignalDirection.CLOSE for signal in closes_seen)

    def test_a_maximum_below_the_step_is_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="at least step"):
            ParabolicSarParams(step=Decimal("0.2"), maximum=Decimal("0.02"))

    def test_shorts_are_refused_unless_enabled(self) -> None:
        closes = [Decimal(100 + index * 2) for index in range(30)]
        closes += [Decimal(158 - index * 6) for index in range(1, 8)]
        bars = series_from(closes, Decimal("1"))
        default = ParabolicSarStrategy()
        enabled = ParabolicSarStrategy(ParabolicSarParams(allow_short=True))
        assert not any(
            default.evaluate(context_from(bars[:cut])).is_entry
            for cut in range(default.warmup_bars, len(bars) + 1)
        )
        assert any(
            enabled.evaluate(context_from(bars[:cut])).direction is SignalDirection.SHORT
            for cut in range(enabled.warmup_bars, len(bars) + 1)
        )


class TestMtfTrend:
    def test_goes_long_when_both_timeframes_agree(self) -> None:
        strategy = MtfTrendStrategy()
        _, signal = first_entry(strategy, mtf_confirmed_cross("0.5"))
        assert signal.direction is SignalDirection.LONG
        assert "higher timeframe" in signal.reason

    def test_a_cross_against_the_higher_timeframe_is_refused(self) -> None:
        """A bounce inside a downtrend: the fast EMAs cross up, the 4-bar trend does not."""
        closes = [Decimal(600) - Decimal(index) * 2 for index in range(160)]
        closes += [closes[-1] + Decimal(index) * 2 for index in range(1, 14)]
        bars = series_from(closes, Decimal("2"))
        strategy = MtfTrendStrategy()
        reasons = [
            strategy.evaluate(context_from(bars[:cut])).reason
            for cut in range(strategy.warmup_bars, len(bars) + 1)
        ]
        assert any("higher timeframe disagrees" in reason for reason in reasons)
        assert not any("confirmed by" in reason for reason in reasons)

    def test_conviction_rises_with_higher_timeframe_separation(self) -> None:
        gentle = entry_conviction(MtfTrendStrategy(), mtf_confirmed_cross("0.1"))
        firm = entry_conviction(MtfTrendStrategy(), mtf_confirmed_cross("0.5"))
        assert gentle < Decimal("1")
        assert firm > gentle

    def test_higher_timeframe_periods_must_increase(self) -> None:
        with pytest.raises(PydanticValidationError, match="htf_fast_period"):
            MtfTrendParams(htf_fast_period=20, htf_slow_period=10)

    def test_warmup_prices_the_higher_timeframe_in_base_bars(self) -> None:
        strategy = MtfTrendStrategy(MtfTrendParams(htf_factor=6, htf_slow_period=10))
        assert strategy.warmup_bars >= 6 * 10


class TestSwingStructure:
    def test_goes_long_on_confirmed_higher_highs_and_higher_lows(self) -> None:
        strategy = SwingStructureStrategy()
        _, signal = first_entry(strategy, rising_zigzag(3))
        assert signal.direction is SignalDirection.LONG
        assert "higher swing high" in signal.reason
        assert "higher swing low" in signal.reason

    def test_lower_highs_and_lower_lows_produce_a_short_only_when_enabled(self) -> None:
        bars = falling_zigzag(3)
        default = SwingStructureStrategy()
        assert not any(
            default.evaluate(context_from(bars[:cut])).is_entry
            for cut in range(default.warmup_bars, len(bars) + 1)
        )
        enabled = SwingStructureStrategy(SwingStructureParams(allow_short=True))
        _, signal = first_entry(enabled, bars)
        assert signal.direction is SignalDirection.SHORT
        assert "lower swing high" in signal.reason

    def test_an_expanding_range_is_not_a_trend(self) -> None:
        """Higher highs with lower lows: no structure, so no entry."""
        closes: list[Decimal] = []
        level = Decimal("200")
        for leg in range(8):
            closes += [level + Decimal(6 + leg * 4) * step / 3 for step in range(1, 5)]
            closes += [level - Decimal(6 + leg * 4) * step / 3 for step in range(1, 5)]
        bars = centred_series(closes, Decimal("1"))
        strategy = SwingStructureStrategy()
        assert not any(
            strategy.evaluate(context_from(bars[:cut])).is_entry
            for cut in range(strategy.warmup_bars, len(bars) + 1)
        )

    def test_a_wider_structural_step_carries_more_conviction(self) -> None:
        narrow = entry_conviction(SwingStructureStrategy(), rising_zigzag(2))
        wide = entry_conviction(SwingStructureStrategy(), rising_zigzag(3))
        assert narrow < Decimal("1")
        assert wide > narrow

    def test_a_long_is_closed_when_price_loses_the_last_swing_low(self) -> None:
        bars = rising_zigzag(3)
        broken = [*bars, candle(len(bars), "150", "151", "80", "82")]
        signal = SwingStructureStrategy().evaluate(
            context_from(broken, position_in(OrderSide.BUY, "150"))
        )
        assert signal.direction is SignalDirection.CLOSE
        assert "swing low" in signal.reason


class TestPullbackContinuation:
    def test_goes_long_when_a_retracement_resumes(self) -> None:
        strategy = PullbackContinuationStrategy()
        _, signal = first_entry(strategy, trend_pullback(28))
        assert signal.direction is SignalDirection.LONG
        assert "resumed above" in signal.reason

    def test_a_deeper_retracement_carries_more_conviction(self) -> None:
        shallow = entry_conviction(PullbackContinuationStrategy(), trend_pullback(20))
        deep = entry_conviction(PullbackContinuationStrategy(), trend_pullback(28))
        assert shallow < Decimal("1")
        assert deep > shallow

    def test_a_retracement_through_the_slow_average_is_not_a_pullback(self) -> None:
        """Price collapses past the 50-period average and bounces: the trend is gone."""
        closes = [Decimal(100 + index * 2) for index in range(120)]
        top = closes[-1]
        closes += [top - 120, top - 130, top - 110, top - 90]
        bars = series_from(closes, Decimal("2"))
        strategy = PullbackContinuationStrategy()
        assert not any(
            strategy.evaluate(context_from(bars[:cut])).is_entry
            for cut in range(strategy.warmup_bars, len(bars) + 1)
        )

    def test_no_entry_without_an_established_trend(self) -> None:
        """A flat, noisy market has a flat slow average, so nothing qualifies."""
        closes = [Decimal(100) + Decimal((index * 7) % 5) for index in range(200)]
        bars = series_from(closes, Decimal("1"))
        strategy = PullbackContinuationStrategy()
        reasons = {
            strategy.evaluate(context_from(bars[:cut])).reason
            for cut in range(strategy.warmup_bars, len(bars) + 1)
        }
        assert "no established trend" in reasons

    def test_a_long_is_closed_below_the_slow_average(self) -> None:
        closes = [Decimal(100 + index * 2) for index in range(120)]
        closes += [Decimal(150)]
        bars = series_from(closes, Decimal("2"))
        signal = PullbackContinuationStrategy().evaluate(
            context_from(bars, position_in(OrderSide.BUY, "300"))
        )
        assert signal.direction is SignalDirection.CLOSE
        assert "slow average" in signal.reason

    def test_fast_period_must_be_below_slow_period(self) -> None:
        with pytest.raises(PydanticValidationError, match="must be below"):
            PullbackContinuationParams(fast_period=60, slow_period=50)
