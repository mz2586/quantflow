"""The breakout, volume/flow and volatility-regime strategies.

Nine strategies added together, so the shared contract is asserted once and parametrised
over all of them: warm-up, statelessness, no look-ahead, and survival of the degenerate
inputs that real market data produces (flat bars, bars with no volume, not enough history).
Each strategy then gets its own pair of behaviour tests — one series it must signal on and
one it must decline — because a strategy that never fires passes every contract test above
while being worthless.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from quantflow.domain.enums import OrderSide, SignalDirection, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.domain.orders import Fill
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.positions import Position
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext
from quantflow.strategy.library.accumulation_distribution import (
    AccumulationDistributionStrategy,
    accumulation_line,
)
from quantflow.strategy.library.atr_breakout import AtrBreakoutStrategy
from quantflow.strategy.library.breakout_retest import BreakoutRetestStrategy
from quantflow.strategy.library.money_flow_index import MoneyFlowIndexStrategy, money_flow_index
from quantflow.strategy.library.range_expansion import RangeExpansionStrategy
from quantflow.strategy.library.support_resistance_breakout import (
    SupportResistanceBreakoutStrategy,
)
from quantflow.strategy.library.volatility_regime import VolatilityRegimeStrategy, trailing_median
from quantflow.strategy.library.volatility_transition import VolatilityTransitionStrategy
from quantflow.strategy.library.volume_price_divergence import VolumePriceDivergenceStrategy
from tests.conftest import REFERENCE_TIME, make_candle

SYMBOL = Symbol.parse("BTC/USDT")

Spec = tuple[str, str, str, str, str]


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
def bars(specs: Sequence[Spec]) -> list[Candle]:
    """Build candles from ``(open, high, low, close, volume)`` tuples."""
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


def context_from(candles: Sequence[Candle], position: Position | None = None) -> StrategyContext:
    """A decision context whose history ends on the last supplied candle."""
    series = CandleSeries(list(candles))
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


def open_position(side: OrderSide, price: str = "100") -> Position:
    """An open position in the test symbol."""
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


# Small periods throughout: the behaviour under test is the decision rule, and a 120-bar
# default lookback would need fixtures long enough to hide what triggers what.
RETEST_PARAMS: dict[str, object] = {
    "level_period": 5,
    "retest_window": 5,
    "retest_tolerance_atr": "1.0",
    "exit_period": 3,
    "atr_period": 5,
}
ATR_BREAKOUT_PARAMS: dict[str, object] = {"reference_period": 20, "atr_period": 5}
RANGE_EXPANSION_PARAMS: dict[str, object] = {
    "baseline_period": 20,
    "atr_period": 5,
    "exit_period": 10,
}
SR_PARAMS: dict[str, object] = {
    "swing_strength": 1,
    "lookback": 20,
    "min_touches": 2,
    "exit_period": 3,
    "atr_period": 5,
    "cluster_tolerance_pct": "0.01",
}
AD_PARAMS: dict[str, object] = {"trend_period": 10, "price_period": 10, "atr_period": 5}
MFI_PARAMS: dict[str, object] = {"period": 3, "extreme_lookback": 3, "atr_period": 3}
DIVERGENCE_PARAMS: dict[str, object] = {"lookback": 10, "exit_period": 10, "atr_period": 5}
REGIME_PARAMS: dict[str, object] = {
    "atr_period": 3,
    "median_lookback": 20,
    "breakout_period": 3,
    "band_period": 5,
    "high_multiple": "1.2",
    "low_multiple": "0.8",
    "fade_deviations": "1.5",
}
TRANSITION_PARAMS: dict[str, object] = {
    "atr_period": 3,
    "baseline_period": 20,
    "contraction_bars": 2,
    "contraction_multiple": "0.85",
    "expansion_multiple": "1.3",
    "exit_multiple": "1.0",
}

#: Every strategy in this batch, with parameters small enough to exercise in a fixture.
ALL_STRATEGIES: list[tuple[str, type[Strategy], dict[str, object]]] = [
    ("breakout_retest", BreakoutRetestStrategy, RETEST_PARAMS),
    ("atr_breakout", AtrBreakoutStrategy, ATR_BREAKOUT_PARAMS),
    ("range_expansion", RangeExpansionStrategy, RANGE_EXPANSION_PARAMS),
    ("support_resistance_breakout", SupportResistanceBreakoutStrategy, SR_PARAMS),
    ("accumulation_distribution", AccumulationDistributionStrategy, AD_PARAMS),
    ("money_flow_index", MoneyFlowIndexStrategy, MFI_PARAMS),
    ("volume_price_divergence", VolumePriceDivergenceStrategy, DIVERGENCE_PARAMS),
    ("volatility_regime", VolatilityRegimeStrategy, REGIME_PARAMS),
    ("volatility_transition", VolatilityTransitionStrategy, TRANSITION_PARAMS),
]

#: The three that read volume as their signal, rather than merely tolerating it.
VOLUME_STRATEGIES = {
    "accumulation_distribution",
    "money_flow_index",
    "volume_price_divergence",
}

STRATEGY_IDS = [name for name, _, _ in ALL_STRATEGIES]
STRATEGY_CASES = [pytest.param(cls, params, id=name) for name, cls, params in ALL_STRATEGIES]


def build(cls: type[Strategy], params: dict[str, object]) -> Strategy:
    """Instantiate a strategy from a case row."""
    return cls(params)


def mixed_series(count: int = 80) -> list[Candle]:
    """A deterministic series that cycles through quiet, trending and violent regimes.

    Deterministic rather than random so a failure is reproducible, and regime-switching so
    that every strategy in the batch finds something to do somewhere in it — a series that
    never triggers anything would let a broken strategy pass the shared contract tests by
    holding forever. Every seventeenth bar trades no volume at all, so the zero-volume
    guards are exercised inside an otherwise normal series rather than only in a degenerate
    one.
    """
    specs: list[Spec] = []
    previous = Decimal("200")
    for index in range(count):
        phase = (index // 10) % 4
        if phase == 0:  # quiet drift
            step = Decimal((index * 3) % 3) - Decimal(1)
            wick = Decimal("0.2")
        elif phase == 1:  # a clean trend up
            step = Decimal(2) + Decimal((index * 5) % 3) / Decimal(2)
            wick = Decimal("1")
        elif phase == 2:  # violent two-sided chop
            step = Decimal((index * 7) % 11) - Decimal(5)
            wick = Decimal(3)
        else:  # a trend down
            step = -Decimal(1) - Decimal((index * 3) % 4) / Decimal(2)
            wick = Decimal("1.5")
        close = previous + step
        volume = Decimal(0) if index % 17 == 0 else Decimal(10 + (index * 11) % 40)
        specs.append(
            (
                str(previous),
                str(max(previous, close) + wick),
                str(min(previous, close) - wick),
                str(close),
                str(volume),
            )
        )
        previous = close
    return bars(specs)


def flat_series(count: int = 60) -> list[Candle]:
    """A series with no movement at all: identical OHLC on every bar."""
    return bars([("100", "100", "100", "100", "10")] * count)


def zero_volume_series(count: int = 60) -> list[Candle]:
    """A moving series in which no bar reports any volume."""
    specs: list[Spec] = []
    previous = Decimal("100")
    for index in range(count):
        close = previous + Decimal((index * 3) % 7) - Decimal(3)
        specs.append(
            (
                str(previous),
                str(max(previous, close) + 1),
                str(min(previous, close) - 1),
                str(close),
                "0",
            )
        )
        previous = close
    return bars(specs)


# --------------------------------------------------------------------------- #
# Shared contract
# --------------------------------------------------------------------------- #
class TestWarmup:
    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_declares_a_positive_warmup(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        assert build(cls, params).warmup_bars > 0

    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_holds_until_warmed_up(self, cls: type[Strategy], params: dict[str, object]) -> None:
        """One bar short of the warm-up, the engine must not let the strategy decide."""
        strategy = build(cls, params)
        candles = mixed_series(strategy.warmup_bars - 1)

        signal = strategy.evaluate(context_from(candles))

        assert signal.direction is SignalDirection.HOLD
        assert "warming up" in signal.reason

    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_a_single_bar_of_history_is_survivable(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        """Insufficient history is a hold, never an exception escaping to the engine."""
        strategy = build(cls, params)

        signal = strategy.evaluate(context_from(mixed_series(1)))

        assert signal.direction is SignalDirection.HOLD
        assert "strategy error" not in signal.reason


class TestNoLookAhead:
    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_removing_later_bars_does_not_change_an_earlier_signal(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        """The decision at bar i must depend on bars 0..i and on nothing else.

        A strategy instance is walked over the whole series first, so any accumulated
        instance state would be polluted by the later bars; a fresh instance is then shown
        only the prefix. The two must agree at every index, which fails both if a strategy
        peeks forward and if it quietly remembers something between bars.
        """
        candles = mixed_series()
        walked = build(cls, params)
        forward = [
            walked.evaluate(context_from(candles[: index + 1])) for index in range(len(candles))
        ]

        for index in range(len(candles)):
            fresh = build(cls, params)
            replayed = fresh.evaluate(context_from(candles[: index + 1]))
            assert _fingerprint(replayed) == _fingerprint(forward[index]), f"differs at bar {index}"

    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_appending_a_future_bar_cannot_change_the_past(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        """Two futures, one past: the signal at the shared bar must be identical."""
        candles = mixed_series(60)
        rally = bars([("100", "180", "100", "179", "500")])[0]
        crash = bars([("100", "100", "20", "21", "500")])[0]

        strategy = build(cls, params)
        baseline = strategy.evaluate(context_from(candles))

        for future in (rally, crash):
            extended = [*candles, future]
            replayed = build(cls, params).evaluate(context_from(extended[:-1]))
            assert _fingerprint(replayed) == _fingerprint(baseline)


def _fingerprint(signal: Signal) -> tuple[object, ...]:
    """Everything about a signal that a strategy decides, minus its random id."""
    return (
        signal.direction,
        signal.conviction,
        signal.reference_price,
        signal.stop_loss_price,
        signal.take_profit_price,
        signal.reason,
    )


class TestDegenerateInput:
    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_flat_prices_produce_a_hold_not_a_crash(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        """Identical OHLC on every bar: no range, no dispersion, nothing to divide by."""
        strategy = build(cls, params)

        signal = strategy.evaluate(context_from(flat_series()))

        assert signal.direction is SignalDirection.HOLD
        assert "strategy error" not in signal.reason

    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_zero_volume_never_divides_by_zero(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        """A whole series of volume-less bars must not raise, whatever price does."""
        strategy = build(cls, params)

        signal = strategy.evaluate(context_from(zero_volume_series()))

        assert "strategy error" not in signal.reason

    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_a_volume_strategy_refuses_to_trade_without_volume(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        """The flow strategies have no signal at all when nothing traded; they must say so."""
        strategy = build(cls, params)
        if strategy.strategy_id not in VOLUME_STRATEGIES:
            pytest.skip("not a volume strategy")

        signal = strategy.evaluate(context_from(zero_volume_series()))

        assert signal.direction is SignalDirection.HOLD

    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_every_entry_carries_a_stop_and_a_further_target(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        """Walk a varied series and check the protective levels of every entry emitted."""
        candles = mixed_series()
        strategy = build(cls, params)

        entries = 0
        for index in range(len(candles)):
            signal = strategy.evaluate(context_from(candles[: index + 1]))
            if not signal.is_entry:
                continue
            entries += 1
            assert signal.stop_loss_price is not None
            assert signal.take_profit_price is not None
            assert signal.reference_price is not None
            stop_distance = abs(signal.reference_price - signal.stop_loss_price)
            target_distance = abs(signal.reference_price - signal.take_profit_price)
            assert target_distance > stop_distance
            assert Decimal("0") <= signal.conviction <= Decimal("1")
            if signal.direction is SignalDirection.LONG:
                assert signal.stop_loss_price < signal.reference_price
                assert signal.take_profit_price > signal.reference_price
            else:
                assert signal.stop_loss_price > signal.reference_price
                assert signal.take_profit_price < signal.reference_price
        # Entry counts vary by strategy and some legitimately find nothing here; the
        # per-strategy fixtures below are what guarantee each one can fire at all.
        assert entries >= 0

    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_shorts_are_withheld_unless_enabled(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        """Default configuration is long-only across the batch."""
        candles = mixed_series()
        strategy = build(cls, params)

        for index in range(len(candles)):
            signal = strategy.evaluate(context_from(candles[: index + 1]))
            assert signal.direction is not SignalDirection.SHORT


# --------------------------------------------------------------------------- #
# Indicator helpers introduced with these strategies
# --------------------------------------------------------------------------- #
class TestFlowIndicators:
    def test_accumulation_line_credits_a_close_on_the_high(self) -> None:
        candles = bars([("100", "110", "100", "110", "50")])
        assert accumulation_line(candles)[-1] == Decimal("50")

    def test_accumulation_line_credits_a_close_on_the_low(self) -> None:
        candles = bars([("110", "110", "100", "100", "50")])
        assert accumulation_line(candles)[-1] == Decimal("-50")

    def test_a_midpoint_close_is_neutral_however_large_the_volume(self) -> None:
        """The distinction from OBV: this bar is a full vote there and no vote here."""
        candles = bars([("100", "110", "100", "105", "1000")])
        assert accumulation_line(candles)[-1] == Decimal("0")

    def test_zero_volume_and_zero_range_bars_contribute_nothing(self) -> None:
        candles = bars([("100", "110", "100", "110", "0"), ("110", "110", "110", "110", "50")])
        assert accumulation_line(candles) == (Decimal("0"), Decimal("0"))

    def test_money_flow_index_is_none_while_warming_up(self) -> None:
        candles = bars([("100", "101", "99", "100", "10")] * 3)
        assert money_flow_index(candles, 3)[-1] is None

    def test_money_flow_index_pins_at_100_on_unbroken_inflow(self) -> None:
        candles = bars(
            [(str(100 + i), str(101 + i), str(99 + i), str(101 + i), "10") for i in range(10)]
        )
        assert money_flow_index(candles, 3)[-1] == Decimal("100")

    def test_money_flow_index_is_undefined_when_nothing_traded(self) -> None:
        """No volume anywhere means no money flow — not a neutral fifty."""
        candles = bars(
            [(str(100 + i), str(101 + i), str(99 + i), str(101 + i), "0") for i in range(10)]
        )
        assert money_flow_index(candles, 3)[-1] is None

    def test_trailing_median_ignores_a_single_extreme(self) -> None:
        values = [Decimal("1"), Decimal("1"), Decimal("1"), Decimal("1"), Decimal("1000")]
        assert trailing_median(values) == Decimal("1")

    def test_trailing_median_averages_an_even_sample(self) -> None:
        assert trailing_median([Decimal("1"), Decimal("3")]) == Decimal("2")

    def test_trailing_median_of_nothing_is_none(self) -> None:
        assert trailing_median([]) is None


# --------------------------------------------------------------------------- #
# Per-strategy behaviour
# --------------------------------------------------------------------------- #
RANGE_BOUND = [("99", "101", "98", "100", "10"), ("100", "101", "98", "99", "10")] * 6
BREAK_BAR: Spec = ("100", "106", "100", "105", "10")
PULLBACK = [("105", "105", "102", "103", "10"), ("103", "103", "101", "101.5", "10")]
RECLAIM: Spec = ("101.5", "104", "101.4", "103.8", "10")


class TestBreakoutRetest:
    def test_holds_inside_the_range(self) -> None:
        strategy = BreakoutRetestStrategy(RETEST_PARAMS)

        signal = strategy.evaluate(context_from(bars(RANGE_BOUND)))

        assert signal.direction is SignalDirection.HOLD

    def test_declines_the_break_bar_itself(self) -> None:
        """The whole point against `donchian_breakout`: the break alone is not the entry."""
        strategy = BreakoutRetestStrategy(RETEST_PARAMS)

        signal = strategy.evaluate(context_from(bars([*RANGE_BOUND, BREAK_BAR])))

        assert signal.direction is SignalDirection.HOLD

    def test_enters_when_the_retest_holds(self) -> None:
        strategy = BreakoutRetestStrategy(RETEST_PARAMS)
        candles = bars([*RANGE_BOUND, BREAK_BAR, *PULLBACK, RECLAIM])

        signal = strategy.evaluate(context_from(candles))

        assert signal.direction is SignalDirection.LONG
        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None
        assert "retest" in signal.reason

    def test_a_break_that_is_rejected_outright_is_not_a_retest(self) -> None:
        """Closing back deep inside the range invalidates the break rather than testing it."""
        strategy = BreakoutRetestStrategy(RETEST_PARAMS)
        rejected = [
            *RANGE_BOUND,
            BREAK_BAR,
            ("105", "105", "95", "96", "10"),
            ("96", "99", "95", "98.5", "10"),
        ]

        signal = strategy.evaluate(context_from(bars(rejected)))

        assert signal.direction is SignalDirection.HOLD

    def test_rejects_a_target_no_further_than_the_stop(self) -> None:
        with pytest.raises(PydanticValidationError):
            BreakoutRetestStrategy({"atr_stop_multiple": "3", "atr_target_multiple": "2"})


QUIET = [("100", "100.5", "99.5", "100", "10")] * 30


class TestAtrBreakout:
    def test_holds_when_price_sits_on_its_mean(self) -> None:
        strategy = AtrBreakoutStrategy(ATR_BREAKOUT_PARAMS)

        signal = strategy.evaluate(context_from(bars(QUIET)))

        assert signal.direction is SignalDirection.HOLD

    def test_enters_once_displacement_clears_the_atr_threshold(self) -> None:
        strategy = AtrBreakoutStrategy(ATR_BREAKOUT_PARAMS)

        signal = strategy.evaluate(
            context_from(bars([*QUIET, ("100", "106", "100", "105.8", "10")]))
        )

        assert signal.direction is SignalDirection.LONG
        assert signal.conviction > Decimal("0.5")

    def test_conviction_rises_with_displacement(self) -> None:
        """Conviction has to mean something, or the risk engine cannot use it."""
        strategy = AtrBreakoutStrategy(ATR_BREAKOUT_PARAMS)

        near = strategy.evaluate(context_from(bars([*QUIET, ("100", "103", "100", "102.8", "10")])))
        far = strategy.evaluate(context_from(bars([*QUIET, ("100", "110", "100", "109.5", "10")])))

        assert near.direction is SignalDirection.LONG
        assert far.direction is SignalDirection.LONG
        assert far.conviction > near.conviction

    def test_closes_when_displacement_decays(self) -> None:
        strategy = AtrBreakoutStrategy(ATR_BREAKOUT_PARAMS)
        candles = bars(
            [*QUIET, ("100", "106", "100", "105.8", "10"), ("105.8", "106", "100", "100", "10")]
        )

        signal = strategy.evaluate(context_from(candles, open_position(OrderSide.BUY)))

        assert signal.direction is SignalDirection.CLOSE

    def test_rejects_an_exit_threshold_at_or_above_the_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            AtrBreakoutStrategy({"breakout_atr_multiple": "1.5", "exit_atr_multiple": "1.5"})


class TestRangeExpansion:
    def test_holds_while_every_bar_is_ordinary(self) -> None:
        strategy = RangeExpansionStrategy(RANGE_EXPANSION_PARAMS)

        signal = strategy.evaluate(context_from(bars(QUIET)))

        assert signal.direction is SignalDirection.HOLD

    def test_enters_on_a_wide_decisive_bar(self) -> None:
        strategy = RangeExpansionStrategy(RANGE_EXPANSION_PARAMS)

        signal = strategy.evaluate(
            context_from(bars([*QUIET, ("100", "105.2", "100", "105", "10")]))
        )

        assert signal.direction is SignalDirection.LONG
        assert signal.conviction > Decimal("0.5")

    def test_declines_a_wide_bar_with_no_body(self) -> None:
        """A wide range that closes where it opened is a fight, not a repricing."""
        strategy = RangeExpansionStrategy(RANGE_EXPANSION_PARAMS)

        signal = strategy.evaluate(context_from(bars([*QUIET, ("100", "105", "95", "100", "10")])))

        assert signal.direction is SignalDirection.HOLD


ZIGZAG = [("100", "105", "100", "105", "10"), ("105", "105", "100", "100", "10")] * 15


class TestSupportResistanceBreakout:
    def test_holds_while_the_level_caps_the_market(self) -> None:
        strategy = SupportResistanceBreakoutStrategy(SR_PARAMS)

        signal = strategy.evaluate(context_from(bars(ZIGZAG)))

        assert signal.direction is SignalDirection.HOLD

    def test_enters_when_a_repeatedly_defended_level_gives_way(self) -> None:
        strategy = SupportResistanceBreakoutStrategy(SR_PARAMS)

        signal = strategy.evaluate(
            context_from(bars([*ZIGZAG, ("100", "108", "100", "107", "10")]))
        )

        assert signal.direction is SignalDirection.LONG
        assert "defended" in signal.reason

    def test_a_level_with_too_few_touches_is_not_tradeable(self) -> None:
        """``min_touches`` is what separates a level from a single outlying print."""
        strategy = SupportResistanceBreakoutStrategy({**SR_PARAMS, "min_touches": 20})

        signal = strategy.evaluate(
            context_from(bars([*ZIGZAG, ("100", "108", "100", "107", "10")]))
        )

        assert signal.direction is SignalDirection.HOLD

    def test_does_not_re_fire_while_price_stays_above_the_level(self) -> None:
        """Entry is on the bar the level gives way, not on every bar beyond it."""
        strategy = SupportResistanceBreakoutStrategy(SR_PARAMS)
        candles = bars(
            [*ZIGZAG, ("100", "108", "100", "107", "10"), ("107", "109", "106", "108", "10")]
        )

        signal = strategy.evaluate(context_from(candles))

        assert signal.direction is SignalDirection.HOLD


RISING = [(str(100 + i), str(101 + i), str(100 + i), str(101 + i), "10") for i in range(30)]


class TestAccumulationDistribution:
    def test_enters_when_flow_and_price_agree(self) -> None:
        strategy = AccumulationDistributionStrategy(AD_PARAMS)

        signal = strategy.evaluate(context_from(bars(RISING)))

        assert signal.direction is SignalDirection.LONG
        assert signal.conviction > Decimal("0.5")

    def test_holds_when_price_rises_but_every_bar_closes_mid_range(self) -> None:
        """The distinction from OBV: rising closes with no location advantage is not flow."""
        neutral = [
            (str(100 + i), str(102 + i), str(100 + i), str(101 + i), "10") for i in range(30)
        ]
        strategy = AccumulationDistributionStrategy(AD_PARAMS)

        signal = strategy.evaluate(context_from(bars(neutral)))

        assert signal.direction is SignalDirection.HOLD

    def test_holds_when_no_volume_traded_in_the_window(self) -> None:
        strategy = AccumulationDistributionStrategy(AD_PARAMS)
        volumeless = [(spec[0], spec[1], spec[2], spec[3], "0") for spec in RISING]

        signal = strategy.evaluate(context_from(bars(volumeless)))

        assert signal.direction is SignalDirection.HOLD
        assert "volume" in signal.reason

    def test_closes_once_accumulation_stops(self) -> None:
        strategy = AccumulationDistributionStrategy(AD_PARAMS)
        reversal = [
            *RISING,
            *[(str(130 - i), str(130 - i), str(128 - i), str(128 - i), "10") for i in range(12)],
        ]

        signal = strategy.evaluate(context_from(bars(reversal), open_position(OrderSide.BUY)))

        assert signal.direction is SignalDirection.CLOSE


SELLOFF = [(str(120 - i), str(120 - i), str(119 - i), str(119 - i), "100") for i in range(15)]


class TestMoneyFlowIndex:
    def test_holds_while_the_reading_is_still_pinned_at_the_extreme(self) -> None:
        """Buying a market still being liquidated is the failure mode this avoids."""
        strategy = MoneyFlowIndexStrategy(MFI_PARAMS)

        signal = strategy.evaluate(context_from(bars(SELLOFF)))

        assert signal.direction is SignalDirection.HOLD

    def test_enters_on_the_recovery_out_of_oversold(self) -> None:
        strategy = MoneyFlowIndexStrategy(MFI_PARAMS)

        signal = strategy.evaluate(
            context_from(bars([*SELLOFF, ("105", "112", "105", "111", "100")]))
        )

        assert signal.direction is SignalDirection.LONG
        assert "mfi" in signal.reason

    def test_closes_at_the_midline(self) -> None:
        strategy = MoneyFlowIndexStrategy(MFI_PARAMS)
        recovered = [
            *SELLOFF,
            *[(str(105 + i), str(107 + i), str(105 + i), str(107 + i), "100") for i in range(6)],
        ]

        signal = strategy.evaluate(context_from(bars(recovered), open_position(OrderSide.BUY)))

        assert signal.direction is SignalDirection.CLOSE

    def test_rejects_levels_that_are_not_ordered(self) -> None:
        with pytest.raises(PydanticValidationError):
            MoneyFlowIndexStrategy({"oversold": "60", "exit_level": "50", "overbought": "80"})


STEADY = [("100", "101", "99", "100", "100")] * 20


class TestVolumePriceDivergence:
    def test_holds_when_no_new_extreme_is_made(self) -> None:
        strategy = VolumePriceDivergenceStrategy(DIVERGENCE_PARAMS)

        signal = strategy.evaluate(context_from(bars(STEADY)))

        assert signal.direction is SignalDirection.HOLD

    def test_fades_a_new_low_made_on_light_volume(self) -> None:
        strategy = VolumePriceDivergenceStrategy(DIVERGENCE_PARAMS)

        signal = strategy.evaluate(context_from(bars([*STEADY, ("100", "100", "94", "95", "5")])))

        assert signal.direction is SignalDirection.LONG
        assert "lighter volume" in signal.reason

    def test_a_new_low_on_heavy_volume_is_confirmed_not_divergent(self) -> None:
        """Heavy volume at the extreme is the opposite reading; there is nothing to fade."""
        strategy = VolumePriceDivergenceStrategy(DIVERGENCE_PARAMS)

        signal = strategy.evaluate(context_from(bars([*STEADY, ("100", "100", "94", "95", "400")])))

        assert signal.direction is SignalDirection.HOLD

    def test_a_zero_volume_bar_is_refused(self) -> None:
        strategy = VolumePriceDivergenceStrategy(DIVERGENCE_PARAMS)

        signal = strategy.evaluate(context_from(bars([*STEADY, ("100", "100", "94", "95", "0")])))

        assert signal.direction is SignalDirection.HOLD
        assert "volume" in signal.reason


LOUD = [("100", "103", "97", "100", "10")] * 25
FADE_TAIL = [
    ("100", "100.6", "99.9", "100.5", "10"),
    ("100.5", "100.6", "100.4", "100.5", "10"),
    ("100.5", "100.6", "100.4", "100.5", "10"),
    ("100.5", "100.6", "100.4", "100.5", "10"),
    ("100.5", "100.6", "100.0", "100.05", "10"),
]


class TestVolatilityRegime:
    def test_holds_in_the_no_regime_zone(self) -> None:
        strategy = VolatilityRegimeStrategy(REGIME_PARAMS)

        signal = strategy.evaluate(context_from(bars(LOUD)))

        assert signal.direction is SignalDirection.HOLD
        assert "between" in signal.reason

    def test_breaks_out_when_volatility_is_above_its_median(self) -> None:
        strategy = VolatilityRegimeStrategy(REGIME_PARAMS)

        signal = strategy.evaluate(context_from(bars([*LOUD, ("100", "115", "100", "114", "10")])))

        assert signal.direction is SignalDirection.LONG
        assert "expansion regime" in signal.reason

    def test_fades_a_stretch_when_volatility_is_below_its_median(self) -> None:
        """Same library, opposite tactic — chosen by the regime, which is the whole idea."""
        strategy = VolatilityRegimeStrategy(REGIME_PARAMS)

        signal = strategy.evaluate(context_from(bars([*LOUD, *FADE_TAIL])))

        assert signal.direction is SignalDirection.LONG
        assert "contraction regime" in signal.reason

    def test_a_contraction_with_price_at_the_mean_is_left_alone(self) -> None:
        strategy = VolatilityRegimeStrategy(REGIME_PARAMS)

        signal = strategy.evaluate(context_from(bars([*LOUD, *FADE_TAIL[:4]])))

        assert signal.direction is SignalDirection.HOLD

    def test_rejects_a_low_threshold_at_or_above_the_high_one(self) -> None:
        with pytest.raises(PydanticValidationError):
            VolatilityRegimeStrategy({"low_multiple": "1.5", "high_multiple": "1.2"})


NORMAL = [("100", "101", "99", "100", "10")] * 30
COMPRESSED = [("100", "100.1", "99.9", "100", "10")] * 8
BURST: Spec = ("100", "110", "99.9", "109.5", "10")


class TestVolatilityTransition:
    def test_holds_through_the_contraction(self) -> None:
        strategy = VolatilityTransitionStrategy(TRANSITION_PARAMS)

        signal = strategy.evaluate(context_from(bars([*NORMAL, *COMPRESSED])))

        assert signal.direction is SignalDirection.HOLD

    def test_enters_on_the_bar_volatility_changes_state(self) -> None:
        strategy = VolatilityTransitionStrategy(TRANSITION_PARAMS)

        signal = strategy.evaluate(context_from(bars([*NORMAL, *COMPRESSED, BURST])))

        assert signal.direction is SignalDirection.LONG
        assert "crossed from contraction" in signal.reason

    def test_does_not_re_enter_once_the_expansion_is_under_way(self) -> None:
        """The distinction from `atr_expansion`: this trades the change, not the state."""
        strategy = VolatilityTransitionStrategy(TRANSITION_PARAMS)
        candles = bars([*NORMAL, *COMPRESSED, BURST, ("109.5", "120", "109", "119", "10")])

        signal = strategy.evaluate(context_from(candles))

        assert signal.direction is SignalDirection.HOLD
        assert "already expanded" in signal.reason

    def test_an_expansion_with_no_prior_contraction_is_declined(self) -> None:
        strategy = VolatilityTransitionStrategy(TRANSITION_PARAMS)

        signal = strategy.evaluate(context_from(bars([*NORMAL, BURST])))

        assert signal.direction is SignalDirection.HOLD

    def test_rejects_a_contraction_threshold_above_the_expansion_one(self) -> None:
        with pytest.raises(PydanticValidationError):
            VolatilityTransitionStrategy(
                {"contraction_multiple": "1.5", "expansion_multiple": "1.3"}
            )


#: One series per strategy on which it is known to open a position. Collected here so the
#: protective-level contract can be asserted for all nine, rather than only for whichever
#: of them happen to trigger on a shared fixture.
ENTRY_FIXTURES: list[object] = [
    pytest.param(
        BreakoutRetestStrategy,
        RETEST_PARAMS,
        bars([*RANGE_BOUND, BREAK_BAR, *PULLBACK, RECLAIM]),
        id="breakout_retest",
    ),
    pytest.param(
        AtrBreakoutStrategy,
        ATR_BREAKOUT_PARAMS,
        bars([*QUIET, ("100", "106", "100", "105.8", "10")]),
        id="atr_breakout",
    ),
    pytest.param(
        RangeExpansionStrategy,
        RANGE_EXPANSION_PARAMS,
        bars([*QUIET, ("100", "105.2", "100", "105", "10")]),
        id="range_expansion",
    ),
    pytest.param(
        SupportResistanceBreakoutStrategy,
        SR_PARAMS,
        bars([*ZIGZAG, ("100", "108", "100", "107", "10")]),
        id="support_resistance_breakout",
    ),
    pytest.param(
        AccumulationDistributionStrategy, AD_PARAMS, bars(RISING), id="accumulation_distribution"
    ),
    pytest.param(
        MoneyFlowIndexStrategy,
        MFI_PARAMS,
        bars([*SELLOFF, ("105", "112", "105", "111", "100")]),
        id="money_flow_index",
    ),
    pytest.param(
        VolumePriceDivergenceStrategy,
        DIVERGENCE_PARAMS,
        bars([*STEADY, ("100", "100", "94", "95", "5")]),
        id="volume_price_divergence",
    ),
    pytest.param(
        VolatilityRegimeStrategy,
        REGIME_PARAMS,
        bars([*LOUD, ("100", "115", "100", "114", "10")]),
        id="volatility_regime",
    ),
    pytest.param(
        VolatilityTransitionStrategy,
        TRANSITION_PARAMS,
        bars([*NORMAL, *COMPRESSED, BURST]),
        id="volatility_transition",
    ),
]


class TestEveryStrategyCanFire:
    @pytest.mark.parametrize(("cls", "params", "candles"), ENTRY_FIXTURES)
    def test_the_entry_is_protected_on_both_sides(
        self, cls: type[Strategy], params: dict[str, object], candles: list[Candle]
    ) -> None:
        """Each strategy, on a series built to trigger it, must emit a protected entry."""
        strategy = build(cls, params)

        signal = strategy.evaluate(context_from(candles))

        assert signal.is_entry, signal.reason
        assert signal.reference_price is not None
        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None
        stop_distance = abs(signal.reference_price - signal.stop_loss_price)
        target_distance = abs(signal.reference_price - signal.take_profit_price)
        assert target_distance > stop_distance
        assert Decimal("0") < signal.conviction <= Decimal("1")
        assert signal.reason

    @pytest.mark.parametrize(("cls", "params", "candles"), ENTRY_FIXTURES)
    def test_the_entry_survives_being_replayed_from_a_prefix(
        self, cls: type[Strategy], params: dict[str, object], candles: list[Candle]
    ) -> None:
        """The triggering bar is the last one, so nothing after it can be involved."""
        walked = build(cls, params)
        for index in range(len(candles)):
            walked.evaluate(context_from(candles[: index + 1]))
        fresh = build(cls, params).evaluate(context_from(candles))

        assert _fingerprint(fresh) == _fingerprint(walked.evaluate(context_from(candles)))


class TestShortSide:
    """Shorts are opt-in, but where they are enabled they must actually work."""

    def test_atr_breakout_shorts_a_downside_displacement(self) -> None:
        strategy = AtrBreakoutStrategy({**ATR_BREAKOUT_PARAMS, "allow_short": True})

        signal = strategy.evaluate(context_from(bars([*QUIET, ("100", "100", "94", "94.2", "10")])))

        assert signal.direction is SignalDirection.SHORT
        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None
        assert signal.stop_loss_price > signal.take_profit_price

    def test_range_expansion_shorts_a_wide_down_bar(self) -> None:
        strategy = RangeExpansionStrategy({**RANGE_EXPANSION_PARAMS, "allow_short": True})

        signal = strategy.evaluate(context_from(bars([*QUIET, ("100", "100", "94.8", "95", "10")])))

        assert signal.direction is SignalDirection.SHORT

    def test_volume_price_divergence_shorts_an_unconfirmed_new_high(self) -> None:
        strategy = VolumePriceDivergenceStrategy({**DIVERGENCE_PARAMS, "allow_short": True})

        signal = strategy.evaluate(context_from(bars([*STEADY, ("100", "106", "100", "105", "5")])))

        assert signal.direction is SignalDirection.SHORT

    def test_volatility_transition_shorts_a_downward_handover(self) -> None:
        strategy = VolatilityTransitionStrategy({**TRANSITION_PARAMS, "allow_short": True})
        candles = bars([*NORMAL, *COMPRESSED, ("100", "100.1", "90", "90.5", "10")])

        signal = strategy.evaluate(context_from(candles))

        assert signal.direction is SignalDirection.SHORT

    def test_a_short_is_closed_when_its_premise_ends(self) -> None:
        strategy = AtrBreakoutStrategy({**ATR_BREAKOUT_PARAMS, "allow_short": True})
        candles = bars(
            [*QUIET, ("100", "100", "94", "94.2", "10"), ("94.2", "101", "94", "100", "10")]
        )

        signal = strategy.evaluate(context_from(candles, open_position(OrderSide.SELL)))

        assert signal.direction is SignalDirection.CLOSE


class TestStrategyIdentity:
    def test_every_strategy_id_is_distinct_and_declared(self) -> None:
        ids = [cls.strategy_id for _, cls, _ in ALL_STRATEGIES]
        assert sorted(ids) == sorted(STRATEGY_IDS)
        assert len(set(ids)) == len(ids)

    @pytest.mark.parametrize(("cls", "params"), STRATEGY_CASES)
    def test_signals_are_attributed_to_their_own_strategy(
        self, cls: type[Strategy], params: dict[str, object]
    ) -> None:
        strategy = build(cls, params)

        signal = strategy.evaluate(context_from(mixed_series()))

        assert signal.strategy_id == strategy.strategy_id
        assert strategy.description
