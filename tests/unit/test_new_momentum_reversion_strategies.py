"""The six momentum and mean-reversion strategies added alongside the trend family.

Three momentum variants that differ in *what they normalise by* — the change in the rate of
change, the standard deviation of returns, and the strategy's own rank history — and three
reversion variants that differ in *what makes a price extreme* — a rank, a volatility unit,
and a percentage gap from an average.

The suite is deliberately heavy on two things. First, warm-up and no-look-ahead, checked
uniformly across all six, because a strategy that quietly reads one bar into the future
produces a backtest that looks like an edge and is not one. Second, a paired
signals/does-not-signal test for each strategy, because a strategy that never trades also
passes every test about not trading wrongly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from quantflow.strategy.indicators import percentile_rank, return_volatility, simple_returns
from quantflow.strategy.library.ma_deviation_reversion import (
    MaDeviationReversionParams,
    MaDeviationReversionStrategy,
)
from quantflow.strategy.library.momentum_acceleration import (
    MomentumAccelerationParams,
    MomentumAccelerationStrategy,
)
from quantflow.strategy.library.normalized_momentum import (
    NormalizedMomentumParams,
    NormalizedMomentumStrategy,
)
from quantflow.strategy.library.percentile_reversion import (
    PercentileReversionParams,
    PercentileReversionStrategy,
)
from quantflow.strategy.library.relative_momentum import (
    RelativeMomentumParams,
    RelativeMomentumStrategy,
)
from quantflow.strategy.library.volatility_normalized_reversion import (
    VolatilityNormalizedReversionParams,
    VolatilityNormalizedReversionStrategy,
)
from tests.conftest import REFERENCE_TIME, make_candle

SYMBOL = Symbol.parse("BTC/USDT")

Factory = Callable[[], Strategy]

#: Every strategy in this batch, for the checks that must hold for all of them.
FACTORIES: list[tuple[str, Factory]] = [
    ("momentum_acceleration", MomentumAccelerationStrategy),
    ("normalized_momentum", NormalizedMomentumStrategy),
    ("relative_momentum", RelativeMomentumStrategy),
    ("percentile_reversion", PercentileReversionStrategy),
    ("volatility_normalized_reversion", VolatilityNormalizedReversionStrategy),
    ("ma_deviation_reversion", MaDeviationReversionStrategy),
]


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
def candles_from(closes: Sequence[Decimal], *, volume: str = "10") -> list[Candle]:
    """Build an hourly series from close prices, with a 1% wick either side."""
    return [
        make_candle(
            SYMBOL,
            open_time=REFERENCE_TIME + Timeframe.H1.delta * index,
            open_price=close,
            high=close * Decimal("1.01"),
            low=close * Decimal("0.99"),
            close=close,
            volume=volume,
        )
        for index, close in enumerate(closes)
    ]


def context_from(candles: Sequence[Candle], position: Position | None = None) -> StrategyContext:
    """A decision context whose history ends at the last supplied bar."""
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


def evaluate(strategy: Strategy, closes: Sequence[Decimal], position: Position | None = None):  # type: ignore[no-untyped-def]
    """Run a strategy over a close series."""
    return strategy.evaluate(context_from(candles_from(closes), position))


def position_at(side: OrderSide, price: str = "100") -> Position:
    """An open position in ``SYMBOL``."""
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


def summarise(
    signal: Signal,
) -> tuple[SignalDirection, Decimal, Decimal | None, Decimal | None, str]:
    """The decision-bearing parts of a signal.

    ``Signal`` carries a random ``signal_id``, so two identical decisions never compare
    equal as objects. Everything that actually reaches the risk engine is here.
    """
    return (
        signal.direction,
        signal.conviction,
        signal.stop_loss_price,
        signal.take_profit_price,
        signal.reason,
    )


def flat(count: int, price: str = "100") -> list[Decimal]:
    """A perfectly flat series — no returns, no dispersion, no rank."""
    return [Decimal(price)] * count


def wiggly(count: int, *, start: str = "100") -> list[Decimal]:
    """A deterministic series that both oscillates and drifts.

    Deterministic rather than random so a failure is reproducible: a flaky market fixture
    turns "this strategy has a look-ahead bug" into "this test fails sometimes".
    """
    base = Decimal(start)
    out: list[Decimal] = []
    for index in range(count):
        oscillation = Decimal((index * 37) % 17) - Decimal(8)
        drift = Decimal(index) / Decimal(20)
        out.append(base + oscillation + drift)
    return out


def accelerating(count: int, *, start: str = "100", step: str = "0.0002") -> list[Decimal]:
    """A rise whose *fractional* growth rate keeps increasing.

    Compounding rather than a fixed increment: a linear ramp has a per-bar increment that
    never changes, so in percentage terms — which is what every strategy here measures —
    it actually decelerates as the denominator grows.
    """
    out = [Decimal(start)]
    for index in range(1, count):
        out.append(out[-1] * (Decimal("1") + Decimal(step) * index))
    return out


def collapsing(count: int, *, start: str = "400", step: str = "0.0002") -> list[Decimal]:
    """A decline whose fractional rate keeps increasing."""
    out = [Decimal(start)]
    for index in range(1, count):
        out.append(out[-1] * (Decimal("1") - Decimal(step) * index))
    return out


def easing_collapse(count: int, *, start: str = "400") -> list[Decimal]:
    """A decline that keeps falling but by a smaller fraction each bar.

    Positive acceleration, firmly negative momentum: the case that separates "the second
    derivative turned up" from "this is worth buying".
    """
    out = [Decimal(start)]
    for index in range(1, count):
        rate = Decimal("0.02") - Decimal(index) * Decimal("0.0002")
        out.append(out[-1] * (Decimal("1") - rate))
    return out


def oscillating(count: int, amplitude: str, *, start: str = "100") -> list[Decimal]:
    """A flat market that ticks up and down by ``amplitude``, setting the volatility level."""
    swing = Decimal(amplitude)
    return [Decimal(start) + swing * (Decimal(index % 5) - Decimal(2)) for index in range(count)]


# --------------------------------------------------------------------------- #
# Appended indicators
# --------------------------------------------------------------------------- #
class TestAppendedIndicators:
    def test_returns_are_aligned_and_start_undefined(self) -> None:
        values = simple_returns([Decimal("100"), Decimal("110"), Decimal("99")])
        assert len(values) == 3
        assert values[0] is None
        assert values[1] == Decimal("0.1")

    def test_a_non_positive_predecessor_yields_none_rather_than_zero(self) -> None:
        """A fabricated zero return would drag a volatility estimate toward calm."""
        assert simple_returns([Decimal("0"), Decimal("100")])[1] is None

    def test_return_volatility_is_zero_on_a_flat_series(self) -> None:
        assert return_volatility(flat(30), 14)[-1] == Decimal("0")

    def test_return_volatility_rises_with_dispersion(self) -> None:
        calm = return_volatility(oscillating(60, "0.1"), 14)[-1]
        violent = return_volatility(oscillating(60, "5"), 14)[-1]
        assert calm is not None
        assert violent is not None
        assert violent > calm

    def test_return_volatility_ignores_wicks_that_atr_would_count(self) -> None:
        """The whole reason this exists next to ATR: it is a close-to-close measure."""
        assert return_volatility(flat(40), 14)[-1] == Decimal("0")

    def test_percentile_rank_places_a_value_in_its_window(self) -> None:
        window = [Decimal(value) for value in range(100)]
        assert percentile_rank(window, Decimal("-1")) == Decimal("0")
        assert percentile_rank(window, Decimal("200")) == Decimal("100")

    def test_percentile_rank_splits_ties(self) -> None:
        """All-identical history must read as the median, not as an extreme."""
        assert percentile_rank(flat(10), Decimal("100")) == Decimal("50")

    def test_percentile_rank_of_an_empty_window_is_none(self) -> None:
        assert percentile_rank([], Decimal("1")) is None


# --------------------------------------------------------------------------- #
# Contract checks that must hold for all six
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "factory"), FACTORIES, ids=[name for name, _ in FACTORIES])
class TestSharedContract:
    def test_declares_its_identity(self, name: str, factory: Factory) -> None:
        strategy = factory()
        assert strategy.strategy_id == name
        assert strategy.description

    def test_warmup_is_positive_and_finite(self, name: str, factory: Factory) -> None:
        assert 1 <= factory().warmup_bars <= 5000

    def test_holds_below_its_own_warmup(self, name: str, factory: Factory) -> None:
        """The engine withholds a strategy until it declares itself ready."""
        strategy = factory()
        signal = evaluate(strategy, wiggly(strategy.warmup_bars - 1))
        assert signal.direction is SignalDirection.HOLD
        assert "warming up" in signal.reason

    def test_decides_something_the_moment_it_is_warm(self, name: str, factory: Factory) -> None:
        """At exactly ``warmup_bars`` the declared window must genuinely be available.

        A strategy that declares 100 bars and then still reports "warming up" at bar 100 has
        understated its warm-up, and the engine's guarantee that indicators are settled by
        the first decision is silently false.
        """
        strategy = factory()
        signal = evaluate(strategy, wiggly(strategy.warmup_bars))
        assert "warming up" not in signal.reason

    def test_three_bars_is_a_hold_not_a_crash(self, name: str, factory: Factory) -> None:
        signal = evaluate(factory(), wiggly(3))
        assert signal.direction is SignalDirection.HOLD

    def test_a_flat_market_produces_no_trade(self, name: str, factory: Factory) -> None:
        """No returns, no dispersion, no rank: nothing here is an opportunity."""
        strategy = factory()
        signal = evaluate(strategy, flat(strategy.warmup_bars + 40))
        assert signal.direction is SignalDirection.HOLD

    def test_zero_volume_bars_are_tolerated(self, name: str, factory: Factory) -> None:
        """None of these read volume, so a dead tape must not take them down."""
        strategy = factory()
        candles = candles_from(wiggly(strategy.warmup_bars + 40), volume="0")
        signal = strategy.evaluate(context_from(candles))
        assert signal.strategy_id == name

    def test_a_target_sits_further_away_than_the_stop(self, name: str, factory: Factory) -> None:
        """Risk below one, reward above it — checked on the parameters themselves."""
        params = factory().params
        assert params.atr_target_multiple > params.atr_stop_multiple  # type: ignore[attr-defined]

    def test_no_lookahead_removing_later_bars_changes_nothing(
        self, name: str, factory: Factory
    ) -> None:
        """Every decision on the shared prefix must be identical with and without the future.

        Two failure modes are covered. A strategy that indexes past ``context.index`` would
        differ between the long and short series; a strategy that caches across calls would
        differ between a fresh instance and one already replayed over the longer series.
        """
        strategy = factory()
        start = strategy.warmup_bars
        long_series = wiggly(start + 90)
        short_series = long_series[: start + 60]

        without_future = walk(factory(), short_series, start)
        with_future = walk(factory(), long_series, start)[: len(without_future)]
        assert with_future == without_future

        replayed = factory()
        walk(replayed, long_series, start)
        assert walk(replayed, short_series, start) == without_future


def walk(
    strategy: Strategy, closes: Sequence[Decimal], start: int
) -> list[tuple[SignalDirection, Decimal, Decimal | None, Decimal | None, str]]:
    """Decisions at every prefix from ``start`` bars to the whole series."""
    candles = candles_from(closes)
    return [
        summarise(strategy.evaluate(context_from(candles[:size])))
        for size in range(start, len(candles) + 1)
    ]


# --------------------------------------------------------------------------- #
# Momentum acceleration
# --------------------------------------------------------------------------- #
class TestMomentumAcceleration:
    def test_rejects_an_exit_threshold_at_or_above_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            MomentumAccelerationParams(
                entry_acceleration=Decimal("0.01"), exit_acceleration=Decimal("0.02")
            )

    def test_buys_a_rise_that_is_speeding_up(self) -> None:
        signal = evaluate(MomentumAccelerationStrategy(), accelerating(80))
        assert signal.direction is SignalDirection.LONG
        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None

    def test_ignores_a_rise_at_constant_speed(self) -> None:
        """The distinction from every other momentum strategy: the *change* in the rate.

        A perfectly linear advance has strong momentum and zero acceleration, so this one
        stands aside where `momentum_roc` would be fully long.
        """
        linear = [Decimal("100") + Decimal(index) for index in range(80)]
        signal = evaluate(MomentumAccelerationStrategy(), linear)
        assert signal.direction is SignalDirection.HOLD

    def test_a_decelerating_collapse_is_not_a_buy(self) -> None:
        """Falling more slowly has positive acceleration and negative momentum.

        Without the ``min_momentum`` gate this is exactly the shape that would be bought.
        """
        signal = evaluate(
            MomentumAccelerationStrategy({"min_momentum": Decimal("0.01")}), easing_collapse(80)
        )
        assert signal.direction is not SignalDirection.LONG

    def test_a_sharper_acceleration_carries_more_conviction(self) -> None:
        gentle = evaluate(MomentumAccelerationStrategy(), accelerating(80, step="0.00006"))
        steep = evaluate(MomentumAccelerationStrategy(), accelerating(80, step="0.0002"))
        assert gentle.direction is SignalDirection.LONG
        assert steep.direction is SignalDirection.LONG
        assert steep.conviction > gentle.conviction

    def test_closes_a_long_once_the_advance_stops_speeding_up(self) -> None:
        linear = [Decimal("100") + Decimal(index) for index in range(80)]
        signal = evaluate(MomentumAccelerationStrategy(), linear, position_at(OrderSide.BUY))
        assert signal.direction is SignalDirection.CLOSE

    def test_shorts_are_refused_unless_enabled(self) -> None:
        falling = collapsing(80)
        assert evaluate(MomentumAccelerationStrategy(), falling).direction is (SignalDirection.HOLD)
        enabled = evaluate(MomentumAccelerationStrategy({"allow_short": True}), falling)
        assert enabled.direction is SignalDirection.SHORT
        assert enabled.stop_loss_price is not None
        assert enabled.stop_loss_price > enabled.reference_price  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# Normalized momentum
# --------------------------------------------------------------------------- #
class TestNormalizedMomentum:
    def test_rejects_an_exit_score_at_or_above_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            NormalizedMomentumParams(entry_score=Decimal("1"), exit_score=Decimal("1"))

    def test_buys_a_move_that_is_large_for_this_series(self) -> None:
        closes = oscillating(60, "0.05")
        for _ in range(24):
            closes.append(closes[-1] * Decimal("1.005"))
        signal = evaluate(NormalizedMomentumStrategy(), closes)
        assert signal.direction is SignalDirection.LONG
        assert signal.conviction > Decimal("0.5")

    def test_the_same_move_is_ignored_when_the_series_is_normally_wild(self) -> None:
        """The point of the normalisation, isolated: identical drift, different backdrop."""
        wild = oscillating(60, "8")
        for _ in range(24):
            wild.append(wild[-1] * Decimal("1.005"))
        assert evaluate(NormalizedMomentumStrategy(), wild).direction is SignalDirection.HOLD

    def test_a_flat_series_never_divides_by_zero_dispersion(self) -> None:
        signal = evaluate(NormalizedMomentumStrategy(), flat(120))
        assert signal.direction is SignalDirection.HOLD
        assert "warming up" in signal.reason

    def test_closes_a_long_when_the_score_decays(self) -> None:
        closes = oscillating(60, "0.05")
        for _ in range(24):
            closes.append(closes[-1] * Decimal("1.005"))
        closes.extend(closes[-1] for _ in range(24))
        signal = evaluate(NormalizedMomentumStrategy(), closes, position_at(OrderSide.BUY))
        assert signal.direction is SignalDirection.CLOSE

    def test_shorts_are_refused_unless_enabled(self) -> None:
        closes = oscillating(60, "0.05")
        for _ in range(24):
            closes.append(closes[-1] * Decimal("0.995"))
        assert evaluate(NormalizedMomentumStrategy(), closes).direction is SignalDirection.HOLD
        assert (
            evaluate(NormalizedMomentumStrategy({"allow_short": True}), closes).direction
            is SignalDirection.SHORT
        )


# --------------------------------------------------------------------------- #
# Relative momentum
# --------------------------------------------------------------------------- #
class TestRelativeMomentum:
    def test_rejects_an_exit_percentile_at_or_above_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            RelativeMomentumParams(entry_percentile=Decimal("90"), exit_percentile=Decimal("95"))

    def test_buys_a_move_that_tops_its_own_history(self) -> None:
        closes = oscillating(130, "0.2")
        for _ in range(12):
            closes.append(closes[-1] * Decimal("1.01"))
        signal = evaluate(RelativeMomentumStrategy(), closes)
        assert signal.direction is SignalDirection.LONG
        assert signal.conviction > Decimal("0.5")

    def test_holds_when_the_move_is_ordinary_for_this_market(self) -> None:
        signal = evaluate(RelativeMomentumStrategy(), oscillating(150, "0.2"))
        assert signal.direction is SignalDirection.HOLD

    def test_a_top_rank_in_a_falling_market_is_not_a_buy(self) -> None:
        """The least-bad decline still ranks 100th percentile, and is still a decline."""
        closes = [Decimal("400") - Decimal(index) for index in range(140)]
        closes.extend(closes[-1] - Decimal(index) / Decimal("100") for index in range(1, 13))
        signal = evaluate(RelativeMomentumStrategy(), closes)
        assert signal.direction is not SignalDirection.LONG

    def test_the_threshold_is_scale_free(self) -> None:
        """A calm market and a wild one must both trade on their own top decile."""

        def series(amplitude: str) -> list[Decimal]:
            closes = oscillating(130, amplitude)
            for _ in range(12):
                closes.append(closes[-1] * Decimal("1.01"))
            return closes

        assert evaluate(RelativeMomentumStrategy(), series("0.2")).direction is (
            SignalDirection.LONG
        )
        assert evaluate(RelativeMomentumStrategy(), series("2")).direction is (SignalDirection.LONG)

    def test_closes_a_long_when_the_rank_falls_back_to_the_median(self) -> None:
        closes = oscillating(130, "0.2")
        for _ in range(12):
            closes.append(closes[-1] * Decimal("1.01"))
        closes.extend(closes[-1] - Decimal(index) for index in range(1, 15))
        signal = evaluate(RelativeMomentumStrategy(), closes, position_at(OrderSide.BUY))
        assert signal.direction is SignalDirection.CLOSE


# --------------------------------------------------------------------------- #
# Percentile reversion
# --------------------------------------------------------------------------- #
class TestPercentileReversion:
    def test_rejects_an_exit_percentile_at_or_below_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            PercentileReversionParams(entry_percentile=Decimal("20"), exit_percentile=Decimal("10"))

    def test_buys_the_bottom_of_the_trailing_distribution(self) -> None:
        closes = oscillating(120, "2")
        closes.append(Decimal("88"))
        signal = evaluate(PercentileReversionStrategy(), closes)
        assert signal.direction is SignalDirection.LONG
        assert signal.conviction > Decimal("0.9")
        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None

    def test_holds_in_the_middle_of_the_distribution(self) -> None:
        closes = oscillating(120, "2")
        closes.append(Decimal("100"))
        signal = evaluate(PercentileReversionStrategy(), closes)
        assert signal.direction is SignalDirection.HOLD

    def test_a_bottom_rank_in_a_motionless_window_is_refused(self) -> None:
        """A rank knows where, not how far; ``min_range_pct`` supplies the how far."""
        closes = oscillating(120, "0.001")
        closes.append(Decimal("99.99"))
        signal = evaluate(PercentileReversionStrategy(), closes)
        assert signal.direction is SignalDirection.HOLD

    def test_one_crash_bar_does_not_suppress_the_next_signal(self) -> None:
        """The behavioural difference from a z-score, which the crash bar would deafen.

        A single outlier inflates a standard deviation for the whole trailing window, so
        `zscore_reversion` goes quiet exactly afterwards. A rank moves by one position.
        """
        closes = oscillating(120, "2")
        closes.append(Decimal("40"))
        closes.append(Decimal("88"))
        signal = evaluate(PercentileReversionStrategy(), closes)
        assert signal.direction is SignalDirection.LONG

    def test_closes_a_long_once_the_rank_normalises(self) -> None:
        closes = oscillating(120, "2")
        closes.append(Decimal("104"))
        signal = evaluate(PercentileReversionStrategy(), closes, position_at(OrderSide.BUY))
        assert signal.direction is SignalDirection.CLOSE

    def test_shorts_are_refused_unless_enabled(self) -> None:
        closes = oscillating(120, "2")
        closes.append(Decimal("115"))
        assert evaluate(PercentileReversionStrategy(), closes).direction is SignalDirection.HOLD
        enabled = evaluate(PercentileReversionStrategy({"allow_short": True}), closes)
        assert enabled.direction is SignalDirection.SHORT


# --------------------------------------------------------------------------- #
# Volatility-normalized reversion
# --------------------------------------------------------------------------- #
class TestVolatilityNormalizedReversion:
    def test_rejects_an_exit_deviation_at_or_above_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            VolatilityNormalizedReversionParams(
                entry_deviation=Decimal("2"), exit_deviation=Decimal("3")
            )

    def test_rejects_a_yardstick_slower_than_the_mean(self) -> None:
        """A volatility window as long as the mean would stop being a *current* measure."""
        with pytest.raises(PydanticValidationError):
            VolatilityNormalizedReversionParams(mean_period=20, volatility_period=20)

    def test_the_same_drop_trades_in_calm_and_is_ignored_in_chaos(self) -> None:
        """The entire thesis of the strategy, as a single paired assertion."""
        calm = oscillating(80, "0.05")
        calm.append(calm[-1] * Decimal("0.97"))
        violent = oscillating(80, "3")
        violent.append(violent[-1] * Decimal("0.97"))

        assert evaluate(VolatilityNormalizedReversionStrategy(), calm).direction is (
            SignalDirection.LONG
        )
        assert evaluate(VolatilityNormalizedReversionStrategy(), violent).direction is (
            SignalDirection.HOLD
        )

    def test_an_entry_carries_protection_and_a_wider_target_than_stop(self) -> None:
        calm = oscillating(80, "0.05")
        calm.append(calm[-1] * Decimal("0.97"))
        signal = evaluate(VolatilityNormalizedReversionStrategy(), calm)
        assert signal.reference_price is not None
        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None
        risk = signal.reference_price - signal.stop_loss_price
        reward = signal.take_profit_price - signal.reference_price
        assert reward > risk

    def test_a_wider_dislocation_carries_more_conviction(self) -> None:
        def drop(factor: str) -> Signal:
            closes = oscillating(80, "0.05")
            closes.append(closes[-1] * Decimal(factor))
            return evaluate(VolatilityNormalizedReversionStrategy(), closes)

        assert drop("0.90").conviction > drop("0.97").conviction

    def test_closes_a_long_as_the_gap_closes(self) -> None:
        closes = oscillating(80, "0.05")
        signal = evaluate(
            VolatilityNormalizedReversionStrategy(), closes, position_at(OrderSide.BUY)
        )
        assert signal.direction is SignalDirection.CLOSE

    def test_shorts_are_refused_unless_enabled(self) -> None:
        closes = oscillating(80, "0.05")
        closes.append(closes[-1] * Decimal("1.03"))
        assert evaluate(VolatilityNormalizedReversionStrategy(), closes).direction is (
            SignalDirection.HOLD
        )
        enabled = evaluate(VolatilityNormalizedReversionStrategy({"allow_short": True}), closes)
        assert enabled.direction is SignalDirection.SHORT


# --------------------------------------------------------------------------- #
# Moving-average deviation reversion
# --------------------------------------------------------------------------- #
class TestMaDeviationReversion:
    def test_rejects_a_trend_filter_no_slower_than_the_signal(self) -> None:
        with pytest.raises(PydanticValidationError):
            MaDeviationReversionParams(ma_period=50, trend_period=50)

    def test_rejects_an_exit_deviation_at_or_above_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            MaDeviationReversionParams(
                entry_deviation=Decimal("0.02"), exit_deviation=Decimal("0.03")
            )

    def test_buys_a_dip_inside_a_range(self) -> None:
        closes = oscillating(160, "0.5")
        closes.append(Decimal("94"))
        signal = evaluate(MaDeviationReversionStrategy(), closes)
        assert signal.direction is SignalDirection.LONG
        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None

    def test_refuses_the_same_dip_inside_a_collapse(self) -> None:
        """The failure mode the trend filter exists for: buying every new low."""
        closes = [Decimal("400") * (Decimal("0.99") ** index) for index in range(160)]
        closes.append(closes[-1] * Decimal("0.90"))
        signal = evaluate(MaDeviationReversionStrategy(), closes)
        assert signal.direction is SignalDirection.HOLD
        assert "fight" in signal.reason

    def test_holds_when_price_sits_on_its_average(self) -> None:
        signal = evaluate(MaDeviationReversionStrategy(), oscillating(160, "0.5"))
        assert signal.direction is SignalDirection.HOLD

    def test_a_wider_stretch_carries_more_conviction(self) -> None:
        def dip(price: str) -> Signal:
            closes = oscillating(160, "0.5")
            closes.append(Decimal(price))
            return evaluate(MaDeviationReversionStrategy(), closes)

        assert dip("90").conviction > dip("96").conviction

    def test_closes_a_long_once_price_returns_to_its_average(self) -> None:
        signal = evaluate(
            MaDeviationReversionStrategy(), oscillating(160, "0.5"), position_at(OrderSide.BUY)
        )
        assert signal.direction is SignalDirection.CLOSE

    def test_shorts_are_refused_unless_enabled(self) -> None:
        closes = oscillating(160, "0.5")
        closes.append(Decimal("106"))
        assert evaluate(MaDeviationReversionStrategy(), closes).direction is SignalDirection.HOLD
        enabled = evaluate(MaDeviationReversionStrategy({"allow_short": True}), closes)
        assert enabled.direction is SignalDirection.SHORT
