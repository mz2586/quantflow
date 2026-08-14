"""Fade a stretched deviation from a moving average — unless the trend explains it.

Price rarely stays far from a moving average for long, and the percentage gap between the
two is about the simplest possible measure of "too far". This strategy buys when price is
some percent below its average and lets go as the gap closes.

On its own that idea has one dominant failure mode, and it is not a subtle one: in a real
downtrend price is *always* below its moving average, and getting further below it every
day. A naive deviation fade buys each new low, all the way down, and every one of those
trades is a loser. The trend filter is the whole reason this strategy is worth running —
it measures the slope of a slower average and simply refuses to buy dips while that slope
is meaningfully negative. The strategy is designed to trade dislocations inside a range or
a mild trend, and to stand aside during the moves that would otherwise destroy it.

Deliberately a *percentage* deviation rather than a volatility-scaled one, which is what
`volatility_normalized_reversion` does: this asks the plainer question of how far price is
from its average in the units a trader actually reads off a chart, and pays for that
simplicity by needing the explicit trend gate that a volatility-scaled version gets partly
for free.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, sma
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class MaDeviationReversionParams(StrategyParams):
    """Parameters for :class:`MaDeviationReversionStrategy`."""

    #: The average price is faded back toward.
    ma_period: int = Field(default=20, ge=2, le=1000)
    #: The slower average whose slope decides whether a dip is worth buying.
    trend_period: int = Field(default=100, ge=5, le=2000)
    #: Bars the trend slope is measured over.
    trend_lookback: int = Field(default=20, ge=1, le=500)
    #: Deviation from the fast average, as a fraction, required to enter.
    entry_deviation: Decimal = Field(default=Decimal("0.03"), gt=0, le=1)
    #: Deviation at which an open position is closed. Above zero so the trade does not wait
    #: for a full round trip to the average that may never arrive.
    exit_deviation: Decimal = Field(default=Decimal("0.005"), ge=0, le=1)
    #: Slope of the trend average over its lookback, as a fraction of price, beyond which
    #: the trend is judged strong enough that fading it is fighting rather than trading.
    max_counter_trend_slope: Decimal = Field(default=Decimal("0.02"), ge=0, le=1)
    #: Deviation beyond the entry threshold at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("0.03"), gt=0, le=1)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if self.exit_deviation >= self.entry_deviation:
            raise ValueError(
                f"exit_deviation ({self.exit_deviation}) must be below entry_deviation "
                f"({self.entry_deviation}), or a position would close on the bar it opened"
            )
        if self.trend_period <= self.ma_period:
            raise ValueError(
                f"trend_period ({self.trend_period}) must exceed ma_period "
                f"({self.ma_period}); a filter that moves as fast as the signal filters nothing"
            )
        return self


@register_strategy
class MaDeviationReversionStrategy(Strategy):
    """Fade a percentage stretch from a moving average, but never against a strong trend."""

    strategy_id = "ma_deviation_reversion"
    description = "Percentage deviation from a moving average, faded behind a trend filter"
    params_model = MaDeviationReversionParams

    params: MaDeviationReversionParams

    @property
    def warmup_bars(self) -> int:
        """The trend average plus the bars its slope is measured over."""
        return max(
            self.params.trend_period + self.params.trend_lookback,
            self.params.ma_period,
            self.params.atr_period + 1,
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Fade the stretch when the trend does not forbid it."""
        index = context.index
        deviation = self._deviation(context, index)
        if deviation is None:
            return context.hold("moving average warming up", self.strategy_id)

        if context.is_long:
            if deviation >= -self.params.exit_deviation:
                return exit_signal(
                    context, self.strategy_id, f"deviation closed to {deviation:.4f}"
                )
            return context.hold(f"deviation {deviation:.4f} still stretched", self.strategy_id)

        if context.is_short:
            if deviation <= self.params.exit_deviation:
                return exit_signal(
                    context, self.strategy_id, f"deviation closed to {deviation:.4f}"
                )
            return context.hold(f"deviation {deviation:.4f} still stretched", self.strategy_id)

        if deviation <= -self.params.entry_deviation:
            direction = SignalDirection.LONG
        elif deviation >= self.params.entry_deviation:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold(
                f"deviation {deviation:.4f} below {self.params.entry_deviation}", self.strategy_id
            )

        slope = self._trend_slope(context, index)
        if slope is None:
            return context.hold("trend filter warming up", self.strategy_id)
        if self._fights_the_trend(direction, slope):
            return context.hold(
                f"trend slope {slope:.4f} makes this a fight, not a fade", self.strategy_id
            )

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"price {deviation:.4f} from its {self.params.ma_period}-bar average",
        )
        return replace_conviction(signal, self._conviction(abs(deviation)))

    def _deviation(self, context: StrategyContext, index: int) -> Decimal | None:
        """Fractional distance between price and the fast moving average."""
        mean = sma(context.closes, self.params.ma_period)[index]
        if mean is None or mean <= ZERO:
            return None
        return (context.price - mean) / mean

    def _trend_slope(self, context: StrategyContext, index: int) -> Decimal | None:
        """Change in the slow average over its lookback, as a fraction of its own level.

        Expressed as a fraction so the threshold is a percentage move and therefore means
        the same thing at any price level.
        """
        trend = sma(context.closes, self.params.trend_period)
        earlier_index = index - self.params.trend_lookback
        if earlier_index < 0:
            return None
        current, earlier = trend[index], trend[earlier_index]
        if current is None or earlier is None or earlier <= ZERO:
            return None
        return (current - earlier) / earlier

    def _fights_the_trend(self, direction: SignalDirection, slope: Decimal) -> bool:
        """Whether taking ``direction`` would mean trading against a strong trend."""
        if direction is SignalDirection.LONG:
            return slope < -self.params.max_counter_trend_slope
        return slope > self.params.max_counter_trend_slope

    def _conviction(self, deviation: Decimal) -> Decimal:
        """A wider stretch reads as a stronger case."""
        excess = deviation - self.params.entry_deviation
        if excess <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5"), ONE)


__all__ = ["MaDeviationReversionParams", "MaDeviationReversionStrategy"]
