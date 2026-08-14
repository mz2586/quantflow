"""Reversion measured in units of *current* volatility.

Price is 3% below its 50-bar mean. Is that a stretched decline worth fading, or a Tuesday?
The answer depends entirely on how much this market has been moving *this week*, and that
is the whole of this strategy: divide the deviation from the mean by the volatility being
realised right now, so one threshold behaves the same in a calm market and a violent one.
In calm conditions a 1% dislocation is already extreme and it trades; in a violent one the
same 1% is inside the noise and it stands aside.

**Why not just use `zscore_reversion`.** That strategy divides by the standard deviation of
*prices inside the same averaging window*. Two consequences follow. First, that
denominator is contaminated by trend: a market that has risen steadily has an enormous
price standard deviation with no volatility in any useful sense, so real dislocations get
divided into invisibility. Second, it is a *backward* measure over the whole window — a
quiet stretch six weeks ago still inflates it today. Here the denominator is the standard
deviation of recent **returns** over a deliberately short, separate window, so it tracks
the regime price is in *now* rather than the regime the averaging window happens to span.
The mean and the yardstick are decoupled on purpose.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, return_volatility, sma
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class VolatilityNormalizedReversionParams(StrategyParams):
    """Parameters for :class:`VolatilityNormalizedReversionStrategy`."""

    #: Window of the mean price reverts toward.
    mean_period: int = Field(default=50, ge=5, le=1000)
    #: Window the *current* return volatility is estimated over. Short by design: it is
    #: meant to describe the regime price is in now, not the one the mean spans.
    volatility_period: int = Field(default=14, ge=2, le=200)
    #: Deviation required to enter, counted in typical recent bar moves.
    entry_deviation: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    #: Deviation at which an open position is closed.
    exit_deviation: Decimal = Field(default=Decimal("0.75"), ge=0, le=20)
    #: Deviation beyond the entry threshold at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_deviations(self) -> Self:
        if self.exit_deviation >= self.entry_deviation:
            raise ValueError(
                f"exit_deviation ({self.exit_deviation}) must be below entry_deviation "
                f"({self.entry_deviation}), or a position would close on the bar it opened"
            )
        if self.volatility_period >= self.mean_period:
            raise ValueError(
                f"volatility_period ({self.volatility_period}) must be shorter than "
                f"mean_period ({self.mean_period}); the point of this strategy is that the "
                "yardstick tracks a faster regime than the mean it measures against"
            )
        return self


@register_strategy
class VolatilityNormalizedReversionStrategy(Strategy):
    """Fade a deviation from the mean that is large relative to today's volatility."""

    strategy_id = "volatility_normalized_reversion"
    description = "Distance from a mean priced in units of current realised volatility"
    params_model = VolatilityNormalizedReversionParams

    params: VolatilityNormalizedReversionParams

    @property
    def warmup_bars(self) -> int:
        """The mean window, plus a bar for the first return, plus the ATR window."""
        return max(self.params.mean_period + 1, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Enter on a volatility-scaled dislocation, exit as it closes."""
        index = context.index
        deviation = self._deviation(context, index)
        if deviation is None:
            return context.hold("mean or volatility warming up", self.strategy_id)

        if context.is_long:
            if deviation >= -self.params.exit_deviation:
                return exit_signal(
                    context, self.strategy_id, f"deviation closed to {deviation:.2f}"
                )
            return context.hold(f"deviation {deviation:.2f} still stretched", self.strategy_id)

        if context.is_short:
            if deviation <= self.params.exit_deviation:
                return exit_signal(
                    context, self.strategy_id, f"deviation closed to {deviation:.2f}"
                )
            return context.hold(f"deviation {deviation:.2f} still stretched", self.strategy_id)

        if deviation <= -self.params.entry_deviation:
            direction = SignalDirection.LONG
        elif deviation >= self.params.entry_deviation:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold(
                f"deviation {deviation:.2f} below {self.params.entry_deviation}", self.strategy_id
            )

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"{abs(deviation):.2f} volatility units from the {self.params.mean_period}-bar mean",
        )
        return replace_conviction(signal, self._conviction(abs(deviation)))

    def _deviation(self, context: StrategyContext, index: int) -> Decimal | None:
        """Signed distance from the mean, counted in typical recent bar moves.

        Deliberately *not* scaled by ``sqrt(mean_period)``. The deviation is a snapshot of
        where price stands right now, not a random walk that has been accumulating variance
        for fifty bars, so the honest unit is "how many ordinary bar moves would it take to
        get back" — a number a trader can act on directly. Scaling by the square root of the
        averaging window would make the threshold depend on ``mean_period``, so lengthening
        the mean would quietly loosen the trigger.
        """
        closes = context.closes
        mean = sma(closes, self.params.mean_period)[index]
        if mean is None or mean <= ZERO:
            return None

        volatility = return_volatility(closes, self.params.volatility_period)[index]
        if volatility is None or volatility <= ZERO:
            # A series with no return dispersion cannot say whether a gap is unusual.
            return None
        return ((context.price - mean) / mean) / volatility

    def _conviction(self, deviation: Decimal) -> Decimal:
        """A wider volatility-scaled dislocation reads as a stronger case."""
        excess = deviation - self.params.entry_deviation
        if excess <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5"), ONE)


__all__ = [
    "VolatilityNormalizedReversionParams",
    "VolatilityNormalizedReversionStrategy",
]
