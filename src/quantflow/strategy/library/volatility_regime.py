"""Volatility regime — classify the market as loud or quiet, then behave accordingly.

Breakouts and fades are not competing opinions; they are the same opinion applied to
different conditions. A break of a range in a market that has gone loud is usually a real
repricing and continues; the identical break in a quiet market is usually the edge of the
range and reverts. A strategy that only ever does one of the two is right about half the
time by construction, and the half it is wrong about is not random — it is systematic and
identifiable in advance.

The classifier is ATR normalised by price, compared against **its own trailing median**
rather than its mean. A median is unmoved by the handful of enormous bars that a crisis
produces, so the threshold does not quietly ratchet upward after every shock and stop
classifying anything as loud for the next hundred bars. The comparison being to the market's
own history is what makes "loud" mean the same thing on a major and on an illiquid pair.

An inherited position is managed under the regime **now in force**, not the one that opened
it. That is the strategy's whole thesis applied consistently: if the conditions that
justified holding a breakout have ended, continuing to hold it on the original reasoning is
exactly the error the classifier exists to prevent. ATR stops bound the cost of the handover.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, normalized_atr, sma, stdev
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class VolatilityRegimeParams(StrategyParams):
    """Parameters for :class:`VolatilityRegimeStrategy`."""

    atr_period: int = Field(default=14, ge=2, le=100)
    #: How much history the trailing median of volatility is taken over.
    median_lookback: int = Field(default=100, ge=10, le=2000)
    #: Volatility at or above this multiple of its median counts as an expansion regime.
    high_multiple: Decimal = Field(default=Decimal("1.2"), gt=0, le=20)
    #: At or below this multiple, a contraction regime.
    low_multiple: Decimal = Field(default=Decimal("0.8"), gt=0, le=20)
    #: Channel broken for entries in the expansion regime.
    breakout_period: int = Field(default=20, ge=2, le=500)
    #: Window for the mean and dispersion used by the contraction-regime fade.
    band_period: int = Field(default=20, ge=2, le=500)
    #: How many standard deviations from the mean the fade requires.
    fade_deviations: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_regimes(self) -> Self:
        if self.low_multiple >= self.high_multiple:
            raise ValueError(
                f"low_multiple ({self.low_multiple}) must be below high_multiple "
                f"({self.high_multiple}); the gap between them is the no-trade zone"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class VolatilityRegimeStrategy(Strategy):
    """Breaks out when volatility is above its median and fades when it is below."""

    strategy_id = "volatility_regime"
    description = "Classifies volatility against its trailing median, then breaks out or fades"
    params_model = VolatilityRegimeParams

    params: VolatilityRegimeParams

    @property
    def warmup_bars(self) -> int:
        """ATR must exist for the whole median window before the classifier means anything."""
        return max(
            self.params.atr_period + self.params.median_lookback + 1,
            self.params.breakout_period + 2,
            self.params.band_period + 1,
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Classify volatility, then break out or fade as that regime warrants."""
        index = context.index
        ratio = self._volatility_ratio(context)
        if ratio is None:
            return context.hold("volatility classifier warming up", self.strategy_id)

        expanding = ratio >= self.params.high_multiple
        contracting = ratio <= self.params.low_multiple

        average = sma(context.closes, self.params.band_period)[index]
        if average is None:
            return context.hold("band average warming up", self.strategy_id)

        if context.has_position:
            return self._manage(context, average, ratio, expanding=expanding)

        if expanding:
            long = self._breakout_direction(context)
            if long is None:
                return context.hold(
                    f"expansion regime ({ratio:.2f}x median), no break", self.strategy_id
                )
            edge = ratio - self.params.high_multiple
            span = self.params.high_multiple
            reason = f"broke the channel in an expansion regime ({ratio:.2f}x median)"
        elif contracting:
            dispersion = stdev(context.closes, self.params.band_period)[index]
            if dispersion is None or dispersion <= ZERO:
                return context.hold("no dispersion to fade", self.strategy_id)
            deviations = (context.price - average) / dispersion
            if deviations <= -self.params.fade_deviations:
                long = True
            elif deviations >= self.params.fade_deviations:
                long = False
            else:
                return context.hold(
                    f"contraction regime ({ratio:.2f}x median), price near the mean",
                    self.strategy_id,
                )
            edge = abs(deviations) - self.params.fade_deviations
            span = self.params.fade_deviations
            reason = f"faded a {deviations:.2f}-sigma stretch in a contraction regime"
        else:
            return context.hold(
                f"volatility {ratio:.2f}x median, between "
                f"{self.params.low_multiple} and {self.params.high_multiple}",
                self.strategy_id,
            )

        if not long and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if long else SignalDirection.SHORT,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            reason,
        )
        return replace_conviction(signal, _conviction(edge, span))

    def _manage(
        self, context: StrategyContext, average: Decimal, ratio: Decimal, *, expanding: bool
    ) -> Signal:
        """Exit on the terms of the regime currently in force."""
        if expanding:
            # Trend-following terms: the mean is the line the move must stay the right side of.
            if context.is_long and context.price < average:
                return exit_signal(context, self.strategy_id, "lost the mean in an expansion")
            if context.is_short and context.price > average:
                return exit_signal(context, self.strategy_id, "lost the mean in an expansion")
            return context.hold(f"holding, volatility {ratio:.2f}x median", self.strategy_id)

        # Mean-reversion terms: reaching the mean is the trade completing.
        if context.is_long and context.price >= average:
            return exit_signal(context, self.strategy_id, "price reverted to the mean")
        if context.is_short and context.price <= average:
            return exit_signal(context, self.strategy_id, "price reverted to the mean")
        return context.hold("holding, waiting for the mean", self.strategy_id)

    def _volatility_ratio(self, context: StrategyContext) -> Decimal | None:
        """Current normalised ATR as a multiple of its trailing median."""
        series = normalized_atr(context.candles, self.params.atr_period)
        current = series[context.index]
        if current is None or current <= ZERO:
            return None
        defined = [value for value in series[: context.index + 1] if value is not None]
        if len(defined) < self.params.median_lookback:
            return None
        middle = trailing_median(defined[-self.params.median_lookback :])
        if middle is None or middle <= ZERO:
            return None
        return current / middle

    def _breakout_direction(self, context: StrategyContext) -> bool | None:
        """``True`` for an upside break, ``False`` for a downside one, ``None`` for neither."""
        index = context.index
        start = index - self.params.breakout_period
        if start < 0:
            return None
        # The channel excludes the current bar: a window containing the bar being tested is
        # breached by definition whenever that bar makes a new extreme.
        window = context.candles[start:index]
        if not window:
            return None
        if context.price > max(candle.high for candle in window):
            return True
        if context.price < min(candle.low for candle in window):
            return False
        return None


def trailing_median(values: Sequence[Decimal]) -> Decimal | None:
    """Median of ``values``, or ``None`` when empty.

    A median rather than a mean because volatility distributions have a long right tail: a
    handful of crisis bars drag a mean upward for as long as they stay in the window, which
    would make the classifier quietly stop recognising expansions after every large move.
    """
    if not values:
        return None
    ordered = sorted(values)
    size = len(ordered)
    half = size // 2
    if size % 2:
        return ordered[half]
    return (ordered[half - 1] + ordered[half]) / Decimal(2)


def _conviction(edge: Decimal, span: Decimal) -> Decimal:
    """How far past its threshold the triggering condition sits, saturating at ``span``."""
    if span <= ZERO or edge <= ZERO:
        return Decimal("0.5")
    return min(Decimal("0.5") + min(edge / span, ONE) * Decimal("0.5"), ONE)


__all__ = ["VolatilityRegimeParams", "VolatilityRegimeStrategy", "trailing_median"]
