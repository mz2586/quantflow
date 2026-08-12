"""Regime-adaptive: classify the market, then apply the tactic that suits it.

Uses ADX to decide whether the market is trending or ranging, and switches behaviour
accordingly — follow the dominant direction when it trends, fade the Bollinger bands when
it ranges. Every other member holds one opinion about market structure and expresses it
regardless of conditions; this one changes its mind, which is the behaviour the research
report identified as the largest missing lever ("six of fourteen are provably
regime-dependent … that is a gating problem, not a signal problem").

The classification uses only closed bars up to the decision bar, like everything else here.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class RegimeAdaptiveParams(StrategyParams):
    """Parameters for :class:`RegimeAdaptiveStrategy`."""

    adx_period: int = Field(default=14, ge=2, le=100)
    #: ADX above this is treated as trending; below, as ranging.
    trend_threshold: Decimal = Field(default=Decimal("25"), gt=0, le=100)
    #: ADX below this is treated as a clean range. Between the two the strategy abstains.
    range_threshold: Decimal = Field(default=Decimal("20"), gt=0, le=100)
    band_period: int = Field(default=20, ge=2, le=200)
    band_deviations: Decimal = Field(default=Decimal("2.0"), gt=0, le=6)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Self:
        if self.range_threshold >= self.trend_threshold:
            raise ValueError(
                f"range_threshold ({self.range_threshold}) must be below trend_threshold "
                f"({self.trend_threshold}); the gap between them is the no-trade zone"
            )
        return self


@register_strategy
class RegimeAdaptiveStrategy(Strategy):
    """Follows trends when ADX is high and fades extremes when it is low."""

    strategy_id = "regime_adaptive"
    description = "Classifies trend vs range with ADX, then trends or fades accordingly"
    params_model = RegimeAdaptiveParams

    params: RegimeAdaptiveParams

    @property
    def warmup_bars(self) -> int:
        """ADX dominates: it is a smoothing of a smoothing."""
        return max(self.params.adx_period * 3 + 1, self.params.band_period + 1)

    def generate(  # noqa: PLR0911, PLR0912 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Classify the regime, then act as that regime warrants."""
        from quantflow.strategy.indicators import atr, bollinger_bands, directional_movement

        index = context.index
        plus_di, minus_di, adx = directional_movement(context.candles, self.params.adx_period)
        strength = adx[index]
        if strength is None:
            return context.hold("adx warming up", self.strategy_id)

        upper, middle, lower = bollinger_bands(
            context.closes, self.params.band_period, self.params.band_deviations
        )
        centre = middle[index]
        if centre is None or upper[index] is None or lower[index] is None:
            return context.hold("bands warming up", self.strategy_id)

        trending = strength >= self.params.trend_threshold
        ranging = strength <= self.params.range_threshold

        if context.has_position:
            # An open position is exited on the terms of the regime that opened it, read
            # from where price now sits relative to the centre line.
            if trending:
                return context.hold(f"holding, adx {strength:.1f}", self.strategy_id)
            if context.is_long and context.price >= centre:
                return exit_signal(context, self.strategy_id, "price reverted to the mean")
            if context.is_short and context.price <= centre:
                return exit_signal(context, self.strategy_id, "price reverted to the mean")
            return context.hold("holding, waiting for the mean", self.strategy_id)

        if not trending and not ranging:
            return context.hold(
                f"adx {strength:.1f} between {self.params.range_threshold} and "
                f"{self.params.trend_threshold}; no regime",
                self.strategy_id,
            )

        volatility = atr(context.candles, self.params.atr_period)[index]
        if trending:
            up = plus_di[index]
            down = minus_di[index]
            if up is None or down is None or up == down:
                return context.hold("no directional dominance", self.strategy_id)
            long = up > down
            reason = f"trending regime (adx {strength:.1f})"
        else:
            band_low = lower[index]
            band_high = upper[index]
            assert band_low is not None
            assert band_high is not None
            if context.price < band_low:
                long = True
            elif context.price > band_high:
                long = False
            else:
                return context.hold("ranging, price inside the bands", self.strategy_id)
            reason = f"ranging regime (adx {strength:.1f}), fading the band"

        if not long and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if long else SignalDirection.SHORT,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            reason,
        )
        return replace_conviction(signal, self._conviction(strength, trending=trending))

    def _conviction(self, strength: Decimal, *, trending: bool) -> Decimal:
        """A clearer regime — further from the ambiguous middle — reads as stronger."""
        if trending:
            excess = strength - self.params.trend_threshold
            span = Decimal("50") - self.params.trend_threshold
        else:
            excess = self.params.range_threshold - strength
            span = self.params.range_threshold
        if span <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + min(excess / span, ONE) * Decimal("0.5"), ONE)


__all__ = ["RegimeAdaptiveParams", "RegimeAdaptiveStrategy"]
