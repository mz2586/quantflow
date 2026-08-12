"""ADX trend-strength following.

Enter in the direction of the dominant directional index, but only while ADX says a trend
actually exists. The other trend members (`ema_cross`, `macd_trend`, `triple_ma`) all infer
trend from price *position* — one average above another — which is equally true in a
listless drift as in a real move. ADX measures trend *strength* independently of direction,
so this one abstains in exactly the chop that produces the others' worst trades.

Exit when ADX falls back through the threshold: the trend it entered on has stopped being
a trend, whatever price is doing.
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


class AdxTrendParams(StrategyParams):
    """Parameters for :class:`AdxTrendStrategy`."""

    period: int = Field(default=14, ge=2, le=100)
    #: ADX above this counts as a trending market.
    trend_threshold: Decimal = Field(default=Decimal("25"), gt=0, le=100)
    #: Exit once ADX decays below this.
    exit_threshold: Decimal = Field(default=Decimal("20"), gt=0, le=100)
    #: ADX at which conviction saturates.
    strong_trend: Decimal = Field(default=Decimal("50"), gt=0, le=100)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("5.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Self:
        if self.exit_threshold >= self.trend_threshold:
            raise ValueError(
                f"exit_threshold ({self.exit_threshold}) must be below trend_threshold "
                f"({self.trend_threshold}), or the strategy would exit on the bar it entered"
            )
        if self.strong_trend <= self.trend_threshold:
            raise ValueError(
                f"strong_trend ({self.strong_trend}) must exceed trend_threshold "
                f"({self.trend_threshold})"
            )
        return self


@register_strategy
class AdxTrendStrategy(Strategy):
    """Trade the dominant DI while ADX confirms a trend is present."""

    strategy_id = "adx_trend"
    description = "Follows +DI/-DI dominance, but only while ADX confirms a trend"
    params_model = AdxTrendParams

    params: AdxTrendParams

    @property
    def warmup_bars(self) -> int:
        """ADX is a smoothing of a smoothing, so it needs roughly twice the period."""
        return max(self.params.period * 3 + 1, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit an ADX-confirmed trend signal."""
        from quantflow.strategy.indicators import atr, directional_movement

        index = context.index
        plus_di, minus_di, adx = directional_movement(context.candles, self.params.period)
        strength = adx[index]
        up = plus_di[index]
        down = minus_di[index]
        if strength is None or up is None or down is None:
            return context.hold("adx warming up", self.strategy_id)

        if context.has_position:
            if strength < self.params.exit_threshold:
                return exit_signal(context, self.strategy_id, f"adx decayed to {strength:.1f}")
            return context.hold(f"holding, adx {strength:.1f}", self.strategy_id)

        if strength < self.params.trend_threshold:
            return context.hold(
                f"adx {strength:.1f} below {self.params.trend_threshold}", self.strategy_id
            )
        if up == down:
            return context.hold("no directional dominance", self.strategy_id)

        long = up > down
        if not long and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if long else SignalDirection.SHORT,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"adx {strength:.1f} with {'+' if long else '-'}DI dominant",
        )
        return replace_conviction(signal, self._conviction(strength, up, down))

    def _conviction(self, strength: Decimal, up: Decimal, down: Decimal) -> Decimal:
        """Trend strength and the margin between the DIs both raise conviction."""
        span = self.params.strong_trend - self.params.trend_threshold
        trend_part = min(max((strength - self.params.trend_threshold) / span, ZERO), ONE)
        total = up + down
        spread_part = abs(up - down) / total if total > ZERO else ZERO
        return min(Decimal("0.4") + trend_part * Decimal("0.4") + spread_part * Decimal("0.2"), ONE)


__all__ = ["AdxTrendParams", "AdxTrendStrategy"]
