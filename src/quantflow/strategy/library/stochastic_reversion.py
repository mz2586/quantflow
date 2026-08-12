"""Stochastic oscillator mean reversion.

Buy when %K crosses back up through its %D line from oversold. The cross matters: entering
the moment %K first prints below the oversold line is entering while the move is still
going, and in a trending market that condition can persist for dozens of bars. Waiting for
%K to turn back through %D asks for evidence the fall has stopped.

Against `rsi_reversion`, the other oscillator member: RSI measures the size of recent gains
against recent losses, while the stochastic measures *where the close sits inside the
period's range*. A market that grinds down in small steps but closes near its highs reads
as oversold on RSI and as strong on the stochastic — they disagree by construction rather
than by parameter choice.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, crossed_above, crossed_below, stochastic
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class StochasticReversionParams(StrategyParams):
    """Parameters for :class:`StochasticReversionStrategy`."""

    period: int = Field(default=14, ge=2, le=200)
    smooth: int = Field(default=3, ge=1, le=50)
    oversold: Decimal = Field(default=Decimal("20"), ge=0, le=50)
    overbought: Decimal = Field(default=Decimal("80"), ge=50, le=100)
    #: Exit once %K has recovered to this level.
    exit_level: Decimal = Field(default=Decimal("55"), ge=0, le=100)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_levels(self) -> Self:
        if self.oversold >= self.overbought:
            raise ValueError(
                f"oversold ({self.oversold}) must be below overbought ({self.overbought})"
            )
        return self


@register_strategy
class StochasticReversionStrategy(Strategy):
    """Enter on a %K/%D cross out of an extreme, exit on recovery."""

    strategy_id = "stochastic_reversion"
    description = "Stochastic %K/%D cross out of oversold or overbought"
    params_model = StochasticReversionParams

    params: StochasticReversionParams

    @property
    def warmup_bars(self) -> int:
        """Oscillator plus its smoothing, with a bar to spare for the cross."""
        return max(self.params.period + self.params.smooth + 1, self.params.atr_period + 1)

    def generate(self, context: StrategyContext) -> Signal:
        """Emit a stochastic reversion signal."""
        index = context.index
        k_line, d_line = stochastic(context.candles, self.params.period, self.params.smooth)
        k_now = k_line[index]
        if k_now is None or d_line[index] is None:
            return context.hold("stochastic warming up", self.strategy_id)

        if context.is_long:
            return (
                exit_signal(context, self.strategy_id, "stochastic recovered")
                if k_now >= self.params.exit_level
                else context.hold("holding long, stochastic still low", self.strategy_id)
            )
        if context.is_short:
            return (
                exit_signal(context, self.strategy_id, "stochastic recovered")
                if k_now <= (Decimal("100") - self.params.exit_level)
                else context.hold("holding short, stochastic still high", self.strategy_id)
            )

        # The cross must happen while the oscillator is still in the extreme zone. A cross
        # in mid-range is not a reversion signal, it is noise.
        bullish = crossed_above(k_line, d_line, index) and k_now <= self.params.oversold
        bearish = crossed_below(k_line, d_line, index) and k_now >= self.params.overbought
        if not (bullish or bearish):
            return context.hold("no cross out of an extreme", self.strategy_id)
        if bearish and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        direction = SignalDirection.LONG if bullish else SignalDirection.SHORT
        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"%K crossed its signal at {k_now:.1f}",
        )
        from quantflow.strategy.library.vwap_reversion import replace_conviction

        return replace_conviction(signal, self._conviction(k_now, long=bullish))

    def _conviction(self, k_value: Decimal, *, long: bool) -> Decimal:
        """Deeper into the extreme reads as a stronger case."""
        if long:
            depth = self.params.oversold - k_value
            span = self.params.oversold if self.params.oversold > ZERO else ONE
        else:
            depth = k_value - self.params.overbought
            span = Decimal("100") - self.params.overbought
            span = span if span > ZERO else ONE
        if depth <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + (depth / span) * Decimal("0.5"), ONE)


__all__ = ["StochasticReversionParams", "StochasticReversionStrategy"]
