"""MACD trend following.

Long when the MACD line crosses above its signal line while the histogram is expanding.
The expansion requirement is the point of difference from a plain crossover: a cross whose
histogram is already shrinking is a cross that has largely happened, and entering there is
how a trend follower ends up systematically buying the end of moves.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, crossed_above, crossed_below, macd
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class MacdTrendParams(StrategyParams):
    """Parameters for :class:`MacdTrendStrategy`."""

    fast_period: int = Field(default=12, ge=2, le=200)
    slow_period: int = Field(default=26, ge=3, le=400)
    signal_period: int = Field(default=9, ge=2, le=100)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("5.0"), gt=0, le=20)
    allow_short: bool = False
    #: Require the histogram to be growing in the direction of the cross.
    require_expansion: bool = True

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be below slow_period ({self.slow_period})"
            )
        return self


@register_strategy
class MacdTrendStrategy(Strategy):
    """MACD signal-line crossover with a histogram-expansion filter."""

    strategy_id = "macd_trend"
    description = "MACD signal-line crossover, entered only while the histogram expands"
    params_model = MacdTrendParams

    params: MacdTrendParams

    @property
    def warmup_bars(self) -> int:
        """Slow EMA plus the signal EMA, doubled so neither is dominated by its seed."""
        return max(
            (self.params.slow_period + self.params.signal_period) * 2, self.params.atr_period + 1
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a MACD crossover signal."""
        index = context.index
        line, signal_line, histogram = macd(
            context.closes,
            self.params.fast_period,
            self.params.slow_period,
            self.params.signal_period,
        )
        if line[index] is None or signal_line[index] is None:
            return context.hold("macd warming up", self.strategy_id)

        bullish = crossed_above(line, signal_line, index)
        bearish = crossed_below(line, signal_line, index)

        if context.is_long:
            return (
                exit_signal(context, self.strategy_id, "macd crossed below its signal")
                if bearish
                else context.hold("holding long", self.strategy_id)
            )
        if context.is_short:
            return (
                exit_signal(context, self.strategy_id, "macd crossed above its signal")
                if bullish
                else context.hold("holding short", self.strategy_id)
            )

        if not (bullish or bearish):
            return context.hold("no crossover", self.strategy_id)

        if self.params.require_expansion and not self._expanding(histogram, index, long=bullish):
            return context.hold("histogram already contracting", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        if bullish:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.LONG,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "macd crossed above its signal with an expanding histogram",
            )
        if not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)
        return entry_signal(
            context,
            self.strategy_id,
            SignalDirection.SHORT,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            "macd crossed below its signal with an expanding histogram",
        )

    @staticmethod
    def _expanding(histogram: tuple[Decimal | None, ...], index: int, *, long: bool) -> bool:
        """Whether the histogram is growing in the direction of the trade."""
        if index < 1:
            return False
        current, previous = histogram[index], histogram[index - 1]
        if current is None or previous is None:
            return False
        return current > previous if long else current < previous and current < ZERO
