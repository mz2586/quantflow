"""Triple moving-average trend alignment.

Requires fast > medium > slow before taking a position, and exits as soon as the fast
crosses back under the medium. The three-way alignment is a stricter filter than a two-MA
cross: it trades far less often, which in a market where fees consume most of the gross
profit is a design decision rather than an accident.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, ema
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class TripleMaParams(StrategyParams):
    """Parameters for :class:`TripleMaStrategy`."""

    fast_period: int = Field(default=10, ge=2, le=200)
    medium_period: int = Field(default=50, ge=3, le=500)
    slow_period: int = Field(default=200, ge=5, le=1000)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("6.0"), gt=0, le=20)

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if not self.fast_period < self.medium_period < self.slow_period:
            raise ValueError(
                f"periods must increase strictly: fast ({self.fast_period}) < "
                f"medium ({self.medium_period}) < slow ({self.slow_period})"
            )
        return self


@register_strategy
class TripleMaStrategy(Strategy):
    """Long only while three EMAs are aligned bullishly."""

    strategy_id = "triple_ma"
    description = "Trend following requiring fast > medium > slow EMA alignment"
    params_model = TripleMaParams

    params: TripleMaParams

    @property
    def warmup_bars(self) -> int:
        """Twice the slow period, so the slowest EMA is no longer tracking its seed."""
        return max(self.params.slow_period * 2, self.params.atr_period + 1)

    def generate(self, context: StrategyContext) -> Signal:
        """Enter on full bullish alignment, exit when the fast loses the medium."""
        index = context.index
        closes = context.closes
        fast = ema(closes, self.params.fast_period)[index]
        medium = ema(closes, self.params.medium_period)[index]
        slow = ema(closes, self.params.slow_period)[index]
        if fast is None or medium is None or slow is None:
            return context.hold("emas warming up", self.strategy_id)

        if context.is_long:
            if fast < medium:
                return exit_signal(context, self.strategy_id, "fast EMA lost the medium EMA")
            return context.hold("alignment intact", self.strategy_id)

        if not (fast > medium > slow):
            return context.hold("emas not aligned", self.strategy_id)

        return entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            "fast > medium > slow EMA alignment",
        )
