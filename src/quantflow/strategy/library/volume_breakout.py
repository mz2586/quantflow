"""Volume-confirmed breakout.

A price breakout on ordinary volume is usually noise; the same breakout on a large
multiple of average volume means something changed. This is the only strategy in the
library that uses the volume field at all, which makes it a genuine test of whether volume
carries information in this market or is merely decoration on a chart.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class VolumeBreakoutParams(StrategyParams):
    """Parameters for :class:`VolumeBreakoutStrategy`."""

    breakout_period: int = Field(default=48, ge=5, le=500)
    exit_period: int = Field(default=24, ge=2, le=500)
    #: Window over which average volume is measured.
    volume_period: int = Field(default=96, ge=5, le=1000)
    #: Multiple of average volume the breakout bar must trade.
    volume_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=50)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)


@register_strategy
class VolumeBreakoutStrategy(Strategy):
    """Break of a rolling high, confirmed by a volume surge."""

    strategy_id = "volume_breakout"
    description = "Rolling-high breakout that must be confirmed by a surge in volume"
    params_model = VolumeBreakoutParams

    params: VolumeBreakoutParams

    @property
    def warmup_bars(self) -> int:
        """The longest of the price, volume and ATR windows."""
        return max(
            self.params.breakout_period + 1,
            self.params.volume_period + 1,
            self.params.atr_period + 1,
        )

    def generate(self, context: StrategyContext) -> Signal:
        """Enter on a volume-confirmed breakout, exit on a rolling low."""
        index = context.index
        candles = context.candles

        if context.is_long:
            lows = [candle.low for candle in candles]
            floor = _rolling_min_excluding_current(lows, self.params.exit_period, index)
            if floor is not None and context.price < floor:
                return exit_signal(
                    context, self.strategy_id, f"closed below the {self.params.exit_period}-bar low"
                )
            return context.hold("holding the breakout", self.strategy_id)

        # The prior high excludes the current bar: comparing a bar's close against a
        # window that already contains that bar's own high is a look-ahead in disguise.
        highs = [candle.high for candle in candles]
        prior_high = _rolling_max_excluding_current(highs, self.params.breakout_period, index)
        if prior_high is None or context.price <= prior_high:
            return context.hold("no breakout", self.strategy_id)

        average_volume = self._average_volume(context, index)
        if average_volume is None or average_volume <= ZERO:
            return context.hold("no volume baseline", self.strategy_id)

        volume = context.candle.volume
        if volume < average_volume * self.params.volume_multiple:
            return context.hold(
                f"volume {volume} below {self.params.volume_multiple}x average", self.strategy_id
            )

        return entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG,
            atr(candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            "breakout confirmed by a volume surge",
        )

    def _average_volume(self, context: StrategyContext, index: int) -> Decimal | None:
        """Mean volume over the window ending on the previous bar."""
        start = index - self.params.volume_period
        if start < 0:
            return None
        window = context.candles[start:index]
        if not window:
            return None
        return sum((candle.volume for candle in window), ZERO) / Decimal(len(window))


def _rolling_max_excluding_current(
    values: list[Decimal], period: int, index: int
) -> Decimal | None:
    """Highest value in the ``period`` bars *before* ``index``."""
    start = index - period
    if start < 0:
        return None
    return max(values[start:index]) if values[start:index] else None


def _rolling_min_excluding_current(
    values: list[Decimal], period: int, index: int
) -> Decimal | None:
    """Lowest value in the ``period`` bars *before* ``index``."""
    start = index - period
    if start < 0:
        return None
    return min(values[start:index]) if values[start:index] else None
