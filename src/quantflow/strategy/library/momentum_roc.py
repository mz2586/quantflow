"""Time-series momentum.

Holds while the rate of change over a lookback window stays positive. This is the classic
cross-asset momentum result applied to a single series: it makes no attempt to time
entries and exits precisely, and its entire premise is that the sign of past return over
weeks predicts the sign of the next return.

Its behaviour is the near-opposite of `rsi_reversion`, which buys weakness. Holding both
in the same leaderboard is deliberate — if the framework only contained variations on one
idea, ranking them would say nothing about which *kind* of edge exists in this market.
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


class MomentumRocParams(StrategyParams):
    """Parameters for :class:`MomentumRocStrategy`."""

    #: Lookback for the rate of change. 720 hourly bars is roughly a month.
    lookback: int = Field(default=720, ge=10, le=5000)
    #: Return above which a long is taken, as a fraction.
    entry_threshold: Decimal = Field(default=Decimal("0.02"), ge=0, le=1)
    #: Return below which an open long is closed. Below the entry threshold on purpose,
    #: so a position is not thrown away by a single bar wobbling across one line.
    exit_threshold: Decimal = Field(default=Decimal("0.0"), ge=-1, le=1)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("6.0"), gt=0, le=20)


@register_strategy
class MomentumRocStrategy(Strategy):
    """Long while trailing return over the lookback window is positive."""

    strategy_id = "momentum_roc"
    description = "Time-series momentum: hold while trailing rate of change stays positive"
    params_model = MomentumRocParams

    params: MomentumRocParams

    @property
    def warmup_bars(self) -> int:
        """One full lookback window, plus a bar to measure from."""
        return max(self.params.lookback + 1, self.params.atr_period + 1)

    def generate(self, context: StrategyContext) -> Signal:
        """Hold while momentum is positive, exit when it decays."""
        index = context.index
        past = context.closes[index - self.params.lookback]
        if past <= ZERO:
            return context.hold("no valid reference price", self.strategy_id)

        change = (context.price - past) / past

        if context.is_long:
            if change <= self.params.exit_threshold:
                return exit_signal(context, self.strategy_id, f"momentum decayed to {change:.4f}")
            return context.hold(f"momentum {change:.4f} still positive", self.strategy_id)

        if change < self.params.entry_threshold:
            return context.hold(f"momentum {change:.4f} below threshold", self.strategy_id)

        return entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"trailing return {change:.4f} above threshold",
        )
