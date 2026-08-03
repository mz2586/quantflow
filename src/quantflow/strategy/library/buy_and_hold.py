"""Buy and hold — the benchmark every other strategy has to beat.

This is not a trading idea; it is the control. A strategy that returns 40% over a period
in which the asset returned 120% has destroyed value, and without this baseline in the
same leaderboard, under the same fees and the same slippage, that fact is invisible. Most
published crypto backtests omit it, which is exactly why so many of them look good.

It takes one position on its first eligible bar and never trades again, so its fee drag is
two fills rather than hundreds — which is itself part of what makes it hard to beat.
"""

from __future__ import annotations

from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.registry import register_strategy


class BuyAndHoldParams(StrategyParams):
    """Parameters for :class:`BuyAndHoldStrategy`. Deliberately empty."""


@register_strategy
class BuyAndHoldStrategy(Strategy):
    """Enter once, hold to the end. The baseline for the leaderboard."""

    strategy_id = "buy_and_hold"
    description = "Benchmark: buy on the first bar and hold to the end"
    params_model = BuyAndHoldParams

    params: BuyAndHoldParams

    @property
    def warmup_bars(self) -> int:
        """One bar. There is nothing to warm up."""
        return 1

    def generate(self, context: StrategyContext) -> Signal:
        """Enter long once, then hold forever.

        No stop is attached on purpose: a stop would make this something other than buy
        and hold, and the comparison would stop being honest. The risk engine may still
        impose its own protection, which is a property of the platform rather than of
        this strategy.
        """
        if context.has_position:
            return context.hold("holding", self.strategy_id)
        return Signal(
            symbol=context.symbol,
            direction=SignalDirection.LONG,
            timestamp=context.now,
            strategy_id=self.strategy_id,
            reference_price=context.price,
            reason="benchmark entry",
        )
