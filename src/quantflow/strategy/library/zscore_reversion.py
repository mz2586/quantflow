"""Z-score mean reversion.

Buys when price is a given number of standard deviations below its own rolling mean and
exits as the z-score returns toward zero. Structurally similar to the Bollinger reversion
but parameterised on the statistic directly, and with a *volatility ceiling*: when
realised volatility is extreme the distribution is not stationary enough for a z-score to
mean anything, and reverting-to-the-mean is precisely the wrong bet during a regime break.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, normalized_atr, sma, stdev
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class ZScoreReversionParams(StrategyParams):
    """Parameters for :class:`ZScoreReversionStrategy`."""

    period: int = Field(default=48, ge=5, le=500)
    #: Standard deviations below the mean required to enter.
    entry_z: Decimal = Field(default=Decimal("2.0"), gt=0, le=6)
    #: Z-score at which an open position is closed.
    exit_z: Decimal = Field(default=Decimal("0.25"), ge=-6, le=6)
    #: Skip entries when ATR/price exceeds this. A regime break is not a mean reversion.
    max_normalized_atr: Decimal = Field(default=Decimal("0.03"), gt=0, le=1)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Self:
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be below entry_z or a position closes immediately")
        return self


@register_strategy
class ZScoreReversionStrategy(Strategy):
    """Buy a statistically stretched decline, exit as it normalises."""

    strategy_id = "zscore_reversion"
    description = "Mean reversion on a rolling z-score, gated by a volatility ceiling"
    params_model = ZScoreReversionParams

    params: ZScoreReversionParams

    @property
    def warmup_bars(self) -> int:
        """Enough bars for both the rolling statistic and the ATR gate."""
        return max(self.params.period * 2, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Enter on a stretched negative z-score; exit as it reverts."""
        index = context.index
        score = self._zscore(context, index)
        if score is None:
            return context.hold("z-score warming up", self.strategy_id)

        if context.is_long:
            if score >= -self.params.exit_z:
                return exit_signal(context, self.strategy_id, f"z-score reverted to {score:.2f}")
            return context.hold(f"z-score {score:.2f} still stretched", self.strategy_id)

        if score > -self.params.entry_z:
            return context.hold(f"z-score {score:.2f} not stretched enough", self.strategy_id)

        volatility_ratio = normalized_atr(context.candles, self.params.atr_period)[index]
        if volatility_ratio is None:
            return context.hold("volatility gate warming up", self.strategy_id)
        if volatility_ratio > self.params.max_normalized_atr:
            # Extreme volatility means the distribution the z-score assumes has broken.
            return context.hold(
                f"volatility {volatility_ratio:.4f} above the ceiling", self.strategy_id
            )

        return entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"z-score {score:.2f} below entry threshold",
        )

    def _zscore(self, context: StrategyContext, index: int) -> Decimal | None:
        """Standard deviations between price and its rolling mean."""
        closes = context.closes
        mean = sma(closes, self.params.period)[index]
        deviation = stdev(closes, self.params.period)[index]
        if mean is None or deviation is None or deviation <= ZERO:
            return None
        return (context.price - mean) / deviation
