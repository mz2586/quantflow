"""Non-parametric mean reversion on the trailing price distribution.

Buy when the current close sits in the bottom few percent of the prices this market has
printed recently, and let go once it climbs back to an ordinary place in that range.

**Why this exists alongside `zscore_reversion`.** A z-score is a *parametric* statistic: it
divides by a standard deviation, and the entry threshold "two sigma" only carries its usual
meaning — a roughly 2.5% tail — if the distribution is approximately normal. Financial
returns are not. They are fat-tailed and skewed, so a two-sigma reading occurs far more
often than the normal assumption implies, and a single violent bar inflates the standard
deviation enough to suppress entries for the entire window afterwards, exactly when the
opportunity is best.

This strategy computes no mean and no standard deviation. It **ranks**: the trigger is
"lower than 95 of the last 100 closes", which is a statement about order, not about shape.
A crash bar moves the rank by one position instead of doubling the denominator, and the
entry rate is stable by construction — a 5th-percentile trigger fires on about 5% of bars
whatever the distribution looks like.

The trade-off is honest: a rank knows *where* price sits but not *how far*, so a 5th
percentile reading in a dead-flat window is a fraction of a percent from the median. The
``min_range_pct`` gate exists for that — no meaningful spread, no trade.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, percentile_rank
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy

#: Percentiles run 0–100; the short trigger mirrors the long one about the median.
FULL_PERCENTILE = Decimal("100")


class PercentileReversionParams(StrategyParams):
    """Parameters for :class:`PercentileReversionStrategy`."""

    #: How many trailing closes form the distribution price is ranked against.
    period: int = Field(default=100, ge=10, le=2000)
    #: Percentile at or below which a long is taken.
    entry_percentile: Decimal = Field(default=Decimal("5"), ge=0, lt=50)
    #: Percentile at which an open long is closed — the median by default, meaning price is
    #: no longer unusual in either direction.
    exit_percentile: Decimal = Field(default=Decimal("50"), gt=0, le=100)
    #: The window's high-low span, as a fraction of price, below which the ranking is
    #: ignored. In a flat window the bottom percentile is noise wearing a statistic's hat.
    min_range_pct: Decimal = Field(default=Decimal("0.02"), ge=0, le=1)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_percentiles(self) -> Self:
        if self.exit_percentile <= self.entry_percentile:
            raise ValueError(
                f"exit_percentile ({self.exit_percentile}) must exceed entry_percentile "
                f"({self.entry_percentile}), or a position would close on the bar it opened"
            )
        return self


@register_strategy
class PercentileReversionStrategy(Strategy):
    """Fade price when it reaches an extreme rank within its own trailing range."""

    strategy_id = "percentile_reversion"
    description = "Rank-based mean reversion: no normality assumption, only order"
    params_model = PercentileReversionParams

    params: PercentileReversionParams

    @property
    def warmup_bars(self) -> int:
        """One full ranking window plus the bar being ranked against it.

        The current close is excluded from its own distribution, so ``period`` prior closes
        must exist alongside it.
        """
        return max(self.params.period + 1, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Enter at a distributional extreme, exit as the rank normalises."""
        index = context.index
        window = self._window(context, index)
        if window is None:
            return context.hold("ranking window warming up", self.strategy_id)

        percentile = percentile_rank(window, context.price)
        if percentile is None:
            return context.hold("no distribution to rank against", self.strategy_id)

        if context.is_long:
            if percentile >= self.params.exit_percentile:
                return exit_signal(context, self.strategy_id, f"rank recovered to {percentile:.1f}")
            return context.hold(f"rank {percentile:.1f} still depressed", self.strategy_id)

        if context.is_short:
            if percentile <= FULL_PERCENTILE - self.params.exit_percentile:
                return exit_signal(context, self.strategy_id, f"rank eased to {percentile:.1f}")
            return context.hold(f"rank {percentile:.1f} still elevated", self.strategy_id)

        if percentile <= self.params.entry_percentile:
            direction = SignalDirection.LONG
        elif percentile >= FULL_PERCENTILE - self.params.entry_percentile:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold(f"rank {percentile:.1f} is unremarkable", self.strategy_id)

        spread = self._range_fraction(window, context.price)
        if spread < self.params.min_range_pct:
            # An extreme rank inside a window that has barely moved is not an extreme price.
            return context.hold(f"window range {spread:.4f} below the minimum", self.strategy_id)

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"price ranks at percentile {percentile:.1f} of its last {self.params.period} closes",
        )
        return replace_conviction(signal, self._conviction(percentile))

    def _window(self, context: StrategyContext, index: int) -> tuple[Decimal, ...] | None:
        """The trailing closes price is ranked against, excluding the current bar.

        Excluded so the reading is "where does today sit among the days before it" rather
        than a self-referential rank that can never reach zero.
        """
        oldest = index - self.params.period
        if oldest < 0:
            return None
        return context.closes[oldest:index]

    def _range_fraction(self, window: tuple[Decimal, ...], price: Decimal) -> Decimal:
        """The window's high-low span as a fraction of the current price."""
        if price <= ZERO:
            return ZERO
        return (max(window) - min(window)) / price

    def _conviction(self, percentile: Decimal) -> Decimal:
        """How deep into the tail the reading sits, measured from the median."""
        distance = abs(percentile - FULL_PERCENTILE / 2) / (FULL_PERCENTILE / 2)
        return min(max(distance, ZERO), ONE)


__all__ = ["PercentileReversionParams", "PercentileReversionStrategy"]
