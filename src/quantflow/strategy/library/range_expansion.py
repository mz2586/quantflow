"""Range expansion — one bar that is much wider than the recent norm, and closes decisively.

A single bar whose high-to-low range is a large multiple of the average range is the market
repricing itself: something arrived that the previous bars did not know. Trading in the
direction that bar closed is a bet that the repricing is incomplete, which is the standard
continuation premise. The body filter is what separates that from a spike-and-reject bar —
a wide range with a tiny body is two-sided fighting, not a repricing, and it is the single
most common way a naive range filter loses money.

Distinct from `atr_expansion`, which compares *smoothed* ATR to its own average. Smoothing
makes that a statement about the volatility regime — it stays true for many bars after the
event and is still true when the widening came from several ordinary bars in a row. This
one looks at the raw range of the decision bar alone, so it fires on the bar the shock
lands and says nothing at all about the bars around it.

Distinct from `volume_breakout`, which needs price to clear a prior extreme; a range
expansion can happen entirely inside an existing range and still be the first sign of it
ending.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, sma
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class RangeExpansionParams(StrategyParams):
    """Parameters for :class:`RangeExpansionStrategy`."""

    #: Window of prior bars whose mean range is the "normal" the current bar is judged against.
    baseline_period: int = Field(default=20, ge=2, le=500)
    #: How many times the average range the current bar must span.
    expansion_multiple: Decimal = Field(default=Decimal("2.0"), gt=1, le=20)
    #: Minimum share of the bar's range its body must occupy, so a wide but indecisive bar
    #: with a tiny body is not mistaken for a directional repricing.
    min_body_ratio: Decimal = Field(default=Decimal("0.5"), gt=0, le=1)
    #: Moving average whose breach closes the position — the continuation premise has failed.
    exit_period: int = Field(default=20, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_multiples(self) -> Self:
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class RangeExpansionStrategy(Strategy):
    """Enter in the direction of a bar whose range dwarfs the recent average."""

    strategy_id = "range_expansion"
    description = "Enters on a bar whose range far exceeds the recent average, in its direction"
    params_model = RangeExpansionParams

    params: RangeExpansionParams

    @property
    def warmup_bars(self) -> int:
        """The baseline window sits entirely before the decision bar."""
        return max(
            self.params.baseline_period + 1, self.params.exit_period + 1, self.params.atr_period + 1
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a range-expansion entry or a moving-average exit."""
        index = context.index
        candle = context.candle

        if context.has_position:
            return self._manage(context)

        # The baseline excludes the current bar: including the wide bar in the average it
        # is being compared against would dilute exactly the thing being measured.
        start = index - self.params.baseline_period
        if start < 0:
            return context.hold("baseline warming up", self.strategy_id)
        window = context.candles[start:index]
        if not window:
            return context.hold("baseline warming up", self.strategy_id)
        baseline = sum((bar.high - bar.low for bar in window), ZERO) / Decimal(len(window))
        if baseline <= ZERO:
            return context.hold("prior bars had no range", self.strategy_id)

        span = candle.high - candle.low
        if span <= ZERO:
            return context.hold("bar has no range", self.strategy_id)
        ratio = span / baseline
        if ratio < self.params.expansion_multiple:
            return context.hold(
                f"range {ratio:.2f}x average, below {self.params.expansion_multiple}x",
                self.strategy_id,
            )

        body = candle.close - candle.open
        body_ratio = abs(body) / span
        if body_ratio < self.params.min_body_ratio:
            return context.hold("wide bar with an indecisive body", self.strategy_id)
        if body == ZERO:
            return context.hold("bar closed unchanged", self.strategy_id)

        if body < ZERO and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        direction = SignalDirection.LONG if body > ZERO else SignalDirection.SHORT
        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"range {ratio:.2f}x the {self.params.baseline_period}-bar average, closing decisively",
        )
        return replace_conviction(signal, self._conviction(ratio, body_ratio))

    def _manage(self, context: StrategyContext) -> Signal:
        """Close once price falls back through the mean the expansion left behind."""
        average = sma(context.closes, self.params.exit_period)[context.index]
        if average is None:
            return context.hold("exit average warming up", self.strategy_id)
        if context.is_long and context.price < average:
            return exit_signal(context, self.strategy_id, "fell back through the moving average")
        if context.is_short and context.price > average:
            return exit_signal(context, self.strategy_id, "rose back through the moving average")
        return context.hold("holding the expansion", self.strategy_id)

    def _conviction(self, ratio: Decimal, body_ratio: Decimal) -> Decimal:
        """A wider bar and a fuller body both read as a stronger repricing."""
        excess = (ratio - self.params.expansion_multiple) / self.params.expansion_multiple
        return min(
            Decimal("0.4") + min(excess, ONE) * Decimal("0.3") + body_ratio * Decimal("0.3"), ONE
        )


__all__ = ["RangeExpansionParams", "RangeExpansionStrategy"]
