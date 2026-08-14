"""Momentum ranked against its own recent history.

The usual "relative momentum" ranks one asset's return against a basket of others and buys
the leaders. No multi-symbol history is available to a strategy here — the context holds
one symbol — so this ranks the current rate of change against **its own** trailing
distribution of rate-of-change readings instead. The question becomes: is this move
unusually strong *for this market lately*?

That reframing keeps the useful property of the cross-sectional version, which is that the
threshold is not a number in price units at all. A fixed 2% trigger fires constantly in a
fast market and never in a slow one; a 90th-percentile trigger fires on roughly a tenth of
bars in both, so the strategy trades at a stable rate across regimes rather than clustering
all its activity in whichever period happened to be volatile.

Being rank-based it is also indifferent to the shape of the return distribution — no
normality, no stationarity of scale, and a single outlier bar shifts a percentile by one
rank rather than dragging a mean.
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

#: Percentiles run 0–100, and the short trigger is the long trigger mirrored about the median.
FULL_PERCENTILE = Decimal("100")


class RelativeMomentumParams(StrategyParams):
    """Parameters for :class:`RelativeMomentumStrategy`."""

    #: Window each rate of change is measured over.
    lookback: int = Field(default=12, ge=2, le=500)
    #: How many past rate-of-change readings the current one is ranked against. Too short
    #: and every mild move is a record; too long and the ranking spans regimes that have
    #: nothing to say about each other.
    rank_window: int = Field(default=100, ge=10, le=2000)
    #: Percentile of its own history the current reading must reach to enter.
    entry_percentile: Decimal = Field(default=Decimal("90"), gt=50, le=100)
    #: Percentile at which an open long is closed — the median by default, meaning the move
    #: is no longer distinguished from a typical one.
    exit_percentile: Decimal = Field(default=Decimal("50"), ge=0, le=100)
    #: The move must also point the right way. A market where every reading is negative
    #: still has a 95th percentile, and it is the least-bad decline, not a rally.
    require_positive_move: bool = True
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_percentiles(self) -> Self:
        if self.exit_percentile >= self.entry_percentile:
            raise ValueError(
                f"exit_percentile ({self.exit_percentile}) must be below entry_percentile "
                f"({self.entry_percentile}), or a position would close on the bar it opened"
            )
        return self


@register_strategy
class RelativeMomentumStrategy(Strategy):
    """Enter when the current rate of change is extreme by its own recent standards."""

    strategy_id = "relative_momentum"
    description = "Rate of change ranked as a percentile of its own trailing history"
    params_model = RelativeMomentumParams

    params: RelativeMomentumParams

    @property
    def warmup_bars(self) -> int:
        """A full rank window of rate-of-change readings, each needing its own lookback.

        The oldest reading in the window is itself measured over ``lookback`` bars, so the
        history has to stretch back one further than the two windows added together.
        """
        return max(self.params.lookback + self.params.rank_window + 1, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Rank the current move against its own history and trade the extremes."""
        index = context.index
        measured = self._ranked_move(context, index)
        if measured is None:
            return context.hold("percentile ranking warming up", self.strategy_id)
        percentile, move = measured

        if context.is_long:
            if percentile <= self.params.exit_percentile:
                return exit_signal(
                    context, self.strategy_id, f"momentum rank fell to {percentile:.1f}"
                )
            return context.hold(f"momentum rank {percentile:.1f} still high", self.strategy_id)

        if context.is_short:
            if percentile >= FULL_PERCENTILE - self.params.exit_percentile:
                return exit_signal(
                    context, self.strategy_id, f"momentum rank rose to {percentile:.1f}"
                )
            return context.hold(f"momentum rank {percentile:.1f} still low", self.strategy_id)

        direction = self._direction(percentile, move)
        if direction is None:
            return context.hold(
                f"momentum rank {percentile:.1f} not extreme enough", self.strategy_id
            )
        if direction is SignalDirection.SHORT and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"move {move:.4f} ranks at percentile {percentile:.1f} of its own history",
        )
        return replace_conviction(signal, self._conviction(percentile))

    def _ranked_move(self, context: StrategyContext, index: int) -> tuple[Decimal, Decimal] | None:
        """The current rate of change and its percentile within the trailing window.

        Only readings strictly *before* the current bar form the comparison window, so the
        current value is ranked against history rather than against a set containing itself.
        """
        oldest = index - self.params.lookback - self.params.rank_window
        if oldest < 0:
            return None

        history: list[Decimal] = []
        for position in range(index - self.params.rank_window, index + 1):
            change = self._rate_of_change(context, position)
            if change is None:
                return None
            history.append(change)

        current = history[-1]
        percentile = percentile_rank(history[:-1], current)
        if percentile is None:
            return None
        return percentile, current

    def _rate_of_change(self, context: StrategyContext, position: int) -> Decimal | None:
        """Fractional change over the lookback ending at ``position``."""
        base = context.closes[position - self.params.lookback]
        if base <= ZERO:
            return None
        return (context.closes[position] - base) / base

    def _direction(self, percentile: Decimal, move: Decimal) -> SignalDirection | None:
        """Which extreme, if either, the current reading sits at."""
        if percentile >= self.params.entry_percentile and (
            move > ZERO or not self.params.require_positive_move
        ):
            return SignalDirection.LONG
        if percentile <= FULL_PERCENTILE - self.params.entry_percentile and (
            move < ZERO or not self.params.require_positive_move
        ):
            return SignalDirection.SHORT
        return None

    def _conviction(self, percentile: Decimal) -> Decimal:
        """How far into the tail the reading sits.

        Distance from the median rather than from the entry threshold, so both the long and
        the short side read off one scale.
        """
        distance = abs(percentile - FULL_PERCENTILE / 2) / (FULL_PERCENTILE / 2)
        return min(max(distance, ZERO), ONE)


__all__ = ["RelativeMomentumParams", "RelativeMomentumStrategy"]
