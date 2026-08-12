"""VWAP reversion.

Fade a stretch away from the volume-weighted average price, on the premise that VWAP marks
where size actually traded and price tends to be drawn back to it. The distinguishing
feature against the other mean-reversion members is *what* the mean is: `zscore_reversion`
and `bollinger_reversion` both revert to an unweighted moving average, which weights a
bar of near-zero volume exactly like a bar that carried the day's real business. VWAP does
not, so the two disagree most in thin trade — precisely where an unweighted mean is least
trustworthy.

Distance is measured in ATR rather than in percent so the trigger means the same thing in
a calm market and a violent one.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, rolling_vwap
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class VwapReversionParams(StrategyParams):
    """Parameters for :class:`VwapReversionStrategy`."""

    vwap_period: int = Field(default=20, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    #: How many ATRs below VWAP price must close before it is considered stretched.
    entry_atr_distance: Decimal = Field(default=Decimal("1.5"), gt=0, le=10)
    #: Exit once price has closed back within this many ATRs of VWAP.
    exit_atr_distance: Decimal = Field(default=Decimal("0.25"), ge=0, le=10)
    #: Distance at which conviction saturates at 1.0, in ATRs beyond the entry threshold.
    conviction_span: Decimal = Field(default=Decimal("1.5"), gt=0, le=10)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_distances(self) -> Self:
        if self.exit_atr_distance >= self.entry_atr_distance:
            raise ValueError(
                f"exit_atr_distance ({self.exit_atr_distance}) must be below "
                f"entry_atr_distance ({self.entry_atr_distance}), or the strategy would "
                "exit on the bar it entered"
            )
        return self


@register_strategy
class VwapReversionStrategy(Strategy):
    """Buy a stretch below rolling VWAP, exit as price returns to it."""

    strategy_id = "vwap_reversion"
    description = "Fades ATR-scaled stretches away from rolling VWAP"
    params_model = VwapReversionParams

    params: VwapReversionParams

    @property
    def warmup_bars(self) -> int:
        """Enough bars for both VWAP and ATR to be defined."""
        return max(self.params.vwap_period, self.params.atr_period) + 1

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a VWAP reversion signal."""
        index = context.index
        vwap = rolling_vwap(context.candles, self.params.vwap_period)[index]
        volatility = atr(context.candles, self.params.atr_period)[index]
        if vwap is None:
            return context.hold("vwap unavailable (no volume in window)", self.strategy_id)
        if volatility is None or volatility <= ZERO:
            return context.hold("atr unavailable", self.strategy_id)

        # Signed: negative means price is below VWAP.
        distance = (context.price - vwap) / volatility

        if context.is_long:
            return (
                exit_signal(context, self.strategy_id, "price returned to vwap")
                if distance >= -self.params.exit_atr_distance
                else context.hold("holding long, still below vwap", self.strategy_id)
            )
        if context.is_short:
            return (
                exit_signal(context, self.strategy_id, "price returned to vwap")
                if distance <= self.params.exit_atr_distance
                else context.hold("holding short, still above vwap", self.strategy_id)
            )

        stretched_low = distance <= -self.params.entry_atr_distance
        stretched_high = distance >= self.params.entry_atr_distance
        if not (stretched_low or stretched_high):
            return context.hold(
                f"within {self.params.entry_atr_distance} ATR of vwap", self.strategy_id
            )
        if stretched_high and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        direction = SignalDirection.LONG if stretched_low else SignalDirection.SHORT
        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"price {abs(distance):.2f} ATR from vwap",
        )
        return replace_conviction(signal, self._conviction(abs(distance)))

    def _conviction(self, distance: Decimal) -> Decimal:
        """Scale conviction with how far beyond the threshold price has stretched.

        A bar that just clears the trigger is a far weaker case than one twice as
        stretched, and collapsing both to 1.0 would hand the orchestrator no way to tell
        them apart.
        """
        excess = distance - self.params.entry_atr_distance
        if excess <= ZERO:
            return Decimal("0.5")
        scaled = Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5")
        return min(scaled, ONE)


def replace_conviction(signal: Signal, conviction: Decimal) -> Signal:
    """Return ``signal`` with a different conviction.

    ``Signal`` is frozen, and the shared ``entry_signal`` helper does not take a conviction
    — keeping it that way means every strategy still builds protective levels identically.
    """
    from dataclasses import replace

    return replace(signal, conviction=conviction)


__all__ = ["VwapReversionParams", "VwapReversionStrategy"]
