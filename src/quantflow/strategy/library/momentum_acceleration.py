"""Momentum acceleration — the second derivative of price.

Every momentum strategy in this library asks whether price *has* moved. This one asks
whether the move is *getting faster*: it compares the rate of change now against the rate
of change a few bars ago and requires the difference to be positive.

The reason to separate the two questions is that they fail in different places. Plain
momentum is at its most confident exactly at the end of a trend, when the trailing return
is largest and the move is running out of buyers. Acceleration is largest at the *start*,
when a range breaks and the trailing return is still modest. A decelerating advance is
still an advance, so this strategy exits while `momentum_roc` is still holding — earlier,
and sometimes far too early, which is the trade it deliberately makes.

Requiring the level as well as the change matters: a market falling less catastrophically
than it was falling has positive acceleration and is not something to buy. The
`min_momentum` gate is what stops that from becoming a long.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class MomentumAccelerationParams(StrategyParams):
    """Parameters for :class:`MomentumAccelerationStrategy`."""

    #: Window each rate of change is measured over.
    lookback: int = Field(default=20, ge=2, le=500)
    #: How far back the comparison rate of change is taken from. Small values make the
    #: measurement jumpy; a value near the lookback compares two nearly disjoint windows.
    gap: int = Field(default=10, ge=1, le=500)
    #: Increase in the rate of change, as a fraction, required to enter.
    entry_acceleration: Decimal = Field(default=Decimal("0.01"), gt=0, le=1)
    #: Acceleration at which an open position is closed. Below the entry threshold so a
    #: position is not thrown away by one bar wobbling across a single line.
    exit_acceleration: Decimal = Field(default=Decimal("0"), ge=-1, le=1)
    #: The trailing return must also clear this, so a decelerating collapse is not read as
    #: a buy simply because it is collapsing more slowly than it was.
    min_momentum: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    #: Acceleration beyond the entry threshold at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("0.02"), gt=0, le=1)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Self:
        if self.exit_acceleration >= self.entry_acceleration:
            raise ValueError(
                f"exit_acceleration ({self.exit_acceleration}) must be below "
                f"entry_acceleration ({self.entry_acceleration}), or a position would close "
                "on the bar it opened"
            )
        return self


@register_strategy
class MomentumAccelerationStrategy(Strategy):
    """Enter when momentum is not merely positive but increasing."""

    strategy_id = "momentum_acceleration"
    description = "Second derivative of price: trade the change in the rate of change"
    params_model = MomentumAccelerationParams

    params: MomentumAccelerationParams

    @property
    def warmup_bars(self) -> int:
        """Both rate-of-change windows must fit, plus the ATR window.

        The earlier window starts ``lookback + gap`` bars back, and one more bar is needed
        to measure from it.
        """
        return max(self.params.lookback + self.params.gap + 1, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Trade the change in the rate of change."""
        index = context.index
        measured = self._rates(context, index)
        if measured is None:
            return context.hold("rate of change warming up", self.strategy_id)
        current, previous = measured
        acceleration = current - previous

        if context.is_long:
            if acceleration <= self.params.exit_acceleration:
                return exit_signal(
                    context, self.strategy_id, f"momentum decelerating ({acceleration:.4f})"
                )
            return context.hold(f"acceleration {acceleration:.4f} intact", self.strategy_id)

        if context.is_short:
            if acceleration >= -self.params.exit_acceleration:
                return exit_signal(
                    context, self.strategy_id, f"decline decelerating ({acceleration:.4f})"
                )
            return context.hold(f"acceleration {acceleration:.4f} intact", self.strategy_id)

        direction = self._direction(current, acceleration)
        if direction is None:
            return context.hold(
                f"acceleration {acceleration:.4f} below threshold at momentum {current:.4f}",
                self.strategy_id,
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
            f"rate of change {previous:.4f} to {current:.4f} over {self.params.gap} bars",
        )
        return replace_conviction(signal, self._conviction(abs(acceleration)))

    def _rates(self, context: StrategyContext, index: int) -> tuple[Decimal, Decimal] | None:
        """The current rate of change and the one from ``gap`` bars ago.

        ``None`` when either window is unavailable or anchored on a non-positive price;
        dividing by such a price would produce a number that looks like a return.
        """
        closes = context.closes
        oldest = index - self.params.lookback - self.params.gap
        if oldest < 0:
            return None
        current_base = closes[index - self.params.lookback]
        previous_base = closes[oldest]
        if current_base <= ZERO or previous_base <= ZERO:
            return None
        current = (context.price - current_base) / current_base
        previous = (closes[index - self.params.gap] - previous_base) / previous_base
        return current, previous

    def _direction(self, momentum: Decimal, acceleration: Decimal) -> SignalDirection | None:
        """Which way an accelerating move points, or ``None`` if it does not qualify."""
        if acceleration >= self.params.entry_acceleration and momentum >= self.params.min_momentum:
            return SignalDirection.LONG
        if (
            acceleration <= -self.params.entry_acceleration
            and momentum <= -self.params.min_momentum
        ):
            return SignalDirection.SHORT
        return None

    def _conviction(self, acceleration: Decimal) -> Decimal:
        """Sharper acceleration reads as a stronger case.

        A move that just clears the trigger is far weaker evidence than one twice as
        steep, and collapsing both to 1.0 leaves the orchestrator no way to tell them apart.
        """
        excess = acceleration - self.params.entry_acceleration
        if excess <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5"), ONE)


__all__ = ["MomentumAccelerationParams", "MomentumAccelerationStrategy"]
