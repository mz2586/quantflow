"""Parabolic SAR — a stop that tightens the longer a trend runs.

Every other exit in the library gives a position a *constant* amount of room: a fixed ATR
multiple, a fixed channel, a fixed moving average. That is the wrong shape for a trend,
because the risk profile of a trade changes as it works. Early on, the position has no
cushion and needs room to survive noise; after a long run it is carrying open profit that
a single reversal bar can take back, and the sensible thing is to give it progressively
less. The acceleration factor encodes exactly that: each new extreme pulls the stop in
faster, so the stop converges on price at a rate set by how much the trend has already
delivered.

The cost is that SAR is always in the market and therefore reverses on every stop-out,
which in a range is a machine for losing money one whipsaw at a time. Included on those
terms: it exists in the library as the *time-varying* stop against which the fixed-distance
exits can be measured, and the leaderboard is the place to find out which shape wins.

Shorts are the same mechanism mirrored — the stop sits above price instead of below — so
they are supported, gated by ``allow_short`` like the rest of the library.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, parabolic_sar
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy

#: The SAR recursion is seeded from the first two bars and needs a run of bars before its
#: level reflects the trend rather than that seed.
SAR_SETTLE_BARS = 25


class ParabolicSarParams(StrategyParams):
    """Parameters for :class:`ParabolicSarStrategy`."""

    #: Wilder's published defaults. Kept as defaults so the leaderboard ranks the idea.
    step: Decimal = Field(default=Decimal("0.02"), gt=0, le=1)
    maximum: Decimal = Field(default=Decimal("0.2"), gt=0, le=1)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    #: Gap between price and the flipped stop, in ATR units, at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_acceleration(self) -> Self:
        if self.maximum < self.step:
            raise ValueError(
                f"maximum ({self.maximum}) must be at least step ({self.step}); a cap below "
                "the step would freeze the acceleration at its first value"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class ParabolicSarStrategy(Strategy):
    """Enters on each SAR reversal and exits when the stop flips against the position."""

    strategy_id = "parabolic_sar"
    description = "Wilder's parabolic stop-and-reverse with an accelerating trailing stop"
    params_model = ParabolicSarParams

    params: ParabolicSarParams

    @property
    def warmup_bars(self) -> int:
        """Enough bars for the recursion to have forgotten its two-bar seed."""
        return max(SAR_SETTLE_BARS, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit an entry on a SAR reversal, or close a position it has reversed against."""
        index = context.index
        if index < 1:
            return context.hold("no prior bar", self.strategy_id)

        sar, direction = parabolic_sar(context.candles, self.params.step, self.params.maximum)
        current = direction[index]
        previous = direction[index - 1]
        stop_level = sar[index]
        if current is None or previous is None or stop_level is None:
            return context.hold("sar warming up", self.strategy_id)

        flipped_up = previous < ZERO < current
        flipped_down = current < ZERO < previous

        if context.is_long:
            if flipped_down:
                return exit_signal(context, self.strategy_id, f"sar reversed to {stop_level}")
            return context.hold("holding long, sar below price", self.strategy_id)

        if context.is_short:
            if flipped_up:
                return exit_signal(context, self.strategy_id, f"sar reversed to {stop_level}")
            return context.hold("holding short, sar above price", self.strategy_id)

        if not flipped_up and not flipped_down:
            return context.hold("sar has not reversed", self.strategy_id)
        if flipped_down and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        long = flipped_up
        volatility = atr(context.candles, self.params.atr_period)[index]
        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if long else SignalDirection.SHORT,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"sar reversed {'up' if long else 'down'} to {stop_level}",
        )
        gap = (context.price - stop_level) if long else (stop_level - context.price)
        return replace_conviction(signal, self._conviction(gap, volatility))

    def _conviction(self, gap: Decimal, volatility: Decimal | None) -> Decimal:
        """Wider separation from the reversed stop means a more decisive turn.

        On a reversal the stop jumps to the prior leg's extreme, so the gap between price
        and the new stop measures how far the market travelled to break that extreme — a
        reversal that barely clears it is the weak case, and conviction should say so
        rather than reporting 1.0 for every flip.
        """
        if volatility is None or volatility <= ZERO or gap <= ZERO:
            return Decimal("0.35")
        scaled = gap / (volatility * self.params.conviction_span)
        return min(Decimal("0.35") + min(scaled, ONE) * Decimal("0.65"), ONE)


__all__ = ["ParabolicSarParams", "ParabolicSarStrategy"]
