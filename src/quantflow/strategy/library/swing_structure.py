"""Market structure — trade the sequence of swings, with no indicator in the decision.

Every other trend member in the library decides what the market is doing by transforming
price into something smoother and then reading the transform. That buys stability and pays
for it twice: an average lags, and it also *invents* a trend wherever the smoothing happens
to slope, including in a drift that no trader would call a trend at all.

This one reads the raw skeleton instead. A market that is going up makes each high above
the last high and each low above the last low; when it stops doing that, it has stopped
going up — regardless of where any average sits. That is the definition a discretionary
trader actually uses, it has no period parameter to overfit, and it produces far fewer
signals than a crossover because a structure break is a rarer event than a line crossing.

The trap is that a swing pivot is only a pivot in hindsight: a high is confirmed as a
swing high once enough bars have failed to exceed it. Reading pivots at the bar they
printed on hands the strategy a level several bars before the market could know it was one,
which backtests beautifully and cannot be traded.
:func:`quantflow.strategy.indicators.swing_pivots` publishes each pivot at its *confirmation*
bar for that reason, and this module never looks anywhere else.

Shorts are exactly symmetric — lower highs with lower lows — and are gated by
``allow_short``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import Series, atr, swing_pivots
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy

#: Two highs and two lows are the minimum needed to say "higher" or "lower" at all.
PIVOTS_FOR_STRUCTURE = 2
#: Swings needed before the structure is considered established, used to size the warm-up.
SWINGS_FOR_WARMUP = 6


class SwingStructureParams(StrategyParams):
    """Parameters for :class:`SwingStructureStrategy`."""

    #: Bars either side of a pivot. ``pivot_right`` is also the confirmation lag, so it is
    #: bought at the cost of entering that many bars after the swing printed.
    pivot_left: int = Field(default=3, ge=1, le=50)
    pivot_right: int = Field(default=3, ge=1, le=50)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    #: Swing expansion, in ATR units, at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_multiples(self) -> Self:
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class SwingStructureStrategy(Strategy):
    """Enters when the swing sequence confirms a trend and exits when it breaks."""

    strategy_id = "swing_structure"
    description = "Market structure from confirmed swing pivots: higher highs and higher lows"
    params_model = SwingStructureParams

    params: SwingStructureParams

    @property
    def warmup_bars(self) -> int:
        """Enough bars to have confirmed several swings, not merely the first two."""
        swing_cost = (self.params.pivot_left + self.params.pivot_right + 1) * SWINGS_FOR_WARMUP
        return max(swing_cost, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit an entry when structure confirms, or an exit when it breaks."""
        index = context.index
        if index < 1:
            return context.hold("no prior bar", self.strategy_id)

        highs, lows = swing_pivots(context.candles, self.params.pivot_left, self.params.pivot_right)
        confirmed_highs = _confirmed(highs, index)
        confirmed_lows = _confirmed(lows, index)
        structure = _structure(confirmed_highs, confirmed_lows)

        if context.is_long:
            if structure < ZERO:
                return exit_signal(context, self.strategy_id, "structure broke to lower swings")
            if confirmed_lows and context.price < confirmed_lows[-1]:
                return exit_signal(
                    context, self.strategy_id, f"closed below swing low {confirmed_lows[-1]}"
                )
            return context.hold("holding long, structure intact", self.strategy_id)

        if context.is_short:
            if structure > ZERO:
                return exit_signal(context, self.strategy_id, "structure broke to higher swings")
            if confirmed_highs and context.price > confirmed_highs[-1]:
                return exit_signal(
                    context, self.strategy_id, f"closed above swing high {confirmed_highs[-1]}"
                )
            return context.hold("holding short, structure intact", self.strategy_id)

        if structure == ZERO:
            return context.hold("no confirmed structure", self.strategy_id)

        # Act only on the bar that confirms a new swing. Between confirmations the same two
        # pivots are being re-read and nothing has been learned, so entering on any other
        # bar would open the same trade repeatedly through a single leg. Gating on the
        # confirmation bar rather than on a *change* of structure also matters: a trend
        # whose structure never breaks would otherwise be enterable exactly once ever, and
        # a stop-out would retire the strategy for the rest of the move.
        if highs[index] is None and lows[index] is None:
            return context.hold("no new swing confirmed on this bar", self.strategy_id)

        long = structure > ZERO
        if not long and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if long else SignalDirection.SHORT,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            (
                f"confirmed {'higher' if long else 'lower'} swing high "
                f"{confirmed_highs[-1]} and {'higher' if long else 'lower'} swing low "
                f"{confirmed_lows[-1]}"
            ),
        )
        expansion = (
            abs(confirmed_highs[-1] - confirmed_highs[-2])
            + abs(confirmed_lows[-1] - confirmed_lows[-2])
        ) / 2
        return replace_conviction(signal, self._conviction(expansion, volatility))

    def _conviction(self, expansion: Decimal, volatility: Decimal | None) -> Decimal:
        """A wide structural step is stronger evidence than a marginal one.

        Higher highs that clear the previous high by a tick are within the noise the pivot
        rule was meant to filter; ones that clear it by an ATR are the trend the strategy
        is trying to catch. Averaging the high and low steps means a break that is only
        half a break — one leg extended, the other barely — scores lower than one where
        both legs moved.
        """
        if volatility is None or volatility <= ZERO or expansion <= ZERO:
            return Decimal("0.35")
        scaled = expansion / (volatility * self.params.conviction_span)
        return min(Decimal("0.35") + min(scaled, ONE) * Decimal("0.65"), ONE)


def _confirmed(series: Series, index: int) -> tuple[Decimal, ...]:
    """Pivot prices confirmed at or before ``index``, oldest first."""
    if index < 0:
        return ()
    return tuple(value for value in series[: index + 1] if value is not None)


def _structure(highs: tuple[Decimal, ...], lows: tuple[Decimal, ...]) -> Decimal:
    """``+1`` for higher highs *and* higher lows, ``-1`` for the mirror, ``0`` otherwise.

    Requiring both legs is the whole point: higher highs with lower lows is an expanding
    range, which is the market saying it has no direction, and calling that an uptrend is
    how a structure rule degenerates into a breakout rule.
    """
    if len(highs) < PIVOTS_FOR_STRUCTURE or len(lows) < PIVOTS_FOR_STRUCTURE:
        return ZERO
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return ONE
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return -ONE
    return ZERO


__all__ = ["SwingStructureParams", "SwingStructureStrategy"]
