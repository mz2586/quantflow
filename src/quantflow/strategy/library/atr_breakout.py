"""ATR breakout — a threshold denominated in volatility rather than in price.

The trigger is "price has travelled more than N ATRs from its own reference level", not
"price exceeded a fixed extreme". That distinction matters because the same 2% move is a
routine hour in one market and a once-a-month event in another; a threshold in ATR units
means the same thing in both, and means the same thing in the same market before and after
volatility doubles.

Distinct from `donchian_breakout`, whose threshold is an N-bar extreme. A channel breakout
fires whenever the range has been narrow enough for price to reach its edge, so it fires
constantly in a quiet drift and almost never after a violent bar has stretched the channel.
This one moves its own bar with volatility: as the market gets noisier the required
displacement grows with it, which is exactly the adjustment a fixed channel cannot make.

Distinct from `atr_expansion`, which is a statement about ATR itself changing. Here ATR is
only the ruler — the signal is about *displacement of price*, measured in that unit.
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


class AtrBreakoutParams(StrategyParams):
    """Parameters for :class:`AtrBreakoutStrategy`."""

    #: Window whose moving average is the reference price the breakout is measured from.
    reference_period: int = Field(default=20, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    #: How many ATRs beyond the reference price a close must sit to count as a breakout.
    breakout_atr_multiple: Decimal = Field(default=Decimal("1.5"), gt=0, le=20)
    #: Exit once displacement has decayed back to this many ATRs — the move that justified
    #: the entry has stopped being one.
    exit_atr_multiple: Decimal = Field(default=Decimal("0.5"), ge=0, le=20)
    #: Displacement beyond the trigger at which conviction saturates, in ATRs.
    conviction_span: Decimal = Field(default=Decimal("1.5"), gt=0, le=20)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_multiples(self) -> Self:
        if self.exit_atr_multiple >= self.breakout_atr_multiple:
            raise ValueError(
                f"exit_atr_multiple ({self.exit_atr_multiple}) must be below "
                f"breakout_atr_multiple ({self.breakout_atr_multiple}), or the strategy "
                "would exit on the bar it entered"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class AtrBreakoutStrategy(Strategy):
    """Enter when price is displaced from its mean by a volatility-scaled threshold."""

    strategy_id = "atr_breakout"
    description = "Breakout threshold measured in ATR units, so it self-scales with volatility"
    params_model = AtrBreakoutParams

    params: AtrBreakoutParams

    @property
    def warmup_bars(self) -> int:
        """The reference average and the ATR that scales it."""
        return max(self.params.reference_period, self.params.atr_period) + 2

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a volatility-scaled breakout entry or a decay exit."""
        index = context.index
        reference = sma(context.closes, self.params.reference_period)[index]
        volatility = atr(context.candles, self.params.atr_period)[index]
        if reference is None:
            return context.hold("reference average warming up", self.strategy_id)
        if volatility is None or volatility <= ZERO:
            return context.hold("atr unavailable", self.strategy_id)

        # Signed displacement in ATR units: positive above the reference, negative below.
        displacement = (context.price - reference) / volatility

        if context.is_long:
            if displacement <= self.params.exit_atr_multiple:
                return exit_signal(
                    context,
                    self.strategy_id,
                    f"displacement decayed to {displacement:.2f} ATR",
                )
            return context.hold(
                f"holding, {displacement:.2f} ATR above reference", self.strategy_id
            )

        if context.is_short:
            if displacement >= -self.params.exit_atr_multiple:
                return exit_signal(
                    context,
                    self.strategy_id,
                    f"displacement decayed to {displacement:.2f} ATR",
                )
            return context.hold(
                f"holding, {displacement:.2f} ATR below reference", self.strategy_id
            )

        if displacement >= self.params.breakout_atr_multiple:
            direction = SignalDirection.LONG
        elif displacement <= -self.params.breakout_atr_multiple:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold(
                f"displacement {displacement:.2f} ATR inside ±{self.params.breakout_atr_multiple}",
                self.strategy_id,
            )

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"price {displacement:.2f} ATR from its {self.params.reference_period}-bar mean",
        )
        return replace_conviction(signal, self._conviction(abs(displacement)))

    def _conviction(self, displacement: Decimal) -> Decimal:
        """Displacement beyond the trigger, saturating at ``conviction_span`` ATRs."""
        excess = displacement - self.params.breakout_atr_multiple
        if excess <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5"), ONE)


__all__ = ["AtrBreakoutParams", "AtrBreakoutStrategy"]
