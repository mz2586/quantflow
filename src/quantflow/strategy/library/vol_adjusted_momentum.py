"""Volatility-adjusted momentum.

Rank momentum by how large the move is *relative to the market's own volatility*, then
require that ratio to clear a threshold. `momentum_roc` uses raw percentage change, which
means the same +3% qualifies as a strong signal on a placid market and as noise on a
violent one. Dividing by ATR makes the threshold mean the same thing everywhere, and is why
this one abstains during high-volatility chop that `momentum_roc` trades.

Exit when the volatility-adjusted move decays back through a lower threshold.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class VolAdjustedMomentumParams(StrategyParams):
    """Parameters for :class:`VolAdjustedMomentumStrategy`."""

    lookback: int = Field(default=20, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    #: Move over the lookback, in ATRs, required to enter.
    entry_atr_move: Decimal = Field(default=Decimal("2.0"), gt=0, le=20)
    #: Exit once the move decays below this many ATRs.
    exit_atr_move: Decimal = Field(default=Decimal("0.5"), ge=0, le=20)
    #: Move at which conviction saturates, in ATRs beyond the entry threshold.
    conviction_span: Decimal = Field(default=Decimal("2.0"), gt=0, le=20)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("5.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_moves(self) -> Self:
        if self.exit_atr_move >= self.entry_atr_move:
            raise ValueError(
                f"exit_atr_move ({self.exit_atr_move}) must be below entry_atr_move "
                f"({self.entry_atr_move}), or the strategy would exit on the bar it entered"
            )
        return self


@register_strategy
class VolAdjustedMomentumStrategy(Strategy):
    """Enter when the lookback move is large relative to volatility."""

    strategy_id = "vol_adjusted_momentum"
    description = "Momentum measured in ATRs rather than percent, so the bar is regime-neutral"
    params_model = VolAdjustedMomentumParams

    params: VolAdjustedMomentumParams

    @property
    def warmup_bars(self) -> int:
        """The lookback plus the ATR window."""
        return self.params.lookback + self.params.atr_period + 1

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a volatility-adjusted momentum signal."""
        from quantflow.strategy.indicators import atr

        index = context.index
        if index < self.params.lookback:
            return context.hold("lookback warming up", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        if volatility is None or volatility <= ZERO:
            return context.hold("atr unavailable", self.strategy_id)

        past = context.closes[index - self.params.lookback]
        move = (context.price - past) / volatility

        if context.is_long:
            return (
                exit_signal(context, self.strategy_id, "momentum decayed")
                if move <= self.params.exit_atr_move
                else context.hold("holding long, momentum intact", self.strategy_id)
            )
        if context.is_short:
            return (
                exit_signal(context, self.strategy_id, "momentum decayed")
                if move >= -self.params.exit_atr_move
                else context.hold("holding short, momentum intact", self.strategy_id)
            )

        if move >= self.params.entry_atr_move:
            direction = SignalDirection.LONG
        elif move <= -self.params.entry_atr_move:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold(
                f"move {move:.2f} ATR below {self.params.entry_atr_move}", self.strategy_id
            )

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"{abs(move):.2f} ATR move over {self.params.lookback} bars",
        )
        return replace_conviction(signal, self._conviction(abs(move)))

    def _conviction(self, move: Decimal) -> Decimal:
        """Larger volatility-adjusted moves read as stronger."""
        excess = move - self.params.entry_atr_move
        if excess <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5"), ONE)


__all__ = ["VolAdjustedMomentumParams", "VolAdjustedMomentumStrategy"]
