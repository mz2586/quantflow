"""RSI mean reversion — a counter-trend reference strategy.

Buys oversold conditions and exits on a return to the midline. Deliberately gated by a
long-term trend filter: mean reversion bought blindly in a downtrend is the classic way to
turn a smooth equity curve into a single catastrophic loss, because "oversold" has no floor
in a market that is genuinely repricing.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, rsi, sma
from quantflow.strategy.registry import register_strategy


class RsiReversionParams(StrategyParams):
    """Parameters for :class:`RsiReversionStrategy`."""

    rsi_period: int = Field(default=14, ge=2, le=100)
    oversold: Decimal = Field(default=Decimal("30"), gt=0, lt=100)
    overbought: Decimal = Field(default=Decimal("70"), gt=0, lt=100)
    exit_level: Decimal = Field(default=Decimal("50"), gt=0, lt=100)
    #: Length of the trend filter. Entries are only taken in the direction of this SMA.
    trend_period: int = Field(default=200, ge=10, le=1000)
    use_trend_filter: bool = True
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_levels(self) -> Self:
        if self.oversold >= self.overbought:
            raise ValueError("oversold must be below overbought")
        if not (self.oversold < self.exit_level < self.overbought):
            raise ValueError("exit_level must sit between oversold and overbought")
        return self


@register_strategy
class RsiReversionStrategy(Strategy):
    """Mean reversion on RSI extremes, filtered by a long-term trend."""

    strategy_id = "rsi_reversion"
    description = "RSI mean reversion with a long-term trend filter and ATR stop"
    params_model = RsiReversionParams

    params: RsiReversionParams

    @property
    def warmup_bars(self) -> int:
        """Longest of the trend, RSI and ATR windows."""
        needed = max(self.params.rsi_period + 1, self.params.atr_period + 1)
        if self.params.use_trend_filter:
            needed = max(needed, self.params.trend_period)
        return needed

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit an entry on an RSI extreme, or an exit at the midline."""
        closes = context.closes
        index = context.index

        strength = rsi(closes, self.params.rsi_period)[index]
        if strength is None:
            return context.hold("rsi warming up", self.strategy_id)

        if context.is_long:
            if strength >= self.params.exit_level:
                return self._exit(context, f"RSI {strength:.1f} recovered to the midline")
            return context.hold(f"holding long, RSI {strength:.1f}", self.strategy_id)

        if context.is_short:
            if strength <= self.params.exit_level:
                return self._exit(context, f"RSI {strength:.1f} fell to the midline")
            return context.hold(f"holding short, RSI {strength:.1f}", self.strategy_id)

        trend_ok_long, trend_ok_short = self._trend_gate(context)

        if strength <= self.params.oversold:
            if not trend_ok_long:
                return context.hold(
                    f"RSI {strength:.1f} oversold but price is below the trend filter",
                    self.strategy_id,
                )
            return self._entry(
                context, SignalDirection.LONG, strength, f"RSI {strength:.1f} oversold"
            )

        if strength >= self.params.overbought and self.params.allow_short:
            if not trend_ok_short:
                return context.hold(
                    f"RSI {strength:.1f} overbought but price is above the trend filter",
                    self.strategy_id,
                )
            return self._entry(
                context, SignalDirection.SHORT, strength, f"RSI {strength:.1f} overbought"
            )

        return context.hold(f"RSI {strength:.1f} is neutral", self.strategy_id)

    def _trend_gate(self, context: StrategyContext) -> tuple[bool, bool]:
        """Whether long and short entries are permitted by the trend filter."""
        if not self.params.use_trend_filter:
            return True, True
        trend = sma(context.closes, self.params.trend_period)[context.index]
        if trend is None:
            return False, False
        return context.price > trend, context.price < trend

    def _conviction(self, strength: Decimal, direction: SignalDirection) -> Decimal:
        """Scale conviction with how far past the threshold RSI has travelled.

        Advisory only: the risk engine may scale size *within* its limits by this, never
        beyond them.
        """
        if direction is SignalDirection.LONG:
            span = self.params.oversold
            if span <= ZERO:
                return Decimal("1")
            depth = (self.params.oversold - strength) / span
        else:
            span = Decimal("100") - self.params.overbought
            if span <= ZERO:
                return Decimal("1")
            depth = (strength - self.params.overbought) / span
        return min(Decimal("1"), max(Decimal("0.25"), Decimal("0.5") + depth))

    def _entry(
        self,
        context: StrategyContext,
        direction: SignalDirection,
        strength: Decimal,
        reason: str,
    ) -> Signal:
        volatility = atr(context.candles, self.params.atr_period)[context.index]
        stop = None
        if volatility is not None and volatility > ZERO:
            distance = volatility * self.params.atr_stop_multiple
            stop = (
                context.price - distance
                if direction is SignalDirection.LONG
                else context.price + distance
            )
            if stop <= ZERO:
                stop = None

        return Signal(
            symbol=context.symbol,
            direction=direction,
            timestamp=context.now,
            strategy_id=self.strategy_id,
            conviction=self._conviction(strength, direction),
            reference_price=context.price,
            stop_loss_price=stop,
            reason=reason,
        )

    def _exit(self, context: StrategyContext, reason: str) -> Signal:
        return Signal(
            symbol=context.symbol,
            direction=SignalDirection.CLOSE,
            timestamp=context.now,
            strategy_id=self.strategy_id,
            reference_price=context.price,
            reason=reason,
        )
