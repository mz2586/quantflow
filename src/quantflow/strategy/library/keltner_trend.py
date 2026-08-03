"""Keltner channel trend following.

A close outside an ATR channel around an EMA. Distinct from the Bollinger family because
the channel is built from *true range* rather than from close-to-close standard deviation:
it therefore widens on gaps and long wicks, which standard deviation of closes ignores
entirely. In crypto, where a violent wick and a quiet drift can produce the same close-to-
close deviation, that difference is not cosmetic.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, ema
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class KeltnerTrendParams(StrategyParams):
    """Parameters for :class:`KeltnerTrendStrategy`."""

    ema_period: int = Field(default=20, ge=5, le=300)
    atr_period: int = Field(default=20, ge=2, le=200)
    #: Channel half-width in ATR multiples.
    channel_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("5.0"), gt=0, le=20)
    allow_short: bool = False


@register_strategy
class KeltnerTrendStrategy(Strategy):
    """Enter on a close outside the Keltner channel, exit back at the centre line."""

    strategy_id = "keltner_trend"
    description = "Keltner channel breakout: EMA centre with an ATR-width channel"
    params_model = KeltnerTrendParams

    params: KeltnerTrendParams

    @property
    def warmup_bars(self) -> int:
        """The EMA needs roughly twice its period before it stops tracking its seed."""
        return max(self.params.ema_period * 2, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a channel breakout signal."""
        index = context.index
        centre = ema(context.closes, self.params.ema_period)[index]
        volatility = atr(context.candles, self.params.atr_period)[index]
        if centre is None or volatility is None:
            return context.hold("channel warming up", self.strategy_id)

        width = volatility * self.params.channel_multiple
        upper, lower = centre + width, centre - width

        if context.is_long:
            if context.price < centre:
                return exit_signal(context, self.strategy_id, "closed back below the centre line")
            return context.hold("holding long", self.strategy_id)
        if context.is_short:
            if context.price > centre:
                return exit_signal(context, self.strategy_id, "closed back above the centre line")
            return context.hold("holding short", self.strategy_id)

        if context.price > upper:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.LONG,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "closed above the Keltner channel",
            )
        if context.price < lower and self.params.allow_short:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.SHORT,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "closed below the Keltner channel",
            )
        return context.hold("inside the channel", self.strategy_id)
