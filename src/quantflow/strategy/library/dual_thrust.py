"""Dual Thrust — asymmetric range breakout.

Builds a range from the last N bars' high-low and close spread, then trades a break of
the current bar's open plus a fraction of that range. Its distinguishing feature is that
the upside and downside triggers can be set asymmetrically, which encodes the view that
breakouts up and breakouts down are not mirror images of one another.

Unlike `donchian_breakout`, the trigger is anchored to the *current bar's open* rather
than to a historical extreme, so the level moves with the market rather than sitting at a
price the market may never revisit.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class DualThrustParams(StrategyParams):
    """Parameters for :class:`DualThrustStrategy`."""

    lookback: int = Field(default=24, ge=2, le=500)
    #: Upside trigger as a fraction of the computed range.
    upper_coefficient: Decimal = Field(default=Decimal("0.5"), gt=0, le=5)
    #: Downside trigger. Separate from the upside on purpose.
    lower_coefficient: Decimal = Field(default=Decimal("0.5"), gt=0, le=5)
    #: Bars to hold before the position is closed regardless of price.
    max_holding_bars: int = Field(default=48, ge=1, le=2000)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False


@register_strategy
class DualThrustStrategy(Strategy):
    """Asymmetric range breakout anchored to the current bar's open."""

    strategy_id = "dual_thrust"
    description = "Dual Thrust: asymmetric range breakout from the current bar's open"
    params_model = DualThrustParams

    params: DualThrustParams

    @property
    def warmup_bars(self) -> int:
        """One full lookback window plus the ATR warm-up."""
        return max(self.params.lookback + 1, self.params.atr_period + 1)

    def generate(self, context: StrategyContext) -> Signal:
        """Trade a break of the Dual Thrust bands."""
        index = context.index

        if context.has_position:
            held = self._bars_held(context)
            if held is not None and held >= self.params.max_holding_bars:
                return exit_signal(
                    context, self.strategy_id, f"held for {self.params.max_holding_bars} bars"
                )
            return context.hold(f"held {held} bars", self.strategy_id)

        span = self._range(context, index)
        if span is None or span <= ZERO:
            return context.hold("no valid range", self.strategy_id)

        open_price = context.candle.open
        upper = open_price + span * self.params.upper_coefficient
        lower = open_price - span * self.params.lower_coefficient
        volatility = atr(context.candles, self.params.atr_period)[index]

        if context.price > upper:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.LONG,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "closed above the upper thrust band",
            )
        if context.price < lower and self.params.allow_short:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.SHORT,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "closed below the lower thrust band",
            )
        return context.hold("inside the thrust bands", self.strategy_id)

    @staticmethod
    def _bars_held(context: StrategyContext) -> int | None:
        """Bars elapsed since the position opened.

        Derived from the position's own timestamp rather than from a counter on the
        strategy: `generate` is required to be pure, and instance state would make the
        same context produce different signals depending on what ran before it — which
        is exactly what makes a backtest irreproducible.
        """
        position = context.position
        if position is None or position.opened_at is None:
            return None
        step = context.timeframe.seconds
        if step <= 0:
            return None
        return int((context.now - position.opened_at).total_seconds() // step)

    def _range(self, context: StrategyContext, index: int) -> Decimal | None:
        """The Dual Thrust range over the bars preceding ``index``.

        ``max(HH - LC, HC - LL)`` where HH/LL are the highest high and lowest low and
        HC/LC the highest and lowest close. The window ends on the previous bar so the
        decision bar never contributes to its own trigger level.
        """
        start = index - self.params.lookback
        if start < 0:
            return None
        window = context.candles[start:index]
        if not window:
            return None

        highest_high = max(candle.high for candle in window)
        lowest_low = min(candle.low for candle in window)
        highest_close = max(candle.close for candle in window)
        lowest_close = min(candle.close for candle in window)
        return max(highest_high - lowest_close, highest_close - lowest_low)
