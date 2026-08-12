"""VWAP momentum.

Trade *with* a decisive break away from rolling VWAP, on the reading that price sustaining
itself above where the volume traded means buyers are paying up rather than reverting.

This is the deliberate opposite of `vwap_reversion`, on the same indicator: one fades the
stretch, the other follows it. Keeping both is not duplication — it is the trend-versus-
reversion question posed on a volume-weighted mean, and the orchestrator scores them
head-to-head on the same bar. The entry conditions are not mirror images, though: this one
additionally requires VWAP itself to be rising, so it cannot fire on a break upward through
a VWAP that is still falling.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class VwapMomentumParams(StrategyParams):
    """Parameters for :class:`VwapMomentumStrategy`."""

    vwap_period: int = Field(default=20, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    #: Distance beyond VWAP, in ATRs, that counts as a decisive break.
    breakout_atr_distance: Decimal = Field(default=Decimal("0.75"), gt=0, le=10)
    #: Bars over which VWAP's own slope is measured.
    slope_lookback: int = Field(default=5, ge=1, le=100)
    #: Distance at which conviction saturates, in ATRs beyond the trigger.
    conviction_span: Decimal = Field(default=Decimal("1.5"), gt=0, le=10)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False


@register_strategy
class VwapMomentumStrategy(Strategy):
    """Follow decisive breaks away from a rising or falling VWAP."""

    strategy_id = "vwap_momentum"
    description = "Follows decisive breaks away from VWAP, requiring VWAP itself to agree"
    params_model = VwapMomentumParams

    params: VwapMomentumParams

    @property
    def warmup_bars(self) -> int:
        """VWAP window plus the slope lookback."""
        return max(self.params.vwap_period + self.params.slope_lookback, self.params.atr_period) + 1

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a VWAP momentum signal."""
        from quantflow.strategy.indicators import atr, rolling_vwap

        index = context.index
        series = rolling_vwap(context.candles, self.params.vwap_period)
        current = series[index]
        earlier = (
            series[index - self.params.slope_lookback]
            if index >= self.params.slope_lookback
            else None
        )
        volatility = atr(context.candles, self.params.atr_period)[index]
        if current is None or earlier is None:
            return context.hold("vwap warming up", self.strategy_id)
        if volatility is None or volatility <= ZERO:
            return context.hold("atr unavailable", self.strategy_id)

        distance = (context.price - current) / volatility
        rising = current > earlier
        falling = current < earlier

        if context.is_long:
            return (
                exit_signal(context, self.strategy_id, "price fell back through vwap")
                if context.price < current
                else context.hold("holding long, above vwap", self.strategy_id)
            )
        if context.is_short:
            return (
                exit_signal(context, self.strategy_id, "price rose back through vwap")
                if context.price > current
                else context.hold("holding short, below vwap", self.strategy_id)
            )

        if distance >= self.params.breakout_atr_distance and rising:
            direction = SignalDirection.LONG
        elif distance <= -self.params.breakout_atr_distance and falling:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold("no confirmed break from vwap", self.strategy_id)

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"price {abs(distance):.2f} ATR beyond a {'rising' if rising else 'falling'} vwap",
        )
        return replace_conviction(signal, self._conviction(abs(distance)))

    def _conviction(self, distance: Decimal) -> Decimal:
        """A more decisive break reads as stronger."""
        excess = distance - self.params.breakout_atr_distance
        if excess <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5"), ONE)


__all__ = ["VwapMomentumParams", "VwapMomentumStrategy"]
