"""Keltner-channel mean reversion.

Buy a close below the lower Keltner band, exit back at the centre line. `keltner_trend`
uses the same channel to do the opposite — it treats a break *through* the band as trend
confirmation. Both readings are defensible and they disagree on exactly the same bar, which
is the point of having both: the orchestrator gets a genuine trend-versus-reversion contest
on identical inputs rather than two variants of one opinion.

Against `bollinger_reversion`: Keltner bands are ATR-width, Bollinger bands are
standard-deviation-width. ATR includes gaps and the full bar range, standard deviation sees
only closes, so the two disagree most in gappy, wick-heavy conditions.
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


class KeltnerReversionParams(StrategyParams):
    """Parameters for :class:`KeltnerReversionStrategy`."""

    ema_period: int = Field(default=20, ge=2, le=200)
    atr_period: int = Field(default=14, ge=2, le=100)
    #: Band width in ATRs either side of the centre line.
    band_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    #: Distance beyond the band, in ATRs, at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("1.0"), gt=0, le=10)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    allow_short: bool = False


@register_strategy
class KeltnerReversionStrategy(Strategy):
    """Fade closes outside the Keltner channel, exit at the centre line."""

    strategy_id = "keltner_reversion"
    description = "Fades closes outside the Keltner channel, exiting at the centre line"
    params_model = KeltnerReversionParams

    params: KeltnerReversionParams

    @property
    def warmup_bars(self) -> int:
        """The EMA needs several multiples of its period to shed its seed."""
        return max(self.params.ema_period * 3, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a Keltner reversion signal."""
        from quantflow.strategy.indicators import atr, ema

        index = context.index
        centre = ema(context.closes, self.params.ema_period)[index]
        volatility = atr(context.candles, self.params.atr_period)[index]
        if centre is None:
            return context.hold("centre line warming up", self.strategy_id)
        if volatility is None or volatility <= ZERO:
            return context.hold("atr unavailable", self.strategy_id)

        width = volatility * self.params.band_multiple
        upper = centre + width
        lower = centre - width
        price = context.price

        if context.is_long:
            return (
                exit_signal(context, self.strategy_id, "price returned to the centre line")
                if price >= centre
                else context.hold("holding long, below centre", self.strategy_id)
            )
        if context.is_short:
            return (
                exit_signal(context, self.strategy_id, "price returned to the centre line")
                if price <= centre
                else context.hold("holding short, above centre", self.strategy_id)
            )

        if price < lower:
            direction = SignalDirection.LONG
            excess = (lower - price) / volatility
        elif price > upper:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
            excess = (price - upper) / volatility
        else:
            return context.hold("price inside the channel", self.strategy_id)

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"price {excess:.2f} ATR outside the channel",
        )
        return replace_conviction(signal, self._conviction(excess))

    def _conviction(self, excess: Decimal) -> Decimal:
        """Further outside the band reads as a stronger reversion case."""
        return min(Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5"), ONE)


__all__ = ["KeltnerReversionParams", "KeltnerReversionStrategy"]
