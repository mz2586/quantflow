"""Volume/price divergence — a new extreme that fewer and fewer people are participating in.

A new high made on less volume than the high before it says the move is running on
inventory rather than on demand: the buyers who were willing to pay up have already paid up.
That is a statement about *participation*, and it is the one thing price alone cannot say.
The trade is against the move — long a new low nobody is selling into, short a new high
nobody is buying.

The comparison is deliberately against the volume of the **bar that made the previous
extreme**, not against a rolling average. An average is dragged around by the quiet bars in
between, so a marginally-weaker peak can still clear it; comparing peak to peak asks the
question a chartist actually asks, "was there less behind this high than the last one". The
average is retained as a second, weaker condition so a pair of equally thin peaks in a dead
market does not count as divergence.

This is the mirror image of `volume_breakout`, which requires the breakout bar to trade
heavily and buys it. Given the same new high, the two take opposite sides — which makes the
pair a direct test of whether volume confirmation is informative in this market at all.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.market import Candle
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, sma
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class VolumePriceDivergenceParams(StrategyParams):
    """Parameters for :class:`VolumePriceDivergenceStrategy`."""

    #: Window whose extreme the current bar must exceed.
    lookback: int = Field(default=20, ge=3, le=500)
    #: The new extreme's volume must be at most this fraction of the prior extreme's.
    volume_ratio: Decimal = Field(default=Decimal("0.7"), gt=0, le=1)
    #: Moving average at which the reversion trade is considered complete.
    exit_period: int = Field(default=20, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.5"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_multiples(self) -> Self:
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class VolumePriceDivergenceStrategy(Strategy):
    """Fade a new price extreme that volume has failed to confirm."""

    strategy_id = "volume_price_divergence"
    description = "Fades a new price extreme made on materially lighter volume"
    params_model = VolumePriceDivergenceParams

    params: VolumePriceDivergenceParams

    @property
    def warmup_bars(self) -> int:
        """The comparison window sits entirely before the decision bar."""
        return max(
            self.params.lookback + 2, self.params.exit_period + 1, self.params.atr_period + 1
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a divergence fade or a mean-reached exit."""
        index = context.index
        if context.has_position:
            return self._manage(context)

        start = index - self.params.lookback
        if start < 0:
            return context.hold("comparison window warming up", self.strategy_id)
        window = context.candles[start:index]
        if not window:
            return context.hold("comparison window warming up", self.strategy_id)

        volume = context.candle.volume
        if volume <= ZERO:
            # A zero-volume bar is a data artefact far more often than it is a real
            # extreme made on no participation. Refusing to trade it is the honest read.
            return context.hold("current bar has no volume", self.strategy_id)

        average_volume = sum((candle.volume for candle in window), ZERO) / Decimal(len(window))
        if average_volume <= ZERO:
            return context.hold("no volume baseline", self.strategy_id)

        low_extreme = _extreme(window, high_side=False)
        if context.price < low_extreme.price and self._diverges(
            volume, low_extreme.volume, average_volume
        ):
            return self._entry(context, SignalDirection.LONG, volume, low_extreme.volume, "low")

        high_extreme = _extreme(window, high_side=True)
        if context.price > high_extreme.price and self._diverges(
            volume, high_extreme.volume, average_volume
        ):
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            return self._entry(context, SignalDirection.SHORT, volume, high_extreme.volume, "high")

        return context.hold("no unconfirmed extreme", self.strategy_id)

    def _manage(self, context: StrategyContext) -> Signal:
        """Close once price has reverted to the mean the extreme departed from."""
        average = sma(context.closes, self.params.exit_period)[context.index]
        if average is None:
            return context.hold("exit average warming up", self.strategy_id)
        if context.is_long and context.price >= average:
            return exit_signal(context, self.strategy_id, "price reverted to the mean")
        if context.is_short and context.price <= average:
            return exit_signal(context, self.strategy_id, "price reverted to the mean")
        return context.hold("holding, waiting for the mean", self.strategy_id)

    def _diverges(self, volume: Decimal, extreme_volume: Decimal, average_volume: Decimal) -> bool:
        """Whether the new extreme traded materially lighter than the one it exceeded."""
        if extreme_volume <= ZERO:
            # Nothing to compare against; the prior extreme carries no information.
            return False
        return volume <= extreme_volume * self.params.volume_ratio and volume < average_volume

    def _entry(
        self,
        context: StrategyContext,
        direction: SignalDirection,
        volume: Decimal,
        extreme_volume: Decimal,
        extreme: str,
    ) -> Signal:
        """Build the fade, with conviction from how much lighter the new extreme is."""
        shortfall = ONE - volume / extreme_volume if extreme_volume > ZERO else ZERO
        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[context.index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"new {self.params.lookback}-bar {extreme} on {shortfall:.0%} lighter volume",
        )
        return replace_conviction(signal, self._conviction(shortfall))

    def _conviction(self, shortfall: Decimal) -> Decimal:
        """A larger participation shortfall is a weaker move."""
        return min(Decimal("0.4") + min(max(shortfall, ZERO), ONE) * Decimal("0.6"), ONE)


class _Extreme:
    """The price of a window's extreme and the volume of the bar that made it."""

    __slots__ = ("price", "volume")

    def __init__(self, price: Decimal, volume: Decimal) -> None:
        self.price = price
        self.volume = volume


def _extreme(window: tuple[Candle, ...], *, high_side: bool) -> _Extreme:
    """The window's extreme close, and the volume traded on the most recent bar to reach it.

    The most recent occurrence rather than the first: when a level has been touched twice,
    the relevant comparison is against the last time the market was there.
    """
    best = window[0]
    for candle in window:
        reaches = candle.close >= best.close if high_side else candle.close <= best.close
        if reaches:
            best = candle
    return _Extreme(best.close, best.volume)


__all__ = ["VolumePriceDivergenceParams", "VolumePriceDivergenceStrategy"]
