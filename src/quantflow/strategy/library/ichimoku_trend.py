"""Ichimoku — a trend trigger that must agree with an older, slower picture of the market.

The Tenkan/Kijun cross on its own is just another fast/slow crossover and behaves like one:
it fires constantly in a range. What makes Ichimoku a different idea from ``ema_cross`` is
the cloud, which is built from a *different* set of bars than the cross that it filters —
the spans in force today were computed 26 bars ago and have not moved since. So the filter
cannot be dragged around by the same recent prices that produced the trigger, which is
exactly the failure mode of a fast-signal-plus-slow-filter built from one window.

The look-ahead trap is the whole difficulty of this indicator. The cloud is drawn 26 bars
*ahead* of the bars that produced it, so on a chart the cloud sitting under today's price
belongs 26 bars in the past. Read it the way it is drawn — take the span value stored at
today's index — and the strategy is filtering today's cross with prices that had not
printed when the cross happened. :func:`quantflow.strategy.indicators.ichimoku` returns the
spans already aligned to the bar they govern precisely so this module cannot make that
mistake.

Shorts are symmetric: price below both spans with a bearish cross is the exact mirror of
the long case, and the cloud is no less informative below price than above it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, crossed_above, crossed_below, ichimoku
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class IchimokuTrendParams(StrategyParams):
    """Parameters for :class:`IchimokuTrendStrategy`."""

    tenkan_period: int = Field(default=9, ge=2, le=200)
    kijun_period: int = Field(default=26, ge=3, le=400)
    senkou_b_period: int = Field(default=52, ge=4, le=800)
    #: Bars the cloud is projected forward. Also the age of the cloud in force today.
    displacement: int = Field(default=26, ge=1, le=400)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if not self.tenkan_period < self.kijun_period < self.senkou_b_period:
            raise ValueError(
                f"periods must increase: tenkan ({self.tenkan_period}) < kijun "
                f"({self.kijun_period}) < senkou_b ({self.senkou_b_period})"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class IchimokuTrendStrategy(Strategy):
    """Tenkan/Kijun crossover, taken only on the side of the cloud price is already on."""

    strategy_id = "ichimoku_trend"
    description = "Tenkan/Kijun cross filtered by price against the displaced Ichimoku cloud"
    params_model = IchimokuTrendParams

    params: IchimokuTrendParams

    @property
    def warmup_bars(self) -> int:
        """The oldest input is Senkou B, which is then read ``displacement`` bars later."""
        return max(
            self.params.senkou_b_period + self.params.displacement + 1,
            self.params.atr_period + 1,
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a cloud-confirmed Tenkan/Kijun signal."""
        index = context.index
        tenkan, kijun, span_a, span_b = ichimoku(
            context.candles,
            self.params.tenkan_period,
            self.params.kijun_period,
            self.params.senkou_b_period,
            self.params.displacement,
        )
        above = span_a[index]
        below = span_b[index]
        if tenkan[index] is None or kijun[index] is None or above is None or below is None:
            return context.hold("ichimoku warming up", self.strategy_id)

        cloud_top = max(above, below)
        cloud_bottom = min(above, below)
        bullish = crossed_above(tenkan, kijun, index)
        bearish = crossed_below(tenkan, kijun, index)

        if context.is_long:
            if bearish:
                return exit_signal(context, self.strategy_id, "tenkan crossed below kijun")
            if context.price < cloud_bottom:
                return exit_signal(context, self.strategy_id, "price fell through the cloud")
            return context.hold("holding long above the cloud", self.strategy_id)

        if context.is_short:
            if bullish:
                return exit_signal(context, self.strategy_id, "tenkan crossed above kijun")
            if context.price > cloud_top:
                return exit_signal(context, self.strategy_id, "price rose through the cloud")
            return context.hold("holding short below the cloud", self.strategy_id)

        if not bullish and not bearish:
            return context.hold("no tenkan/kijun cross", self.strategy_id)

        long = bullish
        if long and context.price <= cloud_top:
            return context.hold("bullish cross but price is not above the cloud", self.strategy_id)
        if not long and context.price >= cloud_bottom:
            return context.hold("bearish cross but price is not below the cloud", self.strategy_id)
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
                f"tenkan crossed {'above' if long else 'below'} kijun with price "
                f"{'above' if long else 'below'} the cloud"
            ),
        )
        clearance = (context.price - cloud_top) if long else (cloud_bottom - context.price)
        return replace_conviction(
            signal, self._conviction(clearance, cloud_top - cloud_bottom, volatility)
        )

    def _conviction(
        self, clearance: Decimal, thickness: Decimal, volatility: Decimal | None
    ) -> Decimal:
        """Clearance from the cloud and the cloud's own thickness both raise conviction.

        Clearance says how unambiguous the filter was; thickness says how much disagreement
        the market would have to overcome to invalidate it — a paper-thin cloud is a level
        price wanders through without meaning anything. Both are scaled by ATR so the
        number is comparable across symbols.
        """
        if volatility is None or volatility <= ZERO:
            return Decimal("0.4")
        clearance_part = min(max(clearance, ZERO) / volatility, ONE)
        thickness_part = min(max(thickness, ZERO) / volatility, ONE)
        return min(
            Decimal("0.4") + clearance_part * Decimal("0.4") + thickness_part * Decimal("0.2"),
            ONE,
        )


__all__ = ["IchimokuTrendParams", "IchimokuTrendStrategy"]
