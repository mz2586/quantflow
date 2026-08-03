"""Bollinger band mean reversion.

Buys a close below the lower band and exits back at the middle band. The trend filter is
what stops this being a knife-catcher: in a sustained downtrend price rides the lower band
for weeks, and a strategy that buys every touch simply averages into the decline. Entries
are therefore suppressed when price is below a long moving average.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, bollinger_bands, sma
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class BollingerReversionParams(StrategyParams):
    """Parameters for :class:`BollingerReversionStrategy`."""

    period: int = Field(default=20, ge=5, le=200)
    deviations: Decimal = Field(default=Decimal("2.0"), gt=0, le=5)
    #: Long moving average used only as a regime filter, never as an entry trigger.
    trend_period: int = Field(default=200, ge=20, le=1000)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    #: Skip entries when price is below the trend filter.
    require_uptrend: bool = True

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if self.trend_period <= self.period:
            raise ValueError("trend_period must exceed period for the filter to mean anything")
        return self


@register_strategy
class BollingerReversionStrategy(Strategy):
    """Buy the lower Bollinger band in an uptrend, exit at the middle band."""

    strategy_id = "bollinger_reversion"
    description = "Mean reversion from the lower Bollinger band, filtered by a long trend MA"
    params_model = BollingerReversionParams

    params: BollingerReversionParams

    @property
    def warmup_bars(self) -> int:
        """The trend filter is the binding constraint."""
        return max(self.params.trend_period + 1, self.params.period * 2, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Buy a close below the lower band; exit at the middle band."""
        index = context.index
        closes = context.closes
        # Only the middle and lower bands matter here: this strategy never trades the
        # upside, so binding the upper band would be dead weight.
        _, middle, lower = bollinger_bands(closes, self.params.period, self.params.deviations)
        middle_value, lower_value = middle[index], lower[index]
        if middle_value is None or lower_value is None:
            return context.hold("bands warming up", self.strategy_id)

        if context.is_long:
            if context.price >= middle_value:
                return exit_signal(context, self.strategy_id, "reverted to the middle band")
            return context.hold("holding for mean reversion", self.strategy_id)

        if context.price >= lower_value:
            return context.hold("price above the lower band", self.strategy_id)

        if self.params.require_uptrend:
            trend = sma(closes, self.params.trend_period)[index]
            if trend is None:
                return context.hold("trend filter warming up", self.strategy_id)
            if context.price < trend:
                # Riding the lower band in a downtrend: the exact case where buying every
                # touch averages into a decline.
                return context.hold("below the trend filter", self.strategy_id)

        return entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            "closed below the lower band while above the trend filter",
        )
