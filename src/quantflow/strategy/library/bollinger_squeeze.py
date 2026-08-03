"""Volatility-squeeze breakout.

Bollinger bandwidth contracting to a multi-period low says the market has gone quiet;
quiet markets are how large moves begin. This trades the *expansion* rather than the
squeeze itself: a squeeze on its own has no direction, so nothing is entered until price
actually breaks out of the compressed range.

Distinct from `donchian_breakout`, which triggers on a price extreme regardless of whether
volatility was compressed first — this one refuses to trade a breakout that comes out of
an already-noisy market, where the "breakout" is usually just noise.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, bollinger_bands
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class BollingerSqueezeParams(StrategyParams):
    """Parameters for :class:`BollingerSqueezeStrategy`."""

    period: int = Field(default=20, ge=5, le=200)
    deviations: Decimal = Field(default=Decimal("2.0"), gt=0, le=5)
    #: Bandwidth must be the lowest in this many bars for the market to count as squeezed.
    squeeze_lookback: int = Field(default=120, ge=10, le=1000)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False


@register_strategy
class BollingerSqueezeStrategy(Strategy):
    """Trade the expansion out of a multi-period volatility squeeze."""

    strategy_id = "bollinger_squeeze"
    description = "Breakout from a Bollinger bandwidth squeeze into expanding volatility"
    params_model = BollingerSqueezeParams

    params: BollingerSqueezeParams

    @property
    def warmup_bars(self) -> int:
        """Enough history for the bandwidth series itself to have a lookback window."""
        return max(
            self.params.period + self.params.squeeze_lookback + 1, self.params.atr_period + 1
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Enter on a break out of a squeezed range."""
        index = context.index
        closes = context.closes
        upper, middle, lower = bollinger_bands(closes, self.params.period, self.params.deviations)

        if context.is_long:
            middle_value = middle[index]
            if middle_value is not None and context.price < middle_value:
                return exit_signal(context, self.strategy_id, "fell back through the middle band")
            return context.hold("holding the expansion", self.strategy_id)
        if context.is_short:
            middle_value = middle[index]
            if middle_value is not None and context.price > middle_value:
                return exit_signal(context, self.strategy_id, "rose back through the middle band")
            return context.hold("holding the expansion", self.strategy_id)

        # The squeeze is measured on the *previous* bar: the breakout bar itself has
        # already widened the bands, so measuring it here would never see a squeeze.
        if not self._was_squeezed(upper, middle, lower, index - 1):
            return context.hold("no prior volatility squeeze", self.strategy_id)

        upper_value, lower_value = upper[index], lower[index]
        if upper_value is None or lower_value is None:
            return context.hold("bands warming up", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        if context.price > upper_value:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.LONG,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "broke above the upper band out of a squeeze",
            )
        if context.price < lower_value and self.params.allow_short:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.SHORT,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "broke below the lower band out of a squeeze",
            )
        return context.hold("squeezed but no breakout", self.strategy_id)

    def _was_squeezed(
        self,
        upper: tuple[Decimal | None, ...],
        middle: tuple[Decimal | None, ...],
        lower: tuple[Decimal | None, ...],
        index: int,
    ) -> bool:
        """Whether bandwidth at ``index`` was the narrowest of the lookback window."""
        if index < self.params.squeeze_lookback:
            return False

        current = _bandwidth(upper, middle, lower, index)
        if current is None:
            return False
        for offset in range(1, self.params.squeeze_lookback + 1):
            previous = _bandwidth(upper, middle, lower, index - offset)
            if previous is not None and previous <= current:
                return False
        return True


def _bandwidth(
    upper: tuple[Decimal | None, ...],
    middle: tuple[Decimal | None, ...],
    lower: tuple[Decimal | None, ...],
    index: int,
) -> Decimal | None:
    """Band width as a fraction of the middle band, or ``None`` when undefined."""
    if index < 0:
        return None
    top, centre, bottom = upper[index], middle[index], lower[index]
    if top is None or centre is None or bottom is None or centre <= ZERO:
        return None
    return (top - bottom) / centre
