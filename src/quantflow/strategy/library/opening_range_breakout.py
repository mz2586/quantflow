"""Opening-range breakout.

Crypto never closes, so "the open" is a convention rather than an event: this uses the
UTC day boundary, which is what most venues and index providers roll on and therefore
where scheduled flow actually concentrates. The range is the first N hours of the UTC day;
a break of that range later in the same day is the trade, and everything is closed at the
day boundary so nothing is carried overnight.

This is the only strategy in the library whose logic depends on *time of day* rather than
on price alone, which is the point of including it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

from pydantic import Field

from quantflow.domain.enums import SignalDirection
from quantflow.domain.market import Candle
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy

#: Last hour of the UTC day. Positions are flattened here so an intraday idea stays
#: intraday rather than quietly becoming an overnight one.
LAST_HOUR_OF_DAY: Final = 23


class OpeningRangeBreakoutParams(StrategyParams):
    """Parameters for :class:`OpeningRangeBreakoutStrategy`."""

    #: Hours from 00:00 UTC that form the range.
    range_hours: int = Field(default=4, ge=1, le=12)
    #: Hour (UTC) after which no new position is opened, so a trade has room to work.
    last_entry_hour: int = Field(default=20, ge=1, le=23)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("1.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    allow_short: bool = False


@register_strategy
class OpeningRangeBreakoutStrategy(Strategy):
    """Break of the first hours of the UTC day, flattened at the day boundary."""

    strategy_id = "opening_range_breakout"
    description = "Breakout of the opening UTC range, closed out at the day boundary"
    params_model = OpeningRangeBreakoutParams

    params: OpeningRangeBreakoutParams

    @property
    def warmup_bars(self) -> int:
        """A full day plus the ATR window, so a complete range is always available."""
        return max(24 + self.params.range_hours, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Trade a break of today's opening range."""
        hour = context.candle.open_time.hour

        if context.has_position:
            # Flat by the end of the day: this is an intraday idea, and holding it
            # overnight would silently turn it into a different strategy.
            if hour >= LAST_HOUR_OF_DAY:
                return exit_signal(context, self.strategy_id, "day boundary reached")
            return context.hold("holding intraday", self.strategy_id)

        if hour < self.params.range_hours:
            return context.hold("still inside the opening range", self.strategy_id)
        if hour > self.params.last_entry_hour:
            return context.hold("too late in the day to open", self.strategy_id)

        bounds = self._range_for_today(context)
        if bounds is None:
            return context.hold("no complete opening range", self.strategy_id)
        high, low = bounds

        volatility = atr(context.candles, self.params.atr_period)[context.index]
        if context.price > high:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.LONG,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "broke above the opening range",
            )
        if context.price < low and self.params.allow_short:
            return entry_signal(
                context,
                self.strategy_id,
                SignalDirection.SHORT,
                volatility,
                self.params.atr_stop_multiple,
                self.params.atr_target_multiple,
                "broke below the opening range",
            )
        return context.hold("inside the opening range", self.strategy_id)

    def _range_for_today(self, context: StrategyContext) -> tuple[Decimal, Decimal] | None:
        """High and low of today's opening bars, or ``None`` if the range is incomplete.

        Walks back from the decision bar rather than scanning the whole history: the
        history can be tens of thousands of bars, and only the current day matters.
        """
        today = context.candle.open_time.date()
        window: list[Candle] = []
        for candle in reversed(context.candles):
            if candle.open_time.date() != today:
                break
            if candle.open_time.hour < self.params.range_hours:
                window.append(candle)

        if len(window) < self.params.range_hours:
            return None
        return max(c.high for c in window), min(c.low for c in window)
