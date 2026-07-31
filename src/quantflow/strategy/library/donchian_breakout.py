"""Donchian channel breakout — a classic trend-following reference strategy.

Enters when price closes beyond the trailing N-bar extreme and exits on the opposite,
shorter channel. The asymmetric exit (a shorter window than the entry) is the point of the
design: it gives a trend room to develop while still exiting promptly once it breaks.

The breakout is evaluated against the channel **excluding the current bar**. Including it
would make the condition trivially true on every new high — a subtle look-ahead that turns
a real strategy into one that buys every bar.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, rolling_max, rolling_min
from quantflow.strategy.registry import register_strategy


class DonchianBreakoutParams(StrategyParams):
    """Parameters for :class:`DonchianBreakoutStrategy`."""

    entry_period: int = Field(default=20, ge=2, le=500)
    exit_period: int = Field(default=10, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    #: Require the breakout to clear the channel by this fraction, filtering marginal
    #: ticks through the level that immediately reverse.
    breakout_buffer_pct: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("0.05"))
    allow_short: bool = False
    use_trailing_stop: bool = True

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if self.exit_period >= self.entry_period:
            raise ValueError(
                f"exit_period ({self.exit_period}) must be below "
                f"entry_period ({self.entry_period}); a symmetric channel exits every "
                "trend the moment it starts"
            )
        return self


@register_strategy
class DonchianBreakoutStrategy(Strategy):
    """Breakout entries on the N-bar extreme, exits on a shorter channel."""

    strategy_id = "donchian_breakout"
    description = "Donchian channel breakout with an asymmetric channel exit and ATR stop"
    params_model = DonchianBreakoutParams

    params: DonchianBreakoutParams

    @property
    def warmup_bars(self) -> int:
        """One bar beyond the entry channel, so the excluded-current-bar window is full."""
        return max(self.params.entry_period + 1, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a breakout entry or a channel exit."""
        index = context.index
        highs = [candle.high for candle in context.candles]
        lows = [candle.low for candle in context.candles]

        # Exclude the current bar: a channel that includes it is breached by definition
        # whenever the bar makes a new extreme.
        prior_high = rolling_max(highs[:-1], self.params.entry_period)
        prior_low = rolling_min(lows[:-1], self.params.entry_period)
        exit_high = rolling_max(highs[:-1], self.params.exit_period)
        exit_low = rolling_min(lows[:-1], self.params.exit_period)

        previous = index - 1
        if previous < 0:
            return context.hold("no prior bar", self.strategy_id)

        upper = prior_high[previous] if previous < len(prior_high) else None
        lower = prior_low[previous] if previous < len(prior_low) else None
        exit_upper = exit_high[previous] if previous < len(exit_high) else None
        exit_lower = exit_low[previous] if previous < len(exit_low) else None

        if context.is_long:
            if exit_lower is not None and context.price < exit_lower:
                return self._exit(context, f"closed below the {self.params.exit_period}-bar low")
            return context.hold("holding long", self.strategy_id)

        if context.is_short:
            if exit_upper is not None and context.price > exit_upper:
                return self._exit(context, f"closed above the {self.params.exit_period}-bar high")
            return context.hold("holding short", self.strategy_id)

        if upper is None or lower is None:
            return context.hold("channel warming up", self.strategy_id)

        buffer_rate = self.params.breakout_buffer_pct
        if context.price > upper * (Decimal("1") + buffer_rate):
            return self._entry(
                context,
                SignalDirection.LONG,
                f"closed above the {self.params.entry_period}-bar high {upper}",
            )
        if self.params.allow_short and context.price < lower * (Decimal("1") - buffer_rate):
            return self._entry(
                context,
                SignalDirection.SHORT,
                f"closed below the {self.params.entry_period}-bar low {lower}",
            )
        return context.hold("inside the channel", self.strategy_id)

    def _entry(self, context: StrategyContext, direction: SignalDirection, reason: str) -> Signal:
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
            reference_price=context.price,
            stop_loss_price=stop,
            reason=reason,
            metadata={"trailing_stop": str(self.params.use_trailing_stop)},
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
