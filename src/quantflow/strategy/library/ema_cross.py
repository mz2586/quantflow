"""EMA crossover — a trend-following reference strategy.

Long when the fast EMA crosses above the slow EMA, flat when it crosses back. Included as a
reference because it is simple enough to verify by hand, which makes it a useful control
when validating the engine itself: if an EMA cross backtest looks wrong, the bug is almost
certainly in the machinery rather than in the strategy.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, crossed_above, crossed_below, ema
from quantflow.strategy.registry import register_strategy


class EmaCrossParams(StrategyParams):
    """Parameters for :class:`EmaCrossStrategy`."""

    fast_period: int = Field(default=12, ge=2, le=200)
    slow_period: int = Field(default=26, ge=3, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    #: Stop distance in ATR multiples. Volatility-scaled rather than a fixed percentage,
    #: which would be far too tight in a violent market and far too wide in a calm one.
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False
    #: Minimum separation between the EMAs, as a fraction of price, before a cross counts.
    #: Filters the chop that produces a stream of immediately-reversed trades.
    min_separation_pct: Decimal = Field(default=Decimal("0.0005"), ge=0, le=Decimal("0.1"))

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be below "
                f"slow_period ({self.slow_period})"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class EmaCrossStrategy(Strategy):
    """Trend following on an EMA crossover with ATR-scaled protective levels."""

    strategy_id = "ema_cross"
    description = "EMA crossover trend following with ATR-based stop and target"
    params_model = EmaCrossParams

    params: EmaCrossParams

    @property
    def warmup_bars(self) -> int:
        """Enough bars for the slow EMA to be meaningful, plus the ATR window.

        The slow EMA is seeded at ``slow_period`` but is still dominated by its seed for
        roughly another period; doubling it avoids trading on a value that has not settled.
        """
        return max(self.params.slow_period * 2, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a long/close signal on an EMA crossover."""
        closes = context.closes
        index = context.index

        fast = ema(closes, self.params.fast_period)
        slow = ema(closes, self.params.slow_period)
        fast_value, slow_value = fast[index], slow[index]
        if fast_value is None or slow_value is None:
            return context.hold("emas warming up", self.strategy_id)

        separation = abs(fast_value - slow_value) / context.price if context.price else ZERO
        volatility = atr(context.candles, self.params.atr_period)[index]

        bullish = crossed_above(fast, slow, index)
        bearish = crossed_below(fast, slow, index)

        if context.is_long:
            if bearish:
                return Signal(
                    symbol=context.symbol,
                    direction=SignalDirection.CLOSE,
                    timestamp=context.now,
                    strategy_id=self.strategy_id,
                    reference_price=context.price,
                    reason="fast EMA crossed below slow EMA",
                )
            return context.hold("holding long", self.strategy_id)

        if context.is_short:
            if bullish:
                return Signal(
                    symbol=context.symbol,
                    direction=SignalDirection.CLOSE,
                    timestamp=context.now,
                    strategy_id=self.strategy_id,
                    reference_price=context.price,
                    reason="fast EMA crossed above slow EMA",
                )
            return context.hold("holding short", self.strategy_id)

        if separation < self.params.min_separation_pct:
            return context.hold(f"EMA separation {separation:.5f} below minimum", self.strategy_id)

        if bullish:
            return self._entry(
                context, SignalDirection.LONG, volatility, "fast EMA crossed above slow EMA"
            )
        if bearish and self.params.allow_short:
            return self._entry(
                context, SignalDirection.SHORT, volatility, "fast EMA crossed below slow EMA"
            )
        return context.hold("no crossover", self.strategy_id)

    def _entry(
        self,
        context: StrategyContext,
        direction: SignalDirection,
        volatility: Decimal | None,
        reason: str,
    ) -> Signal:
        """Build an entry signal with ATR-scaled protective levels.

        When ATR is unavailable the signal carries no stop; the risk engine then applies its
        own default. It is never emitted unprotected.
        """
        stop = target = None
        if volatility is not None and volatility > ZERO:
            stop_distance = volatility * self.params.atr_stop_multiple
            target_distance = volatility * self.params.atr_target_multiple
            if direction is SignalDirection.LONG:
                stop = context.price - stop_distance
                target = context.price + target_distance
            else:
                stop = context.price + stop_distance
                target = context.price - target_distance
            if stop is not None and stop <= ZERO:
                stop = None

        return Signal(
            symbol=context.symbol,
            direction=direction,
            timestamp=context.now,
            strategy_id=self.strategy_id,
            reference_price=context.price,
            stop_loss_price=stop,
            take_profit_price=target,
            reason=reason,
        )
