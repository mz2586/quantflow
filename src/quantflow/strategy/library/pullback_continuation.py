"""Pullback continuation — join an existing trend at a discount instead of at a breakout.

Breakout entries buy the bar that has already moved, which puts the entry at the worst
price of the swing and the stop at the far side of it, and that ratio is what quietly makes
a strategy with a good hit rate unprofitable. A trend does not travel in a straight line: it
periodically hands back part of the move, and an entry taken there has the same thesis, a
much closer invalidation level, and therefore several times the reward-to-risk of the same
thesis expressed at the high.

What stops that being "catch a falling knife" is insisting on all three parts in order.
The trend must already exist (price above a rising slow average — not merely above it, a
flat average with price wandering over it is not a trend). The retracement must be real,
measured in ATR rather than percent so a one-tick dip in a calm market does not qualify.
And the market must have *resumed* — closed back through the fast average — before
anything is done, because a pullback that keeps going is not a pullback, it is the end of
the trend, and a strategy that cannot tell those apart before acting is buying every top.

The pullback must also hold above the slow average throughout. A retracement that goes
through it has not tested the trend, it has broken it, and continuing to call it a pullback
is how this idea turns into an averaging-down machine.

Shorts are symmetric — a rally into the fast average inside a falling trend — and are
gated by ``allow_short``.
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
from quantflow.strategy.indicators import Series, atr, crossed_above, crossed_below, ema
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class PullbackContinuationParams(StrategyParams):
    """Parameters for :class:`PullbackContinuationStrategy`."""

    #: The average price retraces *to*. Entries are triggered by closing back through it.
    fast_period: int = Field(default=20, ge=2, le=200)
    #: The average that defines whether a trend exists at all.
    slow_period: int = Field(default=50, ge=3, le=500)
    #: Bars over which the slow average must have risen for the trend to count.
    trend_slope_bars: int = Field(default=10, ge=1, le=200)
    #: How far back a qualifying retracement may have started.
    pullback_bars: int = Field(default=6, ge=2, le=100)
    #: Minimum depth of the retracement past the fast average, in ATR units.
    min_pullback_atr: Decimal = Field(default=Decimal("0.25"), ge=0, le=10)
    #: Retracement depth, in ATR units, at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("1.5"), gt=0, le=10)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_periods(self) -> Self:
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be below slow_period ({self.slow_period})"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class PullbackContinuationStrategy(Strategy):
    """Buys the resumption of a trend after a measured retracement into the fast average."""

    strategy_id = "pullback_continuation"
    description = "Enters an established trend on a retracement to a moving average that resumes"
    params_model = PullbackContinuationParams

    params: PullbackContinuationParams

    @property
    def warmup_bars(self) -> int:
        """The slow EMA must have settled, and the slope reads back beyond that."""
        return max(
            self.params.slow_period * 2 + self.params.trend_slope_bars,
            self.params.atr_period + 1,
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit an entry when a retracement inside an established trend resumes."""
        index = context.index
        fast = ema(context.closes, self.params.fast_period)
        slow = ema(context.closes, self.params.slow_period)
        anchor = index - self.params.trend_slope_bars
        if anchor < 0:
            return context.hold("not enough history for the trend slope", self.strategy_id)

        fast_value = fast[index]
        slow_value = slow[index]
        earlier_slow = slow[anchor]
        if fast_value is None or slow_value is None or earlier_slow is None:
            return context.hold("emas warming up", self.strategy_id)

        price = context.price
        uptrend = price > slow_value and fast_value > slow_value > earlier_slow
        downtrend = price < slow_value and fast_value < slow_value < earlier_slow

        if context.is_long:
            if price < slow_value:
                return exit_signal(context, self.strategy_id, "closed below the slow average")
            return context.hold("holding long, trend intact", self.strategy_id)

        if context.is_short:
            if price > slow_value:
                return exit_signal(context, self.strategy_id, "closed above the slow average")
            return context.hold("holding short, trend intact", self.strategy_id)

        if not uptrend and not downtrend:
            return context.hold("no established trend", self.strategy_id)
        if downtrend and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        long = uptrend
        resumed = (
            crossed_above(context.closes, fast, index)
            if long
            else crossed_below(context.closes, fast, index)
        )
        if not resumed:
            return context.hold("price has not resumed through the fast average", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        depth = _pullback_depth(
            context.candles, fast, slow, index, self.params.pullback_bars, long=long
        )
        if depth is None:
            return context.hold("the retracement broke the trend average", self.strategy_id)
        threshold = (volatility or ZERO) * self.params.min_pullback_atr
        if depth < threshold:
            return context.hold(f"retracement {depth} shallower than {threshold}", self.strategy_id)

        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if long else SignalDirection.SHORT,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            (
                f"resumed {'above' if long else 'below'} the {self.params.fast_period}-period "
                f"average after a {depth} retracement"
            ),
        )
        return replace_conviction(signal, self._conviction(depth, volatility))

    def _conviction(self, depth: Decimal, volatility: Decimal | None) -> Decimal:
        """A deeper retracement that still resumed is the higher-conviction entry.

        Depth is the entire economic argument for this strategy: it is what moves the entry
        away from the stop. A shallow dip that qualifies on a technicality offers barely
        better reward-to-risk than chasing the high, and conviction should not report it as
        the same trade.
        """
        if volatility is None or volatility <= ZERO or depth <= ZERO:
            return Decimal("0.35")
        scaled = depth / (volatility * self.params.conviction_span)
        return min(Decimal("0.35") + min(scaled, ONE) * Decimal("0.65"), ONE)


def _pullback_depth(
    candles: tuple[Candle, ...],
    fast: Series,
    slow: Series,
    index: int,
    lookback: int,
    *,
    long: bool,
) -> Decimal | None:
    """Deepest excursion past the fast average over the bars before ``index``.

    Returns ``None`` when the retracement closed through the slow average at any point: the
    trend that the entry is meant to be joining was broken, so there is no continuation
    trade here regardless of how the last bar closed.
    """
    start = max(index - lookback, 0)
    deepest = ZERO
    for position in range(start, index):
        fast_value = fast[position]
        slow_value = slow[position]
        if fast_value is None or slow_value is None:
            continue
        candle = candles[position]
        if long:
            if candle.close < slow_value:
                return None
            deepest = max(deepest, fast_value - candle.low)
        else:
            if candle.close > slow_value:
                return None
            deepest = max(deepest, candle.high - fast_value)
    return deepest


__all__ = ["PullbackContinuationParams", "PullbackContinuationStrategy"]
