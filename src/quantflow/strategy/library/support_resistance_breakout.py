"""Support/resistance breakout — levels earned by repeated touches, not by a rolling window.

A Donchian channel treats the single highest bar in a window as *the* level, which means the
level is defined by one print — often an outlier wick that no one traded twice. This builds
levels the way a chartist does: find the confirmed swing points, cluster the ones that sit
at the same price, and keep the clusters that were touched more than once. A price that has
turned the market away three times is a real decision point; a price touched once is a
statistic.

The touch count then does double duty. It gates entry (a level nobody defended is not worth
trading the break of) and it scales conviction, so breaking a four-touch ceiling is a
stronger signal than breaking a two-touch one — a distinction no rolling-extreme breakout
can express at all.

Swing points are confirmed only when ``swing_strength`` bars exist on **both** sides, so the
most recent ``swing_strength`` bars can never contribute a level. That lag is deliberate: a
swing high identified without bars to its right is not a swing high, it is a guess about the
future, and treating it as one is look-ahead in its purest form.
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
from quantflow.strategy.indicators import atr, rolling_max, rolling_min
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class SupportResistanceBreakoutParams(StrategyParams):
    """Parameters for :class:`SupportResistanceBreakoutStrategy`."""

    #: Bars required either side of a bar for it to count as a confirmed swing point.
    swing_strength: int = Field(default=3, ge=1, le=50)
    #: How far back swing points are collected from.
    lookback: int = Field(default=120, ge=10, le=2000)
    #: Swing points within this fraction of each other are the same level.
    cluster_tolerance_pct: Decimal = Field(default=Decimal("0.005"), gt=0, le=Decimal("0.2"))
    #: How many touches a cluster needs before its break is tradeable.
    min_touches: int = Field(default=2, ge=1, le=20)
    #: How far past the level the close must sit, filtering marginal ticks through it.
    confirmation_pct: Decimal = Field(default=Decimal("0.001"), ge=0, le=Decimal("0.1"))
    #: Touch count at which conviction saturates.
    conviction_touches: int = Field(default=5, ge=1, le=50)
    exit_period: int = Field(default=10, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_multiples(self) -> Self:
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class SupportResistanceBreakoutStrategy(Strategy):
    """Break of a level that repeated swing points have defended."""

    strategy_id = "support_resistance_breakout"
    description = "Breaks levels built from clustered swing highs and lows, weighted by touches"
    params_model = SupportResistanceBreakoutParams

    params: SupportResistanceBreakoutParams

    @property
    def warmup_bars(self) -> int:
        """The lookback, plus the confirmation lag at both ends of a swing."""
        return max(
            self.params.lookback + self.params.swing_strength * 2 + 2,
            self.params.exit_period + 2,
            self.params.atr_period + 1,
        )

    def generate(self, context: StrategyContext) -> Signal:
        """Emit a level-break entry or a channel exit."""
        index = context.index
        if context.has_position:
            return self._manage(context)

        previous = index - 1
        if previous < 0:
            return context.hold("no prior bar", self.strategy_id)
        previous_close = context.candles[previous].close

        resistance = self._levels(context.candles, index, high_side=True)
        broken = self._broken_level(resistance, previous_close, context.price, upward=True)
        if broken is not None:
            return self._entry(context, SignalDirection.LONG, broken)

        if self.params.allow_short:
            support = self._levels(context.candles, index, high_side=False)
            broken = self._broken_level(support, previous_close, context.price, upward=False)
            if broken is not None:
                return self._entry(context, SignalDirection.SHORT, broken)

        return context.hold("no defended level broken on this bar", self.strategy_id)

    def _manage(self, context: StrategyContext) -> Signal:
        """Leave on the shorter channel once the broken level fails to hold."""
        previous = context.index - 1
        if previous < 0:
            return context.hold("no prior bar", self.strategy_id)
        candles = context.candles

        if context.is_long:
            lows = [candle.low for candle in candles]
            floor = rolling_min(lows[:-1], self.params.exit_period)[previous]
            if floor is not None and context.price < floor:
                return exit_signal(
                    context, self.strategy_id, "fell back below the broken level's channel"
                )
            return context.hold("holding, the broken level still supports", self.strategy_id)

        highs = [candle.high for candle in candles]
        ceiling = rolling_max(highs[:-1], self.params.exit_period)[previous]
        if ceiling is not None and context.price > ceiling:
            return exit_signal(
                context, self.strategy_id, "rose back above the broken level's channel"
            )
        return context.hold("holding, the broken level still caps", self.strategy_id)

    def _levels(
        self, candles: tuple[Candle, ...], index: int, *, high_side: bool
    ) -> list[tuple[Decimal, int]]:
        """Clustered swing levels as ``(price, touch_count)``, oldest price first."""
        strength = self.params.swing_strength
        start = max(strength, index - self.params.lookback)
        # ``index - strength`` is the newest bar that has a full window on its right; any
        # later bar would need bars that do not exist yet.
        points: list[Decimal] = []
        for position in range(start, index - strength + 1):
            window = candles[position - strength : position + strength + 1]
            if len(window) < strength * 2 + 1:
                continue
            if high_side:
                if candles[position].high >= max(bar.high for bar in window):
                    points.append(candles[position].high)
            elif candles[position].low <= min(bar.low for bar in window):
                points.append(candles[position].low)
        return _cluster(points, self.params.cluster_tolerance_pct)

    def _broken_level(
        self,
        levels: list[tuple[Decimal, int]],
        previous_close: Decimal,
        price: Decimal,
        *,
        upward: bool,
    ) -> tuple[Decimal, int] | None:
        """The best-defended level broken on this bar, or ``None``.

        The previous close must have been on the other side, so a level is traded once, on
        the bar it gives way — not on every bar that price happens to sit beyond it.
        """
        buffer_rate = self.params.confirmation_pct
        best: tuple[Decimal, int] | None = None
        for level, touches in levels:
            if touches < self.params.min_touches or level <= ZERO:
                continue
            if upward:
                fresh = previous_close <= level and price > level * (ONE + buffer_rate)
            else:
                fresh = previous_close >= level and price < level * (ONE - buffer_rate)
            if fresh and (best is None or touches > best[1]):
                best = (level, touches)
        return best

    def _entry(
        self, context: StrategyContext, direction: SignalDirection, broken: tuple[Decimal, int]
    ) -> Signal:
        """Build the entry, with conviction rising in the number of touches."""
        level, touches = broken
        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[context.index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"broke a level at {level} defended {touches} times",
        )
        return replace_conviction(signal, self._conviction(touches))

    def _conviction(self, touches: int) -> Decimal:
        """More touches means a more meaningful level, saturating at ``conviction_touches``."""
        span = max(self.params.conviction_touches - self.params.min_touches, 1)
        excess = Decimal(touches - self.params.min_touches) / Decimal(span)
        return min(Decimal("0.5") + min(excess, ONE) * Decimal("0.5"), ONE)


def _cluster(points: list[Decimal], tolerance: Decimal) -> list[tuple[Decimal, int]]:
    """Group nearby prices into levels, returning ``(mean_price, member_count)``.

    Greedy on a sorted list: each group extends while the next price is within
    ``tolerance`` of the price that opened it. Clustering against the group's *opening*
    price rather than its running mean keeps a long drift of prices from chaining into one
    absurdly wide "level".
    """
    if not points:
        return []
    ordered = sorted(points)
    levels: list[tuple[Decimal, int]] = []
    group: list[Decimal] = [ordered[0]]
    for price in ordered[1:]:
        anchor = group[0]
        if anchor > ZERO and price <= anchor * (ONE + tolerance):
            group.append(price)
            continue
        levels.append((sum(group, ZERO) / Decimal(len(group)), len(group)))
        group = [price]
    levels.append((sum(group, ZERO) / Decimal(len(group)), len(group)))
    return levels


__all__ = ["SupportResistanceBreakoutParams", "SupportResistanceBreakoutStrategy"]
