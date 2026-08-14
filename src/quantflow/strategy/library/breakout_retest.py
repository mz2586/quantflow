"""Breakout retest — wait for the broken level to prove it holds before committing.

Most breaks of a level fail, and the ones that fail do so quickly: price pokes through,
finds no follow-through and falls back inside the range. The retest is the cheapest
available filter on that failure mode. Price must break the level, come back to it, and
then close away from it again — at which point the level has been tested from the other
side and held, which is the only evidence available that the break was real rather than a
liquidity sweep.

Distinct from `donchian_breakout`, which enters **on the break bar itself**. That captures
every genuine breakout, including the fast ones that never look back, at the cost of taking
every false break as well. This one deliberately declines the break bar: it enters later
and at a worse price on the moves that run away, misses them entirely when there is no
pullback, and in exchange never takes the break that immediately reverses. They are the two
halves of the same question — whether the cost of the false breaks exceeds the cost of the
missed ones — and running both is how that gets answered rather than assumed.

Everything is derived from closed bars at or before the decision bar; the break is always
located on a *prior* bar, so the current bar can only ever confirm a retest, never create
one retroactively.
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


class BreakoutRetestParams(StrategyParams):
    """Parameters for :class:`BreakoutRetestStrategy`."""

    #: Window whose extreme defines the level that has to be broken.
    level_period: int = Field(default=20, ge=2, le=500)
    #: How many bars after the break the retest may take. Beyond this the break is stale:
    #: a level revisited twenty bars later is no longer the same event.
    retest_window: int = Field(default=10, ge=1, le=100)
    #: How close, in ATRs, price must come back to the level for it to count as a retest.
    retest_tolerance_atr: Decimal = Field(default=Decimal("0.5"), gt=0, le=5)
    #: Channel used to exit, mirroring the asymmetric exit of the Donchian family.
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
class BreakoutRetestStrategy(Strategy):
    """Enter on a successful retest of a broken level, not on the break itself."""

    strategy_id = "breakout_retest"
    description = "Enters when a broken level is retested from the other side and holds"
    params_model = BreakoutRetestParams

    params: BreakoutRetestParams

    @property
    def warmup_bars(self) -> int:
        """The level window, plus the retest window that sits in front of it."""
        return max(
            self.params.level_period + self.params.retest_window + 2,
            self.params.exit_period + 2,
            self.params.atr_period + 1,
        )

    def generate(self, context: StrategyContext) -> Signal:
        """Emit a retest entry or a channel exit."""
        index = context.index
        candles = context.candles

        if context.has_position:
            return self._manage(context)

        volatility = atr(candles, self.params.atr_period)[index]
        if volatility is None or volatility <= ZERO:
            return context.hold("atr unavailable", self.strategy_id)
        tolerance = volatility * self.params.retest_tolerance_atr

        found = self._find_retest(candles, index, tolerance, long=True)
        if found is not None:
            return self._entry(context, SignalDirection.LONG, volatility, tolerance, found)

        if self.params.allow_short:
            found = self._find_retest(candles, index, tolerance, long=False)
            if found is not None:
                return self._entry(context, SignalDirection.SHORT, volatility, tolerance, found)

        return context.hold("no level has been broken and retested", self.strategy_id)

    def _manage(self, context: StrategyContext) -> Signal:
        """Exit on the shorter channel, the same asymmetric exit the break family uses."""
        candles = context.candles
        previous = context.index - 1
        if previous < 0:
            return context.hold("no prior bar", self.strategy_id)

        if context.is_long:
            lows = [candle.low for candle in candles]
            floor = rolling_min(lows[:-1], self.params.exit_period)[previous]
            if floor is not None and context.price < floor:
                return exit_signal(
                    context,
                    self.strategy_id,
                    f"closed below the {self.params.exit_period}-bar low; the level gave way",
                )
            return context.hold("holding, the retested level still holds", self.strategy_id)

        highs = [candle.high for candle in candles]
        ceiling = rolling_max(highs[:-1], self.params.exit_period)[previous]
        if ceiling is not None and context.price > ceiling:
            return exit_signal(
                context,
                self.strategy_id,
                f"closed above the {self.params.exit_period}-bar high; the level gave way",
            )
        return context.hold("holding, the retested level still holds", self.strategy_id)

    def _find_retest(
        self,
        candles: tuple[Candle, ...],
        index: int,
        tolerance: Decimal,
        *,
        long: bool,
    ) -> tuple[Decimal, Decimal] | None:
        """Locate the most recent break whose retest is confirmed by the current bar.

        Returns ``(level, nearest_gap)`` where ``nearest_gap`` is how close price came back
        to the level during the retest, or ``None`` when no such setup exists.
        """
        current = candles[index]
        # The confirming bar must itself close in the direction of the break; a bar that
        # merely sits above the level without pushing is not evidence the retest held.
        if long and current.close <= current.open:
            return None
        if not long and current.close >= current.open:
            return None

        # Newest break first: a fresher level is the one price is actually reacting to.
        for break_index in range(index - 1, index - self.params.retest_window - 1, -1):
            if break_index - self.params.level_period < 0:
                break
            window = candles[break_index - self.params.level_period : break_index]
            if not window:
                break
            level = (
                max(candle.high for candle in window)
                if long
                else min(candle.low for candle in window)
            )
            if level <= ZERO:
                continue
            gap = self._confirm(candles, break_index, index, level, tolerance, long=long)
            if gap is not None:
                return level, gap
        return None

    def _confirm(  # noqa: PLR0911 - each rejection is a distinct, nameable failure
        self,
        candles: tuple[Candle, ...],
        break_index: int,
        index: int,
        level: Decimal,
        tolerance: Decimal,
        *,
        long: bool,
    ) -> Decimal | None:
        """Whether the break at ``break_index`` was retested and held through ``index``."""
        breaker = candles[break_index]
        if long and breaker.close <= level:
            return None
        if not long and breaker.close >= level:
            return None

        current = candles[index]
        if long and current.close <= level:
            return None
        if not long and current.close >= level:
            return None

        nearest: Decimal | None = None
        for offset in range(break_index + 1, index + 1):
            candle = candles[offset]
            # A decisive close back through the level invalidates the break outright: the
            # market rejected it rather than merely testing it.
            if long and candle.close < level - tolerance:
                return None
            if not long and candle.close > level + tolerance:
                return None
            approach = candle.low - level if long else level - candle.high
            if nearest is None or approach < nearest:
                nearest = approach

        # A retest requires price to have actually come back to the level. Without this the
        # condition degenerates into "price is above a level", which is true most of the time.
        if nearest is None or nearest > tolerance:
            return None
        return nearest

    def _entry(
        self,
        context: StrategyContext,
        direction: SignalDirection,
        volatility: Decimal,
        tolerance: Decimal,
        found: tuple[Decimal, Decimal],
    ) -> Signal:
        """Build the entry, with conviction from how cleanly the retest resolved."""
        level, gap = found
        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"broke {level} and held it on the retest",
        )
        return replace_conviction(
            signal, self._conviction(gap, tolerance, level, context.price, volatility)
        )

    def _conviction(
        self,
        gap: Decimal,
        tolerance: Decimal,
        level: Decimal,
        price: Decimal,
        volatility: Decimal,
    ) -> Decimal:
        """A tighter retest and a firmer close away from the level both read as stronger."""
        proximity = ONE - min(max(gap, ZERO) / tolerance, ONE) if tolerance > ZERO else ZERO
        clearance = min(abs(price - level) / volatility, ONE) if volatility > ZERO else ZERO
        return min(Decimal("0.4") + proximity * Decimal("0.3") + clearance * Decimal("0.3"), ONE)


__all__ = ["BreakoutRetestParams", "BreakoutRetestStrategy"]
