"""Multi-timeframe trend — take the fast signal only when the slow chart agrees.

The standard objection to any crossover strategy is that it cannot tell a genuine trend
from a countertrend bounce inside a larger move against it, because both look identical at
the timeframe it is watching. The usual fix is to lengthen the averages, which does not
work: it trades one lag for another and still reads a single scale. Looking at a genuinely
*coarser* series is a different operation — aggregating four bars into one throws away the
intrabar path, and what survives that discard is precisely the part of the move that was
not noise.

No second data feed is required or wanted. The higher timeframe is built by bucketing the
bars already in hand, which keeps the two views exactly consistent with each other: a
separately-fetched 4h feed can disagree with four 1h bars over boundaries, gaps and
revisions, and then the confirmation is measuring feed alignment rather than trend
agreement.

Only **completed** buckets are used. The partial bucket at the right-hand edge closes at
the current price, so admitting it would let the current bar confirm itself — the
confirmation would be guaranteed to agree with the trigger and the filter would do nothing
at all while appearing to work.

Shorts are symmetric — a bearish cross under a falling higher-timeframe trend — and are
gated by ``allow_short``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import (
    atr,
    crossed_above,
    crossed_below,
    ema,
    higher_timeframe_closes,
)
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class MtfTrendParams(StrategyParams):
    """Parameters for :class:`MtfTrendStrategy`."""

    fast_period: int = Field(default=9, ge=2, le=200)
    slow_period: int = Field(default=21, ge=3, le=400)
    #: Base bars per higher-timeframe bar. Four is the usual step between adjacent
    #: timeframes (15m→1h, 1h→4h, 6h→1d).
    htf_factor: int = Field(default=4, ge=2, le=48)
    htf_fast_period: int = Field(default=5, ge=2, le=100)
    htf_slow_period: int = Field(default=13, ge=3, le=200)
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
        if self.htf_fast_period >= self.htf_slow_period:
            raise ValueError(
                f"htf_fast_period ({self.htf_fast_period}) must be below htf_slow_period "
                f"({self.htf_slow_period})"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class MtfTrendStrategy(Strategy):
    """Lower-timeframe EMA cross, taken only when the aggregated series agrees."""

    strategy_id = "mtf_trend"
    description = "EMA cross confirmed by the trend of bars aggregated to a higher timeframe"
    params_model = MtfTrendParams

    params: MtfTrendParams

    @property
    def warmup_bars(self) -> int:
        """The binding constraint is the higher-timeframe slow EMA, priced in base bars.

        It needs ``htf_slow_period`` completed buckets to seed and about as many again
        before the seed stops dominating, and each bucket costs ``htf_factor`` base bars.
        """
        htf_cost = self.params.htf_slow_period * 2 * self.params.htf_factor
        return max(self.params.slow_period * 2, htf_cost, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a higher-timeframe-confirmed crossover signal."""
        index = context.index
        fast = ema(context.closes, self.params.fast_period)
        slow = ema(context.closes, self.params.slow_period)
        if fast[index] is None or slow[index] is None:
            return context.hold("emas warming up", self.strategy_id)

        higher = higher_timeframe_closes(context.candles, self.params.htf_factor)
        htf_fast = ema(higher, self.params.htf_fast_period)
        htf_slow = ema(higher, self.params.htf_slow_period)
        htf_fast_value = htf_fast[-1] if htf_fast else None
        htf_slow_value = htf_slow[-1] if htf_slow else None
        if htf_fast_value is None or htf_slow_value is None:
            return context.hold("higher timeframe warming up", self.strategy_id)

        htf_up = htf_fast_value > htf_slow_value
        htf_down = htf_fast_value < htf_slow_value
        bullish = crossed_above(fast, slow, index)
        bearish = crossed_below(fast, slow, index)

        if context.has_position:
            return self._manage(
                context, bullish=bullish, bearish=bearish, htf_up=htf_up, htf_down=htf_down
            )

        if bullish and htf_up:
            long = True
        elif bearish and htf_down:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            long = False
        elif bullish or bearish:
            return context.hold("higher timeframe disagrees with the cross", self.strategy_id)
        else:
            return context.hold("no crossover", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if long else SignalDirection.SHORT,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            (
                f"{'bullish' if long else 'bearish'} cross confirmed by "
                f"{self.params.htf_factor}-bar higher timeframe"
            ),
        )
        return replace_conviction(
            signal,
            self._conviction(htf_fast_value, htf_slow_value, context.price, volatility),
        )

    def _manage(
        self,
        context: StrategyContext,
        *,
        bullish: bool,
        bearish: bool,
        htf_up: bool,
        htf_down: bool,
    ) -> Signal:
        """Decide what to do about an open position.

        Either timeframe turning against the trade closes it. Waiting for both to agree
        again means holding through the whole of the move that invalidated the thesis.
        """
        if context.is_long:
            if bearish:
                return exit_signal(context, self.strategy_id, "fast ema crossed below slow")
            if htf_down:
                return exit_signal(context, self.strategy_id, "higher timeframe turned down")
            return context.hold("holding long, both timeframes up", self.strategy_id)
        if bullish:
            return exit_signal(context, self.strategy_id, "fast ema crossed above slow")
        if htf_up:
            return exit_signal(context, self.strategy_id, "higher timeframe turned up")
        return context.hold("holding short, both timeframes down", self.strategy_id)

    def _conviction(
        self,
        htf_fast_value: Decimal,
        htf_slow_value: Decimal,
        price: Decimal,
        volatility: Decimal | None,
    ) -> Decimal:
        """Conviction rises with how far apart the higher-timeframe averages are.

        The cross is a binary event, so the only thing that distinguishes one confirmed
        entry from another is the strength of the confirmation. A higher timeframe whose
        averages are barely separated is on the verge of disagreeing; one whose separation
        is an ATR wide is not.
        """
        separation = abs(htf_fast_value - htf_slow_value)
        # Without ATR, fall back to separation as a fraction of price, where one percent is
        # taken as the saturating figure.
        if volatility is None or volatility <= ZERO:
            if price <= ZERO:
                return Decimal("0.4")
            scaled = separation / price * 100
        else:
            scaled = separation / volatility
        return min(Decimal("0.4") + min(scaled, ONE) * Decimal("0.6"), ONE)


__all__ = ["MtfTrendParams", "MtfTrendStrategy"]
