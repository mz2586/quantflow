"""Volatility-expansion breakout.

Trade the bar on which volatility itself expands, in the direction that bar is travelling.
The entry condition is a statement about ATR, not about price level: current ATR must
exceed its own recent average by a margin, and the bar must close decisively in one
direction.

Against `bollinger_squeeze`, the other volatility member: the squeeze waits for bands to
*contract* and then trades the release, so it requires a quiet period first and misses
expansion that arrives without one. This one has no contraction precondition — it reacts
to expansion wherever it appears, which is most of what a squeeze filter throws away.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, sma
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.registry import register_strategy


class AtrExpansionParams(StrategyParams):
    """Parameters for :class:`AtrExpansionStrategy`."""

    atr_period: int = Field(default=14, ge=2, le=100)
    #: Window over which "normal" volatility is measured.
    baseline_period: int = Field(default=50, ge=5, le=500)
    #: How far above baseline ATR must sit before the bar counts as an expansion.
    expansion_multiple: Decimal = Field(default=Decimal("1.5"), gt=1, le=10)
    #: Minimum share of the bar's range that the body must occupy, so a wide but
    #: indecisive bar with a tiny body does not read as directional.
    min_body_ratio: Decimal = Field(default=Decimal("0.5"), gt=0, le=1)
    #: Exit once volatility has fallen back to this multiple of baseline.
    exit_multiple: Decimal = Field(default=Decimal("1.0"), gt=0, le=10)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False


@register_strategy
class AtrExpansionStrategy(Strategy):
    """Enter in the direction of a bar whose volatility has expanded."""

    strategy_id = "atr_expansion"
    description = "Enters on ATR expansion above baseline, in the bar's own direction"
    params_model = AtrExpansionParams

    params: AtrExpansionParams

    @property
    def warmup_bars(self) -> int:
        """ATR plus the baseline window it is compared against."""
        return self.params.atr_period + self.params.baseline_period + 1

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a volatility-expansion signal."""
        index = context.index
        atr_series = atr(context.candles, self.params.atr_period)
        current = atr_series[index]
        if current is None or current <= ZERO:
            return context.hold("atr unavailable", self.strategy_id)

        # Baseline is the average ATR, so "expansion" is measured against this market's own
        # recent volatility rather than an absolute threshold that would mean different
        # things on BTC and on a sub-dollar altcoin.
        defined = [value for value in atr_series[: index + 1] if value is not None]
        if len(defined) < self.params.baseline_period:
            return context.hold("baseline warming up", self.strategy_id)
        baseline = sma(defined, self.params.baseline_period)[-1]
        if baseline is None or baseline <= ZERO:
            return context.hold("baseline unavailable", self.strategy_id)

        ratio = current / baseline

        if context.has_position:
            if ratio <= self.params.exit_multiple:
                return exit_signal(context, self.strategy_id, "volatility returned to baseline")
            return context.hold("holding, volatility still elevated", self.strategy_id)

        if ratio < self.params.expansion_multiple:
            return context.hold(
                f"atr {ratio:.2f}x baseline, below {self.params.expansion_multiple}x",
                self.strategy_id,
            )

        candle = context.candle
        span = candle.high - candle.low
        if span <= ZERO:
            return context.hold("bar has no range", self.strategy_id)
        body = candle.close - candle.open
        if abs(body) / span < self.params.min_body_ratio:
            return context.hold("expansion bar has no decisive body", self.strategy_id)
        if body == ZERO:
            return context.hold("bar closed unchanged", self.strategy_id)

        long = body > ZERO
        if not long and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        direction = SignalDirection.LONG if long else SignalDirection.SHORT
        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            current,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"atr {ratio:.2f}x baseline with a decisive body",
        )
        from quantflow.strategy.library.vwap_reversion import replace_conviction

        return replace_conviction(signal, self._conviction(ratio, abs(body) / span))

    def _conviction(self, ratio: Decimal, body_ratio: Decimal) -> Decimal:
        """Stronger expansion and a cleaner body both raise conviction."""
        excess = ratio - self.params.expansion_multiple
        volatility_part = min(excess / self.params.expansion_multiple, ONE)
        return min(
            Decimal("0.4") + volatility_part * Decimal("0.3") + body_ratio * Decimal("0.3"), ONE
        )


__all__ = ["AtrExpansionParams", "AtrExpansionStrategy"]
