"""On-balance-volume trend confirmation.

Enter when OBV and price agree on direction: price above its moving average *and* OBV above
its own. Requiring both is the point — price alone can drift up on thinning participation,
and OBV rising while price is flat says accumulation is happening before it shows.

Distinct from `volume_breakout`, which asks whether *this bar* traded unusually heavily.
This asks whether volume has been flowing in the same direction as price over a window, a
question about accumulation rather than about one bar's activity.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class ObvTrendParams(StrategyParams):
    """Parameters for :class:`ObvTrendStrategy`."""

    obv_period: int = Field(default=20, ge=2, le=200)
    price_period: int = Field(default=20, ge=2, le=200)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False


@register_strategy
class ObvTrendStrategy(Strategy):
    """Enter only when price and on-balance volume agree."""

    strategy_id = "obv_trend"
    description = "Requires price and on-balance volume to trend together"
    params_model = ObvTrendParams

    params: ObvTrendParams

    @property
    def warmup_bars(self) -> int:
        """Both moving averages, plus a bar for the comparison."""
        return max(self.params.obv_period, self.params.price_period, self.params.atr_period) + 2

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit an OBV-confirmed trend signal."""
        from quantflow.strategy.indicators import atr, obv, sma

        index = context.index
        flow = obv(context.candles)
        values = [value for value in flow if value is not None]
        if len(values) <= self.params.obv_period:
            return context.hold("obv warming up", self.strategy_id)

        obv_average = sma(values, self.params.obv_period)[-1]
        price_average = sma(context.closes, self.params.price_period)[index]
        current_obv = flow[index]
        if obv_average is None or price_average is None or current_obv is None:
            return context.hold("averages warming up", self.strategy_id)

        price_up = context.price > price_average
        flow_up = current_obv > obv_average

        if context.has_position:
            # Leave as soon as the two stop agreeing: the confirmation that justified the
            # entry is what has gone, regardless of where price is.
            if context.is_long and not (price_up and flow_up):
                return exit_signal(context, self.strategy_id, "price and obv no longer agree")
            if context.is_short and not (not price_up and not flow_up):
                return exit_signal(context, self.strategy_id, "price and obv no longer agree")
            return context.hold("holding, price and obv still agree", self.strategy_id)

        if price_up and flow_up:
            direction = SignalDirection.LONG
        elif not price_up and not flow_up:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold("price and obv disagree", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            "price and on-balance volume agree",
        )
        return replace_conviction(
            signal, self._conviction(current_obv, obv_average, context.price, price_average)
        )

    def _conviction(
        self, flow: Decimal, flow_average: Decimal, price: Decimal, price_average: Decimal
    ) -> Decimal:
        """Wider agreement on both measures reads as a stronger case."""
        price_gap = abs(price - price_average) / price_average if price_average > ZERO else ZERO
        flow_gap = abs(flow - flow_average) / abs(flow_average) if flow_average != ZERO else ZERO
        # Both gaps are unbounded ratios, so cap each before blending.
        return min(
            Decimal("0.4")
            + min(price_gap * 10, ONE) * Decimal("0.3")
            + min(flow_gap, ONE) * Decimal("0.3"),
            ONE,
        )


__all__ = ["ObvTrendParams", "ObvTrendStrategy"]
