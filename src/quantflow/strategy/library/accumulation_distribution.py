"""Accumulation/Distribution — volume weighted by *where in the bar* the close landed.

The close-location value asks a sharper question than the close-to-close change does: of the
range this bar traded, how much of it did buyers hold onto by the end? A bar that opens
weak, trades down and then closes on its high has been accumulated regardless of whether it
closed above the previous bar. Summing that weighting against volume gives a line whose
slope is the net pressure over a window, and entering only when price agrees with it is the
usual guard against reading a flow signal into a market that is simply drifting.

Distinct from `obv_trend`, which uses the same volume but only the **sign** of the
close-to-close change. OBV credits a bar's entire volume to the buyers if it closes one tick
higher and to the sellers if it closes one tick lower, so a violent bar that closes almost
unchanged registers as a full-strength vote in whichever direction it happened to land. A/D
gives that same bar a weight near zero, and gives a quiet bar that closed on its high nearly
full weight. The two therefore disagree most on wide, indecisive bars — which is precisely
the population where "who won this bar" is least obvious and most worth measuring two ways.

Bars with no volume contribute nothing rather than causing a division by zero, and a bar
whose high equals its low has no location to speak of, so it contributes nothing either.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.market import Candle
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, sma
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class AccumulationDistributionParams(StrategyParams):
    """Parameters for :class:`AccumulationDistributionStrategy`."""

    #: Window over which the A/D line's net change is measured.
    trend_period: int = Field(default=20, ge=2, le=500)
    #: Moving average that price must agree with before the flow reading is acted on.
    price_period: int = Field(default=20, ge=2, le=500)
    #: Minimum net flow, as a fraction of the volume that traded in the window, before the
    #: reading counts as pressure rather than noise. Bounded by construction in ``[0, 1]``.
    min_flow_ratio: Decimal = Field(default=Decimal("0.05"), ge=0, le=1)
    #: Flow ratio at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("0.4"), gt=0, le=1)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_multiples(self) -> Self:
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class AccumulationDistributionStrategy(Strategy):
    """Trade with the A/D line when price agrees with it."""

    strategy_id = "accumulation_distribution"
    description = "Close-location-weighted volume flow, confirmed by price"
    params_model = AccumulationDistributionParams

    params: AccumulationDistributionParams

    @property
    def warmup_bars(self) -> int:
        """The flow window and the price average, plus a bar for the comparison."""
        return max(self.params.trend_period, self.params.price_period, self.params.atr_period) + 2

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit an accumulation/distribution signal."""
        index = context.index
        start = index - self.params.trend_period
        if start < 0:
            return context.hold("flow window warming up", self.strategy_id)

        line = accumulation_line(context.candles)
        traded = sum((candle.volume for candle in context.candles[start + 1 : index + 1]), ZERO)
        if traded <= ZERO:
            # No volume in the window means the line cannot have moved for any reason worth
            # trading; normalising by it would also divide by zero.
            return context.hold("no volume in the flow window", self.strategy_id)

        # |change| can never exceed the volume traded, because every bar contributes at most
        # its own volume. That makes the ratio a genuine 0–1 measure of one-sidedness.
        change = line[index] - line[start]
        flow_ratio = change / traded

        average = sma(context.closes, self.params.price_period)[index]
        if average is None:
            return context.hold("price average warming up", self.strategy_id)
        price_up = context.price > average

        if context.has_position:
            if context.is_long and (flow_ratio <= ZERO or not price_up):
                return exit_signal(context, self.strategy_id, "accumulation stopped")
            if context.is_short and (flow_ratio >= ZERO or price_up):
                return exit_signal(context, self.strategy_id, "distribution stopped")
            return context.hold("holding, flow and price still agree", self.strategy_id)

        if flow_ratio >= self.params.min_flow_ratio and price_up:
            direction = SignalDirection.LONG
        elif flow_ratio <= -self.params.min_flow_ratio and not price_up:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold(
                f"net flow {flow_ratio:.3f} of volume traded; no confirmed pressure",
                self.strategy_id,
            )

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"a/d line moved {flow_ratio:.3f} of the volume traded, with price agreeing",
        )
        return replace_conviction(signal, self._conviction(abs(flow_ratio)))

    def _conviction(self, flow_ratio: Decimal) -> Decimal:
        """More one-sided flow reads as stronger, saturating at ``conviction_span``."""
        return min(
            Decimal("0.4") + min(flow_ratio / self.params.conviction_span, ONE) * Decimal("0.6"),
            ONE,
        )


def accumulation_line(candles: Sequence[Candle]) -> tuple[Decimal, ...]:
    """Cumulative close-location-value weighted volume.

    Each bar contributes ``((close - low) - (high - close)) / (high - low) * volume``: full
    volume when the bar closes on its high, minus full volume on its low, nothing at the
    midpoint. A bar with no range has no location within it and a bar with no volume has
    nothing to weight, so both contribute zero — never a division by zero.

    Cumulative from the first bar, so only *changes* in it are meaningful; the absolute
    level depends on where the series happens to start.
    """
    total = ZERO
    out: list[Decimal] = []
    for candle in candles:
        span = candle.high - candle.low
        if span > ZERO and candle.volume > ZERO:
            location = ((candle.close - candle.low) - (candle.high - candle.close)) / span
            total += location * candle.volume
        out.append(total)
    return tuple(out)


__all__ = [
    "AccumulationDistributionParams",
    "AccumulationDistributionStrategy",
    "accumulation_line",
]
