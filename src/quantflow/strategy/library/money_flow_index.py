"""Money Flow Index — RSI's question, asked of money rather than of price.

MFI is built exactly like RSI except that each bar's move is weighted by the money that
changed hands on it (typical price times volume) instead of counting once. The consequence
is the whole point of including it: a drift to an oversold reading on thinning volume barely
registers, while the same drift on heavy volume pushes MFI to an extreme. An oscillator that
can tell those two apart is measuring capitulation rather than merely measuring decline.

The trade is the *recovery* from an extreme, not the extreme itself. Buying because MFI is
below 20 is buying into an ongoing liquidation, and an oscillator can sit pinned at an
extreme for a long time; requiring it to have been there recently and to have come back out
means something has actually changed. Exit is the midline: the mean-reversion premise is
that the reading returns to neutral, and holding past neutral would be a different, unstated
trend-following bet.

Bars with zero volume contribute nothing, and a window whose money flow is entirely zero
leaves MFI undefined rather than defaulting to a neutral 50 — an undefined reading and a
genuinely balanced one must not look the same to the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import HUNDRED, ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.market import Candle
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import Series, atr
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class MoneyFlowIndexParams(StrategyParams):
    """Parameters for :class:`MoneyFlowIndexStrategy`."""

    period: int = Field(default=14, ge=2, le=200)
    #: Reading below which the market counts as washed out.
    oversold: Decimal = Field(default=Decimal("20"), gt=0, lt=100)
    #: Reading above which it counts as saturated.
    overbought: Decimal = Field(default=Decimal("80"), gt=0, lt=100)
    #: Neutral reading at which the reversion trade is complete.
    exit_level: Decimal = Field(default=Decimal("50"), gt=0, lt=100)
    #: How many recent bars are searched for the extreme the recovery comes out of.
    extreme_lookback: int = Field(default=5, ge=1, le=50)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("3.5"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_levels(self) -> Self:
        if not self.oversold < self.exit_level < self.overbought:
            raise ValueError(
                f"levels must satisfy oversold ({self.oversold}) < exit_level "
                f"({self.exit_level}) < overbought ({self.overbought})"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class MoneyFlowIndexStrategy(Strategy):
    """Fade an MFI extreme once the reading has started to recover from it."""

    strategy_id = "money_flow_index"
    description = "Volume-weighted RSI faded on the recovery out of an extreme"
    params_model = MoneyFlowIndexParams

    params: MoneyFlowIndexParams

    @property
    def warmup_bars(self) -> int:
        """The MFI window, plus the run of bars searched for the extreme."""
        return max(
            self.params.period + self.params.extreme_lookback + 2, self.params.atr_period + 1
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a money-flow reversion signal."""
        index = context.index
        readings = money_flow_index(context.candles, self.params.period)
        current = readings[index]
        if current is None:
            return context.hold("money flow index undefined", self.strategy_id)

        if context.has_position:
            if context.is_long and current >= self.params.exit_level:
                return exit_signal(context, self.strategy_id, f"mfi recovered to {current:.1f}")
            if context.is_short and current <= self.params.exit_level:
                return exit_signal(context, self.strategy_id, f"mfi fell back to {current:.1f}")
            return context.hold(f"holding, mfi {current:.1f}", self.strategy_id)

        start = max(index - self.params.extreme_lookback, 0)
        recent = [value for value in readings[start:index] if value is not None]
        if not recent:
            return context.hold("no recent readings to recover from", self.strategy_id)

        trough, peak = min(recent), max(recent)

        if current > self.params.oversold and trough <= self.params.oversold:
            direction = SignalDirection.LONG
            depth = self.params.oversold - trough
            span = self.params.oversold
        elif current < self.params.overbought and peak >= self.params.overbought:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
            depth = peak - self.params.overbought
            span = HUNDRED - self.params.overbought
        else:
            return context.hold(
                f"mfi {current:.1f} with no recent extreme to fade", self.strategy_id
            )

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"mfi recovered to {current:.1f} out of a recent extreme",
        )
        return replace_conviction(signal, self._conviction(depth, span))

    def _conviction(self, depth: Decimal, span: Decimal) -> Decimal:
        """A deeper extreme is a larger imbalance to unwind."""
        if span <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + min(depth / span, ONE) * Decimal("0.5"), ONE)


def money_flow_index(candles: Sequence[Candle], period: int = 14) -> Series:
    """Money Flow Index on a 0–100 scale, aligned to ``candles``.

    Positive money flow is the traded value of bars whose typical price rose; negative flow
    is the same for bars where it fell. Unchanged bars contribute to neither, matching the
    standard definition. Windows in which nothing traded — every bar zero volume, or every
    typical price identical — yield ``None``: with no money flow at all the ratio is not
    zero or fifty, it simply does not exist.
    """
    size = len(candles)
    out: list[Decimal | None] = [None] * size
    if size <= period:
        return tuple(out)

    typical = [(candle.high + candle.low + candle.close) / Decimal(3) for candle in candles]
    positive: list[Decimal] = [ZERO]
    negative: list[Decimal] = [ZERO]
    for position in range(1, size):
        flow = typical[position] * candles[position].volume
        rose = typical[position] > typical[position - 1]
        fell = typical[position] < typical[position - 1]
        positive.append(flow if rose else ZERO)
        negative.append(flow if fell else ZERO)

    for position in range(period, size):
        window = slice(position - period + 1, position + 1)
        inflow = sum(positive[window], ZERO)
        outflow = sum(negative[window], ZERO)
        if inflow <= ZERO and outflow <= ZERO:
            continue
        if outflow <= ZERO:
            out[position] = HUNDRED
        else:
            out[position] = HUNDRED - HUNDRED / (ONE + inflow / outflow)
    return tuple(out)


__all__ = ["MoneyFlowIndexParams", "MoneyFlowIndexStrategy", "money_flow_index"]
