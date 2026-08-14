"""Volatility transition — trade the bar volatility changes state on, and only that bar.

Volatility clusters and it mean-reverts, which together mean the informative moment is not
"volatility is low" or "volatility is high" but the handover between them. A market that has
been quiet for a sustained run has built up a positioning imbalance that nothing has yet
forced anyone to resolve; the first bar on which volatility breaks out of that quiet is the
bar on which resolution starts. Every bar after it is the move already in progress.

So the condition here is a *crossing*, not a level: the previous bar was below the expansion
threshold and this one is at or above it, and the run of bars before that was genuinely
compressed. Requiring the previous bar to be below is what makes it fire once. Requiring the
prior compression is what stops it firing on a market that was already busy.

Distinct from `bollinger_squeeze`, which requires bandwidth to be the *narrowest of a long
lookback* and then waits for **price to close beyond a band**. That is a price trigger with a
volatility precondition, and it can fire many bars after volatility has already expanded —
or never, if the expansion happens without price clearing the band. This one triggers on the
volatility event itself and takes its direction from the transition bar's own body, so it is
in before the price move that a band break is waiting to observe. The single-narrowest-bar
requirement is also relaxed to a sustained run below the median, because "quietest bar in a
hundred and twenty" is one observation and "quiet for five bars running" is five.

Distinct from `atr_expansion`, which has no compression precondition and no crossing
requirement: it will enter on any bar where smoothed ATR happens to sit above its average,
including the fiftieth such bar in a row. The difference is between trading a state and
trading a change of state.
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
from quantflow.strategy.indicators import atr, normalized_atr
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.volatility_regime import trailing_median
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class VolatilityTransitionParams(StrategyParams):
    """Parameters for :class:`VolatilityTransitionStrategy`."""

    atr_period: int = Field(default=14, ge=2, le=100)
    #: History the trailing median of volatility is taken over.
    baseline_period: int = Field(default=60, ge=10, le=1000)
    #: Volatility at or below this multiple of its median counts as compressed.
    contraction_multiple: Decimal = Field(default=Decimal("0.85"), gt=0, le=20)
    #: How many consecutive bars must have been compressed before the transition counts.
    contraction_bars: int = Field(default=5, ge=1, le=100)
    #: Volatility at or above this multiple of its median counts as expanded.
    expansion_multiple: Decimal = Field(default=Decimal("1.3"), gt=0, le=20)
    #: Exit once volatility has fallen back to this multiple; the expansion is over.
    exit_multiple: Decimal = Field(default=Decimal("1.0"), gt=0, le=20)
    #: Minimum share of the transition bar's range that its body must occupy, so the
    #: direction taken from it is a decision rather than a coin flip.
    min_body_ratio: Decimal = Field(default=Decimal("0.4"), gt=0, le=1)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_multiples(self) -> Self:
        if self.contraction_multiple >= self.expansion_multiple:
            raise ValueError(
                f"contraction_multiple ({self.contraction_multiple}) must be below "
                f"expansion_multiple ({self.expansion_multiple}); there is no transition "
                "to trade otherwise"
            )
        if self.exit_multiple >= self.expansion_multiple:
            raise ValueError(
                f"exit_multiple ({self.exit_multiple}) must be below expansion_multiple "
                f"({self.expansion_multiple}), or the strategy would exit on the bar it entered"
            )
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class VolatilityTransitionStrategy(Strategy):
    """Enter on the bar volatility crosses from compression into expansion."""

    strategy_id = "volatility_transition"
    description = "Trades the crossing from a sustained volatility contraction into expansion"
    params_model = VolatilityTransitionParams

    params: VolatilityTransitionParams

    @property
    def warmup_bars(self) -> int:
        """The ATR window, the median baseline and the compression run, end to end."""
        return (
            self.params.atr_period + self.params.baseline_period + self.params.contraction_bars + 2
        )

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit a transition entry or a decay exit."""
        index = context.index
        ratios = self._ratio_series(context.candles)
        current = ratios[index]
        if current is None:
            return context.hold("volatility baseline warming up", self.strategy_id)

        if context.has_position:
            if current <= self.params.exit_multiple:
                return exit_signal(context, self.strategy_id, "the expansion has run its course")
            return context.hold(f"holding, volatility {current:.2f}x median", self.strategy_id)

        if current < self.params.expansion_multiple:
            return context.hold(
                f"volatility {current:.2f}x median, below {self.params.expansion_multiple}x",
                self.strategy_id,
            )

        previous = ratios[index - 1] if index >= 1 else None
        if previous is None:
            return context.hold("no prior reading", self.strategy_id)
        if previous >= self.params.expansion_multiple:
            # Already expanded before this bar: the change of state happened earlier and
            # this is just the move in progress.
            return context.hold("already expanded on the prior bar", self.strategy_id)

        if not self._was_compressed(ratios, index):
            return context.hold("no sustained contraction preceded this", self.strategy_id)

        candle = context.candle
        span = candle.high - candle.low
        if span <= ZERO:
            return context.hold("transition bar has no range", self.strategy_id)
        body = candle.close - candle.open
        body_ratio = abs(body) / span
        if body_ratio < self.params.min_body_ratio or body == ZERO:
            return context.hold("transition bar has no decisive body", self.strategy_id)

        if body < ZERO and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if body > ZERO else SignalDirection.SHORT,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"volatility crossed from contraction to {current:.2f}x its median",
        )
        return replace_conviction(signal, self._conviction(current, body_ratio))

    def _was_compressed(self, ratios: tuple[Decimal | None, ...], index: int) -> bool:
        """Whether the run of bars before the transition was all below the contraction line."""
        start = index - self.params.contraction_bars
        if start < 0:
            return False
        for position in range(start, index):
            ratio = ratios[position]
            if ratio is None or ratio > self.params.contraction_multiple:
                return False
        return True

    def _ratio_series(self, candles: tuple[Candle, ...]) -> tuple[Decimal | None, ...]:
        """Normalised ATR as a multiple of its own trailing median, bar by bar.

        Each bar's median is taken from the readings up to and including that bar only, so
        the series can be read at any index without importing anything from later bars.
        """
        series = normalized_atr(candles, self.params.atr_period)
        out: list[Decimal | None] = [None] * len(series)
        defined: list[Decimal] = []
        for position, value in enumerate(series):
            if value is None or value <= ZERO:
                continue
            defined.append(value)
            if len(defined) < self.params.baseline_period:
                continue
            middle = trailing_median(defined[-self.params.baseline_period :])
            if middle is not None and middle > ZERO:
                out[position] = value / middle
        return tuple(out)

    def _conviction(self, ratio: Decimal, body_ratio: Decimal) -> Decimal:
        """A sharper expansion and a fuller body both read as a cleaner handover."""
        excess = (ratio - self.params.expansion_multiple) / self.params.expansion_multiple
        return min(
            Decimal("0.4")
            + min(max(excess, ZERO), ONE) * Decimal("0.3")
            + body_ratio * Decimal("0.3"),
            ONE,
        )


__all__ = ["VolatilityTransitionParams", "VolatilityTransitionStrategy"]
