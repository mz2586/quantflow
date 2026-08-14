"""Momentum divided by realised volatility of returns.

A +5% move means something entirely different on a symbol that typically moves 0.2% a bar
than on one that routinely moves 3%. This strategy judges the move against how much *this*
series normally moves, so one threshold means the same thing on every symbol and in every
regime.

**How this differs from `vol_adjusted_momentum`, which is not a cosmetic difference.**
That strategy divides the *absolute price change* by **ATR**. ATR is an intrabar range
measure: it counts every wick and every gap, so a market that thrashes violently all day
and closes exactly where it opened reads as extremely volatile. This strategy divides the
*fractional return* by the **standard deviation of close-to-close returns** — a
close-to-close dispersion measure, which reads that same day as calm because no
close-to-close move happened. The two disagree most sharply on choppy, high-range,
low-drift markets: precisely where a momentum bet is worst. Denominating in return
standard deviation and scaling it by ``sqrt(lookback)`` also makes the resulting number a
signal-to-noise ratio — how many standard errors the cumulative drift stands away from
zero — rather than a count of average bar ranges, so the threshold has a statistical
reading that an ATR count does not.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, return_volatility
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class NormalizedMomentumParams(StrategyParams):
    """Parameters for :class:`NormalizedMomentumStrategy`."""

    #: Window the cumulative return is measured over.
    lookback: int = Field(default=24, ge=2, le=500)
    #: Window the per-bar return standard deviation is estimated over. Kept separate from
    #: the lookback so the *scale* of a normal move can be measured over a longer, steadier
    #: sample than the move being judged.
    volatility_period: int = Field(default=48, ge=2, le=500)
    #: Signal-to-noise ratio required to enter, in return standard errors.
    entry_score: Decimal = Field(default=Decimal("1.5"), gt=0, le=20)
    #: Ratio at which an open position is closed.
    exit_score: Decimal = Field(default=Decimal("0.25"), ge=-20, le=20)
    #: Score beyond the entry threshold at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("1.5"), gt=0, le=20)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.5"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("5.0"), gt=0, le=20)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_scores(self) -> Self:
        if self.exit_score >= self.entry_score:
            raise ValueError(
                f"exit_score ({self.exit_score}) must be below entry_score "
                f"({self.entry_score}), or a position would close on the bar it opened"
            )
        return self


@register_strategy
class NormalizedMomentumStrategy(Strategy):
    """Trade the trailing return measured in units of its own return volatility."""

    strategy_id = "normalized_momentum"
    description = "Momentum divided by the standard deviation of close-to-close returns"
    params_model = NormalizedMomentumParams

    params: NormalizedMomentumParams

    @property
    def warmup_bars(self) -> int:
        """The longer of the two windows, plus a bar for the first return."""
        return max(self.params.lookback, self.params.volatility_period, self.params.atr_period) + 1

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Enter when the trailing return is large relative to normal return dispersion."""
        index = context.index
        score = self._score(context, index)
        if score is None:
            return context.hold("normalisation warming up", self.strategy_id)

        if context.is_long:
            if score <= self.params.exit_score:
                return exit_signal(context, self.strategy_id, f"score decayed to {score:.2f}")
            return context.hold(f"score {score:.2f} still strong", self.strategy_id)

        if context.is_short:
            if score >= -self.params.exit_score:
                return exit_signal(context, self.strategy_id, f"score decayed to {score:.2f}")
            return context.hold(f"score {score:.2f} still strong", self.strategy_id)

        if score >= self.params.entry_score:
            direction = SignalDirection.LONG
        elif score <= -self.params.entry_score:
            if not self.params.allow_short:
                return context.hold("short entries disabled", self.strategy_id)
            direction = SignalDirection.SHORT
        else:
            return context.hold(
                f"score {score:.2f} below {self.params.entry_score}", self.strategy_id
            )

        signal = entry_signal(
            context,
            self.strategy_id,
            direction,
            atr(context.candles, self.params.atr_period)[index],
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"{abs(score):.2f} return standard errors over {self.params.lookback} bars",
        )
        return replace_conviction(signal, self._conviction(abs(score)))

    def _score(self, context: StrategyContext, index: int) -> Decimal | None:
        """Trailing return divided by its own horizon-scaled return volatility.

        The per-bar standard deviation is scaled by ``sqrt(lookback)`` because independent
        returns accumulate variance linearly: comparing a 24-bar move against a *one-bar*
        standard deviation would make every trending market look like a five-sigma event.
        """
        closes = context.closes
        if index < self.params.lookback:
            return None
        base = closes[index - self.params.lookback]
        if base <= ZERO:
            return None

        volatility = return_volatility(closes, self.params.volatility_period)[index]
        if volatility is None or volatility <= ZERO:
            # A perfectly flat series has no dispersion, and dividing by it would report
            # an infinite edge from a market that has not moved at all.
            return None

        horizon = volatility * Decimal(self.params.lookback).sqrt()
        if horizon <= ZERO:
            return None
        return ((context.price - base) / base) / horizon

    def _conviction(self, score: Decimal) -> Decimal:
        """A higher signal-to-noise ratio reads as a stronger case."""
        excess = score - self.params.entry_score
        if excess <= ZERO:
            return Decimal("0.5")
        return min(Decimal("0.5") + (excess / self.params.conviction_span) * Decimal("0.5"), ONE)


__all__ = ["NormalizedMomentumParams", "NormalizedMomentumStrategy"]
