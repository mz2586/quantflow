"""SuperTrend — a trailing volatility band that decides which side of the market to be on.

Every other trend member in the library answers "is there a trend?" with an average of
past prices, which necessarily lags and, worse, lags by an amount that changes with
volatility: a 26-period EMA is roughly a week behind in a calm market and roughly a week
behind in a violent one, even though the violent one has already moved three times as far
by then. SuperTrend replaces the fixed lookback with a **distance**: the line sits an ATR
multiple away from price and ratchets closer whenever price advances, so how long it takes
to flip is set by how far the market moves, not by how many bars have printed.

The consequence worth having is that the flip level is knowable *in advance* — it is the
band, and the band is on the chart before the bar that breaks it. That makes it a genuine
stop-and-reverse rule rather than a signal that can only be recognised afterwards.

Shorts are supported and are symmetric here: the mechanism is a band on either side of
price, and there is nothing about the downside band that is weaker than the upside one.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.indicators import atr, supertrend
from quantflow.strategy.library._protection import entry_signal, exit_signal
from quantflow.strategy.library.vwap_reversion import replace_conviction
from quantflow.strategy.registry import register_strategy


class SupertrendParams(StrategyParams):
    """Parameters for :class:`SupertrendStrategy`."""

    #: ATR lookback for the band. 10 with a 3x multiplier is the published default and is
    #: kept as the default here so the leaderboard compares the *idea* against the others
    #: rather than a private tuning of it.
    period: int = Field(default=10, ge=2, le=100)
    band_multiple: Decimal = Field(default=Decimal("3.0"), gt=0, le=20)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiple: Decimal = Field(default=Decimal("2.0"), gt=0, le=10)
    atr_target_multiple: Decimal = Field(default=Decimal("4.0"), gt=0, le=20)
    #: Break of the band, in ATR units, at which conviction saturates.
    conviction_span: Decimal = Field(default=Decimal("1.5"), gt=0, le=10)
    allow_short: bool = False

    @model_validator(mode="after")
    def _validate_multiples(self) -> Self:
        if self.atr_target_multiple <= self.atr_stop_multiple:
            raise ValueError("atr_target_multiple must exceed atr_stop_multiple")
        return self


@register_strategy
class SupertrendStrategy(Strategy):
    """Follows the SuperTrend band, entering on the bar its direction flips."""

    strategy_id = "supertrend"
    description = "ATR-banded trailing stop; enters when the band flips direction"
    params_model = SupertrendParams

    params: SupertrendParams

    @property
    def warmup_bars(self) -> int:
        """The band ratchets, so its first readings still carry their seed.

        ATR needs ``period`` bars before it means anything, and the band built on it needs
        roughly as long again before the ratchet — not the seed — is what sets its level.
        """
        return max(self.params.period * 3, self.params.atr_period + 1)

    def generate(  # noqa: PLR0911 - a flat chain of guard clauses is clearer here
        self, context: StrategyContext
    ) -> Signal:
        """Emit an entry on a band flip, or close a position the band has flipped against."""
        index = context.index
        if index < 1:
            return context.hold("no prior bar", self.strategy_id)

        line, direction = supertrend(context.candles, self.params.period, self.params.band_multiple)
        current = direction[index]
        previous = direction[index - 1]
        breached = line[index - 1]
        if current is None or previous is None or breached is None:
            return context.hold("supertrend warming up", self.strategy_id)

        flipped_up = previous < ZERO < current
        flipped_down = current < ZERO < previous

        if context.is_long:
            if flipped_down:
                return exit_signal(context, self.strategy_id, "supertrend flipped down")
            return context.hold("holding long, supertrend up", self.strategy_id)

        if context.is_short:
            if flipped_up:
                return exit_signal(context, self.strategy_id, "supertrend flipped up")
            return context.hold("holding short, supertrend down", self.strategy_id)

        if not flipped_up and not flipped_down:
            return context.hold("supertrend unchanged", self.strategy_id)
        if flipped_down and not self.params.allow_short:
            return context.hold("short entries disabled", self.strategy_id)

        volatility = atr(context.candles, self.params.atr_period)[index]
        long = flipped_up
        signal = entry_signal(
            context,
            self.strategy_id,
            SignalDirection.LONG if long else SignalDirection.SHORT,
            volatility,
            self.params.atr_stop_multiple,
            self.params.atr_target_multiple,
            f"close {context.price} crossed the supertrend band at {breached}",
        )
        excess = (context.price - breached) if long else (breached - context.price)
        return replace_conviction(signal, self._conviction(excess, volatility))

    def _conviction(self, excess: Decimal, volatility: Decimal | None) -> Decimal:
        """Scale conviction with how decisively the band was broken.

        A close that clears the band by a tick is the same *event* as one that clears it by
        a full ATR, and collapsing both to 1.0 would leave the orchestrator no way to
        prefer the second. Measured in ATR units so the number means the same thing across
        symbols and volatility regimes.
        """
        if volatility is None or volatility <= ZERO or excess <= ZERO:
            return Decimal("0.35")
        scaled = excess / (volatility * self.params.conviction_span)
        return min(Decimal("0.35") + min(scaled, ONE) * Decimal("0.65"), ONE)


__all__ = ["SupertrendParams", "SupertrendStrategy"]
