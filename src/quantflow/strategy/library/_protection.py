"""Shared construction of ATR-scaled protective levels.

Every strategy in the library attaches a stop and a target scaled to volatility rather
than to a fixed percentage, which would be far too tight in a violent market and far too
wide in a calm one. Factored out so ten strategies express that identically: a subtle
difference in how one of them places its stop would show up in the leaderboard as if it
were a difference in the *idea*, which is exactly the confound a research framework
exists to remove.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.core.precision import ZERO
from quantflow.domain.enums import SignalDirection
from quantflow.domain.signals import Signal
from quantflow.strategy.base import StrategyContext


def protective_levels(
    price: Decimal,
    direction: SignalDirection,
    volatility: Decimal | None,
    stop_multiple: Decimal,
    target_multiple: Decimal,
) -> tuple[Decimal | None, Decimal | None]:
    """Stop and target at ATR multiples either side of ``price``.

    Returns ``(None, None)`` when ATR is unavailable or non-positive. The signal then
    carries no stop of its own and the risk engine applies its default — it is never
    emitted unprotected, and a strategy is never allowed to invent a stop out of a
    volatility estimate that does not exist.
    """
    if volatility is None or volatility <= ZERO:
        return None, None

    stop_distance = volatility * stop_multiple
    target_distance = volatility * target_multiple
    if direction is SignalDirection.LONG:
        stop = price - stop_distance
        target = price + target_distance
    else:
        stop = price + stop_distance
        target = price - target_distance

    # A stop at or below zero is not a stop. Better to hand the decision to the risk
    # engine than to attach a level that can never be reached.
    if stop <= ZERO:
        return None, None
    if target <= ZERO:
        return stop, None
    return stop, target


def entry_signal(
    context: StrategyContext,
    strategy_id: str,
    direction: SignalDirection,
    volatility: Decimal | None,
    stop_multiple: Decimal,
    target_multiple: Decimal,
    reason: str,
) -> Signal:
    """An entry signal with ATR-scaled protection attached."""
    stop, target = protective_levels(
        context.price, direction, volatility, stop_multiple, target_multiple
    )
    return Signal(
        symbol=context.symbol,
        direction=direction,
        timestamp=context.now,
        strategy_id=strategy_id,
        reference_price=context.price,
        stop_loss_price=stop,
        take_profit_price=target,
        reason=reason,
    )


def exit_signal(context: StrategyContext, strategy_id: str, reason: str) -> Signal:
    """A signal closing whatever position is open."""
    return Signal(
        symbol=context.symbol,
        direction=SignalDirection.CLOSE,
        timestamp=context.now,
        strategy_id=strategy_id,
        reference_price=context.price,
        reason=reason,
    )
