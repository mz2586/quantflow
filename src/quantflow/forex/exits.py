"""Intrabar exit evaluation.

A bar is four numbers, not a path. When both the stop and the target sit inside the same
bar's range there is no way to know which was touched first, so this module always resolves
the ambiguity **against** the trade and flags it, rather than quietly booking the winner —
that single choice is the difference between a backtest that survives contact with a live
account and one that does not.

Gaps are handled explicitly too: if a bar *opens* through the stop, the fill is modelled at
the open, not at the stop level, because that is where the venue would have filled it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import OrderSide
from quantflow.forex.protocol import ForexBar, ForexPosition


class IntrabarOutcome(StrEnum):
    """What, if anything, closed the position inside a bar."""

    NONE = "none"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass(frozen=True, slots=True)
class IntrabarExit:
    """The modelled outcome of one bar against one position's protective levels."""

    outcome: IntrabarOutcome
    price: Decimal | None = None
    ambiguous: bool = False
    gapped: bool = False

    def __bool__(self) -> bool:
        """Truthy when the bar closed the position."""
        return self.outcome is not IntrabarOutcome.NONE


def evaluate_intrabar_exit(
    bar: ForexBar,
    side: OrderSide,
    stop_loss: Decimal | None = None,
    take_profit: Decimal | None = None,
) -> IntrabarExit:
    """Decide whether ``bar`` would have taken out a stop or a target.

    When both levels fall inside the bar's range the stop wins and ``ambiguous`` is set, so
    a caller can count how much of a result rests on that assumption.
    """
    if side is OrderSide.BUY:
        stop_hit = stop_loss is not None and bar.low <= stop_loss
        target_hit = take_profit is not None and bar.high >= take_profit
    else:
        stop_hit = stop_loss is not None and bar.high >= stop_loss
        target_hit = take_profit is not None and bar.low <= take_profit

    if stop_hit and stop_loss is not None:
        gapped = bar.open <= stop_loss if side is OrderSide.BUY else bar.open >= stop_loss
        return IntrabarExit(
            outcome=IntrabarOutcome.STOP_LOSS,
            price=bar.open if gapped else stop_loss,
            ambiguous=target_hit,
            gapped=gapped,
        )
    if target_hit and take_profit is not None:
        gapped = bar.open >= take_profit if side is OrderSide.BUY else bar.open <= take_profit
        return IntrabarExit(
            outcome=IntrabarOutcome.TAKE_PROFIT,
            price=bar.open if gapped else take_profit,
            gapped=gapped,
        )
    return IntrabarExit(outcome=IntrabarOutcome.NONE)


def evaluate_position_exit(bar: ForexBar, position: ForexPosition) -> IntrabarExit:
    """Apply :func:`evaluate_intrabar_exit` to a live position's own levels."""
    if bar.symbol != position.symbol:
        raise ValidationError(
            "bar and position refer to different symbols",
            bar_symbol=bar.symbol,
            position_symbol=position.symbol,
        )
    return evaluate_intrabar_exit(bar, position.side, position.stop_loss, position.take_profit)
