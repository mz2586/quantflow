"""Exposure that has been committed but not yet filled.

A resting entry order is exposure the account has already decided to take: it becomes a
position with no further decision, at a price already chosen. Leaving it out of the risk
caps let one symbol pass the position check twice on its way past it.

Measured on 2026-08-17: a WLD entry was placed at 01:00 while the symbol was flat and
passed the 20% position cap at 17.7%. It rested for 5.4 hours and filled at 06:25, on top
of a position opened at 01:30 by a later order that had also passed at 17.7%. The symbol
finished at 35.4% of equity against a 20% cap, and every individual check had passed
honestly on the information it was given.

Lives here, rather than in either engine, because paper and backtest must compute it the
same way. Two copies of this would be two risk models, and the invariant this codebase
rests on is that the live path is the one the backtest already exercised.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Any, Protocol

from quantflow.core.precision import ZERO
from quantflow.domain.orders import Order


class _MarkSource(Protocol):
    """Anything that can price a symbol for valuation."""

    def mark_price(self, symbol: Any) -> Decimal | None:
        """Latest mark for ``symbol``, or ``None`` when unknown."""
        ...


def resting_entry_notional(
    open_orders: Iterable[Order] | Sequence[Order], marks: _MarkSource
) -> dict[str, Decimal]:
    """Notional of unfilled entry orders, keyed by symbol.

    Args:
        open_orders: Orders still working at the venue or in the simulator.
        marks: Price source, used only for orders carrying no limit price.

    Returns:
        Symbol to committed-but-unfilled notional. Symbols with none are absent rather
        than zero, so a caller can distinguish "nothing resting" from "not considered".

    """
    totals: dict[str, Decimal] = {}
    for order in open_orders:
        # Reduce-only orders are protection: a stop or target *removes* exposure, and
        # charging it against the risk budget would penalise protecting a position.
        if order.reduce_only:
            continue
        price = order.price or marks.mark_price(order.symbol)
        if price is None or price <= ZERO:
            continue
        remaining = order.quantity - order.filled_quantity
        if remaining <= ZERO:
            continue
        key = str(order.symbol)
        totals[key] = totals.get(key, ZERO) + remaining * price
    return totals


__all__ = ["resting_entry_notional"]
