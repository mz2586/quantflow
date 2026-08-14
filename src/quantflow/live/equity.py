"""Resolve the equity a live session should size against.

Position size is a percentage of equity, so equity is the number every risk limit is
ultimately expressed in. Taking it from a constant while the account holds something else
does not loosen the limits — it silently redefines what they are a percentage *of*.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.portfolio import Balance

logger = get_logger(__name__)


def resolve_starting_equity(
    balances: dict[str, Balance], *, configured: Decimal, quote: str
) -> Decimal:
    """The quote-currency capital actually available, or ``configured`` if unknown.

    Free plus locked: margin held against an open position is still this account's money,
    and excluding it would shrink apparent equity every time a position was opened.

    Only the quote currency counts. A BTC balance on a USDT-quoted book is inventory, not
    buying power, and converting it would mean marking it — a price this function has no
    business fetching.

    Falls back rather than guessing. An unreadable or empty balance yields the configured
    figure: sizing off capital the account does not have is the failure that matters, and
    it is worse than sizing off too little.
    """
    balance = balances.get(quote)
    available = (balance.free + balance.locked) if balance is not None else ZERO

    if available <= ZERO:
        logger.warning(
            "equity.venue_balance_unavailable",
            quote=quote,
            configured=str(configured),
            reason="no positive quote balance; falling back to the configured equity",
        )
        return configured

    logger.info(
        "equity.resolved_from_venue",
        quote=quote,
        venue_balance=str(available),
        configured=str(configured),
    )
    return available


__all__ = ["resolve_starting_equity"]
