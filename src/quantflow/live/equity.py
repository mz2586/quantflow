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
    balances: dict[str, Balance],
    *,
    configured: Decimal,
    quote: str,
    allocation: Decimal | None = None,
) -> Decimal:
    """The quote-currency capital this session may size against.

    Free plus locked: margin held against an open position is still this account's money,
    and excluding it would shrink apparent equity every time a position was opened.

    Only the quote currency counts. A BTC balance on a USDT-quoted book is inventory, not
    buying power, and converting it would mean marking it — a price this function has no
    business fetching.

    Falls back rather than guessing. An unreadable or empty balance yields the configured
    figure: sizing off capital the account does not have is the failure that matters, and
    it is worse than sizing off too little.

    Args:
        balances: Venue balances by asset.
        configured: The fallback figure when the venue cannot be read.
        quote: The quote currency the book is denominated in.
        allocation: An upper bound on the capital this session may use. The wallet may hold
            more; the session is scoped to this. Applied last, and to the fallback as well
            as to the venue reading — an unreadable venue must not become the one moment
            the cap stops applying. Never raises equity above what the account holds:
            allocating capital that is not there would size positions against margin that
            cannot be posted.

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
        return min(configured, allocation) if allocation is not None else configured

    logger.info(
        "equity.resolved_from_venue",
        quote=quote,
        venue_balance=str(available),
        configured=str(configured),
    )
    if allocation is None:
        return available

    capped = min(available, allocation)
    logger.critical(
        "equity.allocation_applied",
        quote=quote,
        venue_balance=str(available),
        allocation=str(allocation),
        session_equity=str(capped),
        note=(
            "the session sizes and risk-limits against this figure, not the wallet; "
            "no funds were moved"
        ),
    )
    return capped


__all__ = ["resolve_starting_equity"]
