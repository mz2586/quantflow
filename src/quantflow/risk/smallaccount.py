"""Whether an account is large enough to hold a symbol's smallest legal position.

Exchanges sell in indivisible lots. That is normally invisible — one BTC lot is a rounding
error against a fifty-thousand-dollar book — but on a small account it becomes the dominant
fact about portfolio construction.

Measured live on 2026-08-21 against a 100 USDT allocation:

* one BTC lot (0.001) cost **76.25 USDT** — 76% of the entire account in a single
  indivisible position;
* one ETH lot (0.01) cost **23.70 USDT** — 24%, which leaves room for a portfolio.

The consequence showed up in the results before it showed up in the design. Eight trades in,
win rate was 62.5% and the account was still down 1.95: the five ETH trades netted +0.09
between them while three BTC trades lost 2.03, because a 1.5% adverse move on a 78%
position outweighs any number of wins on a 24% one. Nothing was wrong with the strategies.
The account simply could not express a BTC opinion at a survivable size.

This is a **market-access rule, not a strategy threshold**. It answers "can this account
own the smallest legal unit of this instrument without that unit dominating the book" — a
question about lot sizes and capital, which no amount of signal quality can change.

It is recalculated from the live price every bar, so a symbol becomes eligible on its own
the moment the arithmetic allows: either the allocation grows or the price falls. Nothing
is hardcoded off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal

from quantflow.core.precision import ZERO

#: The largest share of the allocation a single minimum lot may occupy.
#:
#: At 35% an account can hold roughly three minimum positions, which is enough for the
#: existing correlation and pyramiding rules to mean anything. Above that the "portfolio"
#: is one position wearing a hat: a single adverse move decides the account, and the risk
#: framework's percentages stop describing anything real.
#:
#: Override with QF_MAX_MIN_LOT_FRACTION.
MAX_MIN_LOT_FRACTION = Decimal(os.environ.get("QF_MAX_MIN_LOT_FRACTION", "0.35"))

#: Reason code recorded when a symbol is refused by this rule.
MIN_LOT_TOO_LARGE = "MIN_LOT_TOO_LARGE_FOR_SMALL_ACCOUNT"


@dataclass(frozen=True, slots=True)
class LotEligibility:
    """Whether one symbol may be ENTERED on this account, and the arithmetic behind it."""

    symbol: str
    configured_allocation: Decimal
    symbol_min_notional: Decimal
    symbol_min_lot_fraction: Decimal
    symbol_tradeable: bool
    reason_if_disabled: str | None
    #: Allocation at which this symbol becomes eligible at the current price. Logged so the
    #: operator can see how far away activation is instead of having to derive it.
    allocation_required: Decimal

    def to_dict(self) -> dict[str, str | bool]:
        """Wire form for the runtime status block."""
        return {
            "symbol": self.symbol,
            "configured_allocation": str(self.configured_allocation),
            "symbol_min_notional": str(self.symbol_min_notional),
            "symbol_min_lot_fraction": f"{self.symbol_min_lot_fraction:.2%}",
            "symbol_tradeable": self.symbol_tradeable,
            "reason_if_disabled": self.reason_if_disabled or "",
            "allocation_required": str(self.allocation_required),
        }


def lot_eligibility(
    *,
    symbol: str,
    min_quantity: Decimal,
    price: Decimal,
    allocation: Decimal | None,
    max_fraction: Decimal = MAX_MIN_LOT_FRACTION,
) -> LotEligibility:
    """Judge one symbol against the small-account rule.

    Args:
        symbol: The instrument, for the status record.
        min_quantity: The venue's smallest legal order size.
        price: Current price, so the judgement tracks the market rather than a stale figure.
        allocation: Capital this session may deploy. ``None`` means the account is not
            scoped, in which case the rule does not apply — it exists for small accounts.
        max_fraction: Largest share of the allocation one minimum lot may occupy.

    Returns:
        The verdict and every number behind it.

    """
    min_notional = min_quantity * price
    if allocation is None or allocation <= ZERO:
        return LotEligibility(
            symbol=symbol,
            configured_allocation=allocation or ZERO,
            symbol_min_notional=min_notional,
            symbol_min_lot_fraction=ZERO,
            symbol_tradeable=True,
            reason_if_disabled=None,
            allocation_required=ZERO,
        )

    fraction = min_notional / allocation
    required = min_notional / max_fraction if max_fraction > ZERO else ZERO
    eligible = fraction <= max_fraction
    return LotEligibility(
        symbol=symbol,
        configured_allocation=allocation,
        symbol_min_notional=min_notional,
        symbol_min_lot_fraction=fraction,
        symbol_tradeable=eligible,
        reason_if_disabled=None if eligible else MIN_LOT_TOO_LARGE,
        allocation_required=required,
    )


__all__ = ["MAX_MIN_LOT_FRACTION", "MIN_LOT_TOO_LARGE", "LotEligibility", "lot_eligibility"]
