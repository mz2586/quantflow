"""Perpetual funding costs.

A perpetual has no expiry, so it is tethered to spot by a periodic payment between the two
sides of the book. Every 8 hours the venue settles it: the position pays or receives
``rate x notional``, and the money leaves or enters the wallet for real.

Nothing charged this before. Backtest and paper both reported PnL as though holding a perp
were free, which flatters every result in proportion to how long positions are held — a
trend follower holding for days was being credited a cost it would certainly have paid.
Expect results to get worse once this is on. That is the correction, not a regression.

**The sign is the part that is easy to get backwards and expensive to get wrong.**
Convention, matching every major venue including Bybit:

- rate **positive** -> longs **pay**, shorts **receive**
- rate **negative** -> longs **receive**, shorts **pay**

A sign error here does not merely mis-state a cost; it turns a systematic drain into a
systematic credit, which makes an unprofitable strategy look profitable for exactly as long
as funding stays one-sided.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.positions import Position

logger = get_logger(__name__)

#: Hours between funding settlements. Bybit, Binance and OKX all settle every 8 hours.
FUNDING_INTERVAL_HOURS = 8

#: The UTC hours at which funding settles: 00:00, 08:00 and 16:00.
FUNDING_HOURS = (0, 8, 16)


@dataclass(frozen=True, slots=True)
class FundingCharge:
    """One settled funding payment.

    ``amount`` is signed from the account's perspective: negative is money paid away,
    positive is money received.
    """

    symbol: Symbol
    settled_at: datetime
    rate: Decimal
    notional: Decimal
    amount: Decimal

    @property
    def is_payment(self) -> bool:
        """Whether the account paid rather than received."""
        return self.amount < ZERO


def funding_stamps(start: datetime, end: datetime) -> list[datetime]:
    """Every funding settlement strictly after ``start`` and at or before ``end``.

    Exclusive of ``start`` so a position opened exactly on a stamp is not charged for a
    period it did not hold through; inclusive of ``end`` so one closed exactly on a stamp
    pays for the period it did.
    """
    if end <= start:
        return []

    stamps: list[datetime] = []
    cursor = start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    # Step back to a known boundary, then walk forward.
    cursor = cursor.replace(hour=(cursor.hour // FUNDING_INTERVAL_HOURS) * FUNDING_INTERVAL_HOURS)
    while cursor <= end:
        if cursor > start:
            stamps.append(cursor)
        cursor += timedelta(hours=FUNDING_INTERVAL_HOURS)
    return stamps


def funding_amount(
    *, side: PositionSide, quantity: Decimal, price: Decimal, rate: Decimal
) -> Decimal:
    """Signed funding for one settlement, from the account's perspective.

    Positive rate: a long pays (negative amount), a short receives (positive amount).
    Negative rate: reversed.
    """
    notional = abs(quantity) * price
    payment = notional * rate
    if side is PositionSide.LONG:
        return -payment
    if side is PositionSide.SHORT:
        return payment
    return ZERO


def charge_for(
    position: Position,
    *,
    price: Decimal,
    rate: Decimal,
    settled_at: datetime,
) -> FundingCharge | None:
    """Build the charge for one position at one settlement, or ``None`` if flat."""
    if position.is_flat or position.quantity == ZERO:
        return None
    amount = funding_amount(side=position.side, quantity=position.quantity, price=price, rate=rate)
    return FundingCharge(
        symbol=position.symbol,
        settled_at=settled_at,
        rate=rate,
        notional=abs(position.quantity) * price,
        amount=amount,
    )


def total_funding(charges: Iterable[FundingCharge]) -> Decimal:
    """Net funding across charges: negative means the account paid overall."""
    return sum((charge.amount for charge in charges), ZERO)


class FundingSchedule:
    """Historical funding rates for one symbol, looked up by settlement time.

    Backtests use the actual rate that applied at each stamp. Where a stamp has no recorded
    rate the schedule returns ``None`` and the caller charges nothing — inventing a rate
    would be fabricating a cost, which is no better than ignoring a real one.
    """

    __slots__ = ("_rates",)

    def __init__(self, rates: Sequence[tuple[datetime, Decimal]] | None = None) -> None:
        self._rates: dict[datetime, Decimal] = {
            stamp.astimezone(UTC): rate for stamp, rate in (rates or ())
        }

    def rate_at(self, settled_at: datetime) -> Decimal | None:
        """The rate that applied at a settlement, or ``None`` if unknown."""
        return self._rates.get(settled_at.astimezone(UTC))

    def add(self, settled_at: datetime, rate: Decimal) -> None:
        """Record a rate for a settlement."""
        self._rates[settled_at.astimezone(UTC)] = rate

    def __len__(self) -> int:
        return len(self._rates)


def borrow_cost(
    *, notional: Decimal, leverage: Decimal, rate_per_period: Decimal = ZERO
) -> Decimal:
    """Financing cost on the borrowed portion of a position.

    At 1x nothing is borrowed, so this is zero by construction rather than by assumption:
    ``notional - notional/leverage`` is the borrowed amount, and at leverage 1 that is
    exactly nothing. The hook exists so that enabling leverage later cannot silently skip
    the cost — but it fabricates nothing today, and the default rate is zero because no
    venue borrow rate is currently wired.
    """
    if leverage <= Decimal("1") or notional <= ZERO:
        return ZERO
    borrowed = notional - (notional / leverage)
    return -(borrowed * rate_per_period)


__all__ = [
    "FUNDING_HOURS",
    "FUNDING_INTERVAL_HOURS",
    "FundingCharge",
    "FundingSchedule",
    "borrow_cost",
    "charge_for",
    "funding_amount",
    "funding_stamps",
    "total_funding",
]
