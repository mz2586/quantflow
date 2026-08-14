"""Market-neutral funding capture.

Short the perpetual, hold the equivalent notional of spot, and collect funding. The legs
offset, so price direction is not a source of profit or loss and the trade earns exactly
one thing: the funding flow, less what it costs to put the hedge on and take it off.

The arithmetic that decides whether this is a strategy or a fee generator:

* Funding settles every 8 hours. Bybit's typical rate is around 0.01% per settlement,
  roughly 0.03% a day when it stays positive.
* A round trip crosses the spread four times — perp and spot to open, perp and spot to
  close. At 0.06% taker that is ~0.24% of notional, before slippage.

So a position needs on the order of **eight consecutive days** of favourable funding just
to return to zero. That is not a tuning problem; it is the shape of the trade. Everything
here exists to measure it honestly rather than to make it look better.

**Sign convention.** A positive funding rate means longs pay shorts. This strategy is short
the perpetual, so positive rates are revenue and negative rates are a bill.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO

#: Legs crossed per round trip: perp and spot to open, perp and spot to close.
LEGS_PER_ROUND_TRIP = 4

#: Funding settlements per day on Bybit.
SETTLEMENTS_PER_DAY = 3


@dataclass(frozen=True, slots=True)
class FundingCaptureParams:
    """Inputs to one simulated funding-capture book.

    ``notional`` is per leg. A delta-neutral pair holds the same notional short in the perp
    and long in spot, so the capital committed is the notional, not twice it — but the fee
    is paid on both legs, which is why cost is four crossings rather than two.
    """

    notional: Decimal
    taker_fee: Decimal
    slippage_bps: Decimal = ZERO

    def __post_init__(self) -> None:
        if self.notional <= ZERO:
            raise ValidationError("funding capture needs a positive notional")
        if self.taker_fee < ZERO:
            raise ValidationError("taker fee cannot be negative")
        if self.slippage_bps < ZERO:
            raise ValidationError("slippage cannot be negative")


@dataclass(frozen=True, slots=True)
class FundingCaptureResult:
    """What one simulated book earned, and what it paid to earn it."""

    funding_collected: Decimal
    costs_paid: Decimal
    net_pnl: Decimal
    round_trips: int
    settlements_held: int
    settlements_total: int

    @property
    def cost_coverage(self) -> Decimal:
        """Funding collected per unit of cost paid. Below 1 means it did not pay for itself."""
        if self.costs_paid == ZERO:
            return ZERO
        return self.funding_collected / self.costs_paid

    def to_dict(self) -> dict[str, object]:
        """Serialise for the validation report and the dashboard."""
        return {
            "funding_collected": str(self.funding_collected),
            "costs_paid": str(self.costs_paid),
            "net_pnl": str(self.net_pnl),
            "round_trips": self.round_trips,
            "settlements_held": self.settlements_held,
            "settlements_total": self.settlements_total,
            "cost_coverage": str(round(self.cost_coverage, 4)),
        }


def funding_payment(notional: Decimal, rate: Decimal) -> Decimal:
    """Funding received by a short perp position at one settlement.

    Positive rate → longs pay shorts → revenue. Negative rate → the short pays.
    """
    return notional * rate


def round_trip_cost(
    notional: Decimal, *, taker_fee: Decimal, slippage_bps: Decimal = ZERO
) -> Decimal:
    """Cost of opening and closing one delta-neutral pair.

    Four crossings, not two: the hedge is two instruments, and both are entered and exited.
    Slippage is charged on every crossing for the same reason — a hedge that only pays fees
    on one leg is not a hedge that was actually put on.
    """
    fee_cost = notional * taker_fee * LEGS_PER_ROUND_TRIP
    slippage_cost = notional * (slippage_bps / Decimal("10000")) * LEGS_PER_ROUND_TRIP
    return fee_cost + slippage_cost


def simulate_funding_capture(
    settlements: list[tuple[Decimal, bool]], *, params: FundingCaptureParams
) -> FundingCaptureResult:
    """Run a book through a series of ``(rate, hold)`` settlements.

    ``hold`` is the decision for that settlement, supplied by the caller rather than
    computed here: the decision rule belongs to the validation harness, so this function
    cannot quietly become the place a threshold gets fitted.

    Costs are charged per *position*, not per settlement — opening and closing is what
    costs money; holding through a settlement is free. Getting that wrong in either
    direction is the difference between a strategy and an artefact.
    """
    funding = ZERO
    costs = ZERO
    round_trips = 0
    held = 0
    in_position = False

    for rate, hold in settlements:
        if hold and not in_position:
            # Opening. The full round trip is charged now, since a position that is opened
            # will be closed: recognising only half would flatter every unclosed book.
            costs += round_trip_cost(
                params.notional,
                taker_fee=params.taker_fee,
                slippage_bps=params.slippage_bps,
            )
            round_trips += 1
            in_position = True
        elif not hold and in_position:
            in_position = False

        if in_position:
            funding += funding_payment(params.notional, rate)
            held += 1

    return FundingCaptureResult(
        funding_collected=funding,
        costs_paid=costs,
        net_pnl=funding - costs,
        round_trips=round_trips,
        settlements_held=held,
        settlements_total=len(settlements),
    )


__all__ = [
    "LEGS_PER_ROUND_TRIP",
    "SETTLEMENTS_PER_DAY",
    "FundingCaptureParams",
    "FundingCaptureResult",
    "funding_payment",
    "round_trip_cost",
    "simulate_funding_capture",
]
