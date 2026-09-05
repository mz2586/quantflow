"""FX cost model — spread, commission, slippage and overnight financing.

FX costs are not a single taker fee. A trade pays the spread on entry, a per-lot commission
on raw-spread accounts, slippage on market orders, and then *keeps paying* a swap every
night it is held — including a triple charge on one weekday, which is how the venue bills
the weekend it does not roll through. On the tight stops FX strategies use, these four
together routinely exceed the gross edge, so :meth:`TradeCosts.net_edge` is what a decision
should be taken on, never the gross number.

Sign convention: every field on :class:`TradeCosts` is a **cost** — positive means money
out. Venue swap *rates* keep their native sign (negative = charged), so a positive swap
rate produces a negative swap cost, i.e. a credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from typing import Final

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide
from quantflow.forex.instruments import ForexInstrument
from quantflow.forex.sessions import require_utc, require_weekday

#: Commission Bybit's documentation quotes for its MT5 Tight-Spread account: USD 6 per lot,
#: round turn.
#:
#: This is a **reference value we have not verified against a live statement**. It is not a
#: default — :class:`ForexCostModel` starts at zero commission, and an operator must set
#: this deliberately after checking their own account's contract specification.
BYBIT_TIGHT_SPREAD_COMMISSION_PER_LOT_ROUND_TURN: Final = Decimal("6")

#: Broker rollover happens at 00:00 in the venue's server time, which for most FX brokers
#: is 21:00 or 22:00 UTC. 21:00 UTC lines up with the New York close used elsewhere here.
DEFAULT_ROLLOVER_TIME: Final = time(21, 0)

#: A position held across the weekend is not charged three extra nights on top of the
#: triple-swap day — that charge *is* the weekend. Rollovers on these weekdays are skipped,
#: minus whichever day the instrument nominates as its triple-swap day.
_WEEKEND_ROLLOVER_WEEKDAYS: Final = frozenset({4, 5, 6})

TRIPLE_SWAP_MULTIPLIER: Final = Decimal("3")


def swap_nights(
    opened_at: datetime,
    closed_at: datetime,
    triple_swap_weekday: int,
    rollover_time: time = DEFAULT_ROLLOVER_TIME,
) -> Decimal:
    """Number of swap charges a position accrues between two instants.

    Counts rollover instants in ``(opened_at, closed_at]`` — a position opened exactly on a
    rollover is not billed for it, one closed exactly on a rollover is. The rollover falling
    on ``triple_swap_weekday`` counts three; those falling on the remaining weekend days
    count nothing, because the triple charge already covers them.
    """
    opened_at = require_utc(opened_at, field="opened_at")
    closed_at = require_utc(closed_at, field="closed_at")
    require_weekday(triple_swap_weekday, field="triple_swap_weekday")
    if closed_at < opened_at:
        raise ValidationError("closed_at must not precede opened_at")

    skipped = _WEEKEND_ROLLOVER_WEEKDAYS - {triple_swap_weekday}
    nights = ZERO
    rollover = datetime.combine(opened_at.date(), rollover_time, tzinfo=UTC)
    if rollover <= opened_at:
        rollover += timedelta(days=1)
    while rollover <= closed_at:
        weekday = rollover.weekday()
        if weekday == triple_swap_weekday:
            nights += TRIPLE_SWAP_MULTIPLIER
        elif weekday not in skipped:
            nights += Decimal("1")
        rollover += timedelta(days=1)
    return nights


@dataclass(frozen=True, slots=True)
class TradeCosts:
    """A cost breakdown in account currency. Positive means money out."""

    spread: Decimal
    commission: Decimal
    slippage: Decimal
    swap: Decimal

    @property
    def total(self) -> Decimal:
        """Sum of every cost component."""
        return self.spread + self.commission + self.slippage + self.swap

    def net_edge(self, gross_edge: Decimal) -> Decimal:
        """Expected edge after every cost — the only number worth deciding on."""
        return gross_edge - self.total


@dataclass(frozen=True, slots=True)
class ForexCostModel:
    """Per-account cost parameters.

    Defaults are all zero. Nothing here is guessed on the operator's behalf: an unset
    commission that silently defaulted to a plausible number would make every backtest and
    every net-edge gate quietly wrong in the same direction.
    """

    commission_per_lot_round_turn: Decimal = ZERO
    slippage_points: Decimal = ZERO
    spread_points_override: Decimal | None = None
    include_swap: bool = True
    rollover_time: time = DEFAULT_ROLLOVER_TIME

    def __post_init__(self) -> None:
        """Reject negative cost parameters."""
        if self.commission_per_lot_round_turn < ZERO:
            raise ValidationError("commission_per_lot_round_turn must not be negative")
        if self.slippage_points < ZERO:
            raise ValidationError("slippage_points must not be negative")
        if self.spread_points_override is not None and self.spread_points_override < ZERO:
            raise ValidationError("spread_points_override must not be negative")

    def spread_cost(self, instrument: ForexInstrument, lots: Decimal) -> Decimal:
        """Cost of crossing the spread once, in account currency."""
        points = (
            instrument.spread_points
            if self.spread_points_override is None
            else self.spread_points_override
        )
        return points * instrument.value_per_point_per_lot * lots

    def commission_cost(self, lots: Decimal) -> Decimal:
        """Round-turn commission for ``lots``."""
        return self.commission_per_lot_round_turn * lots

    def slippage_cost(self, instrument: ForexInstrument, lots: Decimal) -> Decimal:
        """Expected slippage cost for ``lots``."""
        return self.slippage_points * instrument.value_per_point_per_lot * lots

    def swap_cost(
        self,
        instrument: ForexInstrument,
        lots: Decimal,
        side: OrderSide,
        opened_at: datetime | None = None,
        closed_at: datetime | None = None,
    ) -> Decimal:
        """Overnight financing cost for holding ``lots`` between two instants.

        Returns zero when either instant is unknown or when swap is disabled. A positive
        venue swap rate yields a negative cost — that is a credit, and it is meant to
        improve the net edge.
        """
        if not self.include_swap or opened_at is None or closed_at is None:
            return ZERO
        nights = swap_nights(
            opened_at,
            closed_at,
            instrument.triple_swap_weekday,
            self.rollover_time,
        )
        rate = instrument.swap_long if side is OrderSide.BUY else instrument.swap_short
        return -(rate * lots * nights)

    def estimate(
        self,
        instrument: ForexInstrument,
        lots: Decimal,
        side: OrderSide,
        opened_at: datetime | None = None,
        closed_at: datetime | None = None,
    ) -> TradeCosts:
        """Full cost breakdown for one round-turn trade."""
        if lots <= ZERO:
            raise ValidationError(f"lots must be positive, got {lots}", symbol=instrument.symbol)
        return TradeCosts(
            spread=self.spread_cost(instrument, lots),
            commission=self.commission_cost(lots),
            slippage=self.slippage_cost(instrument, lots),
            swap=self.swap_cost(instrument, lots, side, opened_at, closed_at),
        )


def expected_net_edge(gross_edge: Decimal, costs: TradeCosts) -> Decimal:
    """Gross edge less every cost component."""
    return costs.net_edge(gross_edge)


def break_even_points(instrument: ForexInstrument, lots: Decimal, costs: TradeCosts) -> Decimal:
    """How far price must move in your favour, in points, just to cover ``costs``."""
    point_value = instrument.value_per_point_per_lot * lots
    if point_value <= ZERO:
        raise ValidationError(
            "cannot compute break-even without a point value", symbol=instrument.symbol
        )
    return costs.total / point_value


__all__ = [
    "BYBIT_TIGHT_SPREAD_COMMISSION_PER_LOT_ROUND_TURN",
    "DEFAULT_ROLLOVER_TIME",
    "TRIPLE_SWAP_MULTIPLIER",
    "ForexCostModel",
    "TradeCosts",
    "break_even_points",
    "expected_net_edge",
    "swap_nights",
]
