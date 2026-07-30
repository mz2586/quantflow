"""Positions and lot-level accounting.

Positions are built by folding fills. Realised PnL uses **FIFO** lot matching, which matches
how most jurisdictions expect crypto gains to be reported and is deterministic — average-cost
accounting hides the timing of individual entries, which makes trade-level attribution and
holding-period statistics impossible to reconstruct.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.enums import OrderSide, PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill


@dataclass(frozen=True, slots=True)
class Lot:
    """An open tranche of a position, retained for FIFO matching."""

    quantity: Decimal
    price: Decimal
    opened_at: datetime
    fee: Decimal = ZERO

    def __post_init__(self) -> None:
        """Validate the lot."""
        if self.quantity <= ZERO:
            raise ValidationError(f"lot quantity must be positive, got {self.quantity}")
        if self.price <= ZERO:
            raise ValidationError(f"lot price must be positive, got {self.price}")

    @property
    def cost(self) -> Decimal:
        """Gross cost of the lot, excluding fees."""
        return self.quantity * self.price

    def take(self, quantity: Decimal) -> tuple[Lot, Lot | None]:
        """Split the lot into ``(taken, remainder)``.

        ``remainder`` is ``None`` when the lot is fully consumed.
        """
        if quantity <= ZERO or quantity > self.quantity:
            raise ValidationError(
                f"cannot take {quantity} from a lot of {self.quantity}",
            )
        taken_fee = self.fee * safe_divide(quantity, self.quantity)
        taken = Lot(quantity=quantity, price=self.price, opened_at=self.opened_at, fee=taken_fee)
        if quantity == self.quantity:
            return taken, None
        remainder = Lot(
            quantity=self.quantity - quantity,
            price=self.price,
            opened_at=self.opened_at,
            fee=self.fee - taken_fee,
        )
        return taken, remainder


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """A round-trip: one entry lot matched against one exit.

    This is the unit the analytics layer and the AI trade-journal operate on.
    """

    symbol: Symbol
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    entry_time: datetime
    exit_time: datetime
    gross_pnl: Decimal
    fees: Decimal
    strategy_id: str | None = None

    @property
    def net_pnl(self) -> Decimal:
        """PnL after fees."""
        return self.gross_pnl - self.fees

    @property
    def return_pct(self) -> Decimal:
        """Net return on the capital committed to the entry."""
        return safe_divide(self.net_pnl, self.quantity * self.entry_price)

    @property
    def holding_period(self) -> Decimal:
        """Holding period in seconds."""
        return Decimal(str((self.exit_time - self.entry_time).total_seconds()))

    @property
    def is_win(self) -> bool:
        """Whether the round-trip was profitable net of fees."""
        return self.net_pnl > ZERO


@dataclass(frozen=True, slots=True)
class Position:
    """Net exposure in a single symbol, with FIFO lots and realised PnL.

    Immutable — :meth:`apply_fill` returns a new position plus any round-trips the fill
    closed. Position flips (long to short in one fill) are handled by closing the existing
    exposure first and opening the residual on the other side.
    """

    symbol: Symbol
    quantity: Decimal = ZERO
    """Signed: positive long, negative short."""
    lots: tuple[Lot, ...] = field(default_factory=tuple)
    realized_pnl: Decimal = ZERO
    fees_paid: Decimal = ZERO
    opened_at: datetime | None = None
    updated_at: datetime | None = None
    strategy_id: str | None = None
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate lot/quantity coherence."""
        lot_total = sum((lot.quantity for lot in self.lots), ZERO)
        if lot_total != abs(self.quantity):
            raise ValidationError(
                f"lot total {lot_total} does not match position size {abs(self.quantity)}",
                symbol=str(self.symbol),
            )

    # ------------------------------------------------------------------ #
    # Derived state
    # ------------------------------------------------------------------ #
    @property
    def side(self) -> PositionSide:
        """Direction of the exposure."""
        return PositionSide.from_signed_quantity(self.quantity)

    @property
    def is_flat(self) -> bool:
        """Whether there is no exposure."""
        return self.quantity == ZERO

    @property
    def absolute_quantity(self) -> Decimal:
        """Unsigned size."""
        return abs(self.quantity)

    @property
    def average_entry_price(self) -> Decimal:
        """Weighted average price of the open lots."""
        total = sum((lot.quantity for lot in self.lots), ZERO)
        if total == ZERO:
            return ZERO
        cost = sum((lot.cost for lot in self.lots), ZERO)
        return cost / total

    @property
    def cost_basis(self) -> Decimal:
        """Gross cost of the open exposure."""
        return sum((lot.cost for lot in self.lots), ZERO)

    def market_value(self, price: Decimal) -> Decimal:
        """Signed mark-to-market value at ``price``."""
        return self.quantity * price

    def notional(self, price: Decimal) -> Decimal:
        """Unsigned mark-to-market exposure at ``price``."""
        return abs(self.quantity) * price

    def unrealized_pnl(self, price: Decimal) -> Decimal:
        """Mark-to-market PnL on the open exposure at ``price``."""
        if self.is_flat:
            return ZERO
        return (price - self.average_entry_price) * self.quantity

    def unrealized_pnl_pct(self, price: Decimal) -> Decimal:
        """Unrealised PnL as a fraction of the cost basis."""
        return safe_divide(self.unrealized_pnl(price), self.cost_basis)

    def total_pnl(self, price: Decimal) -> Decimal:
        """Realised plus unrealised PnL, net of fees already paid."""
        return self.realized_pnl + self.unrealized_pnl(price)

    def is_stop_breached(self, price: Decimal) -> bool:
        """Whether ``price`` has reached the position's stop loss."""
        if self.stop_loss_price is None or self.is_flat:
            return False
        if self.side is PositionSide.LONG:
            return price <= self.stop_loss_price
        return price >= self.stop_loss_price

    def is_target_reached(self, price: Decimal) -> bool:
        """Whether ``price`` has reached the position's take-profit."""
        if self.take_profit_price is None or self.is_flat:
            return False
        if self.side is PositionSide.LONG:
            return price >= self.take_profit_price
        return price <= self.take_profit_price

    def with_protection(
        self,
        *,
        stop_loss_price: Decimal | None = None,
        take_profit_price: Decimal | None = None,
    ) -> Position:
        """Attach or update protective levels."""
        return replace(
            self,
            stop_loss_price=(
                stop_loss_price if stop_loss_price is not None else self.stop_loss_price
            ),
            take_profit_price=(
                take_profit_price if take_profit_price is not None else self.take_profit_price
            ),
        )

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def apply_fill(self, fill: Fill) -> tuple[Position, tuple[ClosedTrade, ...]]:
        """Fold a fill into the position.

        Returns:
            ``(new_position, closed_trades)``. ``closed_trades`` is empty when the fill only
            adds exposure.

        Raises:
            ValidationError: if the fill is for a different symbol.

        """
        if fill.symbol != self.symbol:
            raise ValidationError(
                f"fill for {fill.symbol} cannot be applied to a {self.symbol} position"
            )

        signed = fill.signed_quantity
        is_increase = self.is_flat or (signed > ZERO) == (self.quantity > ZERO)

        if is_increase:
            return self._increase(fill, signed), ()
        return self._reduce(fill, signed)

    def _increase(self, fill: Fill, signed: Decimal) -> Position:
        lot = Lot(
            quantity=fill.quantity,
            price=fill.price,
            opened_at=fill.timestamp,
            fee=fill.fee,
        )
        return replace(
            self,
            quantity=self.quantity + signed,
            lots=(*self.lots, lot),
            fees_paid=self.fees_paid + fill.fee,
            opened_at=self.opened_at or fill.timestamp,
            updated_at=fill.timestamp,
        )

    def _reduce(self, fill: Fill, signed: Decimal) -> tuple[Position, tuple[ClosedTrade, ...]]:
        side = self.side
        closing_quantity = min(fill.quantity, self.absolute_quantity)
        remaining_lots = deque(self.lots)
        closed: list[ClosedTrade] = []
        to_close = closing_quantity
        gross_total = ZERO

        # Exit fees are apportioned across the lots this fill closes.
        exit_fee_rate = safe_divide(fill.fee, fill.quantity)

        while to_close > ZERO and remaining_lots:
            lot = remaining_lots.popleft()
            take_quantity = min(to_close, lot.quantity)
            taken, remainder = lot.take(take_quantity)
            if remainder is not None:
                remaining_lots.appendleft(remainder)

            direction = Decimal(side.sign)
            gross = (fill.price - taken.price) * take_quantity * direction
            gross_total += gross
            closed.append(
                ClosedTrade(
                    symbol=self.symbol,
                    side=side,
                    quantity=take_quantity,
                    entry_price=taken.price,
                    exit_price=fill.price,
                    entry_time=taken.opened_at,
                    exit_time=fill.timestamp,
                    gross_pnl=gross,
                    fees=taken.fee + exit_fee_rate * take_quantity,
                    strategy_id=self.strategy_id,
                )
            )
            to_close -= take_quantity

        residual = fill.quantity - closing_quantity
        base = replace(
            self,
            quantity=self.quantity + signed if residual == ZERO else ZERO,
            lots=tuple(remaining_lots),
            realized_pnl=self.realized_pnl + gross_total,
            fees_paid=self.fees_paid + fill.fee,
            updated_at=fill.timestamp,
        )

        if residual == ZERO:
            if base.is_flat:
                base = replace(base, opened_at=None, stop_loss_price=None, take_profit_price=None)
            return base, tuple(closed)

        # Position flip: the fill closed everything and opened the other side.
        flipped = replace(
            base,
            quantity=ZERO,
            lots=(),
            opened_at=None,
            stop_loss_price=None,
            take_profit_price=None,
        )
        flip_lot = Lot(quantity=residual, price=fill.price, opened_at=fill.timestamp, fee=ZERO)
        flipped = replace(
            flipped,
            quantity=residual * fill.side.sign,
            lots=(flip_lot,),
            opened_at=fill.timestamp,
        )
        return flipped, tuple(closed)

    @classmethod
    def from_fills(
        cls, symbol: Symbol, fills: Iterable[Fill], *, strategy_id: str | None = None
    ) -> tuple[Position, tuple[ClosedTrade, ...]]:
        """Rebuild a position by replaying fills in order.

        Used for crash recovery and for reconciling against the exchange.
        """
        position = cls(symbol=symbol, strategy_id=strategy_id)
        all_closed: list[ClosedTrade] = []
        for fill in sorted(fills, key=lambda item: item.timestamp):
            position, closed = position.apply_fill(fill)
            all_closed.extend(closed)
        return position, tuple(all_closed)

    def closing_side(self) -> OrderSide | None:
        """The order side that would flatten this position, or ``None`` if flat."""
        if self.is_flat:
            return None
        return self.side.exit_side


def net_exposure(positions: Sequence[Position], prices: dict[Symbol, Decimal]) -> Decimal:
    """Signed sum of position market values.

    Raises:
        ValidationError: if a price is missing for a non-flat position — silently treating
        it as zero would understate exposure and defeat the risk limits.

    """
    total = ZERO
    for position in positions:
        if position.is_flat:
            continue
        price = prices.get(position.symbol)
        if price is None:
            raise ValidationError(
                f"missing mark price for {position.symbol}", symbol=str(position.symbol)
            )
        total += position.market_value(price)
    return total


def gross_exposure(positions: Sequence[Position], prices: dict[Symbol, Decimal]) -> Decimal:
    """Sum of absolute position market values."""
    total = ZERO
    for position in positions:
        if position.is_flat:
            continue
        price = prices.get(position.symbol)
        if price is None:
            raise ValidationError(
                f"missing mark price for {position.symbol}", symbol=str(position.symbol)
            )
        total += position.notional(price)
    return total
