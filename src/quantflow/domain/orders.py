"""Orders, fills and the OMS state machine transition table."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Final, Self

from quantflow.core.errors import InvalidOrderTransitionError, ValidationError
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.enums import (
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quantflow.domain.instruments import Symbol

#: Legal OMS transitions. Anything absent here raises
#: :class:`~quantflow.core.errors.InvalidOrderTransitionError`, which is what stops a
#: late websocket message from resurrecting a cancelled order.
ORDER_TRANSITIONS: Final[dict[OrderStatus, frozenset[OrderStatus]]] = {
    OrderStatus.PENDING_NEW: frozenset(
        {
            OrderStatus.NEW,
            OrderStatus.REJECTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.NEW: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.PENDING_CANCEL: frozenset(
        {
            OrderStatus.CANCELLED,
            OrderStatus.FILLED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}


def can_transition(current: OrderStatus, target: OrderStatus) -> bool:
    """Whether ``current -> target`` is a legal OMS transition."""
    return target in ORDER_TRANSITIONS[current]


def new_client_order_id(prefix: str = "qf") -> str:
    r"""Generate a venue-safe client order id.

    Binance allows up to 36 characters matching ``^[\.A-Z\:/a-z0-9_-]{1,36}$``; a
    hex UUID4 with a short prefix fits and is collision-safe across processes.
    """
    return f"{prefix}-{uuid.uuid4().hex}"[:36]


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """An intent to trade, before it reaches the risk engine or the venue.

    The stop-loss and take-profit fields are part of the *request* rather than bolted on
    afterwards: the risk engine can then refuse any request lacking protection, and there
    is no window in which an unprotected position exists.
    """

    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    trigger_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    reduce_only: bool = False
    #: Passive-only. The venue must reject this order rather than let it cross the
    #: spread and pay the taker fee. Modelled explicitly because a post-only order
    #: that quietly fills as taker is the difference between a 0.01% and a 0.06%
    #: entry, which is the whole economics of maker-first execution.
    post_only: bool = False
    client_order_id: str = field(default_factory=new_client_order_id)
    strategy_id: str | None = None
    signal_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate structural coherence of the request."""
        if self.quantity <= ZERO:
            raise ValidationError(
                f"order quantity must be positive, got {self.quantity}",
                symbol=str(self.symbol),
            )
        if self.order_type.requires_price and self.price is None:
            raise ValidationError(
                f"{self.order_type} requires a limit price", symbol=str(self.symbol)
            )
        if self.order_type.requires_trigger_price and self.trigger_price is None:
            raise ValidationError(
                f"{self.order_type} requires a trigger price", symbol=str(self.symbol)
            )
        if self.price is not None and self.price <= ZERO:
            raise ValidationError(f"price must be positive, got {self.price}")
        if self.trigger_price is not None and self.trigger_price <= ZERO:
            raise ValidationError(f"trigger price must be positive, got {self.trigger_price}")
        if self.stop_loss_price is not None and self.stop_loss_price <= ZERO:
            raise ValidationError(f"stop loss must be positive, got {self.stop_loss_price}")
        if self.take_profit_price is not None and self.take_profit_price <= ZERO:
            raise ValidationError(f"take profit must be positive, got {self.take_profit_price}")
        self._validate_protective_sides()

    def _validate_protective_sides(self) -> None:
        """A stop must sit on the losing side of the entry, a target on the winning side."""
        reference = self.price or self.trigger_price
        if reference is None:
            return
        if self.side is OrderSide.BUY:
            if self.stop_loss_price is not None and self.stop_loss_price >= reference:
                raise ValidationError(
                    f"long stop loss {self.stop_loss_price} must be below entry {reference}",
                    symbol=str(self.symbol),
                )
            if self.take_profit_price is not None and self.take_profit_price <= reference:
                raise ValidationError(
                    f"long take profit {self.take_profit_price} must be above entry {reference}",
                    symbol=str(self.symbol),
                )
        else:
            if self.stop_loss_price is not None and self.stop_loss_price <= reference:
                raise ValidationError(
                    f"short stop loss {self.stop_loss_price} must be above entry {reference}",
                    symbol=str(self.symbol),
                )
            if self.take_profit_price is not None and self.take_profit_price >= reference:
                raise ValidationError(
                    f"short take profit {self.take_profit_price} must be below entry {reference}",
                    symbol=str(self.symbol),
                )

    @property
    def has_stop_loss(self) -> bool:
        """Whether the request carries downside protection."""
        return self.stop_loss_price is not None

    @property
    def is_entry(self) -> bool:
        """Whether this order opens or increases exposure."""
        return not self.reduce_only

    def reference_price(self, fallback: Decimal) -> Decimal:
        """Best available price estimate, used for pre-trade notional checks."""
        return self.price or self.trigger_price or fallback

    def notional(self, fallback_price: Decimal) -> Decimal:
        """Estimated quote-currency value of the request."""
        return self.quantity * self.reference_price(fallback_price)

    def with_quantity(self, quantity: Decimal) -> OrderRequest:
        """Return a copy with a different quantity (used by position sizing)."""
        return replace(self, quantity=quantity)

    def with_stop_loss(self, stop_loss_price: Decimal) -> OrderRequest:
        """Return a copy with a stop-loss attached (used by the risk engine)."""
        return replace(self, stop_loss_price=stop_loss_price)


@dataclass(frozen=True, slots=True)
class Fill:
    """A single execution against an order."""

    fill_id: str
    order_id: str
    symbol: Symbol
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime
    role: LiquidityRole = LiquidityRole.TAKER
    #: Realised PnL the venue attributed to this execution, when it reported one.
    #: ``None`` means the venue said nothing — deliberately not zero, since "no figure"
    #: and "closed flat" are different facts and only one of them is a result.
    realized_pnl: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate the fill."""
        if self.quantity <= ZERO:
            raise ValidationError(f"fill quantity must be positive, got {self.quantity}")
        if self.price <= ZERO:
            raise ValidationError(f"fill price must be positive, got {self.price}")
        if self.fee < ZERO:
            raise ValidationError(f"fill fee cannot be negative, got {self.fee}")
        if self.timestamp.tzinfo is None:
            raise ValidationError("fill timestamp must be timezone-aware UTC")

    @property
    def notional(self) -> Decimal:
        """Gross quote-currency value of the fill, before fees."""
        return self.quantity * self.price

    @property
    def signed_quantity(self) -> Decimal:
        """Quantity signed by side: positive for buys, negative for sells."""
        return self.quantity * self.side.sign


@dataclass(frozen=True, slots=True)
class Order:
    """A live or historical order, including its fill history.

    Immutable: every state change returns a new instance via :meth:`transition_to`,
    :meth:`apply_fill` or :meth:`acknowledge`. That makes the OMS trivially auditable and
    removes any possibility of two coroutines observing a half-mutated order.
    """

    order_id: str
    client_order_id: str
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    price: Decimal | None = None
    trigger_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    filled_quantity: Decimal = ZERO
    average_fill_price: Decimal = ZERO
    fees_paid: Decimal = ZERO
    fills: tuple[Fill, ...] = field(default_factory=tuple)
    venue_order_id: str | None = None
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    reduce_only: bool = False
    #: Passive-only. The venue must reject this order rather than let it cross the
    #: spread and pay the taker fee. Modelled explicitly because a post-only order
    #: that quietly fills as taker is the difference between a 0.01% and a 0.06%
    #: entry, which is the whole economics of maker-first execution.
    post_only: bool = False
    strategy_id: str | None = None
    signal_id: str | None = None
    reject_reason: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate accounting invariants."""
        if self.quantity <= ZERO:
            raise ValidationError(f"order quantity must be positive, got {self.quantity}")
        if self.filled_quantity < ZERO:
            raise ValidationError("filled_quantity cannot be negative")
        if self.filled_quantity > self.quantity:
            raise ValidationError(
                f"filled_quantity {self.filled_quantity} exceeds order quantity {self.quantity}",
                order_id=self.order_id,
            )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_request(
        cls, request: OrderRequest, *, now: datetime, order_id: str | None = None
    ) -> Self:
        """Create a ``PENDING_NEW`` order from a validated request."""
        return cls(
            order_id=order_id or uuid.uuid4().hex,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            status=OrderStatus.PENDING_NEW,
            created_at=now,
            updated_at=now,
            price=request.price,
            trigger_price=request.trigger_price,
            time_in_force=request.time_in_force,
            stop_loss_price=request.stop_loss_price,
            take_profit_price=request.take_profit_price,
            reduce_only=request.reduce_only,
            post_only=request.post_only,
            strategy_id=request.strategy_id,
            signal_id=request.signal_id,
            metadata=dict(request.metadata),
        )

    # ------------------------------------------------------------------ #
    # Derived state
    # ------------------------------------------------------------------ #
    @property
    def remaining_quantity(self) -> Decimal:
        """Quantity still working on the venue."""
        return self.quantity - self.filled_quantity

    @property
    def fill_ratio(self) -> Decimal:
        """Fraction of the order that has been filled."""
        return safe_divide(self.filled_quantity, self.quantity)

    @property
    def is_open(self) -> bool:
        """Whether the order can still receive fills."""
        return self.status.is_open

    @property
    def is_terminal(self) -> bool:
        """Whether the order has reached a final state."""
        return self.status.is_terminal

    @property
    def filled_notional(self) -> Decimal:
        """Quote-currency value of everything filled so far."""
        return self.filled_quantity * self.average_fill_price

    # ------------------------------------------------------------------ #
    # Transitions
    # ------------------------------------------------------------------ #
    def transition_to(
        self, status: OrderStatus, *, now: datetime, reason: str | None = None
    ) -> Order:
        """Move to ``status``, enforcing the OMS transition table.

        Raises:
            InvalidOrderTransitionError: if the transition is not permitted.

        """
        if status is self.status and status is not OrderStatus.PARTIALLY_FILLED:
            return self
        if not can_transition(self.status, status):
            raise InvalidOrderTransitionError(
                f"cannot transition order from {self.status} to {status}",
                order_id=self.order_id,
                current=self.status.value,
                target=status.value,
            )
        return replace(
            self,
            status=status,
            updated_at=now,
            reject_reason=reason if status is OrderStatus.REJECTED else self.reject_reason,
        )

    def acknowledge(self, venue_order_id: str, *, now: datetime) -> Order:
        """Record the venue's acknowledgement and move to ``NEW``."""
        acknowledged = self.transition_to(OrderStatus.NEW, now=now)
        return replace(acknowledged, venue_order_id=venue_order_id)

    def apply_fill(self, fill: Fill) -> Order:
        """Fold a fill into the order, recomputing VWAP, fees and status.

        Raises:
            ValidationError: if the fill does not belong to this order, or would overfill it.

        """
        if fill.order_id != self.order_id:
            raise ValidationError(
                f"fill {fill.fill_id} belongs to order {fill.order_id}, not {self.order_id}"
            )
        if fill.side is not self.side:
            raise ValidationError(
                f"fill side {fill.side} does not match order side {self.side}",
                order_id=self.order_id,
            )
        if any(existing.fill_id == fill.fill_id for existing in self.fills):
            # Idempotency: exchanges re-deliver execution reports on reconnect.
            return self
        if self.status.is_terminal:
            raise InvalidOrderTransitionError(
                f"cannot fill order in terminal state {self.status}",
                order_id=self.order_id,
            )

        new_filled = self.filled_quantity + fill.quantity
        if new_filled > self.quantity:
            raise ValidationError(
                f"fill would overfill order: {new_filled} > {self.quantity}",
                order_id=self.order_id,
                fill_id=fill.fill_id,
            )

        gross = self.filled_quantity * self.average_fill_price + fill.notional
        status = OrderStatus.FILLED if new_filled == self.quantity else OrderStatus.PARTIALLY_FILLED
        if not can_transition(self.status, status):
            raise InvalidOrderTransitionError(
                f"cannot transition order from {self.status} to {status} on fill",
                order_id=self.order_id,
            )

        return replace(
            self,
            filled_quantity=new_filled,
            average_fill_price=gross / new_filled,
            fees_paid=self.fees_paid + fill.fee,
            fills=(*self.fills, fill),
            status=status,
            updated_at=fill.timestamp,
        )
