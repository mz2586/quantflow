"""Simulated venue: fee and slippage models plus a matching engine.

This is the single fill model used by **both** the backtester and the paper-trading engine.
Having one implementation is the point: if backtest fills and paper fills came from
different code, a backtested edge would not be evidence about the paper results, and paper
results would not be evidence about live.

Fill assumptions are deliberately pessimistic — a market order pays the spread *and*
slippage, limit orders only fill when price trades strictly through them, and a stop is
filled at the worse of the trigger and the bar's traded range. Optimistic assumptions are
how backtests come to promise returns that never materialise.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from quantflow.core.errors import ValidationError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.enums import LiquidityRole, OrderSide, OrderStatus, OrderType
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, Ticker
from quantflow.domain.orders import Fill, Order, OrderRequest

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Slippage
# --------------------------------------------------------------------------- #
class SlippageModel(Protocol):
    """Estimates the price actually achieved versus the reference price."""

    def apply(
        self, *, reference_price: Decimal, side: OrderSide, quantity: Decimal, candle: Candle | None
    ) -> Decimal:
        """Return the effective fill price."""
        ...


@dataclass(frozen=True, slots=True)
class FixedSlippage:
    """A constant fractional cost, always against the trader."""

    rate: Decimal = Decimal("0.0005")

    def apply(
        self,
        *,
        reference_price: Decimal,
        side: OrderSide,
        quantity: Decimal,
        candle: Candle | None = None,
    ) -> Decimal:
        """Move the price against ``side`` by ``rate``."""
        del quantity, candle
        adjustment = reference_price * self.rate
        return (
            reference_price + adjustment if side is OrderSide.BUY else reference_price - adjustment
        )


@dataclass(frozen=True, slots=True)
class VolumeShareSlippage:
    """Slippage that scales with the share of a bar's volume the order consumes.

    A model that charges a flat cost regardless of size lets a backtest "fill" an order
    larger than everything that traded in the bar at the mid price — the single most
    common way a backtest overstates a strategy's capacity.
    """

    base_rate: Decimal = Decimal("0.0002")
    impact_coefficient: Decimal = Decimal("0.1")
    max_volume_share: Decimal = Decimal("0.1")

    def apply(
        self,
        *,
        reference_price: Decimal,
        side: OrderSide,
        quantity: Decimal,
        candle: Candle | None = None,
    ) -> Decimal:
        """Return the effective price including a size-dependent impact term."""
        share = ZERO
        if candle is not None and candle.volume > ZERO:
            share = min(safe_divide(quantity, candle.volume), Decimal("1"))
        impact = self.base_rate + self.impact_coefficient * share
        adjustment = reference_price * impact
        return (
            reference_price + adjustment if side is OrderSide.BUY else reference_price - adjustment
        )

    def is_oversized(self, quantity: Decimal, candle: Candle) -> bool:
        """Whether the order would consume an implausible share of the bar's volume."""
        if candle.volume <= ZERO:
            return True
        return safe_divide(quantity, candle.volume) > self.max_volume_share


@dataclass(frozen=True, slots=True)
class SpreadSlippage:
    """Cross the quoted spread, for use when live ticker data is available."""

    extra_rate: Decimal = Decimal("0.0")

    def apply_with_ticker(self, *, ticker: Ticker, side: OrderSide) -> Decimal:
        """Pay the offer on a buy, hit the bid on a sell, plus any extra cost."""
        price = ticker.price_for(side)
        adjustment = price * self.extra_rate
        return price + adjustment if side is OrderSide.BUY else price - adjustment

    def apply(
        self,
        *,
        reference_price: Decimal,
        side: OrderSide,
        quantity: Decimal,
        candle: Candle | None = None,
    ) -> Decimal:
        """Fall back to a fixed cost when no ticker is available."""
        del quantity, candle
        adjustment = reference_price * self.extra_rate
        return (
            reference_price + adjustment if side is OrderSide.BUY else reference_price - adjustment
        )


# --------------------------------------------------------------------------- #
# Fees
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FeeModel:
    """Maker/taker fees, overridable per instrument."""

    maker_rate: Decimal | None = None
    taker_rate: Decimal | None = None

    def rate_for(self, instrument: Instrument, *, role: LiquidityRole) -> Decimal:
        """Fee rate for a liquidity role, preferring the explicit override."""
        if role is LiquidityRole.MAKER:
            return self.maker_rate if self.maker_rate is not None else instrument.maker_fee
        return self.taker_rate if self.taker_rate is not None else instrument.taker_fee

    def compute(
        self, instrument: Instrument, *, quantity: Decimal, price: Decimal, role: LiquidityRole
    ) -> Decimal:
        """Fee in quote currency for a fill."""
        return instrument.notional(quantity, price) * self.rate_for(instrument, role=role)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FillDecision:
    """The outcome of testing one order against one bar."""

    filled: bool
    price: Decimal = ZERO
    role: LiquidityRole = LiquidityRole.TAKER
    reason: str = ""


def match_against_candle(  # noqa: PLR0911 - one explicit branch per order type
    order: Order, candle: Candle, *, slippage: SlippageModel
) -> FillDecision:
    """Decide whether ``order`` fills within ``candle``, and at what price.

    Rules, all chosen to avoid flattering the strategy:

    - **Market**: fills at the bar's open plus slippage. Not the close — a decision made on
      the previous bar's close can only be executed at the next bar's open.
    - **Limit buy**: fills only if ``low < limit``; strictly through, because touching a
      price does not guarantee a fill at it. Fills at the limit (maker, no slippage).
    - **Stop**: triggers if the bar's range reaches the stop, and fills at the *worse* of
      the stop price and the bar open, plus slippage — gaps go against the trader.
    """
    if candle.symbol != order.symbol:
        raise ValidationError(f"candle for {candle.symbol} cannot fill an order for {order.symbol}")

    quantity = order.remaining_quantity
    if quantity <= ZERO:
        return FillDecision(filled=False, reason="nothing remaining")

    if order.order_type is OrderType.MARKET:
        price = slippage.apply(
            reference_price=candle.open, side=order.side, quantity=quantity, candle=candle
        )
        return FillDecision(filled=True, price=price, role=LiquidityRole.TAKER)

    if order.order_type is OrderType.LIMIT:
        limit = order.price
        if limit is None:
            return FillDecision(filled=False, reason="limit order without a price")
        if order.side is OrderSide.BUY and candle.low < limit:
            return FillDecision(filled=True, price=limit, role=LiquidityRole.MAKER)
        if order.side is OrderSide.SELL and candle.high > limit:
            return FillDecision(filled=True, price=limit, role=LiquidityRole.MAKER)
        return FillDecision(filled=False, reason="limit not reached")

    if order.order_type.requires_trigger_price:
        trigger = order.trigger_price
        if trigger is None:
            return FillDecision(filled=False, reason="stop order without a trigger")
        triggered = candle.high >= trigger if order.side is OrderSide.BUY else candle.low <= trigger
        if not triggered:
            return FillDecision(filled=False, reason="trigger not reached")
        # A gap through the stop fills at the open, not the stop price.
        worst = (
            max(trigger, candle.open) if order.side is OrderSide.BUY else min(trigger, candle.open)
        )
        price = slippage.apply(
            reference_price=worst, side=order.side, quantity=quantity, candle=candle
        )
        return FillDecision(filled=True, price=price, role=LiquidityRole.TAKER)

    return FillDecision(filled=False, reason=f"unsupported order type {order.order_type}")


@dataclass
class SimulatedBroker:
    """In-memory matching engine shared by backtest and paper trading.

    Holds working orders, matches them against incoming bars, and produces fills with
    realistic fees and slippage. It knows nothing about strategies, risk or portfolios —
    it is purely the venue's half of the conversation.
    """

    instruments: dict[Symbol, Instrument]
    slippage: SlippageModel = field(default_factory=VolumeShareSlippage)
    fees: FeeModel = field(default_factory=FeeModel)
    reject_oversized: bool = True
    _open_orders: dict[str, Order] = field(default_factory=dict, init=False)
    _fill_sequence: int = field(default=0, init=False)

    @property
    def open_orders(self) -> tuple[Order, ...]:
        """Every order still working."""
        return tuple(self._open_orders.values())

    def instrument_for(self, symbol: Symbol) -> Instrument:
        """Look up an instrument, raising if unknown."""
        instrument = self.instruments.get(symbol)
        if instrument is None:
            raise ValidationError(f"no instrument configured for {symbol}", symbol=str(symbol))
        return instrument

    def submit(
        self, request: OrderRequest, *, now: datetime, reference_price: Decimal | None = None
    ) -> Order:
        """Accept an order into the book, validated against venue rules.

        The simulated venue enforces the same lot, tick and notional rules as the real one.
        Skipping that would let a backtest fill orders Binance would have rejected, which is
        exactly the kind of divergence this shared fill model exists to prevent.

        Args:
            request: The order to accept.
            now: Current engine time.
            reference_price: Price used for the notional check on market orders, which
                carry no price of their own. Validation is skipped when neither is known.

        """
        instrument = self.instrument_for(request.symbol)
        explicit_price = request.price or request.trigger_price
        check_price = explicit_price or reference_price
        if check_price is not None:
            instrument.validate_order(
                request.quantity, check_price, check_price_tick=explicit_price is not None
            )
        order = Order.from_request(request, now=now)
        order = order.acknowledge(f"sim-{uuid.uuid4().hex[:12]}", now=now)
        self._open_orders[order.order_id] = order
        return order

    def cancel(self, order_id: str, *, now: datetime) -> Order:
        """Cancel a working order."""
        order = self._open_orders.get(order_id)
        if order is None:
            raise ValidationError(f"order {order_id} is not working", order_id=order_id)
        cancelled = order.transition_to(OrderStatus.CANCELLED, now=now)
        del self._open_orders[order_id]
        return cancelled

    def cancel_all(self, *, symbol: Symbol | None = None, now: datetime) -> list[Order]:
        """Cancel every working order, optionally limited to one symbol."""
        targets = [
            order
            for order in self._open_orders.values()
            if symbol is None or order.symbol == symbol
        ]
        return [self.cancel(order.order_id, now=now) for order in targets]

    def process_candle(self, candle: Candle) -> list[tuple[Order, Fill]]:
        """Match every working order for the bar's symbol.

        Returns:
            ``(order, fill)`` pairs for each execution, in submission order.

        """
        results: list[tuple[Order, Fill]] = []
        instrument = self.instruments.get(candle.symbol)
        if instrument is None:
            return results

        for order_id in list(self._open_orders):
            order = self._open_orders.get(order_id)
            if order is None or order.symbol != candle.symbol:
                continue

            decision = match_against_candle(order, candle, slippage=self.slippage)
            if not decision.filled:
                continue

            quantity = order.remaining_quantity
            if self._is_oversized(quantity, candle):
                logger.warning(
                    "simulator.order_exceeds_bar_volume",
                    order_id=order.order_id,
                    symbol=str(candle.symbol),
                    quantity=str(quantity),
                    bar_volume=str(candle.volume),
                )
                if self.reject_oversized:
                    rejected = order.transition_to(
                        OrderStatus.REJECTED,
                        now=candle.close_time,
                        reason="order size exceeds available bar liquidity",
                    )
                    del self._open_orders[order_id]
                    results.append((rejected, _null_fill(rejected, candle)))
                    continue

            fill = self._build_fill(order, decision, quantity, instrument, candle.close_time)
            filled = order.apply_fill(fill)
            if filled.is_terminal:
                del self._open_orders[order_id]
            else:
                self._open_orders[order_id] = filled
            results.append((filled, fill))

        return results

    def fill_at_market(
        self, request: OrderRequest, *, ticker: Ticker, now: datetime
    ) -> tuple[Order, Fill]:
        """Immediately fill a market order against a live ticker.

        Used by the paper engine, where a real quote is available and waiting for the next
        bar would misrepresent how a market order behaves.
        """
        instrument = self.instrument_for(request.symbol)
        order = Order.from_request(request, now=now).acknowledge(
            f"sim-{uuid.uuid4().hex[:12]}", now=now
        )
        price = SpreadSlippage().apply_with_ticker(ticker=ticker, side=request.side)
        fee = self.fees.compute(
            instrument, quantity=request.quantity, price=price, role=LiquidityRole.TAKER
        )
        fill = Fill(
            fill_id=self._next_fill_id(),
            order_id=order.order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=price,
            fee=fee,
            fee_currency=request.symbol.quote,
            timestamp=now,
            role=LiquidityRole.TAKER,
        )
        return order.apply_fill(fill), fill

    def _is_oversized(self, quantity: Decimal, candle: Candle) -> bool:
        model = self.slippage
        if isinstance(model, VolumeShareSlippage):
            return model.is_oversized(quantity, candle)
        return False

    def _build_fill(
        self,
        order: Order,
        decision: FillDecision,
        quantity: Decimal,
        instrument: Instrument,
        timestamp: datetime,
    ) -> Fill:
        fee = self.fees.compute(
            instrument, quantity=quantity, price=decision.price, role=decision.role
        )
        return Fill(
            fill_id=self._next_fill_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=decision.price,
            fee=fee,
            fee_currency=order.symbol.quote,
            timestamp=timestamp,
            role=decision.role,
        )

    def _next_fill_id(self) -> str:
        self._fill_sequence += 1
        return f"sim-fill-{self._fill_sequence:08d}"


def _null_fill(order: Order, candle: Candle) -> Fill:
    """A zero-quantity placeholder is not representable, so rejections carry the reference bar.

    Callers must check ``order.status`` before consuming the fill.
    """
    return Fill(
        fill_id=f"rejected-{order.order_id}",
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.remaining_quantity,
        price=candle.open,
        fee=ZERO,
        fee_currency=order.symbol.quote,
        timestamp=candle.close_time,
    )
