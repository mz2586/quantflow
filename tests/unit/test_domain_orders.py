"""Order requests, fills and the OMS state machine."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from quantflow.core.errors import InvalidOrderTransitionError, ValidationError
from quantflow.domain.enums import (
    OPEN_ORDER_STATUSES,
    TERMINAL_ORDER_STATUSES,
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import (
    ORDER_TRANSITIONS,
    Fill,
    Order,
    OrderRequest,
    can_transition,
    new_client_order_id,
)
from tests.conftest import REFERENCE_TIME


def make_request(symbol: Symbol, **overrides: object) -> OrderRequest:
    kwargs: dict[str, object] = {
        "symbol": symbol,
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": Decimal("1"),
    }
    kwargs.update(overrides)
    return OrderRequest(**kwargs)  # type: ignore[arg-type]


def make_fill(
    order: Order, quantity: str, price: str, *, fill_id: str = "f1", fee: str = "0"
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency=order.symbol.quote,
        timestamp=REFERENCE_TIME,
        role=LiquidityRole.TAKER,
    )


class TestClientOrderId:
    def test_fits_binance_limit(self) -> None:
        assert len(new_client_order_id()) <= 36

    def test_is_unique(self) -> None:
        assert len({new_client_order_id() for _ in range(1000)}) == 1000


class TestOrderRequest:
    def test_market_order_needs_no_price(self, btc: Symbol) -> None:
        assert make_request(btc).price is None

    def test_limit_order_requires_price(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="requires a limit price"):
            make_request(btc, order_type=OrderType.LIMIT)

    def test_stop_order_requires_trigger(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="requires a trigger price"):
            make_request(btc, order_type=OrderType.STOP_MARKET)

    def test_stop_limit_requires_both(self, btc: Symbol) -> None:
        request = make_request(
            btc,
            order_type=OrderType.STOP_LIMIT,
            price=Decimal("100"),
            trigger_price=Decimal("101"),
        )
        assert request.price == Decimal("100")

    def test_rejects_non_positive_quantity(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="quantity must be positive"):
            make_request(btc, quantity=Decimal("0"))

    def test_long_stop_must_be_below_entry(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match=r"stop loss .* must be below entry"):
            make_request(
                btc,
                order_type=OrderType.LIMIT,
                price=Decimal("100"),
                stop_loss_price=Decimal("105"),
            )

    def test_long_target_must_be_above_entry(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match=r"take profit .* must be above entry"):
            make_request(
                btc,
                order_type=OrderType.LIMIT,
                price=Decimal("100"),
                take_profit_price=Decimal("95"),
            )

    def test_short_stop_must_be_above_entry(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match=r"stop loss .* must be above entry"):
            make_request(
                btc,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100"),
                stop_loss_price=Decimal("95"),
            )

    def test_short_target_must_be_below_entry(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match=r"take profit .* must be below entry"):
            make_request(
                btc,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100"),
                take_profit_price=Decimal("105"),
            )

    def test_valid_protective_levels_accepted(self, btc: Symbol) -> None:
        request = make_request(
            btc,
            order_type=OrderType.LIMIT,
            price=Decimal("100"),
            stop_loss_price=Decimal("98"),
            take_profit_price=Decimal("106"),
        )
        assert request.has_stop_loss

    def test_market_order_skips_side_validation_without_reference(self, btc: Symbol) -> None:
        # A market request has no price to compare against; the risk engine attaches the
        # stop once the fill price is known.
        assert make_request(btc, stop_loss_price=Decimal("1")).has_stop_loss

    def test_notional_uses_fallback_for_market_orders(self, btc: Symbol) -> None:
        assert make_request(btc, quantity=Decimal("2")).notional(Decimal("50")) == Decimal("100")

    def test_notional_prefers_limit_price(self, btc: Symbol) -> None:
        request = make_request(
            btc, order_type=OrderType.LIMIT, price=Decimal("10"), quantity=Decimal("2")
        )
        assert request.notional(Decimal("999")) == Decimal("20")

    def test_with_quantity_and_with_stop_loss_are_copies(self, btc: Symbol) -> None:
        request = make_request(btc)
        resized = request.with_quantity(Decimal("5"))
        protected = request.with_stop_loss(Decimal("1"))
        assert request.quantity == Decimal("1")
        assert resized.quantity == Decimal("5")
        assert protected.stop_loss_price == Decimal("1")
        assert request.stop_loss_price is None

    def test_reduce_only_is_not_an_entry(self, btc: Symbol) -> None:
        assert not make_request(btc, reduce_only=True).is_entry


class TestTransitionTable:
    def test_every_status_has_an_entry(self) -> None:
        assert set(ORDER_TRANSITIONS) == set(OrderStatus)

    def test_terminal_statuses_have_no_successors(self) -> None:
        for status in TERMINAL_ORDER_STATUSES:
            assert ORDER_TRANSITIONS[status] == frozenset()
            assert status.is_terminal
            assert not status.is_open

    def test_open_statuses_are_not_terminal(self) -> None:
        for status in OPEN_ORDER_STATUSES:
            assert status.is_open
            assert not status.is_terminal

    def test_statuses_partition_into_open_and_terminal(self) -> None:
        assert set(OrderStatus) == OPEN_ORDER_STATUSES | TERMINAL_ORDER_STATUSES
        assert not OPEN_ORDER_STATUSES & TERMINAL_ORDER_STATUSES

    @pytest.mark.parametrize(
        ("current", "target", "allowed"),
        [
            (OrderStatus.PENDING_NEW, OrderStatus.NEW, True),
            (OrderStatus.PENDING_NEW, OrderStatus.REJECTED, True),
            (OrderStatus.NEW, OrderStatus.FILLED, True),
            (OrderStatus.NEW, OrderStatus.PENDING_NEW, False),
            (OrderStatus.FILLED, OrderStatus.CANCELLED, False),
            (OrderStatus.CANCELLED, OrderStatus.NEW, False),
            (OrderStatus.REJECTED, OrderStatus.FILLED, False),
            (OrderStatus.PENDING_CANCEL, OrderStatus.FILLED, True),
        ],
    )
    def test_can_transition(self, current: OrderStatus, target: OrderStatus, allowed: bool) -> None:
        assert can_transition(current, target) is allowed


class TestOrderLifecycle:
    def test_from_request_starts_pending_new(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        assert order.status is OrderStatus.PENDING_NEW
        assert order.filled_quantity == Decimal("0")
        assert order.is_open

    def test_acknowledge_records_venue_id(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        acknowledged = order.acknowledge("BINANCE-123", now=REFERENCE_TIME)
        assert acknowledged.status is OrderStatus.NEW
        assert acknowledged.venue_order_id == "BINANCE-123"
        assert order.venue_order_id is None  # original untouched

    def test_illegal_transition_raises(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        filled = order.transition_to(OrderStatus.FILLED, now=REFERENCE_TIME)
        with pytest.raises(InvalidOrderTransitionError, match="cannot transition"):
            filled.transition_to(OrderStatus.CANCELLED, now=REFERENCE_TIME)

    def test_transition_to_same_status_is_a_noop(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        assert order.transition_to(OrderStatus.PENDING_NEW, now=REFERENCE_TIME) is order

    def test_reject_reason_is_recorded(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        rejected = order.transition_to(
            OrderStatus.REJECTED, now=REFERENCE_TIME, reason="insufficient balance"
        )
        assert rejected.reject_reason == "insufficient balance"


class TestOrderFills:
    def test_partial_fill(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc, quantity=Decimal("2")), now=REFERENCE_TIME)
        order = order.acknowledge("v1", now=REFERENCE_TIME)
        order = order.apply_fill(make_fill(order, "1", "100", fee="0.1"))
        assert order.status is OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("1")
        assert order.remaining_quantity == Decimal("1")
        assert order.average_fill_price == Decimal("100")
        assert order.fees_paid == Decimal("0.1")
        assert order.fill_ratio == Decimal("0.5")

    def test_full_fill_via_two_partials_computes_vwap(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc, quantity=Decimal("3")), now=REFERENCE_TIME)
        order = order.acknowledge("v1", now=REFERENCE_TIME)
        order = order.apply_fill(make_fill(order, "1", "100", fill_id="f1", fee="0.1"))
        order = order.apply_fill(make_fill(order, "2", "103", fill_id="f2", fee="0.2"))
        assert order.status is OrderStatus.FILLED
        assert order.filled_quantity == Decimal("3")
        assert order.average_fill_price == Decimal("102")  # (100 + 206) / 3
        assert order.fees_paid == Decimal("0.3")
        assert order.filled_notional == Decimal("306")
        assert order.is_terminal

    def test_duplicate_fill_is_idempotent(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc, quantity=Decimal("2")), now=REFERENCE_TIME)
        order = order.acknowledge("v1", now=REFERENCE_TIME)
        fill = make_fill(order, "1", "100")
        once = order.apply_fill(fill)
        twice = once.apply_fill(fill)
        assert twice is once
        assert twice.filled_quantity == Decimal("1")

    def test_overfill_is_rejected(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc, quantity=Decimal("1")), now=REFERENCE_TIME)
        order = order.acknowledge("v1", now=REFERENCE_TIME)
        with pytest.raises(ValidationError, match="overfill"):
            order.apply_fill(make_fill(order, "2", "100"))

    def test_fill_for_wrong_order_is_rejected(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        other = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        with pytest.raises(ValidationError, match="belongs to order"):
            order.apply_fill(make_fill(other, "1", "100"))

    def test_fill_with_wrong_side_is_rejected(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        bad = Fill(
            fill_id="f1",
            order_id=order.order_id,
            symbol=btc,
            side=OrderSide.SELL,
            quantity=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0"),
            fee_currency="USDT",
            timestamp=REFERENCE_TIME,
        )
        with pytest.raises(ValidationError, match="does not match order side"):
            order.apply_fill(bad)

    def test_fill_on_terminal_order_is_rejected(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        cancelled = order.transition_to(OrderStatus.CANCELLED, now=REFERENCE_TIME)
        with pytest.raises(InvalidOrderTransitionError, match="terminal state"):
            cancelled.apply_fill(make_fill(order, "1", "100"))

    def test_fill_after_pending_cancel_still_applies(self, btc: Symbol) -> None:
        # A cancel request can lose the race with a fill; the OMS must accept the fill.
        order = Order.from_request(make_request(btc), now=REFERENCE_TIME)
        order = order.acknowledge("v1", now=REFERENCE_TIME)
        order = order.transition_to(OrderStatus.PENDING_CANCEL, now=REFERENCE_TIME)
        filled = order.apply_fill(make_fill(order, "1", "100"))
        assert filled.status is OrderStatus.FILLED

    def test_fill_updates_timestamp(self, btc: Symbol) -> None:
        order = Order.from_request(make_request(btc, quantity=Decimal("2")), now=REFERENCE_TIME)
        order = order.acknowledge("v1", now=REFERENCE_TIME)
        later = REFERENCE_TIME + timedelta(seconds=30)
        fill = Fill(
            fill_id="f1",
            order_id=order.order_id,
            symbol=btc,
            side=order.side,
            quantity=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0"),
            fee_currency="USDT",
            timestamp=later,
        )
        assert order.apply_fill(fill).updated_at == later


class TestFillValidation:
    def test_rejects_non_positive_quantity(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="quantity must be positive"):
            Fill(
                fill_id="f",
                order_id="o",
                symbol=btc,
                side=OrderSide.BUY,
                quantity=Decimal("0"),
                price=Decimal("1"),
                fee=Decimal("0"),
                fee_currency="USDT",
                timestamp=REFERENCE_TIME,
            )

    def test_rejects_negative_fee(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="fee cannot be negative"):
            Fill(
                fill_id="f",
                order_id="o",
                symbol=btc,
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                price=Decimal("1"),
                fee=Decimal("-1"),
                fee_currency="USDT",
                timestamp=REFERENCE_TIME,
            )

    def test_signed_quantity_follows_side(self, btc: Symbol) -> None:
        common = {
            "fill_id": "f",
            "order_id": "o",
            "symbol": btc,
            "quantity": Decimal("2"),
            "price": Decimal("10"),
            "fee": Decimal("0"),
            "fee_currency": "USDT",
            "timestamp": REFERENCE_TIME,
        }
        buy = Fill(side=OrderSide.BUY, **common)  # type: ignore[arg-type]
        sell = Fill(side=OrderSide.SELL, **common)  # type: ignore[arg-type]
        assert buy.signed_quantity == Decimal("2")
        assert sell.signed_quantity == Decimal("-2")
        assert buy.notional == sell.notional == Decimal("20")


class TestEnumSemantics:
    def test_side_opposite_and_sign(self) -> None:
        assert OrderSide.BUY.opposite is OrderSide.SELL
        assert OrderSide.SELL.opposite is OrderSide.BUY
        assert OrderSide.BUY.sign == 1
        assert OrderSide.SELL.sign == -1

    @pytest.mark.parametrize(
        ("order_type", "needs_price", "needs_trigger", "is_market"),
        [
            (OrderType.MARKET, False, False, True),
            (OrderType.LIMIT, True, False, False),
            (OrderType.STOP_MARKET, False, True, True),
            (OrderType.STOP_LIMIT, True, True, False),
            (OrderType.TAKE_PROFIT_MARKET, False, True, True),
            (OrderType.TAKE_PROFIT_LIMIT, True, True, False),
        ],
    )
    def test_order_type_requirements(
        self, order_type: OrderType, needs_price: bool, needs_trigger: bool, is_market: bool
    ) -> None:
        assert order_type.requires_price is needs_price
        assert order_type.requires_trigger_price is needs_trigger
        assert order_type.is_market is is_market

    def test_time_in_force_values(self) -> None:
        assert TimeInForce.GTC.value == "gtc"
        assert {member.value for member in TimeInForce} == {"gtc", "ioc", "fok", "gtd"}
