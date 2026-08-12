"""Phase 5: GET /account/fills serialises without raising.

The defect: the router read `fill.liquidity_role`, but the domain field is `fill.role`
(`domain/orders.py:201`). Every request to the endpoint raised AttributeError -> HTTP 500.
"""

from __future__ import annotations

import inspect
from decimal import Decimal

import pytest

from quantflow.domain.enums import LiquidityRole, OrderSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill
from tests.conftest import REFERENCE_TIME

BTC = Symbol.parse("BTC/USDT")


def a_fill() -> Fill:
    return Fill(
        fill_id="f1",
        order_id="o1",
        symbol=BTC,
        side=OrderSide.BUY,
        quantity=Decimal("0.01"),
        price=Decimal("50000"),
        fee=Decimal("0.5"),
        fee_currency="USDT",
        timestamp=REFERENCE_TIME,
        role=LiquidityRole.TAKER,
    )


class TestFillSerialisation:
    def test_the_domain_field_is_role_not_liquidity_role(self) -> None:
        """The attribute the router used simply does not exist."""
        fill = a_fill()
        assert fill.role is LiquidityRole.TAKER
        with pytest.raises(AttributeError):
            _ = fill.liquidity_role  # type: ignore[attr-defined]

    def test_the_router_reads_the_field_that_exists(self) -> None:
        from quantflow.api.routers import account

        source = inspect.getsource(account)
        assert "fill.liquidity_role" not in source
        assert "fill.role.value" in source

    def test_a_fill_payload_serialises_cleanly(self) -> None:
        """Exercises the exact expression the endpoint builds."""
        import json

        fill = a_fill()
        payload = {
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "side": fill.side.value,
            "price": str(fill.price),
            "quantity": str(fill.quantity),
            "fee": str(fill.fee),
            "fee_currency": fill.fee_currency,
            "role": fill.role.value,
            "timestamp": fill.timestamp.isoformat(),
        }
        assert json.loads(json.dumps(payload))["role"] == "taker"
        # Money crosses the wire as strings; a JSON number would be corrupted by JS floats.
        assert isinstance(payload["price"], str)
        assert isinstance(payload["fee"], str)
