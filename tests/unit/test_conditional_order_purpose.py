"""A working order must say what it is for, not just which way it points.

Four bracketed positions produce eight working orders, and CCXT normalises every one of
them to "sell market" — the stop and the target look identical in a list. That is how a
correct protective bracket came to be read as a duplicate-exit bug, and how a real
diagnosis was spent ruling one out.

Bybit already distinguishes them: ``stopOrderType`` is ``StopLoss`` or ``TakeProfit``. The
information was arriving and being dropped at the mapping boundary.

``None`` for an ordinary order is deliberate. An entry is not a stop with a missing label,
and defaulting to one or the other would put a purpose on an order that has none.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.exchange.bybit.mapping import parse_order


def raw_order(stop_order_type: str | None) -> dict[str, object]:
    info: dict[str, object] = {}
    if stop_order_type is not None:
        info["stopOrderType"] = stop_order_type
    return {
        "id": "venue-1",
        "symbol": "SOL/USDT:USDT",
        "side": "sell",
        "type": "market",
        "amount": "2.6",
        "status": "open",
        "timestamp": 1_700_000_000_000,
        "triggerPrice": "75.81",
        "reduceOnly": True,
        "info": info,
    }


class TestConditionalPurpose:
    def test_stop_loss_is_identified(self) -> None:
        order = parse_order(raw_order("StopLoss"))

        assert order.metadata.get("purpose") == "stop_loss"

    def test_take_profit_is_identified(self) -> None:
        order = parse_order(raw_order("TakeProfit"))

        assert order.metadata.get("purpose") == "take_profit"

    def test_an_ordinary_order_has_no_purpose(self) -> None:
        """An entry is not a stop with a missing label."""
        order = parse_order(raw_order(None))

        assert order.metadata.get("purpose") is None

    def test_stop_and_target_are_distinguishable(self) -> None:
        """The whole point: two 'sell market' orders that are not the same thing."""
        stop = parse_order(raw_order("StopLoss"))
        target = parse_order(raw_order("TakeProfit"))

        assert stop.metadata.get("purpose") != target.metadata.get("purpose")

    def test_trigger_price_survives_as_decimal(self) -> None:
        order = parse_order(raw_order("StopLoss"))

        assert order.trigger_price == Decimal("75.81")
