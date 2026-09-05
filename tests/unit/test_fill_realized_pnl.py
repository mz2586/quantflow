"""A fill must carry the realised PnL the venue reported with it.

``/api/v1/account/fills`` summed ``fill.realized_pnl`` over a field that does not exist on
:class:`Fill`, so every call raised ``AttributeError`` and the dashboard's realised-PnL
panel failed outright.

The tempting fix — ``getattr(fill, "realized_pnl", ZERO)`` — is worse than the crash: it
reports a confident 0.00 realised PnL for an account that has closed trades. Bybit sends
``closedPnl`` on each execution, so the number exists; it was simply being dropped at the
mapping boundary.

``None`` is preserved rather than coerced to zero. A venue that did not report a figure and
a venue that reported exactly zero are different facts, and only one of them means "flat".
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.domain.instruments import Symbol
from quantflow.exchange.bybit.mapping import parse_fill

BTC = Symbol.parse("BTC/USDT")


def raw_fill(closed_pnl: str | None) -> dict[str, object]:
    info: dict[str, object] = {}
    if closed_pnl is not None:
        info["closedPnl"] = closed_pnl
    return {
        "id": "exec-1",
        "side": "sell",
        "amount": "0.01",
        "price": "50000",
        "fee": {"cost": "0.25", "currency": "USDT"},
        "timestamp": 1_700_000_000_000,
        "takerOrMaker": "taker",
        "info": info,
    }


class TestRealizedPnl:
    def test_closed_pnl_is_carried_onto_the_fill(self) -> None:
        fill = parse_fill(raw_fill("12.34"), order_id="o-1", symbol=BTC)

        assert fill.realized_pnl == Decimal("12.34")

    def test_a_loss_keeps_its_sign(self) -> None:
        fill = parse_fill(raw_fill("-7.5"), order_id="o-1", symbol=BTC)

        assert fill.realized_pnl == Decimal("-7.5")

    def test_absent_closed_pnl_is_none_not_zero(self) -> None:
        """Unreported and 'exactly zero' are different claims."""
        fill = parse_fill(raw_fill(None), order_id="o-1", symbol=BTC)

        assert fill.realized_pnl is None

    def test_realized_pnl_is_decimal(self) -> None:
        fill = parse_fill(raw_fill("12.34"), order_id="o-1", symbol=BTC)

        assert isinstance(fill.realized_pnl, Decimal)
