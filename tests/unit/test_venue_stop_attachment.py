"""Phase 2: protective stops must exist on the venue, not just in memory.

The defect: `submit_order` passed `stop_loss_price`/`take_profit_price` only to
`parse_order` (the local record). Bybit never received them, so every filled entry sat
naked on the exchange while the in-memory portfolio reported it as protected.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import SecretStr

from quantflow.core.config import ExchangeSettings, MarketType
from quantflow.core.errors import ExchangeError, ValidationError
from quantflow.domain.enums import OrderSide, OrderType, TimeInForce
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Ticker
from quantflow.domain.orders import OrderRequest
from quantflow.exchange.bybit.rest import STOP_TRIGGER_BY, BybitGateway
from tests.conftest import REFERENCE_TIME

SYMBOL = Symbol.parse("BTC/USDT")


def instrument() -> Instrument:
    return Instrument(
        symbol=SYMBOL,
        price_tick=Decimal("0.1"),
        quantity_step=Decimal("0.001"),
        min_quantity=Decimal("0.001"),
        min_notional=Decimal("5"),
    )


def entry_request(*, stop: str | None = "49000", target: str | None = "52000") -> OrderRequest:
    return OrderRequest(
        symbol=SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        time_in_force=TimeInForce.GTC,
        stop_loss_price=Decimal(stop) if stop is not None else None,
        take_profit_price=Decimal(target) if target is not None else None,
    )


class FakeClient:
    """Captures the params dict that would reach Bybit's create_order."""

    def __init__(self, *, stop_in_response: str | None = "49000") -> None:
        self.calls: list[dict[str, Any]] = []
        self.orders: list[tuple[Any, ...]] = []
        self._stop_in_response = stop_in_response

    async def fetch_order(
        self, order_id: str, symbol: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """The authoritative read-back the gateway performs after an ack."""
        return {
            "id": order_id,
            "symbol": symbol,
            "side": "buy",
            "type": "market",
            "amount": 0.01,
            "filled": 0.01,
            "average": 50000.0,
            "status": "closed",
            "info": {"stopLoss": self._stop_in_response or ""},
        }

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.orders.append((symbol, order_type, side, amount, price))
        self.calls.append(params)
        return {
            "id": f"venue-{len(self.calls)}",
            "clientOrderId": params.get("clientOrderId"),
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "filled": amount,
            "status": "closed",
            "info": {"stopLoss": self._stop_in_response or ""},
        }


class StubGateway(BybitGateway):
    """BybitGateway with the two network reads stubbed.

    Subclassed rather than monkeypatched: the gateway uses ``__slots__``, so instance
    attribute assignment on a method is not possible.
    """

    def __init__(self, client: FakeClient, positions: list[dict[str, Any]] | None = None) -> None:
        super().__init__(
            ExchangeSettings(
                name="bybit",
                api_key=SecretStr("k" * 18),
                api_secret=SecretStr("s" * 36),
                testnet=True,
                market_type=MarketType.FUTURE,
            )
        )
        self._client = client
        self._instruments.put(instrument())
        type(self)._positions_response = positions or []  # type: ignore[attr-defined]

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:
        return Ticker(
            symbol=symbol,
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000"),
            timestamp=REFERENCE_TIME,
        )

    async def fetch_positions(self) -> list[dict[str, Any]]:
        return list(type(self)._positions_response)  # type: ignore[attr-defined]


def gateway_with(client: FakeClient, positions: list[dict[str, Any]] | None = None) -> BybitGateway:
    return StubGateway(client, positions)


class TestStopReachesTheVenue:
    async def test_stop_loss_and_take_profit_are_sent_to_create_order(self) -> None:
        """The core fix: these must appear in the params Bybit actually receives."""
        client = FakeClient()
        gateway = gateway_with(client)

        await gateway.submit_order(entry_request())

        params = client.calls[0]
        assert params["stopLoss"] == "49000"
        assert params["takeProfit"] == "52000"
        assert params["slTriggerBy"] == STOP_TRIGGER_BY

    async def test_stop_is_sent_as_a_string_preserving_decimal_precision(self) -> None:
        """A float round-trip would move the exact price the risk engine computed."""
        client = FakeClient(stop_in_response="49123.7")
        gateway = gateway_with(client)

        await gateway.submit_order(entry_request(stop="49123.7"))

        assert client.calls[0]["stopLoss"] == "49123.7"
        assert isinstance(client.calls[0]["stopLoss"], str)


class TestUnprotectedEntriesAreRefused:
    async def test_entry_without_a_stop_is_rejected_before_leaving_the_process(self) -> None:
        client = FakeClient()
        gateway = gateway_with(client)

        with pytest.raises(ValidationError, match="unprotected entry"):
            await gateway.submit_order(entry_request(stop=None))

        assert client.calls == [], "nothing may reach the venue"

    async def test_reduce_only_exit_without_a_stop_is_allowed(self) -> None:
        """An exit closes risk; requiring a stop on it would block flattening."""
        client = FakeClient(stop_in_response=None)
        gateway = gateway_with(client)

        request = OrderRequest(
            symbol=SYMBOL,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        await gateway.submit_order(request)
        assert client.calls[0]["reduceOnly"] is True


class TestStopAttachFailureClosesTheEntry:
    async def test_entry_is_closed_reduce_only_when_no_stop_can_be_confirmed(self) -> None:
        """An unprotected live position is worse than a flat one."""
        client = FakeClient(stop_in_response=None)
        gateway = gateway_with(client, positions=[])

        with pytest.raises(ExchangeError, match="closed reduce-only"):
            await gateway.submit_order(entry_request())

        # Two calls: the entry, then the emergency reduce-only close.
        assert len(client.calls) == 2
        assert client.calls[1]["reduceOnly"] is True
        assert client.orders[1][2] == "sell", "the close must be the opposite side"

    async def test_a_confirmed_stop_leaves_the_position_open(self) -> None:
        client = FakeClient(stop_in_response="49000")
        gateway = gateway_with(client)

        await gateway.submit_order(entry_request())

        assert len(client.calls) == 1, "no emergency close when the stop is confirmed"
