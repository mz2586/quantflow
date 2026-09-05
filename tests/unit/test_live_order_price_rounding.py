"""Live order prices must reach the venue on its tick grid, rounded the safe way.

Two defects, both on the LIVE path only, both found by running the demo bot:

1. A MARKET order carries no price and no trigger, so ``submit_order`` fell back to the
   *ticker* for a validation reference — and that price was never snapped to the grid.
   Every market entry on ETH/USDT died with ``price 1893.93 is not a multiple of tick
   0.1``, the session went ``failed``, and the supervisor restarted it into the same wall.

2. ``normalize_order`` rounded ``stop_loss_price`` with the *closing* side's convention
   (``not side_is_buy``). A long is closed by a sell, and a sell price rounds up, so a
   long's stop was rounded **up — toward the entry**. A short's was rounded down, also
   toward the entry. A stop is the one price that must never drift toward the position it
   protects: it tightens the risk the engine sized for, and at a wide tick it can land on
   the wrong side of its own trigger.

Take-profit is the opposite case and was already correct: rounding a target *away* from
entry means never accepting less than the strategy asked for.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import SecretStr

from quantflow.core.config import ExchangeSettings, MarketType
from quantflow.domain.enums import OrderSide, OrderType, TimeInForce
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Ticker
from quantflow.domain.orders import OrderRequest
from quantflow.exchange.bybit.rest import BybitGateway
from tests.conftest import REFERENCE_TIME

ETH = Symbol.parse("ETH/USDT")

#: The exact price from the crash: not a multiple of the 0.1 tick.
OFF_GRID_PRICE = Decimal("1893.93")


def eth_instrument() -> Instrument:
    """ETH/USDT as the demo venue actually describes it."""
    return Instrument(
        symbol=ETH,
        price_tick=Decimal("0.1"),
        quantity_step=Decimal("0.1"),
        min_quantity=Decimal("0.1"),
        min_notional=Decimal("1E-8"),
        market_type=MarketType.FUTURE,
    )


class FakeClient:
    """Captures what would reach Bybit's create_order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.orders: list[tuple[Any, ...]] = []

    async def fetch_order(
        self, order_id: str, symbol: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "id": order_id,
            "symbol": symbol,
            "side": "buy",
            "type": "market",
            "amount": 0.1,
            "filled": 0.1,
            "average": float(OFF_GRID_PRICE),
            "status": "closed",
            "info": {"stopLoss": "1893.9"},
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
            "info": {"stopLoss": params.get("stopLoss", "")},
        }


class StubGateway(BybitGateway):
    """Gateway with its two network reads stubbed, returning an OFF-GRID ticker.

    The off-grid ticker is the point: a real venue quotes on its own grid, but the last
    trade, a mid-price, or any derived reference need not be, and the gateway must not
    assume it is.
    """

    def __init__(self, client: FakeClient) -> None:
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
        self._instruments.put(eth_instrument())

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:
        return Ticker(
            symbol=symbol,
            bid=OFF_GRID_PRICE,
            ask=OFF_GRID_PRICE,
            last=OFF_GRID_PRICE,
            timestamp=REFERENCE_TIME,
        )

    async def fetch_positions(self) -> list[dict[str, Any]]:
        return []


def market_entry(
    side: OrderSide,
    *,
    stop: Decimal | None,
    target: Decimal | None = None,
    quantity: Decimal = Decimal("0.1"),
) -> OrderRequest:
    return OrderRequest(
        symbol=ETH,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        time_in_force=TimeInForce.GTC,
        stop_loss_price=stop,
        take_profit_price=target,
    )


class TestOffGridReferencePrice:
    """Defect 1: the ticker fallback reference was never snapped to the tick grid."""

    async def test_market_order_with_off_grid_ticker_does_not_raise(self) -> None:
        """The reported crash: 1893.93 on a 0.1 tick must snap, not explode."""
        gateway = StubGateway(FakeClient())

        order = await gateway.submit_order(market_entry(OrderSide.BUY, stop=Decimal("1800.05")))

        assert order is not None

    async def test_order_actually_reaches_the_venue(self) -> None:
        """Not raising is not enough — the order must have been submitted."""
        client = FakeClient()
        gateway = StubGateway(client)

        await gateway.submit_order(market_entry(OrderSide.BUY, stop=Decimal("1800.05")))

        assert len(client.calls) == 1


class TestStopRoundsAwayFromTheEntry:
    """Defect 2: a stop must never be rounded toward the position it protects."""

    async def test_long_stop_rounds_down(self) -> None:
        """A long's stop sits below entry; rounding up would tighten it."""
        client = FakeClient()
        gateway = StubGateway(client)

        await gateway.submit_order(market_entry(OrderSide.BUY, stop=OFF_GRID_PRICE))

        assert Decimal(client.calls[0]["stopLoss"]) == Decimal("1893.9")

    async def test_short_stop_rounds_up(self) -> None:
        """A short's stop sits above entry; rounding down would tighten it."""
        client = FakeClient()
        gateway = StubGateway(client)

        await gateway.submit_order(market_entry(OrderSide.SELL, stop=OFF_GRID_PRICE))

        assert Decimal(client.calls[0]["stopLoss"]) == Decimal("1894.0")

    @pytest.mark.parametrize(
        ("side", "expected"),
        [(OrderSide.BUY, Decimal("1893.9")), (OrderSide.SELL, Decimal("1894.0"))],
    )
    async def test_stop_never_lands_on_the_wrong_side_of_its_trigger(
        self, side: OrderSide, expected: Decimal
    ) -> None:
        """Whatever the rounding, the stop stays on its own side of the raw price."""
        client = FakeClient()
        gateway = StubGateway(client)

        await gateway.submit_order(market_entry(side, stop=OFF_GRID_PRICE))

        sent = Decimal(client.calls[0]["stopLoss"])
        assert sent == expected
        if side is OrderSide.BUY:
            assert sent <= OFF_GRID_PRICE
        else:
            assert sent >= OFF_GRID_PRICE


class TestTakeProfitStaysConservative:
    """The target rounds away from entry: never accept less than was asked for."""

    async def test_long_target_rounds_up(self) -> None:
        client = FakeClient()
        gateway = StubGateway(client)

        await gateway.submit_order(
            market_entry(OrderSide.BUY, stop=Decimal("1800.05"), target=OFF_GRID_PRICE)
        )

        assert Decimal(client.calls[0]["takeProfit"]) == Decimal("1894.0")

    async def test_short_target_rounds_down(self) -> None:
        client = FakeClient()
        gateway = StubGateway(client)

        await gateway.submit_order(
            market_entry(OrderSide.SELL, stop=Decimal("2000.05"), target=OFF_GRID_PRICE)
        )

        assert Decimal(client.calls[0]["takeProfit"]) == Decimal("1893.9")


class TestQuantityAndTypes:
    async def test_quantity_snaps_down_to_the_lot_step(self) -> None:
        """Down, never up: an order must not exceed the size that was risk-sized."""
        client = FakeClient()
        gateway = StubGateway(client)

        await gateway.submit_order(
            market_entry(OrderSide.BUY, stop=Decimal("1800.05"), quantity=Decimal("0.17"))
        )

        assert Decimal(str(client.orders[0][3])) == Decimal("0.1")

    async def test_prices_are_decimal_all_the_way_to_the_venue(self) -> None:
        """A float anywhere in this path reintroduces the drift the tick grid exists to stop."""
        client = FakeClient()
        gateway = StubGateway(client)

        await gateway.submit_order(market_entry(OrderSide.BUY, stop=OFF_GRID_PRICE))

        stop_param = client.calls[0]["stopLoss"]
        assert isinstance(stop_param, str)
        assert Decimal(stop_param) == Decimal("1893.9")
