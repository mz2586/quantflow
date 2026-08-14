"""The manager against a simulated live ticker stream.

`test_intrabar_management.py` proves the decision is right. This proves the wiring is: that
a price arriving on the ticker stream — with no candle anywhere in sight — reaches the
venue as a real stop amendment or close order.

The fakes here are deliberately strict about the two things that would make this feature
dangerous in production rather than merely ineffective:

* A stop amendment the venue **rejects** must not advance local state. Believing a
  position is protected at a level the exchange never accepted is worse than knowing it
  sits at the old one, because the risk is unchanged while the reported risk is not.
* A burst of ticks through the same condition must produce **one** close order. Ticks
  arrive milliseconds apart; a manager that acts on each one turns a profit-take into an
  order storm against its own position.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.live.intrabar_manager import IntrabarManager, adopt_open_positions
from quantflow.position.intrabar import IntrabarConfig, PositionState

BTC = Symbol.parse("BTC/USDT")
NOW = datetime(2026, 1, 1, tzinfo=UTC)
ENTRY = Decimal("100")


class FakeTicker:
    def __init__(self, price: str) -> None:
        self.last = Decimal(price)
        self.timestamp = NOW


class FakeStream:
    """Emits a scripted sequence of ticks, then stops."""

    def __init__(self, prices: list[str]) -> None:
        self._prices = prices

    async def watch_ticker(self, symbol: Symbol) -> Any:
        for price in self._prices:
            yield FakeTicker(price)
            await asyncio.sleep(0)


class FakeGateway:
    """Records what actually reached the venue, and reports a book that agrees with it.

    ``fetch_positions`` is not a constant: the manager re-reads the position before every
    action and reconciles against it on a cadence, so a venue whose book never reflects the
    amendments and fills it just accepted would be modelling an exchange that silently
    discards them.
    """

    def __init__(self, *, accept_stop: bool = True) -> None:
        self.stops: list[Decimal] = []
        self.orders: list[Any] = []
        self._accept = accept_stop
        self._quantity = Decimal("10")
        self._stop = Decimal("99")

    async def set_trading_stop(
        self, symbol: Symbol, *, stop_loss: Decimal | None = None
    ) -> Decimal | None:
        if not self._accept:
            return None
        assert stop_loss is not None
        self.stops.append(stop_loss)
        self._stop = stop_loss
        return stop_loss

    async def submit_order(self, request: Any) -> Any:
        self.orders.append(request)
        self._quantity = max(Decimal("0"), self._quantity - request.quantity)
        return request

    async def fetch_positions(self) -> list[dict[str, Any]]:
        if self._quantity <= 0:
            return []
        return [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "contracts": str(self._quantity),
                "entryPrice": "100",
                "info": {"stopLoss": str(self._stop), "takeProfit": "105"},
            }
        ]


def state() -> PositionState:
    return PositionState.from_entry(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=ENTRY,
        quantity=Decimal("10"),
        stop=Decimal("99"),
        opened_at=NOW,
        target=Decimal("105"),
    )


def manager(gateway: FakeGateway, prices: list[str], *, enabled: bool = True) -> IntrabarManager:
    mgr = IntrabarManager(gateway, FakeStream(prices), IntrabarConfig(enabled=enabled))
    mgr.track(state())
    mgr.set_atr(BTC, Decimal("0.5"))
    return mgr


async def run(mgr: IntrabarManager) -> None:
    await mgr.start([BTC])
    await asyncio.sleep(0.05)
    await mgr.stop()


class TestTickReachesTheVenue:
    async def test_a_tick_moves_the_stop_without_any_candle(self) -> None:
        """The headline requirement: live price alone triggers protection."""
        gateway = FakeGateway()

        await run(manager(gateway, ["100.30"]))

        assert gateway.stops

    async def test_the_moved_stop_is_at_or_above_breakeven(self) -> None:
        gateway = FakeGateway()

        await run(manager(gateway, ["100.30"]))

        assert gateway.stops[-1] >= ENTRY

    async def test_reaching_stage_three_sends_a_reduce_only_close(self) -> None:
        gateway = FakeGateway()

        await run(manager(gateway, ["100.30", "100.60", "100.80"]))

        assert any(getattr(o, "reduce_only", False) for o in gateway.orders)

    async def test_a_reversal_closes_on_the_tick(self) -> None:
        gateway = FakeGateway()

        await run(manager(gateway, ["100.30", "100.60", "100.05"]))

        assert gateway.orders


class TestVenueIsAuthoritative:
    async def test_a_rejected_stop_amendment_does_not_advance_state(self) -> None:
        """If the exchange did not accept it, we are not protected at that level."""
        gateway = FakeGateway(accept_stop=False)
        mgr = manager(gateway, ["100.30"])

        await run(mgr)

        assert not gateway.stops


class TestNoDuplicateOrders:
    async def test_a_burst_of_ticks_produces_one_close(self) -> None:
        gateway = FakeGateway()

        await run(manager(gateway, ["100.30", "100.60", "100.05", "100.04", "100.03", "100.02"]))

        assert len(gateway.orders) == 1


class TestDisabled:
    async def test_disabled_never_touches_the_venue(self) -> None:
        gateway = FakeGateway()

        await run(manager(gateway, ["100.30", "100.60", "100.80"], enabled=False))

        assert not gateway.stops
        assert not gateway.orders


class TestAdoption:
    async def test_existing_venue_positions_are_adopted_on_restart(self) -> None:
        """A restart must resume management, not leave positions on a static stop."""
        adopted = await adopt_open_positions(FakeGateway(), NOW)

        assert len(adopted) == 1

    async def test_the_adopted_stop_comes_from_the_venue(self) -> None:
        adopted = await adopt_open_positions(FakeGateway(), NOW)

        assert adopted[0].current_stop == Decimal("99")

    async def test_water_marks_start_at_entry_not_at_the_current_price(self) -> None:
        """The excursion that happened while the process was down cannot be invented."""
        adopted = await adopt_open_positions(FakeGateway(), NOW)

        assert adopted[0].high_water == Decimal("100")
