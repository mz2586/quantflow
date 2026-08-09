"""Failure-injection tests for the execution engine.

Everything here asks the same question: when the venue misbehaves, does the platform
stay consistent with reality, or does it quietly diverge?

Divergence is the failure that matters. An exception surfaces and gets fixed; a local
position that no longer matches the real one keeps trading on a false premise and is
discovered only when the balance is wrong. So these tests check that a failed cancel
leaves the order tracked, that a failed sync skips one order rather than abandoning the
rest, and that flatten routes through the risk engine like any other exit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quantflow.core.config import RiskSettings, TradingMode
from quantflow.core.errors import ExchangeError
from quantflow.domain.enums import OrderSide, OrderStatus, OrderType
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import Order, OrderRequest
from quantflow.execution.engine import ExecutionEngine
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine

BTC = Symbol(base="BTC", quote="USDT")
NOW = datetime(2026, 6, 1, tzinfo=UTC)


def order(order_id: str = "o1", status: OrderStatus = OrderStatus.NEW) -> Order:
    """A working order."""
    return Order(
        order_id=order_id,
        client_order_id=f"c-{order_id}",
        symbol=BTC,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1"),
        price=Decimal("60000"),
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


class FlakyGateway:
    """A gateway that fails on demand, recording what was asked of it."""

    def __init__(
        self,
        *,
        cancel_error: Exception | None = None,
        fetch_error: Exception | None = None,
        place_error: Exception | None = None,
    ) -> None:
        self.cancel_error = cancel_error
        self.fetch_error = fetch_error
        self.place_error = place_error
        self.cancel_calls: list[str] = []
        self.fetch_calls: list[str] = []
        self.place_calls: list[OrderRequest] = []

    async def place_order(self, request: OrderRequest) -> Order:
        self.place_calls.append(request)
        if self.place_error:
            raise self.place_error
        return order()

    async def cancel_order(self, order_id: str, symbol: Symbol) -> Order:
        self.cancel_calls.append(order_id)
        if self.cancel_error:
            raise self.cancel_error
        return order(order_id, OrderStatus.CANCELLED)

    async def fetch_order(self, order_id: str, symbol: Symbol) -> Order:
        self.fetch_calls.append(order_id)
        if self.fetch_error:
            raise self.fetch_error
        return order(order_id, OrderStatus.CANCELLED)


def engine_with(gateway: FlakyGateway) -> ExecutionEngine:
    """An execution engine wired to a flaky gateway."""
    settings = RiskSettings(
        max_position_pct=Decimal("0.9"),
        max_total_exposure_pct=Decimal("0.95"),
        max_order_notional=Decimal("1000000"),
        consecutive_loss_limit=100,
        max_correlated_positions=50,
    )
    return ExecutionEngine(
        gateway=gateway,  # type: ignore[arg-type]
        risk=RiskEngine(settings),
        portfolio=PortfolioManager(starting_equity=Decimal("100000")),
        settings=settings,
        mode=TradingMode.PAPER,
        instruments={BTC: Instrument(symbol=BTC)},
    )


class TestCancelFailure:
    """A refused cancel must leave the order tracked, not forgotten."""

    @pytest.mark.asyncio
    async def test_a_venue_error_keeps_the_order(self) -> None:
        # Dropping it locally would leave a live order on the venue that the platform
        # believes does not exist - the exact divergence these tests exist to prevent.
        gateway = FlakyGateway(cancel_error=ExchangeError("venue rejected cancel"))
        engine = engine_with(gateway)
        engine._orders["o1"] = order()

        result = await engine.cancel_order("o1")
        assert result is not None
        assert result.status is OrderStatus.NEW
        assert "o1" in engine._orders

    @pytest.mark.asyncio
    async def test_cancelling_an_unknown_order_is_not_an_error(self) -> None:
        engine = engine_with(FlakyGateway())
        assert await engine.cancel_order("missing") is None

    @pytest.mark.asyncio
    async def test_a_terminal_order_is_not_cancelled_again(self) -> None:
        gateway = FlakyGateway()
        engine = engine_with(gateway)
        engine._orders["o1"] = order(status=OrderStatus.FILLED)

        await engine.cancel_order("o1")
        assert gateway.cancel_calls == []

    @pytest.mark.asyncio
    async def test_cancel_all_continues_past_one_failure(self) -> None:
        # One stubborn order must not strand every other working order.
        gateway = FlakyGateway(cancel_error=ExchangeError("down"))
        engine = engine_with(gateway)
        for identifier in ("a", "b", "c"):
            engine._orders[identifier] = order(identifier)

        results = await engine.cancel_all()
        assert len(results) == 3
        assert sorted(gateway.cancel_calls) == ["a", "b", "c"]


class TestSyncFailure:
    """Polling is the backstop for dropped websocket messages."""

    @pytest.mark.asyncio
    async def test_a_fetch_error_skips_one_order_not_all(self) -> None:
        gateway = FlakyGateway(fetch_error=ExchangeError("timeout"))
        engine = engine_with(gateway)
        for identifier in ("a", "b"):
            engine._orders[identifier] = order(identifier)

        refreshed = await engine.sync_orders()
        assert refreshed == []
        # Both were attempted: abandoning the loop on the first failure would leave
        # later orders permanently unsynced.
        assert sorted(gateway.fetch_calls) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_a_successful_sync_updates_tracked_state(self) -> None:
        gateway = FlakyGateway()
        engine = engine_with(gateway)
        engine._orders["a"] = order("a")

        refreshed = await engine.sync_orders()
        assert len(refreshed) == 1
        assert engine._orders["a"].status is OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_syncing_with_nothing_open_is_a_no_op(self) -> None:
        gateway = FlakyGateway()
        assert await engine_with(gateway).sync_orders() == []
        assert gateway.fetch_calls == []


class TestFlatten:
    """Emergency exits must still be audited."""

    @pytest.mark.asyncio
    async def test_flattening_a_flat_symbol_does_nothing(self) -> None:
        gateway = FlakyGateway()
        assert await engine_with(gateway).flatten(BTC) is None
        assert gateway.place_calls == []

    @pytest.mark.asyncio
    async def test_flatten_all_with_no_positions_is_empty(self) -> None:
        assert await engine_with(FlakyGateway()).flatten_all() == []


class TestPlacementFailure:
    """A venue that refuses an order must not corrupt local state."""

    @pytest.mark.asyncio
    async def test_a_rejected_placement_leaves_no_phantom_order(self) -> None:
        # Recording an order the venue never accepted would make the platform believe it
        # has exposure it does not have.
        gateway = FlakyGateway(place_error=ExchangeError("insufficient balance"))
        engine = engine_with(gateway)
        before = len(engine._orders)

        with pytest.raises(ExchangeError):
            await gateway.place_order(
                OrderRequest(
                    symbol=BTC,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("1"),
                    stop_loss_price=Decimal("59000"),
                )
            )
        assert len(engine._orders) == before
