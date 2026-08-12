"""Phase 4: a submit that raises may still have executed on the venue.

The defect: `ratelimit.py` retries timeouts; when exhausted `submit_order` raises and
`_submit` returned `submitted=False`. But a timed-out request can still have been accepted,
leaving a real position the system believes it does not hold — orphaned and unprotected.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.core.clock import FrozenClock
from quantflow.core.config import RiskSettings, TradingMode
from quantflow.core.errors import ExchangeError
from quantflow.domain.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    SignalDirection,
    TimeInForce,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import Order, OrderRequest
from quantflow.domain.signals import Signal
from quantflow.execution.engine import ExecutionEngine
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine
from tests.conftest import REFERENCE_TIME

BTC = Symbol.parse("BTC/USDT")


def instrument() -> Instrument:
    return Instrument(
        symbol=BTC,
        price_tick=Decimal("0.1"),
        quantity_step=Decimal("0.001"),
        min_quantity=Decimal("0.001"),
        min_notional=Decimal("5"),
    )


def a_request(client_id: str = "qf-timeout-1") -> OrderRequest:
    return OrderRequest(
        symbol=BTC,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        time_in_force=TimeInForce.GTC,
        client_order_id=client_id,
        stop_loss_price=Decimal("49000"),
        take_profit_price=Decimal("52000"),
    )


def venue_order(client_id: str) -> Order:
    return Order(
        order_id="local-1",
        client_order_id=client_id,
        symbol=BTC,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        status=OrderStatus.NEW,
        created_at=REFERENCE_TIME,
        updated_at=REFERENCE_TIME,
    )


class TimingOutGateway:
    """Raises on submit, then reports whatever `open_orders` was seeded with."""

    def __init__(self, open_orders: list[Order] | None = None) -> None:
        self._open = open_orders or []
        self.open_order_queries = 0

    async def submit_order(self, request: OrderRequest) -> Order:
        raise ExchangeError("request timed out after retries", symbol=str(request.symbol))

    async def fetch_open_orders(self, symbol: Symbol | None = None) -> list[Order]:
        self.open_order_queries += 1
        return list(self._open)


def engine_for(gateway: object) -> ExecutionEngine:
    settings = RiskSettings()
    clock = FrozenClock(REFERENCE_TIME)
    return ExecutionEngine(
        gateway=gateway,  # type: ignore[arg-type]
        risk=RiskEngine(settings, clock=clock),
        portfolio=PortfolioManager(starting_equity=Decimal("10000"), clock=clock),
        settings=settings,
        mode=TradingMode.PAPER,
        instruments={BTC: instrument()},
        clock=clock,
    )


def a_signal() -> Signal:
    return Signal(
        symbol=BTC,
        direction=SignalDirection.LONG,
        timestamp=REFERENCE_TIME,
        strategy_id="test",
        reference_price=Decimal("50000"),
        stop_loss_price=Decimal("49000"),
        take_profit_price=Decimal("52000"),
    )


class TestReconcileAfterFailedSubmit:
    async def test_a_timed_out_but_filled_order_is_adopted(self) -> None:
        """The orphan case: the venue has it, we raised, we must not declare failure."""
        client_id = "qf-timeout-1"
        gateway = TimingOutGateway(open_orders=[venue_order(client_id)])
        engine = engine_for(gateway)

        found = await engine._reconcile_after_failed_submit(a_request(client_id))

        assert found is not None
        assert found.client_order_id == client_id
        assert gateway.open_order_queries == 1

    async def test_nothing_is_adopted_when_the_venue_has_no_such_order(self) -> None:
        """A genuine failure must still read as a failure."""
        gateway = TimingOutGateway(open_orders=[])
        engine = engine_for(gateway)

        assert await engine._reconcile_after_failed_submit(a_request()) is None

    async def test_an_unrelated_venue_order_is_not_adopted(self) -> None:
        """Matching is on our client id, not 'any order on this symbol'."""
        gateway = TimingOutGateway(open_orders=[venue_order("someone-elses-id")])
        engine = engine_for(gateway)

        assert await engine._reconcile_after_failed_submit(a_request("qf-mine")) is None

    async def test_reconciliation_survives_a_failing_lookup(self) -> None:
        """If the venue cannot be queried we report failure, never a false adoption."""

        class UnreachableGateway(TimingOutGateway):
            async def fetch_open_orders(self, symbol: Symbol | None = None) -> list[Order]:
                raise ExchangeError("venue unreachable")

        engine = engine_for(UnreachableGateway())
        assert await engine._reconcile_after_failed_submit(a_request()) is None

    async def test_adopted_order_gets_its_protection_applied(self) -> None:
        """An adopted orphan must not be left unprotected."""
        client_id = "qf-timeout-2"
        gateway = TimingOutGateway(open_orders=[venue_order(client_id)])
        engine = engine_for(gateway)
        request = a_request(client_id)

        from quantflow.risk.engine import RiskDecision

        result = await engine._submit(a_signal(), RiskDecision(approved=True, request=request))

        assert result.submitted is True, "an order found on the venue is not a failure"
        assert result.order is not None
        assert "adopted" in (result.reason or "")
