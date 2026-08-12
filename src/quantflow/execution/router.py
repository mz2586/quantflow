"""Order routers: where an approved order actually goes.

The paper engine was written against :class:`SimulatedBroker` directly, with no seam to
substitute anything else. That is the mechanism behind the defect this module exists to
close: a session configured as ``TradingMode.LIVE`` still constructed a simulated broker,
and the runner logged ``live_trading_armed`` over the top of it. The system announced real
orders and simulated them.

A router is the seam. Two implementations, and the engine cannot tell them apart:

- :class:`SimulatedOrderRouter` — the existing bar-matching simulator, for backtest and paper.
- :class:`LiveOrderRouter` — an approved order goes to the venue through ``ExecutionEngine``
  and ``BybitGateway.submit_order``. Fills arrive from the exchange, not from replaying a bar,
  so ``process_candle`` yields nothing.

``submit`` is async on the protocol because a real submit is a network call. The simulated
router satisfies it without doing any IO.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from quantflow.core.errors import ExecutionError
from quantflow.core.logging import get_logger
from quantflow.domain.market import Candle
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.exchange.simulator import SimulatedBroker

logger = get_logger(__name__)


@runtime_checkable
class OrderRouter(Protocol):
    """Where the engine sends an approved order."""

    @property
    def is_simulated(self) -> bool:
        """Whether fills are invented locally rather than coming from a venue.

        Load-bearing: the live runner refuses to arm when this is true, which is the check
        that makes "armed" mean what it says.
        """
        ...

    async def submit(
        self, request: OrderRequest, *, now: datetime, reference_price: Decimal | None = None
    ) -> Order:
        """Send the order and return it in its post-submit state."""
        ...

    def process_candle(self, candle: Candle) -> Iterable[tuple[Order, Fill]]:
        """Match resting orders against a bar. Empty for venue-driven routers."""
        ...

    def open_orders(self, symbol: object | None = None) -> Sequence[Order]:
        """Orders still working."""
        ...


class SimulatedOrderRouter:
    """Routes to :class:`SimulatedBroker`. Backtest and paper only."""

    __slots__ = ("_broker",)

    def __init__(self, broker: SimulatedBroker) -> None:
        self._broker = broker

    @property
    def is_simulated(self) -> bool:
        """Always true. Fills here are invented from bars."""
        return True

    @property
    def broker(self) -> SimulatedBroker:
        """The underlying simulator."""
        return self._broker

    async def submit(
        self, request: OrderRequest, *, now: datetime, reference_price: Decimal | None = None
    ) -> Order:
        """Accept the order into the simulated book. No IO, despite being async."""
        return self._broker.submit(request, now=now, reference_price=reference_price)

    def process_candle(self, candle: Candle) -> Iterable[tuple[Order, Fill]]:
        """Match resting simulated orders against the bar."""
        return self._broker.process_candle(candle)

    def open_orders(self, symbol: object | None = None) -> Sequence[Order]:
        """Resting simulated orders."""
        orders: Sequence[Order] = self._broker.open_orders
        if symbol is None:
            return orders
        return [order for order in orders if order.symbol == symbol]


class LiveOrderRouter:
    """Routes to the real venue via ``ExecutionEngine`` and ``BybitGateway``.

    Deliberately thin. Sizing, protection and every risk rule have already run by the time
    an order reaches here — this only carries it to the exchange and reports what came back.
    """

    __slots__ = ("_gateway", "_orders")

    def __init__(self, gateway: object) -> None:
        if not hasattr(gateway, "submit_order"):
            raise ExecutionError(
                "live order router needs a gateway exposing submit_order; "
                f"got {type(gateway).__name__}"
            )
        self._gateway = gateway
        self._orders: dict[str, Order] = {}

    @property
    def is_simulated(self) -> bool:
        """Always false. Orders here reach a real venue."""
        return False

    async def submit(
        self,
        request: OrderRequest,
        *,
        now: datetime,  # noqa: ARG002 - protocol parity; the venue stamps its own time
        reference_price: Decimal | None = None,  # noqa: ARG002 - simulator-only concern
    ) -> Order:
        """Send the order to the venue.

        ``now`` and ``reference_price`` are part of the protocol for the simulator's benefit
        and are not sent to the exchange, which timestamps and prices the order itself.
        """
        logger.critical(
            "router.live_submit",
            symbol=str(request.symbol),
            side=request.side.value,
            quantity=str(request.quantity),
            order_type=request.order_type.value,
        )
        submit = self._gateway.submit_order
        order: Order = await submit(request)
        self._orders[order.order_id] = order
        return order

    def process_candle(
        self,
        candle: Candle,  # noqa: ARG002 - bars never create fills on a real venue
    ) -> Iterable[tuple[Order, Fill]]:
        """Nothing. A real venue reports its own fills; bars do not create them here."""
        return ()

    def open_orders(self, symbol: object | None = None) -> Sequence[Order]:
        """Orders submitted through this router that are not yet terminal."""
        return [
            order
            for order in self._orders.values()
            if not order.status.is_terminal and (symbol is None or order.symbol == symbol)
        ]


__all__ = ["LiveOrderRouter", "OrderRouter", "SimulatedOrderRouter"]
