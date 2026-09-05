"""Phase 1: a LIVE session places real orders, or refuses to arm.

The defect these cover: `live/runner.py` built a `PaperTradingEngine` (and therefore a
`SimulatedBroker`) regardless of `TradingMode`, while the arming gate logged
`runner.live_trading_armed`. A live-labelled session simulated its fills and said nothing.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.core.errors import LiveTradingNotArmedError
from quantflow.domain.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Order, OrderRequest
from quantflow.execution.router import LiveOrderRouter, SimulatedOrderRouter
from tests.conftest import REFERENCE_TIME

SYMBOL = Symbol.parse("BTC/USDT")


def a_request() -> OrderRequest:
    return OrderRequest(
        symbol=SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        time_in_force=TimeInForce.GTC,
        stop_loss_price=Decimal("49000"),
        take_profit_price=Decimal("52000"),
    )


class RecordingGateway:
    """A stand-in for BybitGateway that records what was submitted."""

    def __init__(self) -> None:
        self.submitted: list[OrderRequest] = []

    async def submit_order(self, request: OrderRequest) -> Order:
        self.submitted.append(request)
        return Order(
            order_id="venue-1",
            client_order_id="qf-1",
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            status=OrderStatus.NEW,
            created_at=REFERENCE_TIME,
            updated_at=REFERENCE_TIME,
        )


class TestLiveOrderRouter:
    async def test_live_router_calls_gateway_submit_order(self) -> None:
        """The whole point: a live submit must reach the venue gateway."""
        gateway = RecordingGateway()
        router = LiveOrderRouter(gateway)
        request = a_request()

        order = await router.submit(request, now=REFERENCE_TIME, reference_price=Decimal("50000"))

        assert len(gateway.submitted) == 1
        assert gateway.submitted[0] is request
        assert order.order_id == "venue-1"

    def test_live_router_declares_itself_not_simulated(self) -> None:
        assert LiveOrderRouter(RecordingGateway()).is_simulated is False

    def test_live_router_rejects_a_gateway_that_cannot_submit(self) -> None:
        """A misconfigured live path must fail loudly, not fall back to simulation."""
        from quantflow.core.errors import ExecutionError

        class NotAGateway:
            pass

        with pytest.raises(ExecutionError, match="submit_order"):
            LiveOrderRouter(NotAGateway())

    def test_live_router_invents_no_fills_from_bars(self) -> None:
        """Bars must not manufacture fills on a venue-driven router."""
        from tests.unit.test_new_strategies import bars

        router = LiveOrderRouter(RecordingGateway())
        assert list(router.process_candle(bars([("100", "101", "99", "100", "10")])[0])) == []


class TestSimulatedRouterIsMarked:
    def test_simulated_router_declares_itself_simulated(self) -> None:
        from quantflow.exchange.simulator import SimulatedBroker

        router = SimulatedOrderRouter(SimulatedBroker(instruments={}))
        assert router.is_simulated is True


class TestRunnerRefusesToSimulateUnderLive:
    """`_assert_live_execution_wired` is what makes the 'armed' log honest."""

    def _runner(self, mode: object) -> object:
        from quantflow.core.config import Settings
        from quantflow.domain.enums import Timeframe
        from quantflow.live.runner import RunnerConfig, TradingRunner

        return TradingRunner(
            Settings(),
            RunnerConfig(
                strategy_id="ema_cross",
                symbols=(SYMBOL,),
                timeframe=Timeframe.H1,
                mode=mode,  # type: ignore[arg-type]
            ),
        )

    def test_live_session_with_a_simulated_router_raises(self) -> None:
        """A LIVE session holding a SimulatedBroker must refuse, never simulate."""
        from quantflow.core.config import TradingMode
        from quantflow.exchange.simulator import SimulatedBroker

        runner = self._runner(TradingMode.LIVE)

        class FakeEngine:
            router = SimulatedOrderRouter(SimulatedBroker(instruments={}))

        runner._engine = FakeEngine()  # type: ignore[attr-defined]
        with pytest.raises(LiveTradingNotArmedError, match="simulates fills"):
            runner._assert_live_execution_wired()  # type: ignore[attr-defined]

    def test_live_session_without_an_engine_raises(self) -> None:
        from quantflow.core.config import TradingMode

        runner = self._runner(TradingMode.LIVE)
        runner._engine = None  # type: ignore[attr-defined]
        with pytest.raises(LiveTradingNotArmedError, match="no execution engine"):
            runner._assert_live_execution_wired()  # type: ignore[attr-defined]

    def test_live_session_with_a_real_router_passes(self) -> None:
        from quantflow.core.config import TradingMode

        runner = self._runner(TradingMode.LIVE)

        class FakeEngine:
            router = LiveOrderRouter(RecordingGateway())

        runner._engine = FakeEngine()  # type: ignore[attr-defined]
        runner._assert_live_execution_wired()  # type: ignore[attr-defined]

    def test_paper_session_is_unaffected(self) -> None:
        """Paper legitimately simulates; the assertion must not touch it."""
        from quantflow.core.config import TradingMode
        from quantflow.exchange.simulator import SimulatedBroker

        runner = self._runner(TradingMode.PAPER)

        class FakeEngine:
            router = SimulatedOrderRouter(SimulatedBroker(instruments={}))

        runner._engine = FakeEngine()  # type: ignore[attr-defined]
        runner._assert_live_execution_wired()  # type: ignore[attr-defined]

    def test_arming_gate_no_longer_claims_armed_on_its_own(self) -> None:
        """The gate proves intent, not routing; it must not log 'armed' by itself."""
        import inspect

        from quantflow.live.runner import TradingRunner

        source = inspect.getsource(TradingRunner._assert_mode_permitted)
        assert "live_trading_armed" not in source


class TestPaperEngineRouterSeam:
    async def test_engine_defaults_to_a_simulated_router(self) -> None:
        from quantflow.domain.enums import Timeframe
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine
        from quantflow.strategy.registry import load_builtin_strategies

        engine = PaperTradingEngine(
            load_builtin_strategies().create("ema_cross"),
            PaperConfig(symbols=(SYMBOL,), timeframe=Timeframe.H1, persist=False),
            instruments={},
        )
        assert engine.router.is_simulated is True

    async def test_engine_accepts_an_injected_live_router(self) -> None:
        """Without this seam a live session could not route anywhere else."""
        from quantflow.domain.enums import Timeframe
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine
        from quantflow.strategy.registry import load_builtin_strategies

        engine = PaperTradingEngine(
            load_builtin_strategies().create("ema_cross"),
            PaperConfig(symbols=(SYMBOL,), timeframe=Timeframe.H1, persist=False),
            instruments={},
            router=LiveOrderRouter(RecordingGateway()),
        )
        assert engine.router.is_simulated is False


class TestOrderSyncAgainstTheVenue:
    """A resting order the venue no longer holds must stop counting as exposure."""

    class _Gateway(RecordingGateway):
        def __init__(self, live: list[Order] | None = None) -> None:
            super().__init__()
            self._live = live or []

        async def fetch_open_orders(self) -> list[Order]:
            return list(self._live)

    async def test_a_vanished_order_is_dropped(self) -> None:
        # The live regression, 2026-08-18 03:00: the venue held zero open orders while the
        # router still remembered ~8,670 of ETH as resting, so every candidate for four
        # hours was refused "would reach 35.35% of equity, above the 20.00% limit
        # (open 0, resting 8670.62, ...)". Thirty-two selections, zero orders placed.
        gateway = self._Gateway(live=[])
        router = LiveOrderRouter(gateway)
        await router.submit(a_request(), now=REFERENCE_TIME, reference_price=Decimal("50000"))
        assert len(router.open_orders()) == 1

        await router.sync_open_orders()
        assert router.open_orders() == []

    async def test_an_order_the_venue_still_holds_is_kept(self) -> None:
        gateway = self._Gateway()
        router = LiveOrderRouter(gateway)
        order = await router.submit(
            a_request(), now=REFERENCE_TIME, reference_price=Decimal("50000")
        )
        gateway._live = [order]

        await router.sync_open_orders()
        assert len(router.open_orders()) == 1

    async def test_a_gateway_without_the_endpoint_is_a_no_op(self) -> None:
        router = LiveOrderRouter(RecordingGateway())
        await router.submit(a_request(), now=REFERENCE_TIME, reference_price=Decimal("50000"))
        await router.sync_open_orders()
        assert len(router.open_orders()) == 1

    async def test_a_failing_read_leaves_state_untouched(self) -> None:
        class Broken(RecordingGateway):
            async def fetch_open_orders(self) -> list[Order]:
                raise RuntimeError("venue down")

        router = LiveOrderRouter(Broken())
        await router.submit(a_request(), now=REFERENCE_TIME, reference_price=Decimal("50000"))
        await router.sync_open_orders()
        assert len(router.open_orders()) == 1
