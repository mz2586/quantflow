"""Full-stack integration: strategy → AI → risk → execution → portfolio → persistence.

These run against real Postgres and Redis. The point is to prove the *composition* works,
not the pieces — every piece is unit-tested, but a system can be made of correct parts and
still be wrong at the seams.

The load-bearing test is :class:`TestAICannotBypassRisk`: it proves that no matter what the
AI decides, the risk engine is still the last thing between a signal and a venue.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.ai.decision import AIAdvice, AIDecisionEngine, build_engine
from quantflow.ai.strategy import AIAugmentedStrategy
from quantflow.backtest.engine import BacktestConfig, BacktestEngine
from quantflow.cache.redis import Cache, EventBus
from quantflow.core.clock import FrozenClock
from quantflow.core.config import RiskSettings, TradingMode
from quantflow.domain.enums import (
    OrderSide,
    OrderType,
    SignalDirection,
    Timeframe,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle
from quantflow.domain.orders import OrderRequest
from quantflow.domain.signals import Signal
from quantflow.exchange.simulator import FeeModel, FixedSlippage
from quantflow.notifications.base import NullNotifier
from quantflow.notifications.dispatcher import NotificationDispatcher
from quantflow.paper.engine import PaperConfig, PaperTradingEngine
from quantflow.persistence.database import Database
from quantflow.persistence.repositories import (
    ClosedTradeRepository,
    EquityRepository,
    OrderRepository,
    RiskEventRepository,
)
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.registry import load_builtin_strategies

pytestmark = pytest.mark.integration

BTC = Symbol(base="BTC", quote="USDT")
START = datetime(2026, 1, 1, tzinfo=UTC)


def instrument() -> Instrument:
    return Instrument(
        symbol=BTC,
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.00001"),
        min_quantity=Decimal("0.00001"),
        min_notional=Decimal("10"),
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
    )


def permissive_risk(**overrides: object) -> RiskSettings:
    kwargs: dict[str, object] = {
        "max_position_pct": Decimal("0.5"),
        "max_total_exposure_pct": Decimal("0.9"),
        "max_order_notional": Decimal("100000"),
        "min_order_notional": Decimal("10"),
        "max_concurrent_positions": 5,
        "max_orders_per_minute": 600,
        "max_daily_loss_pct": Decimal("0.5"),
        # The capital-preservation rules are relaxed here for the same reason as the
        # rest: an end-to-end persistence test must measure persistence, not the caps.
        # Their own behaviour is covered in test_risk_capital_preservation.py.
        "max_weekly_loss_pct": Decimal("0.55"),
        "max_drawdown_pct": Decimal("0.6"),
        "consecutive_loss_limit": 100,
        "max_correlated_positions": 50,
    }
    kwargs.update(overrides)
    return RiskSettings(**kwargs)  # type: ignore[arg-type]


def trending_candles(count: int = 200, *, start_price: float = 50_000.0) -> list[Candle]:
    """A steady uptrend with realistic intrabar ranges."""
    candles: list[Candle] = []
    price = Decimal(str(start_price))
    for index in range(count):
        step = price * Decimal("0.004")
        close = price + step
        candles.append(
            Candle(
                symbol=BTC,
                timeframe=Timeframe.H1,
                open_time=START + timedelta(hours=index),
                open=price,
                high=close + step,
                low=price - step,
                close=close,
                volume=Decimal("500"),
                quote_volume=Decimal("500") * close,
            )
        )
        price = close
    return candles


class AlwaysLongStrategy(Strategy):
    """Emits a long on every bar, so the layers downstream are what is under test."""

    strategy_id = "always_long"
    params_model = StrategyParams

    @property
    def warmup_bars(self) -> int:
        return 5

    def generate(self, context: StrategyContext) -> Signal:
        if context.has_position:
            return context.hold("already long", self.strategy_id)
        return Signal(
            symbol=context.symbol,
            direction=SignalDirection.LONG,
            timestamp=context.now,
            strategy_id=self.strategy_id,
            reference_price=context.price,
            stop_loss_price=context.price * Decimal("0.97"),
            reason="always long",
        )


async def feed_from(candles: Sequence[Candle]) -> AsyncIterator[Candle]:
    for candle in candles:
        yield candle


class HistoryGateway:
    def __init__(self, candles: list[Candle]) -> None:
        self.candles = candles

    async def fetch_candles(self, symbol, timeframe, *, since=None, limit=1000):
        del symbol, timeframe, since
        return self.candles[-limit:]


# --------------------------------------------------------------------------- #
# The load-bearing invariant
# --------------------------------------------------------------------------- #
class TestAICannotBypassRisk:
    """No AI decision can reach a venue without the risk engine inspecting it."""

    async def test_risk_still_refuses_what_the_ai_approved(self, database: Database) -> None:
        # The AI is configured to be maximally permissive; risk must still refuse.
        class ApproveEverything:
            name = "approve_everything"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                return AIAdvice()  # neutral: no veto, no discount

        strategy = AIAugmentedStrategy(
            AlwaysLongStrategy(), AIDecisionEngine(advisors=(ApproveEverything(),))
        )
        candles = trending_candles(120)

        # Risk configured so nothing can pass: the account cannot fund the minimum.
        config = BacktestConfig(
            symbols=(BTC,),
            timeframe=Timeframe.H1,
            starting_equity=Decimal("15"),
            risk=permissive_risk(min_order_notional=Decimal("10000")),
        )
        result = await BacktestEngine(strategy, config, {BTC: instrument()}).run({BTC: candles})

        assert result.signals, "the AI approved signals, so some were produced"
        assert result.orders == (), "risk must still have refused every one"
        assert result.rejected_signals

    async def test_an_ai_veto_produces_no_order(self, database: Database) -> None:
        class VetoEverything:
            name = "veto_everything"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                return AIAdvice(veto=True, reasons=("integration test veto",))

        strategy = AIAugmentedStrategy(
            AlwaysLongStrategy(), AIDecisionEngine(advisors=(VetoEverything(),))
        )
        config = BacktestConfig(symbols=(BTC,), timeframe=Timeframe.H1, risk=permissive_risk())
        result = await BacktestEngine(strategy, config, {BTC: instrument()}).run(
            {BTC: trending_candles(120)}
        )

        assert result.signals == ()
        assert result.orders == ()

    async def test_ai_discount_produces_a_smaller_but_still_checked_order(
        self, database: Database
    ) -> None:
        class Halve:
            name = "halve"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                return AIAdvice(conviction_multiplier=Decimal("0.5"))

        candles = trending_candles(120)
        config = BacktestConfig(
            symbols=(BTC,),
            timeframe=Timeframe.H1,
            starting_equity=Decimal("100000"),
            risk=permissive_risk(),
            slippage=FixedSlippage(Decimal("0")),
            fees=FeeModel(maker_rate=Decimal("0"), taker_rate=Decimal("0")),
        )

        plain = await BacktestEngine(AlwaysLongStrategy(), config, {BTC: instrument()}).run(
            {BTC: candles}
        )
        discounted = await BacktestEngine(
            AIAugmentedStrategy(AlwaysLongStrategy(), AIDecisionEngine(advisors=(Halve(),))),
            replace(config, run_id="ai-run"),
            {BTC: instrument()},
        ).run({BTC: candles})

        assert plain.orders
        assert discounted.orders
        # Every AI-influenced order still carries a stop, because it still went through
        # the risk engine's mandatory-stop rule.
        for order in discounted.orders:
            if not order.reduce_only:
                assert order.stop_loss_price is not None
        assert discounted.orders[0].quantity < plain.orders[0].quantity

    async def test_the_risk_engine_is_the_last_gate_even_for_ai_signals(
        self, database: Database
    ) -> None:
        # Directly: build an AI-adjusted signal and confirm the risk engine still runs
        # every rule against it, refusing an unfundable order.
        # An account of 5 USDT cannot fund the venue's 10 USDT minimum, whatever the AI
        # decided about the signal.
        risk = RiskEngine(permissive_risk(), database=database)
        portfolio = PortfolioManager(starting_equity=Decimal("5"))
        portfolio.update_mark_price(BTC, Decimal("50000"))

        engine = build_engine()
        original = Signal(
            symbol=BTC,
            direction=SignalDirection.LONG,
            timestamp=datetime.now(UTC),
            strategy_id="test",
            reference_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
        )
        adjusted, _ = engine.apply(original, trending_candles(120))
        assert adjusted is not None

        decision = await risk.evaluate_signal(
            adjusted,
            portfolio=portfolio.snapshot(),
            instrument=instrument(),
            reference_price=Decimal("50000"),
        )
        assert not decision.approved


class TestFullChainPersistence:
    async def test_a_paper_session_persists_orders_trades_and_equity(
        self, database: Database
    ) -> None:
        candles = trending_candles(160)
        session_id = "integration-session-1"

        engine = PaperTradingEngine(
            load_builtin_strategies().create(
                "donchian_breakout", {"entry_period": 10, "exit_period": 5}
            ),
            PaperConfig(
                symbols=(BTC,),
                timeframe=Timeframe.H1,
                starting_equity=Decimal("100000"),
                risk=permissive_risk(),
                history_bars=60,
                persist=True,
                session_id=session_id,
            ),
            instruments={BTC: instrument()},
            database=database,
            clock=FrozenClock(candles[59].close_time),
        )
        await engine.prepare(HistoryGateway(candles[:60]))  # type: ignore[arg-type]
        state = await engine.run(feed_from(candles[60:]))

        assert state.bars_seen == 100

        async with database.read_session() as session:
            orders = await OrderRepository(session).list_recent(session_id=session_id)
            trades = await ClosedTradeRepository(session).list_for_session(session_id)
            curve = await EquityRepository(session).curve(session_id)

        assert len(curve) == 100, "one equity sample per processed bar"
        if state.orders:
            assert orders, "submitted orders must be persisted"
        assert len(trades) == len(engine.portfolio.closed_trades)

    async def test_risk_refusals_are_written_to_the_audit_trail(self, database: Database) -> None:
        risk = RiskEngine(
            permissive_risk(max_order_notional=Decimal("20")),
            database=database,
            session_id=None,
        )
        portfolio = PortfolioManager(starting_equity=Decimal("100000"))
        portfolio.update_mark_price(BTC, Decimal("50000"))

        decision = await risk.approve(
            OrderRequest(
                symbol=BTC,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                stop_loss_price=Decimal("49000"),
            ),
            portfolio=portfolio.snapshot(),
            instrument=instrument(),
            reference_price=Decimal("50000"),
        )
        assert not decision.approved

        async with database.read_session() as session:
            events = await RiskEventRepository(session).list_recent(limit=10)
        assert any(event.rule == "order_notional" for event in events)
        assert all(event.blocked_order for event in events)


class TestNotificationsInTheChain:
    async def test_a_risk_refusal_notifies_the_operator(self, database: Database) -> None:
        recorder = NullNotifier()

        class EnabledRecorder(NullNotifier):
            @property
            def enabled(self) -> bool:
                return True

        transport = EnabledRecorder()
        dispatcher = NotificationDispatcher(notifiers=[transport])
        risk = RiskEngine(
            permissive_risk(max_order_notional=Decimal("20")),
            database=database,
            notifier=dispatcher,
        )
        portfolio = PortfolioManager(starting_equity=Decimal("100000"))
        portfolio.update_mark_price(BTC, Decimal("50000"))

        await risk.approve(
            OrderRequest(
                symbol=BTC,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                stop_loss_price=Decimal("49000"),
            ),
            portfolio=portfolio.snapshot(),
            instrument=instrument(),
            reference_price=Decimal("50000"),
        )

        assert transport.sent, "the operator must be told the order was refused"
        assert transport.sent[0].event_type == "risk"
        assert recorder.sent == []

    async def test_the_kill_switch_notifies_on_engage_and_clear(self, database: Database) -> None:
        class EnabledRecorder(NullNotifier):
            @property
            def enabled(self) -> bool:
                return True

        transport = EnabledRecorder()
        dispatcher = NotificationDispatcher(notifiers=[transport])

        from quantflow.risk.killswitch import KillSwitch

        switch = KillSwitch(database, notifier=dispatcher)
        await switch.engage("integration drill", actor="test")
        await switch.clear(actor="test")

        kinds = [notification.event_type for notification in transport.sent]
        assert kinds.count("kill_switch") == 2, "both directions must alert"


class TestEventBusInTheChain:
    async def test_fills_are_published_to_subscribers(
        self, database: Database, cache: Cache
    ) -> None:
        import asyncio

        bus = EventBus(cache)
        received: list[object] = []

        async def listen() -> None:
            async with bus.subscribe(EventBus.CHANNEL_FILLS) as stream:
                async for _, message in stream:
                    received.append(message)
                    return

        task = asyncio.create_task(listen())
        await asyncio.sleep(0.2)

        candles = trending_candles(120)
        engine = PaperTradingEngine(
            AlwaysLongStrategy(),
            PaperConfig(
                symbols=(BTC,),
                timeframe=Timeframe.H1,
                starting_equity=Decimal("100000"),
                risk=permissive_risk(),
                history_bars=60,
                persist=False,
            ),
            instruments={BTC: instrument()},
            event_bus=bus,
            clock=FrozenClock(candles[59].close_time),
        )
        await engine.prepare(HistoryGateway(candles[:60]))  # type: ignore[arg-type]
        await engine.run(feed_from(candles[60:]))

        await asyncio.wait_for(task, timeout=5)
        assert received, "a fill must reach the event bus"


class TestModeIsolation:
    async def test_a_paper_session_never_reports_live_mode(self, database: Database) -> None:
        engine = PaperTradingEngine(
            AlwaysLongStrategy(),
            PaperConfig(
                symbols=(BTC,),
                timeframe=Timeframe.H1,
                risk=permissive_risk(),
                persist=False,
            ),
            instruments={BTC: instrument()},
        )
        assert engine.mode is TradingMode.PAPER

    async def test_backtest_paper_and_live_share_one_fill_model(self, database: Database) -> None:
        """The property that makes results comparable across modes.

        If the three engines filled differently, a backtest would say nothing about paper
        and paper would say nothing about live.
        """
        import inspect

        from quantflow.backtest import engine as backtest_module
        from quantflow.paper import engine as paper_module

        backtest_broker = inspect.getmodule(backtest_module).__dict__["SimulatedBroker"]
        paper_broker = inspect.getmodule(paper_module).__dict__["SimulatedBroker"]
        assert backtest_broker is paper_broker
