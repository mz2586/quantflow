"""Phase 3: loss limits are evaluated continuously, not only on new orders.

The defect: MaxDrawdownRule / MaxDailyLossRule / MaxWeeklyLossRule ran only inside
`RiskEngine.approve`. A position already open and losing produces no new order, so nothing
evaluated them and nothing acted — the account had no backstop between signals.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.core.clock import FrozenClock
from quantflow.core.config import RiskSettings
from quantflow.domain.enums import OrderSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.positions import Position
from quantflow.risk.engine import RiskEngine
from quantflow.risk.monitor import LossMonitor, evaluate_limits
from tests.conftest import REFERENCE_TIME

BTC = Symbol.parse("BTC/USDT")


def open_position() -> Position:
    position, _ = Position(symbol=BTC).apply_fill(
        Fill(
            fill_id="f1",
            order_id="o1",
            symbol=BTC,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            fee=Decimal("0"),
            fee_currency="USDT",
            timestamp=REFERENCE_TIME,
        )
    )
    return position


def snapshot(
    *, equity: Decimal, peak: Decimal, day_start: Decimal | None = None
) -> PortfolioSnapshot:
    """A snapshot whose total equity is exactly ``equity``.

    Equity is cash plus the position's mark value, so cash is set to the difference - a
    naive ``cash=equity`` would silently add the position's 500 on top and report no
    drawdown at all.
    """
    position_value = Decimal("0.01") * Decimal("50000")
    return PortfolioSnapshot(
        timestamp=REFERENCE_TIME,
        base_currency="USDT",
        cash=equity - position_value,
        positions=(open_position(),),
        mark_prices={BTC: Decimal("50000")},
        peak_equity=peak,
        day_start_equity=day_start if day_start is not None else peak,
    )


class FlattenSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, reason: str) -> list[object]:
        self.calls.append(reason)
        return []


def monitor(settings: RiskSettings | None = None) -> tuple[LossMonitor, RiskEngine, FlattenSpy]:
    config = settings or RiskSettings()
    risk = RiskEngine(config, clock=FrozenClock(REFERENCE_TIME))
    spy = FlattenSpy()
    return LossMonitor(risk, config, flatten=spy), risk, spy


class TestEvaluateLimits:
    def test_no_breach_when_within_every_limit(self) -> None:
        assert (
            evaluate_limits(snapshot(equity=Decimal("9900"), peak=Decimal("10000")), RiskSettings())
            is None
        )

    def test_drawdown_breach_is_detected(self) -> None:
        """15% default limit; 20% down must breach."""
        breach = evaluate_limits(
            snapshot(equity=Decimal("8000"), peak=Decimal("10000")), RiskSettings()
        )
        assert breach is not None
        assert breach.rule == "max_drawdown"

    def test_daily_loss_breach_is_detected(self) -> None:
        breach = evaluate_limits(
            snapshot(equity=Decimal("9600"), peak=Decimal("9700"), day_start=Decimal("10000")),
            RiskSettings(),
        )
        assert breach is not None
        assert breach.rule == "max_daily_loss"

    def test_weekly_loss_breach_is_detected(self) -> None:
        breach = evaluate_limits(
            snapshot(equity=Decimal("9100"), peak=Decimal("9150"), day_start=Decimal("9150")),
            RiskSettings(),
            week_start_equity=Decimal("10000"),
        )
        assert breach is not None
        assert breach.rule == "max_weekly_loss"


class TestMonitorHaltsAndFlattens:
    async def test_open_loser_with_no_new_signal_trips_kill_switch_and_flattens(self) -> None:
        """The exact gap: nothing proposes an order, so nothing used to check."""
        loss_monitor, risk, spy = monitor()
        await risk.start()
        assert not risk.kill_switch.engaged

        breach = await loss_monitor.check(snapshot(equity=Decimal("8000"), peak=Decimal("10000")))

        assert breach is not None
        assert breach.rule == "max_drawdown"
        assert risk.kill_switch.engaged, "the switch must latch"
        assert spy.calls, "positions must be flattened"

    async def test_no_action_while_inside_the_limits(self) -> None:
        loss_monitor, risk, spy = monitor()
        await risk.start()

        assert (
            await loss_monitor.check(snapshot(equity=Decimal("9900"), peak=Decimal("10000")))
            is None
        )
        assert not risk.kill_switch.engaged
        assert spy.calls == []

    async def test_the_switch_latches_before_flattening(self) -> None:
        """If the close fails, nothing may open on top of the loss."""
        loss_monitor, risk, _ = monitor()
        await risk.start()
        seen: list[bool] = []

        async def failing_flatten(reason: str) -> list[object]:
            seen.append(risk.kill_switch.engaged)
            raise RuntimeError("venue unreachable")

        loss_monitor._flatten = failing_flatten
        await loss_monitor.check(snapshot(equity=Decimal("8000"), peak=Decimal("10000")))

        assert seen == [True], "switch must already be latched when flatten runs"
        assert risk.kill_switch.engaged

    async def test_it_fires_once_not_every_bar(self) -> None:
        """A persistent breach must not re-issue closes on an already-flat book."""
        loss_monitor, risk, spy = monitor()
        await risk.start()
        breached = snapshot(equity=Decimal("8000"), peak=Decimal("10000"))

        assert await loss_monitor.check(breached) is not None
        assert await loss_monitor.check(breached) is None
        assert len(spy.calls) == 1


class TestEngineWiring:
    async def test_the_paper_engine_owns_a_loss_monitor(self) -> None:
        """It must be wired into the candle loop, not merely importable."""
        import inspect

        from quantflow.paper.engine import PaperTradingEngine

        source = inspect.getsource(PaperTradingEngine.on_candle)
        assert "_loss_monitor" in source, "the monitor must run on every equity sample"
