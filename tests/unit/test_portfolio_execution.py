"""Portfolio accounting and the execution path.

The invariants here are the ones that decide whether reported equity is real: cash and
positions must move together, and no fill may ever be counted twice.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from quantflow.core.clock import FrozenClock
from quantflow.core.config import RiskSettings, TradingMode
from quantflow.core.errors import ExchangeError, ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import LiquidityRole, OrderSide, OrderType, SignalDirection
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.domain.portfolio import Balance
from quantflow.domain.signals import Signal
from quantflow.execution.engine import (
    ExecutionEngine,
    ExecutionStats,
    build_exit_request,
    protective_exit_price,
    should_trigger_stop,
)
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine
from tests.conftest import REFERENCE_TIME


def fill(
    symbol: Symbol,
    side: OrderSide,
    quantity: str,
    price: str,
    *,
    fill_id: str = "f1",
    fee: str = "0",
    order_id: str = "o1",
    offset_seconds: int = 0,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency=symbol.quote,
        timestamp=REFERENCE_TIME + timedelta(seconds=offset_seconds),
        role=LiquidityRole.TAKER,
    )


@pytest.fixture
def manager(clock: FrozenClock) -> PortfolioManager:
    return PortfolioManager(base_currency="USDT", starting_equity=Decimal("10000"), clock=clock)


class TestCashAccounting:
    def test_buy_debits_notional_plus_fee(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000", fee="5"))
        assert manager.cash == Decimal("10000") - Decimal("5000") - Decimal("5")

    def test_sell_credits_notional_minus_fee(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000", fee="5", fill_id="a"))
        cash_after_buy = manager.cash
        manager.apply_fill(
            fill(btc, OrderSide.SELL, "0.1", "51000", fee="5", fill_id="b", offset_seconds=60)
        )
        assert manager.cash == cash_after_buy + Decimal("5100") - Decimal("5")

    def test_equity_is_conserved_across_a_flat_round_trip(
        self, manager: PortfolioManager, btc: Symbol
    ) -> None:
        # Buying and selling at the same price with no fees must leave equity unchanged.
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000", fill_id="a"))
        manager.apply_fill(
            fill(btc, OrderSide.SELL, "0.1", "50000", fill_id="b", offset_seconds=60)
        )
        assert manager.equity() == Decimal("10000")

    def test_fees_are_the_only_leak(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000", fee="5", fill_id="a"))
        manager.apply_fill(
            fill(btc, OrderSide.SELL, "0.1", "50000", fee="5", fill_id="b", offset_seconds=60)
        )
        assert manager.equity() == Decimal("9990")
        assert manager.fees_paid == Decimal("10")

    def test_equity_tracks_the_mark_price(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000"))
        assert manager.equity() == Decimal("10000")  # marked at the fill price
        manager.update_mark_price(btc, Decimal("60000"))
        assert manager.equity() == Decimal("11000")

    def test_valuation_without_a_mark_price_raises(
        self, manager: PortfolioManager, btc: Symbol
    ) -> None:
        # Silently valuing at zero would understate exposure and defeat every risk limit.
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000"))
        manager._mark_prices.clear()
        with pytest.raises(ValidationError, match="no mark price"):
            manager.equity()

    def test_rejects_a_non_positive_mark_price(
        self, manager: PortfolioManager, btc: Symbol
    ) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            manager.update_mark_price(btc, ZERO)

    def test_rejects_non_positive_starting_equity(self) -> None:
        with pytest.raises(ValidationError, match="starting equity"):
            PortfolioManager(starting_equity=ZERO)


class TestIdempotency:
    def test_a_duplicate_fill_is_ignored(self, manager: PortfolioManager, btc: Symbol) -> None:
        # Exchanges re-deliver execution reports on reconnect; counting one twice would
        # corrupt both the position and the cash balance.
        duplicate = fill(btc, OrderSide.BUY, "0.1", "50000", fee="5")
        manager.apply_fill(duplicate)
        cash = manager.cash
        position = manager.position_for(btc)
        assert position is not None
        quantity = position.quantity

        manager.apply_fill(duplicate)

        assert manager.cash == cash
        after = manager.position_for(btc)
        assert after is not None
        assert after.quantity == quantity

    def test_distinct_fills_both_apply(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000", fill_id="a"))
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000", fill_id="b", offset_seconds=1))
        position = manager.position_for(btc)
        assert position is not None
        assert position.quantity == Decimal("0.2")


class TestPositionsAndTrades:
    def test_round_trip_records_a_closed_trade(
        self, manager: PortfolioManager, btc: Symbol
    ) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "1", "50000", fill_id="a"))
        manager.apply_fill(
            fill(btc, OrderSide.SELL, "1", "51000", fill_id="b", offset_seconds=3600)
        )
        assert len(manager.closed_trades) == 1
        assert manager.closed_trades[0].gross_pnl == Decimal("1000")
        assert manager.realized_pnl == Decimal("1000")
        assert manager.position_for(btc) is None

    def test_apply_fills_orders_by_timestamp(self, manager: PortfolioManager, btc: Symbol) -> None:
        closed = manager.apply_fills(
            [
                fill(btc, OrderSide.SELL, "1", "51000", fill_id="b", offset_seconds=60),
                fill(btc, OrderSide.BUY, "1", "50000", fill_id="a", offset_seconds=0),
            ]
        )
        assert len(closed) == 1
        assert closed[0].gross_pnl == Decimal("1000")

    def test_set_protection(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "1", "50000"))
        manager.set_protection(btc, stop_loss_price=Decimal("49000"))
        position = manager.position_for(btc)
        assert position is not None
        assert position.stop_loss_price == Decimal("49000")

    def test_set_protection_on_a_flat_symbol_is_a_noop(
        self, manager: PortfolioManager, btc: Symbol
    ) -> None:
        manager.set_protection(btc, stop_loss_price=Decimal("1"))
        assert manager.position_for(btc) is None


class TestEquityCurve:
    def test_records_points_and_tracks_the_peak(
        self, manager: PortfolioManager, btc: Symbol, clock: FrozenClock
    ) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000"))
        manager.record_equity()

        clock.advance(seconds=3600)
        manager.update_mark_price(btc, Decimal("60000"))
        peak = manager.record_equity()

        clock.advance(seconds=3600)
        manager.update_mark_price(btc, Decimal("55000"))
        drawdown_point = manager.record_equity()

        assert len(manager.equity_curve) == 3
        assert peak.equity == Decimal("11000")
        assert manager.peak_equity == Decimal("11000")
        assert drawdown_point.drawdown_pct == Decimal("500") / Decimal("11000")

    def test_daily_baseline_rolls_at_utc_midnight(
        self, manager: PortfolioManager, btc: Symbol, clock: FrozenClock
    ) -> None:
        manager.record_equity()
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000"))
        manager.update_mark_price(btc, Decimal("60000"))

        clock.advance(delta=timedelta(days=1))
        manager.record_equity()

        snapshot = manager.snapshot()
        # The new day starts from the raised equity, so yesterday's gain does not count
        # toward today's loss budget.
        assert snapshot.day_start_equity == Decimal("11000")
        assert snapshot.daily_pnl == ZERO


class TestSnapshotAndRecovery:
    def test_snapshot_reflects_live_state(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "0.1", "50000", fee="5"))
        snapshot = manager.snapshot()
        assert snapshot.cash == manager.cash
        assert snapshot.position_count == 1
        assert snapshot.fees_paid == Decimal("5")

    def test_restore_rebuilds_state_including_seen_fills(
        self, manager: PortfolioManager, btc: Symbol
    ) -> None:
        applied = fill(btc, OrderSide.BUY, "0.1", "50000")
        manager.apply_fill(applied)
        position = manager.position_for(btc)
        assert position is not None

        fresh = PortfolioManager(starting_equity=Decimal("10000"))
        fresh.restore(
            cash=manager.cash,
            positions=[position],
            peak_equity=Decimal("10000"),
            applied_fill_ids=[applied.fill_id],
        )
        fresh.update_mark_price(btc, Decimal("50000"))

        assert fresh.cash == manager.cash
        assert fresh.position_for(btc) is not None

        # The re-delivered fill must not be applied a second time after recovery.
        cash_before = fresh.cash
        fresh.apply_fill(applied)
        assert fresh.cash == cash_before

    def test_reconcile_reports_mismatches(
        self, manager: PortfolioManager, btc: Symbol, eth: Symbol
    ) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "1", "50000"))
        discrepancies = manager.reconcile({btc: Decimal("1.5"), eth: Decimal("2")})
        assert discrepancies[btc] == Decimal("0.5")
        assert discrepancies[eth] == Decimal("2")

    def test_reconcile_is_empty_when_aligned(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "1", "50000"))
        assert manager.reconcile({btc: Decimal("1")}) == {}

    def test_summary(self, manager: PortfolioManager, btc: Symbol) -> None:
        manager.apply_fill(fill(btc, OrderSide.BUY, "1", "50000", fill_id="a"))
        manager.apply_fill(fill(btc, OrderSide.SELL, "1", "51000", fill_id="b", offset_seconds=60))
        summary = manager.summary()
        assert summary["closed_trades"] == 1
        assert summary["wins"] == 1


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
class RecordingGateway:
    """A trading gateway that fills every order instantly at a fixed price."""

    def __init__(
        self,
        *,
        fill_price: Decimal = Decimal("50000"),
        fail: bool = False,
        is_testnet: bool = True,
        supports_trading: bool = True,
    ) -> None:
        self.fill_price = fill_price
        self.fail = fail
        self.is_testnet = is_testnet
        self.supports_trading = supports_trading
        self.submitted: list[OrderRequest] = []
        self.cancelled: list[str] = []

    async def submit_order(self, request: OrderRequest) -> Order:
        if self.fail:
            raise ExchangeError("venue rejected the order")
        self.submitted.append(request)
        order = Order.from_request(request, now=REFERENCE_TIME).acknowledge(
            f"venue-{len(self.submitted)}", now=REFERENCE_TIME
        )
        return order.apply_fill(
            Fill(
                fill_id=f"fill-{len(self.submitted)}",
                order_id=order.order_id,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                price=self.fill_price,
                fee=request.quantity * self.fill_price * Decimal("0.001"),
                fee_currency=request.symbol.quote,
                timestamp=REFERENCE_TIME,
            )
        )

    async def cancel_order(self, order_id: str, symbol: Symbol) -> Order:
        del symbol
        self.cancelled.append(order_id)
        raise ExchangeError("not implemented for this fake")

    async def fetch_order(self, order_id: str, symbol: Symbol) -> Order:  # pragma: no cover
        raise NotImplementedError

    async def fetch_open_orders(
        self, symbol: Symbol | None = None
    ) -> list[Order]:  # pragma: no cover
        return []

    async def fetch_my_trades(
        self, symbol: Symbol, **kwargs: object
    ) -> list[Fill]:  # pragma: no cover
        return []

    async def fetch_balances(self) -> dict[str, Balance]:  # pragma: no cover
        return {}


def build_engine(
    btc: Symbol,
    clock: FrozenClock,
    *,
    gateway: RecordingGateway | None = None,
    mode: TradingMode = TradingMode.PAPER,
) -> tuple[ExecutionEngine, PortfolioManager, RecordingGateway]:
    settings = RiskSettings(
        max_position_pct=Decimal("0.5"),
        max_total_exposure_pct=Decimal("1"),
        max_order_notional=Decimal("100000"),
    )
    portfolio = PortfolioManager(starting_equity=Decimal("10000"), clock=clock)
    portfolio.update_mark_price(btc, Decimal("50000"))
    venue = gateway or RecordingGateway()
    engine = ExecutionEngine(
        gateway=venue,
        risk=RiskEngine(settings, clock=clock),
        portfolio=portfolio,
        settings=settings,
        mode=mode,
        instruments={btc: Instrument(symbol=btc, min_notional=Decimal("5"))},
        clock=clock,
    )
    return engine, portfolio, venue


def long_signal(btc: Symbol, **overrides: object) -> Signal:
    kwargs: dict[str, object] = {
        "symbol": btc,
        "direction": SignalDirection.LONG,
        "timestamp": REFERENCE_TIME,
        "strategy_id": "test",
        "reference_price": Decimal("50000"),
        "stop_loss_price": Decimal("49000"),
    }
    kwargs.update(overrides)
    return Signal(**kwargs)  # type: ignore[arg-type]


class TestExecutionEngine:
    async def test_happy_path_submits_and_updates_the_portfolio(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        engine, portfolio, gateway = build_engine(btc, clock)
        result = await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        assert result.succeeded
        assert len(gateway.submitted) == 1
        assert portfolio.position_for(btc) is not None
        assert portfolio.cash < Decimal("10000")

    async def test_stop_loss_is_attached_to_the_position(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        engine, portfolio, _ = build_engine(btc, clock)
        await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        position = portfolio.position_for(btc)
        assert position is not None
        assert position.stop_loss_price == Decimal("49000")

    async def test_hold_signals_are_not_submitted(self, btc: Symbol, clock: FrozenClock) -> None:
        engine, _, gateway = build_engine(btc, clock)
        result = await engine.execute_signal(
            Signal.hold(btc, REFERENCE_TIME, "test"), reference_price=Decimal("50000")
        )
        assert not result.submitted
        assert gateway.submitted == []

    async def test_stale_signals_are_discarded(self, btc: Symbol, clock: FrozenClock) -> None:
        # The price a stale signal was computed against may be long gone.
        engine, _, gateway = build_engine(btc, clock)
        clock.advance(seconds=300)
        result = await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        assert not result.submitted
        assert "stale" in result.reason
        assert gateway.submitted == []

    async def test_risk_rejection_stops_submission(self, btc: Symbol, clock: FrozenClock) -> None:
        engine, _, gateway = build_engine(btc, clock)
        await engine._risk.kill_switch.engage("test halt")
        result = await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        assert not result.submitted
        assert result.rejected_by_risk
        assert gateway.submitted == []

    async def test_unknown_instrument_is_refused(
        self, btc: Symbol, eth: Symbol, clock: FrozenClock
    ) -> None:
        engine, _, gateway = build_engine(btc, clock)
        result = await engine.execute_signal(
            long_signal(btc, symbol=eth), reference_price=Decimal("2500")
        )
        assert not result.submitted
        assert "no instrument metadata" in result.reason
        assert gateway.submitted == []

    async def test_exchange_errors_are_reported_not_raised(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        engine, portfolio, _ = build_engine(btc, clock, gateway=RecordingGateway(fail=True))
        result = await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        assert not result.succeeded
        assert result.error is not None
        assert portfolio.position_for(btc) is None

    async def test_paper_mode_refuses_a_production_gateway(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        # The last line of defence against pointing a paper run at real money.
        engine, _, gateway = build_engine(
            btc,
            clock,
            gateway=RecordingGateway(is_testnet=False, supports_trading=True),
            mode=TradingMode.PAPER,
        )
        result = await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        assert not result.submitted
        assert "refusing to submit" in result.reason
        assert gateway.submitted == []

    async def test_live_mode_permits_a_production_gateway(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        engine, _, gateway = build_engine(
            btc,
            clock,
            gateway=RecordingGateway(is_testnet=False, supports_trading=True),
            mode=TradingMode.LIVE,
        )
        result = await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        assert result.succeeded
        assert len(gateway.submitted) == 1

    async def test_submitted_orders_count_toward_the_rate_limit(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        engine, _, _ = build_engine(btc, clock)
        await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        assert engine._risk.orders_in_last_minute() == 1

    async def test_flatten_closes_a_position(self, btc: Symbol, clock: FrozenClock) -> None:
        engine, portfolio, gateway = build_engine(btc, clock)
        await engine.execute_signal(long_signal(btc), reference_price=Decimal("50000"))
        assert portfolio.position_for(btc) is not None

        result = await engine.flatten(btc, reason="test")
        assert result is not None
        assert result.succeeded
        assert gateway.submitted[-1].reduce_only
        assert portfolio.position_for(btc) is None

    async def test_flatten_without_a_position_returns_none(
        self, btc: Symbol, clock: FrozenClock
    ) -> None:
        engine, _, _ = build_engine(btc, clock)
        assert await engine.flatten(btc) is None

    async def test_track_adopts_recovered_orders(self, btc: Symbol, clock: FrozenClock) -> None:
        engine, _, _ = build_engine(btc, clock)
        order = Order.from_request(
            OrderRequest(
                symbol=btc,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.01"),
                price=Decimal("49000"),
            ),
            now=REFERENCE_TIME,
        ).acknowledge("v1", now=REFERENCE_TIME)
        engine.track([order])
        assert len(engine.open_orders) == 1


class TestExecutionHelpers:
    def test_protective_exit_price(self) -> None:
        assert protective_exit_price(True, Decimal("100"), Decimal("0.02")) == Decimal("98")
        assert protective_exit_price(False, Decimal("100"), Decimal("0.02")) == Decimal("102")

    def test_protective_exit_rejects_a_non_positive_entry(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            protective_exit_price(True, ZERO, Decimal("0.02"))

    def test_stop_triggers_intrabar_not_on_the_close(self) -> None:
        # Only testing the close would let a strategy sail through a 20% intrabar spike
        # in a backtest while being stopped out in reality.
        assert should_trigger_stop(
            position_side_is_long=True,
            stop_price=Decimal("95"),
            candle_low=Decimal("94"),
            candle_high=Decimal("105"),
        )
        assert not should_trigger_stop(
            position_side_is_long=True,
            stop_price=Decimal("95"),
            candle_low=Decimal("96"),
            candle_high=Decimal("105"),
        )

    def test_short_stop_uses_the_high(self) -> None:
        assert should_trigger_stop(
            position_side_is_long=False,
            stop_price=Decimal("105"),
            candle_low=Decimal("95"),
            candle_high=Decimal("106"),
        )

    def test_build_exit_request_inverts_the_side(self, btc: Symbol) -> None:
        instrument = Instrument(symbol=btc)
        long_exit = build_exit_request(btc, Decimal("1"), instrument, strategy_id="s")
        short_exit = build_exit_request(btc, Decimal("-1"), instrument, strategy_id="s")
        assert long_exit.side is OrderSide.SELL
        assert short_exit.side is OrderSide.BUY
        assert long_exit.reduce_only

    def test_build_exit_request_rejects_a_flat_position(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="flat position"):
            build_exit_request(btc, ZERO, Instrument(symbol=btc), strategy_id=None)

    def test_stats_accumulate(self, btc: Symbol) -> None:
        from quantflow.execution.engine import ExecutionResult

        stats = ExecutionStats()
        stats = stats.observe(ExecutionResult(long_signal(btc), submitted=True, order=None))
        stats = stats.observe(
            ExecutionResult(long_signal(btc), submitted=False, reason="signal is stale")
        )
        assert stats.signals_seen == 2
        assert stats.orders_submitted == 1
        assert stats.stale_signals == 1
        assert "signals_seen" in stats.to_dict()
