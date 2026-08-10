"""Repository round-trips against a real Postgres."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from quantflow.core.errors import NotFoundError
from quantflow.domain.enums import (
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    RunStatus,
    Timeframe,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import Position
from quantflow.persistence.database import Database
from quantflow.persistence.repositories import (
    BacktestRepository,
    CandleRepository,
    ClosedTradeRepository,
    EquityRepository,
    InstrumentRepository,
    OrderRepository,
    PositionRepository,
    RiskEventRepository,
    TradingSessionRepository,
)

pytestmark = pytest.mark.integration

START = datetime(2026, 1, 1, tzinfo=UTC)
BTC = Symbol(base="BTC", quote="USDT")


def candle(index: int, *, close: str = "50000", volume: str = "1.5") -> Candle:
    price = Decimal(close)
    return Candle(
        symbol=BTC,
        timeframe=Timeframe.H1,
        open_time=START + timedelta(hours=index),
        open=price,
        high=price + Decimal("100"),
        low=price - Decimal("100"),
        close=price,
        volume=Decimal(volume),
        quote_volume=Decimal(volume) * price,
        trades=42,
    )


class TestCandleRepository:
    async def test_upsert_and_fetch_round_trip(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        written = await repo.upsert_many([candle(i) for i in range(5)])
        await session.commit()

        assert written == 5
        stored = await repo.fetch(BTC, Timeframe.H1)
        assert len(stored) == 5
        assert stored[0].open_time == START
        assert stored[0].close == Decimal("50000")
        assert stored[0].volume == Decimal("1.5")
        assert stored[0].trades == 42

    async def test_decimal_precision_survives_the_round_trip(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        precise = Candle(
            symbol=BTC,
            timeframe=Timeframe.H1,
            open_time=START,
            open=Decimal("50000.123456789012"),
            high=Decimal("50000.123456789012"),
            low=Decimal("50000.123456789012"),
            close=Decimal("50000.123456789012"),
            volume=Decimal("0.000000010000"),
        )
        await repo.upsert_many([precise])
        await session.commit()

        stored = (await repo.fetch(BTC, Timeframe.H1))[0]
        assert stored.close == Decimal("50000.123456789012")
        assert stored.volume == Decimal("0.000000010000")

    async def test_upsert_is_idempotent(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        await repo.upsert_many([candle(i) for i in range(3)])
        await repo.upsert_many([candle(i) for i in range(3)])
        await session.commit()
        assert await repo.count(BTC, Timeframe.H1) == 3

    async def test_upsert_updates_in_place(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        await repo.upsert_many([candle(0, close="50000")])
        await repo.upsert_many([candle(0, close="51000")])
        await session.commit()

        stored = await repo.fetch(BTC, Timeframe.H1)
        assert len(stored) == 1
        assert stored[0].close == Decimal("51000")

    async def test_fetch_range_is_half_open(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        await repo.upsert_many([candle(i) for i in range(10)])
        await session.commit()

        stored = await repo.fetch(
            BTC, Timeframe.H1, start=START + timedelta(hours=2), end=START + timedelta(hours=5)
        )
        assert [c.open_time for c in stored] == [
            START + timedelta(hours=2),
            START + timedelta(hours=3),
            START + timedelta(hours=4),
        ]

    async def test_newest_first_limit_returns_chronological_tail(
        self, session: AsyncSession
    ) -> None:
        repo = CandleRepository(session)
        await repo.upsert_many([candle(i) for i in range(10)])
        await session.commit()

        stored = await repo.fetch(BTC, Timeframe.H1, limit=3, newest_first=True)
        assert [c.open_time for c in stored] == [
            START + timedelta(hours=7),
            START + timedelta(hours=8),
            START + timedelta(hours=9),
        ]

    async def test_boundary_times(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        assert await repo.latest_open_time(BTC, Timeframe.H1) is None
        await repo.upsert_many([candle(i) for i in range(4)])
        await session.commit()

        assert await repo.earliest_open_time(BTC, Timeframe.H1) == START
        assert await repo.latest_open_time(BTC, Timeframe.H1) == START + timedelta(hours=3)

    async def test_timeframes_are_isolated(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        hourly = candle(0)
        daily = Candle(
            symbol=BTC,
            timeframe=Timeframe.D1,
            open_time=START,
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("1"),
            close=Decimal("1"),
            volume=Decimal("1"),
        )
        await repo.upsert_many([hourly, daily])
        await session.commit()

        assert await repo.count(BTC, Timeframe.H1) == 1
        assert await repo.count(BTC, Timeframe.D1) == 1

    async def test_integrity_report_detects_gaps(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        await repo.upsert_many([candle(0), candle(1), candle(5), candle(6)])
        await session.commit()

        report = await repo.integrity_report(BTC, Timeframe.H1)
        assert not report.is_clean
        assert report.candle_count == 4
        assert report.gaps == ((START + timedelta(hours=2), START + timedelta(hours=5)),)
        assert report.missing_bar_count == 3

    async def test_integrity_report_on_contiguous_data(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        await repo.upsert_many([candle(i) for i in range(6)])
        await session.commit()
        assert (await repo.integrity_report(BTC, Timeframe.H1)).is_clean

    async def test_integrity_report_on_empty_range(self, session: AsyncSession) -> None:
        report = await CandleRepository(session).integrity_report(BTC, Timeframe.H1)
        assert report.candle_count == 0
        assert report.start is None

    async def test_delete_range(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        await repo.upsert_many([candle(i) for i in range(10)])
        await session.commit()

        removed = await repo.delete_range(
            BTC, Timeframe.H1, start=START + timedelta(hours=2), end=START + timedelta(hours=5)
        )
        await session.commit()
        assert removed == 3
        assert await repo.count(BTC, Timeframe.H1) == 7

    async def test_available_series(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        await repo.upsert_many([candle(i) for i in range(3)])
        await session.commit()
        series = await repo.available_series()
        assert (BTC, Timeframe.H1, 3) in series

    async def test_large_batch_crosses_chunk_boundary(self, session: AsyncSession) -> None:
        repo = CandleRepository(session)
        written = await repo.upsert_many([candle(i) for i in range(6_000)])
        await session.commit()
        assert written == 6_000
        assert await repo.count(BTC, Timeframe.H1) == 6_000


class TestInstrumentRepository:
    async def test_upsert_and_get(self, session: AsyncSession) -> None:
        repo = InstrumentRepository(session)
        instrument = Instrument(
            symbol=BTC,
            price_tick=Decimal("0.01"),
            quantity_step=Decimal("0.00001"),
            min_quantity=Decimal("0.00001"),
            min_notional=Decimal("5"),
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001"),
        )
        await repo.upsert(instrument)
        await session.commit()

        stored = await repo.get(BTC)
        assert stored is not None
        assert stored.price_tick == Decimal("0.01")
        assert stored.min_notional == Decimal("5")

    async def test_upsert_refreshes_rules(self, session: AsyncSession) -> None:
        repo = InstrumentRepository(session)
        await repo.upsert(Instrument(symbol=BTC, min_notional=Decimal("5")))
        await repo.upsert(Instrument(symbol=BTC, min_notional=Decimal("10")))
        await session.commit()

        stored = await repo.get(BTC)
        assert stored is not None
        assert stored.min_notional == Decimal("10")

    async def test_get_missing_returns_none(self, session: AsyncSession) -> None:
        assert await InstrumentRepository(session).get(BTC) is None


class TestOrderRepository:
    def _order(self) -> Order:
        request = OrderRequest(
            symbol=BTC,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            strategy_id="ema_cross",
        )
        return Order.from_request(request, now=START)

    async def test_save_and_load(self, session: AsyncSession) -> None:
        repo = OrderRepository(session)
        order = self._order()
        await repo.save(order)
        await session.commit()

        stored = await repo.get(order.order_id)
        assert stored is not None
        assert stored.symbol == BTC
        assert stored.quantity == Decimal("0.5")
        assert stored.price == Decimal("50000")
        assert stored.stop_loss_price == Decimal("49000")
        assert stored.status is OrderStatus.PENDING_NEW
        assert stored.strategy_id == "ema_cross"

    async def test_save_persists_fills_and_is_idempotent(self, session: AsyncSession) -> None:
        repo = OrderRepository(session)
        order = self._order().acknowledge("venue-1", now=START)
        fill = Fill(
            fill_id="venue-fill-1",
            order_id=order.order_id,
            symbol=BTC,
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            fee=Decimal("25"),
            fee_currency="USDT",
            timestamp=START + timedelta(seconds=1),
            role=LiquidityRole.MAKER,
        )
        order = order.apply_fill(fill)

        await repo.save(order)
        await session.commit()
        await repo.save(order)  # re-saving must not duplicate the fill
        await session.commit()

        stored = await repo.get(order.order_id)
        assert stored is not None
        assert stored.status is OrderStatus.FILLED
        assert len(stored.fills) == 1
        assert stored.fills[0].fee == Decimal("25")
        assert stored.fills[0].role is LiquidityRole.MAKER
        assert stored.average_fill_price == Decimal("50000")

    async def test_get_by_client_id(self, session: AsyncSession) -> None:
        repo = OrderRepository(session)
        order = self._order()
        await repo.save(order)
        await session.commit()

        stored = await repo.get_by_client_id(order.client_order_id)
        assert stored is not None
        assert stored.order_id == order.order_id

    async def test_require_raises_when_absent(self, session: AsyncSession) -> None:
        with pytest.raises(NotFoundError, match="not found"):
            await OrderRepository(session).require("missing")

    async def test_list_open_excludes_terminal_orders(self, session: AsyncSession) -> None:
        repo = OrderRepository(session)
        live = self._order().acknowledge("v1", now=START)
        dead = self._order().transition_to(OrderStatus.CANCELLED, now=START)
        await repo.save(live)
        await repo.save(dead)
        await session.commit()

        open_orders = await repo.list_open()
        assert [order.order_id for order in open_orders] == [live.order_id]

    async def test_count_since(self, session: AsyncSession) -> None:
        repo = OrderRepository(session)
        for _ in range(3):
            await repo.save(self._order())
        await session.commit()

        assert await repo.count_since(datetime(2020, 1, 1, tzinfo=UTC)) == 3
        assert await repo.count_since(datetime(2099, 1, 1, tzinfo=UTC)) == 0


class TestPositionRepository:
    async def test_save_and_load_open_position(self, session: AsyncSession) -> None:
        repo = PositionRepository(session)
        fill = Fill(
            fill_id="f1",
            order_id="o1",
            symbol=BTC,
            side=OrderSide.BUY,
            quantity=Decimal("2"),
            price=Decimal("50000"),
            fee=Decimal("50"),
            fee_currency="USDT",
            timestamp=START,
        )
        position, _ = Position(symbol=BTC, strategy_id="ema_cross").apply_fill(fill)
        position = position.with_protection(stop_loss_price=Decimal("49000"))

        await repo.save(position)
        await session.commit()

        stored = await repo.get_open(BTC)
        assert stored is not None
        assert stored.quantity == Decimal("2")
        assert stored.average_entry_price == Decimal("50000")
        assert stored.stop_loss_price == Decimal("49000")
        assert len(stored.lots) == 1
        assert stored.lots[0].fee == Decimal("50")

    async def test_lots_survive_the_json_round_trip(self, session: AsyncSession) -> None:
        repo = PositionRepository(session)
        position = Position(symbol=BTC)
        for index in range(3):
            position, _ = position.apply_fill(
                Fill(
                    fill_id=f"f{index}",
                    order_id="o1",
                    symbol=BTC,
                    side=OrderSide.BUY,
                    quantity=Decimal("1"),
                    price=Decimal(str(50000 + index * 100)),
                    fee=Decimal("1"),
                    fee_currency="USDT",
                    timestamp=START + timedelta(minutes=index),
                )
            )
        await repo.save(position)
        await session.commit()

        stored = await repo.get_open(BTC)
        assert stored is not None
        assert [lot.price for lot in stored.lots] == [
            Decimal("50000"),
            Decimal("50100"),
            Decimal("50200"),
        ]
        assert stored.average_entry_price == position.average_entry_price

    async def test_closing_a_position_removes_it_from_open(self, session: AsyncSession) -> None:
        repo = PositionRepository(session)
        position, _ = Position(symbol=BTC).apply_fill(
            Fill(
                fill_id="f1",
                order_id="o1",
                symbol=BTC,
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                price=Decimal("50000"),
                fee=Decimal("0"),
                fee_currency="USDT",
                timestamp=START,
            )
        )
        await repo.save(position)
        await session.commit()

        closed, _ = position.apply_fill(
            Fill(
                fill_id="f2",
                order_id="o2",
                symbol=BTC,
                side=OrderSide.SELL,
                quantity=Decimal("1"),
                price=Decimal("51000"),
                fee=Decimal("0"),
                fee_currency="USDT",
                timestamp=START + timedelta(hours=1),
            )
        )
        await repo.save(closed)
        await session.commit()

        assert await repo.get_open(BTC) is None
        assert await repo.list_open() == []

    async def test_saving_a_flat_position_creates_nothing(self, session: AsyncSession) -> None:
        repo = PositionRepository(session)
        await repo.save(Position(symbol=BTC))
        await session.commit()
        assert await repo.list_open() == []


class TestSessionsEquityAndRisk:
    async def test_session_lifecycle(self, session: AsyncSession) -> None:
        repo = TradingSessionRepository(session)
        await repo.create(
            session_id="s1",
            mode="paper",
            strategy_id="ema_cross",
            symbols=[BTC],
            timeframe=Timeframe.H1,
            starting_equity=Decimal("10000"),
            strategy_params={"fast": 12, "slow": 26},
        )
        await session.commit()

        await repo.finish(
            "s1",
            status=RunStatus.COMPLETED,
            final_equity=Decimal("11000"),
            metrics={"sharpe": 1.42},
        )
        await session.commit()

        record = await repo.get("s1")
        assert record is not None
        assert record.status is RunStatus.COMPLETED
        assert record.final_equity == Decimal("11000")
        assert record.metrics["sharpe"] == 1.42
        assert record.strategy_params["fast"] == "12"

    async def test_finish_unknown_session_raises(self, session: AsyncSession) -> None:
        with pytest.raises(NotFoundError):
            await TradingSessionRepository(session).finish("nope", status=RunStatus.FAILED)

    async def test_equity_curve_round_trip(self, session: AsyncSession) -> None:
        await TradingSessionRepository(session).create(
            session_id="s1",
            mode="backtest",
            strategy_id="ema_cross",
            symbols=[BTC],
            timeframe=Timeframe.H1,
            starting_equity=Decimal("10000"),
        )
        repo = EquityRepository(session)
        for index in range(5):
            await repo.add(
                "s1",
                EquityPoint(
                    timestamp=START + timedelta(hours=index),
                    equity=Decimal(10000 + index * 100),
                    cash=Decimal("5000"),
                    position_count=1,
                ),
            )
        await session.commit()

        curve = await repo.curve("s1")
        assert len(curve) == 5
        assert curve[-1].equity == Decimal("10400")
        latest = await repo.latest("s1")
        assert latest is not None
        assert latest.timestamp == START + timedelta(hours=4)

    async def test_equity_upsert_replaces_duplicate_timestamp(self, session: AsyncSession) -> None:
        await TradingSessionRepository(session).create(
            session_id="s1",
            mode="backtest",
            strategy_id="x",
            symbols=[BTC],
            timeframe=Timeframe.H1,
            starting_equity=Decimal("1"),
        )
        repo = EquityRepository(session)
        point = EquityPoint(
            timestamp=START, equity=Decimal("100"), cash=Decimal("100"), position_count=0
        )
        await repo.add("s1", point)
        await repo.add(
            "s1",
            EquityPoint(
                timestamp=START, equity=Decimal("200"), cash=Decimal("200"), position_count=0
            ),
        )
        await session.commit()

        curve = await repo.curve("s1")
        assert len(curve) == 1
        assert curve[0].equity == Decimal("200")

    async def test_risk_events_are_recorded(self, session: AsyncSession) -> None:
        repo = RiskEventRepository(session)
        await repo.record(
            rule="max_drawdown",
            message="drawdown 18% exceeds limit 15%",
            severity="critical",
            symbol=BTC,
            observed_value=Decimal("0.18"),
            limit_value=Decimal("0.15"),
            blocked_order=True,
            halted_trading=True,
        )
        await session.commit()

        events = await repo.list_recent()
        assert len(events) == 1
        assert events[0].rule == "max_drawdown"
        assert events[0].halted_trading is True
        assert events[0].observed_value == Decimal("0.18")

    async def test_kill_switch_latches_and_clears(self, session: AsyncSession) -> None:
        repo = RiskEventRepository(session)
        assert (await repo.get_kill_switch()).engaged is False

        await repo.set_kill_switch(engaged=True, reason="max drawdown", actor="risk_engine")
        await session.commit()

        record = await repo.get_kill_switch()
        assert record.engaged is True
        assert record.reason == "max drawdown"
        assert record.engaged_at is not None
        assert record.engaged_by == "risk_engine"

        await repo.set_kill_switch(engaged=False, actor="operator")
        await session.commit()

        record = await repo.get_kill_switch()
        assert record.engaged is False
        assert record.cleared_by == "operator"


class TestUnitOfWork:
    async def test_commits_on_clean_exit(self, database: Database) -> None:
        async with database.unit_of_work() as uow:
            await uow.candles.upsert_many([candle(0)])

        async with database.read_session() as session:
            assert await CandleRepository(session).count(BTC, Timeframe.H1) == 1

    async def test_rolls_back_on_exception(self, database: Database) -> None:
        async def write_then_fail() -> None:
            async with database.unit_of_work() as uow:
                await uow.candles.upsert_many([candle(0)])
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await write_then_fail()

        async with database.read_session() as session:
            assert await CandleRepository(session).count(BTC, Timeframe.H1) == 0

    async def test_multi_repository_write_is_atomic(self, database: Database) -> None:
        request = OrderRequest(
            symbol=BTC, side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("1")
        )
        order = Order.from_request(request, now=START)

        async def write_both_then_fail() -> None:
            async with database.unit_of_work() as uow:
                await uow.orders.save(order)
                await uow.candles.upsert_many([candle(0)])
                raise RuntimeError("fail after both writes")

        with pytest.raises(RuntimeError, match="fail after both writes"):
            await write_both_then_fail()

        async with database.read_session() as session:
            assert await OrderRepository(session).get(order.order_id) is None
            assert await CandleRepository(session).count(BTC, Timeframe.H1) == 0

    async def test_session_outside_context_raises(self, database: Database) -> None:
        from quantflow.core.errors import DatabaseError

        uow = database.unit_of_work()
        with pytest.raises(DatabaseError, match="not active"):
            _ = uow.session

    async def test_ping_and_extension_check(self, database: Database) -> None:
        assert await database.ping() is True
        assert await database.has_extension("definitely_not_installed") is False


class TestClosedTrades:
    async def test_add_and_aggregate(self, session: AsyncSession) -> None:
        position = Position(symbol=BTC, strategy_id="ema_cross")
        position, _ = position.apply_fill(
            Fill(
                fill_id="f1",
                order_id="o1",
                symbol=BTC,
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                price=Decimal("50000"),
                fee=Decimal("10"),
                fee_currency="USDT",
                timestamp=START,
            )
        )
        _, closed = position.apply_fill(
            Fill(
                fill_id="f2",
                order_id="o2",
                symbol=BTC,
                side=OrderSide.SELL,
                quantity=Decimal("1"),
                price=Decimal("51000"),
                fee=Decimal("10"),
                fee_currency="USDT",
                timestamp=START + timedelta(hours=2),
            )
        )

        repo = ClosedTradeRepository(session)
        assert await repo.add_many(closed) == 1
        await session.commit()

        stored = await repo.list_between(START, START + timedelta(days=1))
        assert len(stored) == 1
        assert stored[0].gross_pnl == Decimal("1000")
        assert stored[0].net_pnl == Decimal("980")

        realized = await repo.realized_pnl_since(START)
        assert realized == Decimal("980")


class TestBacktestRepository:
    async def test_save_and_list(self, session: AsyncSession) -> None:
        from quantflow.persistence.models import BacktestRunRecord

        repo = BacktestRepository(session)
        record = BacktestRunRecord(
            id="bt1",
            strategy_id="ema_cross",
            symbols=["BTC/USDT"],
            timeframe=Timeframe.H1,
            start=START,
            end=START + timedelta(days=30),
            status=RunStatus.COMPLETED,
            starting_equity=Decimal("10000"),
            final_equity=Decimal("12000"),
            sharpe_ratio=Decimal("1.85"),
            trade_count=42,
        )
        await repo.save(record)
        await session.commit()

        stored = await repo.get("bt1")
        assert stored is not None
        assert stored.sharpe_ratio == Decimal("1.85")
        assert len(await repo.list_recent(strategy_id="ema_cross")) == 1
        assert await repo.list_recent(strategy_id="other") == []


class TestSessionRestartSafety:
    """A paper session must survive a restart under the same id."""

    async def test_reopening_the_same_session_id_does_not_raise(
        self, session: AsyncSession
    ) -> None:
        # The failure this guards: a dropped websocket, an operator restart or a container
        # bounce all bring the session back with the same id, and a plain INSERT turned
        # every one of those into a unique-key violation at startup — the platform dying
        # exactly when it was trying to recover.
        repo = TradingSessionRepository(session)
        params = {
            "session_id": "restart-me",
            "mode": "paper",
            "strategy_id": "donchian_breakout",
            "symbols": [BTC],
            "timeframe": Timeframe.M5,
            "starting_equity": Decimal("10000"),
        }
        first = await repo.create(**params)  # type: ignore[arg-type]
        await session.commit()
        assert first.id == "restart-me"

        second = await repo.create(**params)  # type: ignore[arg-type]
        await session.commit()
        assert second.id == "restart-me"
        assert second.status is RunStatus.RUNNING

    async def test_restart_clears_the_previous_terminal_state(self, session: AsyncSession) -> None:
        # A finished session that restarts must not keep reporting its old final equity,
        # or the dashboard shows a completed run while a live one is underway.
        repo = TradingSessionRepository(session)
        params = {
            "session_id": "restart-after-finish",
            "mode": "paper",
            "strategy_id": "ema_cross",
            "symbols": [BTC],
            "timeframe": Timeframe.M5,
            "starting_equity": Decimal("10000"),
        }
        await repo.create(**params)  # type: ignore[arg-type]
        await repo.finish(
            "restart-after-finish", status=RunStatus.COMPLETED, final_equity=Decimal("9000")
        )
        await session.commit()

        reopened = await repo.create(**params)  # type: ignore[arg-type]
        await session.commit()
        assert reopened.status is RunStatus.RUNNING
        assert reopened.final_equity is None
        assert reopened.finished_at is None


class TestEquityCurveWindow:
    """A live chart must track the present, not freeze at the session start."""

    async def test_the_limit_returns_the_newest_points(self, session: AsyncSession) -> None:
        # Ordering ascending and then limiting silently freezes a running chart: once the
        # session exceeds the limit every later point falls outside the window and the
        # curve stops advancing while still looking healthy.
        sessions = TradingSessionRepository(session)
        await sessions.create(
            session_id="curve-window",
            mode="paper",
            strategy_id="ema_cross",
            symbols=[BTC],
            timeframe=Timeframe.M5,
            starting_equity=Decimal("10000"),
        )
        equity = EquityRepository(session)
        base = datetime(2026, 6, 1, tzinfo=UTC)
        for i in range(20):
            await equity.add(
                "curve-window",
                EquityPoint(
                    timestamp=base + timedelta(minutes=5 * i),
                    equity=Decimal("10000") + Decimal(i),
                    cash=Decimal("10000"),
                    position_count=0,
                ),
            )
        await session.commit()

        window = await equity.curve("curve-window", limit=5)
        assert len(window) == 5
        # Newest five, still oldest-first within the window.
        assert window[-1].equity == Decimal("10019")
        assert window[0].equity == Decimal("10015")
