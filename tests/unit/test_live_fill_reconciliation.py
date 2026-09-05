"""Live fills must come back into the portfolio, or the risk engine guards nothing.

A simulated router hands the engine ``(order, fill)`` pairs when a bar is processed. A live
router cannot: a real venue matches when it matches, and says so through its own endpoints.
The consequence, before :class:`LiveReconciler` existed, was that ``LiveOrderRouter`` sent
real orders that really filled and ``process_candle`` returned ``()`` — so nothing ever
called ``PortfolioManager.apply_fill``. The local book stayed empty for the entire life of
a session while the exchange held real, protected positions.

That is not a reporting bug. Every risk limit is measured against the portfolio snapshot:
position count, gross exposure, drawdown, daily loss. A permanently flat book means every
one of them is evaluated against an account that does not exist, and the engine will keep
approving entries it believes are its first.

These tests are written against a fake venue in the style of
``test_intrabar_runner_integration.py``, and they are deliberately strict about the three
ways a reconciler can be worse than none at all:

* applying the same execution twice, which doubles a position with no order behind it;
* believing the fill stream over the venue's own statement of the book;
* repairing state from the position endpoint so eagerly that an execution still in flight
  is later applied on top of the repair.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import SecretStr

from quantflow.core.config import MarketType, RiskSettings
from quantflow.domain.enums import (
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import Fill, Order, OrderRequest
from quantflow.domain.portfolio import Balance
from quantflow.live.reconcile import LiveReconciler
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine, exposure_headroom
from quantflow.risk.monitor import evaluate_limits

BTC = Symbol.parse("BTC/USDT")
ETH = Symbol.parse("ETH/USDT")
NOW = datetime(2026, 8, 14, 9, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# The fake venue
# --------------------------------------------------------------------------- #
def venue_position(
    symbol: Symbol,
    *,
    side: str = "long",
    quantity: str = "1",
    entry: str = "100",
    stop: str | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """One row shaped like Bybit's ``fetch_positions`` output."""
    info: dict[str, Any] = {
        "size": quantity,
        "side": "Buy" if side == "long" else "Sell",
        "avgPrice": entry,
        "leverage": "1",
    }
    if stop is not None:
        info["stopLoss"] = stop
    if target is not None:
        info["takeProfit"] = target
    return {
        "symbol": f"{symbol.slashed}:{symbol.quote}",
        "side": side,
        "contracts": quantity,
        "entryPrice": entry,
        "info": info,
    }


def execution(
    symbol: Symbol,
    *,
    fill_id: str,
    order_id: str,
    side: OrderSide = OrderSide.BUY,
    quantity: str = "1",
    price: str = "100",
    fee: str = "0.06",
    at: datetime | None = None,
    realized_pnl: str | None = None,
) -> Fill:
    """One execution as ``fetch_my_trades`` returns it, already parsed."""
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency=symbol.quote,
        timestamp=at or NOW,
        role=LiquidityRole.TAKER,
        realized_pnl=None if realized_pnl is None else Decimal(realized_pnl),
    )


def local_order(
    symbol: Symbol,
    *,
    order_id: str = "local-1",
    side: OrderSide = OrderSide.BUY,
    quantity: str = "1",
    status: OrderStatus = OrderStatus.NEW,
) -> Order:
    """An order as the engine holds it immediately after a live submit."""
    return Order(
        order_id=order_id,
        client_order_id=f"qf-{order_id}",
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=Decimal(quantity),
        status=status,
        created_at=NOW,
        updated_at=NOW,
        time_in_force=TimeInForce.GTC,
        venue_order_id=f"venue-{order_id}",
        strategy_id="ema_cross",
    )


class FakeGateway:
    """Answers the three reads the reconciler makes, and records that it was asked.

    The reads are independent on purpose. A real venue's position endpoint and execution
    endpoint do not update in the same instant, and a reconciler that assumes they do is one
    that will eventually synthesise a close for a fill it was about to be told about.
    """

    def __init__(self) -> None:
        self.positions: list[dict[str, Any]] = []
        self.positions_error: Exception | None = None
        self.executions: dict[Symbol, list[Fill]] = {}
        self.orders: dict[str, Order] = {}
        self.wallet = Decimal("10000")
        #: Raised by ``fetch_order`` when set — a read-back that timed out.
        self.order_read_error: Exception | None = None
        self.order_reads = 0
        self.registered: list[tuple[str, str]] = []

    async def fetch_order(self, order_id: str, symbol: Symbol) -> Order:
        self.order_reads += 1
        if self.order_read_error is not None:
            raise self.order_read_error
        found = self.orders.get(order_id)
        if found is None:
            raise LookupError(f"order {order_id} not found for {symbol}")
        return found

    async def fetch_my_trades(
        self, symbol: Symbol, *, since: datetime | None = None, limit: int = 100
    ) -> list[Fill]:
        rows = self.executions.get(symbol, [])
        if since is not None:
            rows = [fill for fill in rows if fill.timestamp >= since]
        return rows[:limit]

    async def fetch_positions(self) -> list[dict[str, Any]]:
        if self.positions_error is not None:
            raise self.positions_error
        return list(self.positions)

    async def fetch_balances(self) -> dict[str, Balance]:
        return {"USDT": Balance(asset="USDT", free=self.wallet, locked=Decimal("0"))}

    def register_venue_id(self, order_id: str, venue_order_id: str) -> None:
        self.registered.append((order_id, venue_order_id))


def portfolio(equity: str = "10000") -> PortfolioManager:
    """A futures portfolio, the accounting a linear-perp venue actually uses."""
    return PortfolioManager(
        starting_equity=Decimal(equity),
        market_type=MarketType.FUTURE,
        leverage=Decimal("1"),
    )


def reconciler(
    gateway: FakeGateway,
    book: PortfolioManager,
    *,
    symbols: tuple[Symbol, ...] = (BTC,),
    confirmations: int = 1,
) -> LiveReconciler:
    """A reconciler with the confirmation delay collapsed unless a test wants it."""
    return LiveReconciler(
        gateway,
        book,
        symbols=symbols,
        clock=lambda: NOW + timedelta(minutes=1),
        repair_confirmations=confirmations,
    )


def instrument(symbol: Symbol = BTC) -> Instrument:
    return Instrument(
        symbol=symbol,
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.001"),
        min_quantity=Decimal("0.001"),
        min_notional=Decimal("5"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0006"),
    )


# --------------------------------------------------------------------------- #
# Fills reaching the local book
# --------------------------------------------------------------------------- #
class TestVenueFillsCreatePositions:
    """The break itself: a real fill has to become a real local position."""

    @pytest.mark.asyncio
    async def test_a_filled_buy_creates_a_local_long(self) -> None:
        gateway = FakeGateway()
        order = local_order(BTC)
        gateway.orders[order.order_id] = order
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id=order.order_id, quantity="0.5", price="60000")
        ]
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()

        await reconciler(gateway, book).reconcile([order])

        position = book.position_for(BTC)
        assert position is not None
        assert position.side is PositionSide.LONG
        assert position.quantity == Decimal("0.5")
        assert position.average_entry_price == Decimal("60000")

    @pytest.mark.asyncio
    async def test_a_filled_sell_creates_a_local_short(self) -> None:
        gateway = FakeGateway()
        order = local_order(BTC, side=OrderSide.SELL)
        gateway.orders[order.order_id] = order
        gateway.executions[BTC] = [
            execution(
                BTC,
                fill_id="e1",
                order_id=order.order_id,
                side=OrderSide.SELL,
                quantity="0.5",
                price="60000",
            )
        ]
        gateway.positions = [venue_position(BTC, side="short", quantity="0.5", entry="60000")]
        book = portfolio()

        await reconciler(gateway, book).reconcile([order])

        position = book.position_for(BTC)
        assert position is not None
        assert position.side is PositionSide.SHORT
        assert position.quantity == Decimal("-0.5")

    @pytest.mark.asyncio
    async def test_a_partial_fill_updates_the_local_quantity(self) -> None:
        """Half of a one-BTC order filled is half a BTC held, not one and not none."""
        gateway = FakeGateway()
        order = local_order(BTC, quantity="1")
        gateway.orders[order.order_id] = order
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id=order.order_id, quantity="0.4", price="60000")
        ]
        gateway.positions = [venue_position(BTC, quantity="0.4", entry="60000")]
        book = portfolio()

        outcome = await reconciler(gateway, book).reconcile([order])

        position = book.position_for(BTC)
        assert position is not None
        assert position.quantity == Decimal("0.4")
        reconciled = next(item for item in outcome.orders if item.order_id == order.order_id)
        assert reconciled.status is OrderStatus.PARTIALLY_FILLED
        assert reconciled.filled_quantity == Decimal("0.4")

    @pytest.mark.asyncio
    async def test_a_second_fill_updates_the_average_entry_price(self) -> None:
        gateway = FakeGateway()
        order = local_order(BTC, quantity="1")
        gateway.orders[order.order_id] = order
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id=order.order_id, quantity="0.5", price="60000")
        ]
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()
        engine = reconciler(gateway, book)
        first = await engine.reconcile([order])
        # The caller holds the reconciled order, exactly as the engine does: fills
        # accumulate on the record rather than being rediscovered from scratch.
        order = next(item for item in first.orders if item.order_id == order.order_id)

        gateway.executions[BTC].append(
            execution(
                BTC,
                fill_id="e2",
                order_id=order.order_id,
                quantity="0.5",
                price="61000",
                at=NOW + timedelta(seconds=30),
            )
        )
        gateway.positions = [venue_position(BTC, quantity="1", entry="60500")]
        outcome = await engine.reconcile([order])

        position = book.position_for(BTC)
        assert position is not None
        assert position.quantity == Decimal("1")
        assert position.average_entry_price == Decimal("60500")
        reconciled = next(item for item in outcome.orders if item.order_id == order.order_id)
        assert reconciled.status is OrderStatus.FILLED
        assert reconciled.average_fill_price == Decimal("60500")

    @pytest.mark.asyncio
    async def test_a_duplicate_execution_is_ignored(self) -> None:
        """The venue re-delivers on every poll; counting one twice doubles the position."""
        gateway = FakeGateway()
        order = local_order(BTC)
        gateway.orders[order.order_id] = order
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id=order.order_id, quantity="0.5", price="60000")
        ]
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()
        engine = reconciler(gateway, book)

        await engine.reconcile([order])
        second = await engine.reconcile([order])

        position = book.position_for(BTC)
        assert position is not None
        assert position.quantity == Decimal("0.5")
        assert second.fills == []
        assert len(book.closed_trades) == 0

    @pytest.mark.asyncio
    async def test_an_order_whose_read_back_timed_out_is_settled_by_its_fill(self) -> None:
        """The submit response never came. The execution still did, and it is enough."""
        gateway = FakeGateway()
        gateway.order_read_error = TimeoutError("venue read timed out")
        order = local_order(BTC, quantity="0.5")
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id=order.order_id, quantity="0.5", price="60000")
        ]
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()

        outcome = await reconciler(gateway, book).reconcile([order])

        assert gateway.order_reads == 1
        reconciled = next(item for item in outcome.orders if item.order_id == order.order_id)
        assert reconciled.status is OrderStatus.FILLED
        assert reconciled.filled_quantity == Decimal("0.5")
        position = book.position_for(BTC)
        assert position is not None
        assert position.quantity == Decimal("0.5")


class TestClosingRoundTrips:
    """A closed position is only a *result* if it leaves a closed trade behind."""

    @pytest.mark.asyncio
    async def test_a_full_close_creates_a_closed_trade(self) -> None:
        gateway = FakeGateway()
        entry = local_order(BTC, order_id="entry", quantity="0.5")
        gateway.orders[entry.order_id] = entry
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id="entry", quantity="0.5", price="60000")
        ]
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()
        engine = reconciler(gateway, book)
        await engine.reconcile([entry])

        gateway.executions[BTC].append(
            execution(
                BTC,
                fill_id="e2",
                order_id="venue-stop-1",
                side=OrderSide.SELL,
                quantity="0.5",
                price="61000",
                at=NOW + timedelta(minutes=5),
                realized_pnl="500",
            )
        )
        gateway.positions = []
        outcome = await engine.reconcile([entry])

        assert book.position_for(BTC) is None
        assert len(outcome.closed_trades) == 1
        trade = outcome.closed_trades[0]
        assert trade.side is PositionSide.LONG
        assert trade.quantity == Decimal("0.5")
        assert trade.entry_price == Decimal("60000")
        assert trade.exit_price == Decimal("61000")

    @pytest.mark.asyncio
    async def test_the_realized_pnl_of_the_round_trip_is_recorded(self) -> None:
        gateway = FakeGateway()
        entry = local_order(BTC, order_id="entry", quantity="0.5")
        gateway.orders[entry.order_id] = entry
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id="entry", quantity="0.5", price="60000", fee="18")
        ]
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()
        engine = reconciler(gateway, book)
        await engine.reconcile([entry])

        gateway.executions[BTC].append(
            execution(
                BTC,
                fill_id="e2",
                order_id="venue-stop-1",
                side=OrderSide.SELL,
                quantity="0.5",
                price="61000",
                fee="18.3",
                at=NOW + timedelta(minutes=5),
                realized_pnl="500",
            )
        )
        gateway.positions = []
        outcome = await engine.reconcile([entry])

        trade = outcome.closed_trades[0]
        # 0.5 BTC from 60,000 to 61,000 is 500 gross, and the venue agrees.
        assert trade.gross_pnl == Decimal("500")
        assert book.realized_pnl == Decimal("500")
        assert trade.net_pnl == Decimal("500") - trade.fees

    @pytest.mark.asyncio
    async def test_the_fees_the_venue_charged_are_carried_through(self) -> None:
        """Both legs. A round trip charged twice that shows one fee is a fictional edge."""
        gateway = FakeGateway()
        entry = local_order(BTC, order_id="entry", quantity="0.5")
        gateway.orders[entry.order_id] = entry
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id="entry", quantity="0.5", price="60000", fee="18")
        ]
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()
        engine = reconciler(gateway, book)
        outcome = await engine.reconcile([entry])
        reconciled = next(item for item in outcome.orders if item.order_id == "entry")
        assert reconciled.fees_paid == Decimal("18")
        assert reconciled.fills[0].fee == Decimal("18")

        gateway.executions[BTC].append(
            execution(
                BTC,
                fill_id="e2",
                order_id="venue-stop-1",
                side=OrderSide.SELL,
                quantity="0.5",
                price="61000",
                fee="18.3",
                at=NOW + timedelta(minutes=5),
            )
        )
        gateway.positions = []
        outcome = await engine.reconcile([entry])

        assert outcome.closed_trades[0].fees == Decimal("36.3")
        assert book.fees_paid == Decimal("36.3")
        # The venue-side stop had no local order; it still gets one, or its fill cannot be
        # persisted at all — the fills table is keyed by order.
        adopted = next(item for item in outcome.orders if item.order_id == "venue-stop-1")
        assert adopted.status is OrderStatus.FILLED
        assert adopted.fees_paid == Decimal("18.3")
        assert adopted.fills[0].fill_id == "e2"


class TestTheVenueIsAuthoritative:
    """When the fill stream and the venue's own book disagree, the book wins."""

    @pytest.mark.asyncio
    async def test_a_restart_re_adopts_the_positions_the_venue_holds(self) -> None:
        """A fresh process, an empty local book, and three real positions on the account."""
        gateway = FakeGateway()
        gateway.positions = [
            venue_position(BTC, quantity="0.039", entry="60000", stop="59000", target="61000"),
            venue_position(ETH, side="short", quantity="4.1", entry="3000", stop="3100"),
        ]
        book = portfolio()

        outcome = await reconciler(gateway, book, symbols=(BTC, ETH)).reconcile([], initial=True)

        assert sorted(str(symbol) for symbol in outcome.adopted) == ["BTC/USDT", "ETH/USDT"]
        btc_position = book.position_for(BTC)
        eth_position = book.position_for(ETH)
        assert btc_position is not None
        assert eth_position is not None
        assert btc_position.quantity == Decimal("0.039")
        assert btc_position.average_entry_price == Decimal("60000")
        assert btc_position.stop_loss_price == Decimal("59000")
        assert btc_position.take_profit_price == Decimal("61000")
        assert eth_position.side is PositionSide.SHORT
        assert eth_position.quantity == Decimal("-4.1")
        assert eth_position.average_entry_price == Decimal("3000")

    @pytest.mark.asyncio
    async def test_a_venue_position_the_book_does_not_know_about_is_adopted(self) -> None:
        """Mid-session. Nothing filled through us, and the account holds a position anyway."""
        gateway = FakeGateway()
        gateway.positions = [venue_position(BTC, quantity="0.25", entry="60000", stop="59000")]
        book = portfolio()
        assert book.positions == ()

        outcome = await reconciler(gateway, book).reconcile([])

        assert [str(symbol) for symbol in outcome.adopted] == ["BTC/USDT"]
        position = book.position_for(BTC)
        assert position is not None
        assert position.quantity == Decimal("0.25")
        assert position.average_entry_price == Decimal("60000")

    @pytest.mark.asyncio
    async def test_a_position_that_vanished_from_the_venue_is_closed_locally(self) -> None:
        gateway = FakeGateway()
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()
        engine = reconciler(gateway, book)
        await engine.reconcile([], initial=True)
        assert book.position_for(BTC) is not None
        book.update_mark_price(BTC, Decimal("61000"))

        gateway.positions = []
        outcome = await engine.reconcile([])

        assert [str(symbol) for symbol in outcome.orphaned] == ["BTC/USDT"]
        assert book.position_for(BTC) is None
        assert len(outcome.closed_trades) == 1
        assert outcome.closed_trades[0].exit_price == Decimal("61000")

    @pytest.mark.asyncio
    async def test_the_stop_the_venue_holds_replaces_the_one_that_was_asked_for(self) -> None:
        """The exchange snaps a stop to its tick; the snapped one is what will fill."""
        gateway = FakeGateway()
        gateway.positions = [
            venue_position(BTC, quantity="0.5", entry="60000", stop="59000", target="61000")
        ]
        book = portfolio()
        engine = reconciler(gateway, book)
        await engine.reconcile([], initial=True)
        # Something local asked for an unsnapped level, as the risk engine's raw output does.
        book.set_protection(BTC, stop_loss_price=Decimal("58999.7431"))

        outcome = await engine.reconcile([])

        position = book.position_for(BTC)
        assert position is not None
        assert position.stop_loss_price == Decimal("59000")
        assert position.take_profit_price == Decimal("61000")
        # Reported, so the row on disk carries the venue's level rather than the request.
        assert [str(item.symbol) for item in outcome.positions] == ["BTC/USDT"]

    @pytest.mark.asyncio
    async def test_a_disagreement_must_persist_before_state_is_repaired(self) -> None:
        """The two endpoints are not simultaneous.

        Repairing on the first disagreement would synthesise a close, and the execution that
        actually closed the position would then be applied to a flat book — opening a
        position nobody asked for. One observation is a race; two is a fact.
        """
        gateway = FakeGateway()
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio()
        engine = reconciler(gateway, book, confirmations=2)

        first = await engine.reconcile([])
        assert first.adopted == []
        assert book.position_for(BTC) is None

        second = await engine.reconcile([])
        assert [str(symbol) for symbol in second.adopted] == ["BTC/USDT"]
        assert book.position_for(BTC) is not None


# --------------------------------------------------------------------------- #
# What the risk engine then sees
# --------------------------------------------------------------------------- #
class TestRiskSeesTheRealBook:
    """The point of all of the above: the limits have to be measured against reality."""

    @pytest.mark.asyncio
    async def test_the_risk_snapshot_carries_the_actual_open_positions(self) -> None:
        gateway = FakeGateway()
        gateway.positions = [
            venue_position(BTC, quantity="0.5", entry="60000"),
            venue_position(ETH, side="short", quantity="4", entry="3000"),
        ]
        book = portfolio()
        await reconciler(gateway, book, symbols=(BTC, ETH)).reconcile([], initial=True)

        snapshot = book.snapshot(NOW)

        assert snapshot.position_count == 2
        assert {str(position.symbol) for position in snapshot.open_positions} == {
            "BTC/USDT",
            "ETH/USDT",
        }

    @pytest.mark.asyncio
    async def test_max_concurrent_positions_counts_the_venues_positions(self) -> None:
        """The rule that decides whether a sixth position may be opened, against a book
        that used to report zero no matter how many the account held."""
        gateway = FakeGateway()
        gateway.positions = [
            venue_position(BTC, quantity="0.5", entry="60000"),
            venue_position(ETH, quantity="4", entry="3000"),
        ]
        book = portfolio()
        await reconciler(gateway, book, symbols=(BTC, ETH)).reconcile([], initial=True)
        book.update_mark_prices({BTC: Decimal("60000"), ETH: Decimal("3000")})

        settings = RiskSettings(max_concurrent_positions=2, max_total_exposure_pct=Decimal("10"))
        engine = RiskEngine(settings)
        decision = await engine.approve(
            OrderRequest(
                symbol=Symbol.parse("SOL/USDT"),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                stop_loss_price=Decimal("90"),
            ),
            portfolio=book.snapshot(NOW),
            instrument=instrument(Symbol.parse("SOL/USDT")),
            reference_price=Decimal("100"),
        )

        assert not decision.approved
        assert decision.blocking_rule == "max_concurrent_positions"

    @pytest.mark.asyncio
    async def test_gross_exposure_counts_the_venues_positions(self) -> None:
        gateway = FakeGateway()
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000")]
        book = portfolio(equity="100000")
        await reconciler(gateway, book).reconcile([], initial=True)
        book.update_mark_price(BTC, Decimal("60000"))

        snapshot = book.snapshot(NOW)
        settings = RiskSettings(max_total_exposure_pct=Decimal("0.6"))

        assert snapshot.gross_exposure == Decimal("30000")
        # 60% of a ~100k account is ~60k of allowance, and 30k of it is already committed.
        assert (
            exposure_headroom(snapshot, settings)
            < snapshot.equity * settings.max_total_exposure_pct
        )

    @pytest.mark.asyncio
    async def test_drawdown_is_measured_against_the_real_account_state(self) -> None:
        """A losing venue position has to show up as drawdown, or the monitor never fires."""
        gateway = FakeGateway()
        gateway.positions = [venue_position(BTC, quantity="1", entry="60000")]
        book = portfolio(equity="10000")
        await reconciler(gateway, book).reconcile([], initial=True)
        book.record_equity(NOW)

        # The position moves against the account by 2,000 on a 10,000 book.
        book.update_mark_price(BTC, Decimal("58000"))
        point = book.record_equity(NOW + timedelta(minutes=15))

        assert point.position_count == 1
        assert point.unrealized_pnl == Decimal("-2000")
        breach = evaluate_limits(
            book.snapshot(NOW + timedelta(minutes=15)),
            RiskSettings(max_drawdown_pct=Decimal("0.15")),
        )
        assert breach is not None
        assert breach.rule == "max_drawdown"


# --------------------------------------------------------------------------- #
# The wiring, end to end through the engine
# --------------------------------------------------------------------------- #
class RecordingUnitOfWork:
    """Captures what the engine wrote, in place of a database."""

    def __init__(self, store: dict[str, list[Any]]) -> None:
        self._store = store
        self.orders = _Recorder(store, "orders")
        self.positions = _Recorder(store, "positions")
        self.trades = _Recorder(store, "trades")
        self.equity = _Recorder(store, "equity")
        self.sessions = _Recorder(store, "sessions")

    async def __aenter__(self) -> RecordingUnitOfWork:
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False


class _Recorder:
    def __init__(self, store: dict[str, list[Any]], bucket: str) -> None:
        self._store = store
        self._bucket = bucket

    async def save(self, item: Any, **_: Any) -> None:
        self._store.setdefault(self._bucket, []).append(item)

    async def add(self, _session_id: str, item: Any, **__: Any) -> None:
        self._store.setdefault(self._bucket, []).append(item)

    async def add_many(self, items: Any, **_: Any) -> int:
        rows = list(items)
        self._store.setdefault(self._bucket, []).extend(rows)
        return len(rows)

    async def create(self, **kwargs: Any) -> None:
        self._store.setdefault(self._bucket, []).append(kwargs)

    async def finish(self, *_: Any, **__: Any) -> None:
        return None


class RecordingDatabase:
    """A database that records writes and has nothing to restore."""

    def __init__(self) -> None:
        self.written: dict[str, list[Any]] = {}

    def unit_of_work(self) -> RecordingUnitOfWork:
        return RecordingUnitOfWork(self.written)

    def read_session(self) -> Any:
        raise RuntimeError("no prior state")


class LiveFakeGateway(FakeGateway):
    """The venue plus the market-data reads ``PaperTradingEngine.prepare`` makes."""

    def __init__(self, candles: list[Any]) -> None:
        super().__init__()
        self._candles = candles
        self.submitted: list[OrderRequest] = []

    async def fetch_candles(self, symbol: Symbol, timeframe: Any, *, limit: int = 500) -> list[Any]:
        return list(self._candles)

    async def submit_order(self, request: OrderRequest) -> Order:
        self.submitted.append(request)
        return local_order(request.symbol, order_id="submitted", quantity=str(request.quantity))


class HoldStrategy:
    """Stands aside on every bar, so the test measures reconciliation and nothing else."""

    strategy_id = "hold_only"

    def __init__(self) -> None:
        self.params = _NoParams()
        self.warmup_bars = 1

    def on_start(self, _symbols: Any) -> None:
        return None

    def on_restore(self, _positions: Any) -> None:
        return None

    def on_trade_closed(self, _trade: Any) -> None:
        return None

    def on_finish(self) -> None:
        return None

    def evaluate(self, context: Any) -> Any:
        return context.hold("test strategy holds", self.strategy_id)


class _NoParams:
    def to_dict(self) -> dict[str, Any]:
        return {}


class TestTheOrderReadBackReachesTheVenue:
    """``fetch_order`` was being called with the wrong arity and never left the process.

    CCXT's bybit signature is ``fetch_order(id, symbol, params)``. The gateway passed a
    fourth positional — the ``since`` slot several *other* endpoints have — so every call
    raised ``TypeError`` before any network IO. Two consequences, and the second is the one
    that matters: post-submit enrichment silently fell back to the acknowledgement, and a
    ``REJECTED`` or ``CANCELLED`` order was undetectable, because neither produces an
    execution for the fill stream to report.
    """

    @pytest.mark.asyncio
    async def test_the_call_matches_ccxts_signature(self) -> None:
        import inspect

        import ccxt.async_support as ccxt_async

        from quantflow.core.config import ExchangeSettings
        from quantflow.exchange.bybit.rest import BybitGateway

        signature = inspect.signature(ccxt_async.bybit.fetch_order)
        captured: dict[str, Any] = {}

        async def stub_fetch_order(
            order_id: str, symbol: str | None = None, params: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            # Bound against the real signature, so a shape CCXT would reject fails here.
            signature.bind(object(), order_id, symbol, params)
            captured["args"] = (order_id, symbol, params)
            return {
                "id": "venue-1",
                "clientOrderId": "qf-1",
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "type": "market",
                "amount": 0.5,
                "filled": 0.5,
                "average": 60000,
                "status": "closed",
            }

        gateway = BybitGateway(
            ExchangeSettings(
                api_key=SecretStr("k"), api_secret=SecretStr("s"), market_type=MarketType.FUTURE
            )
        )
        gateway._client.fetch_order = stub_fetch_order
        gateway.register_venue_id("local-1", "venue-1")

        order = await gateway.fetch_order("local-1", BTC)

        assert captured["args"][0] == "venue-1"
        assert captured["args"][2] == {"acknowledged": True}
        assert order.order_id == "local-1"
        assert order.status is OrderStatus.FILLED
        assert order.filled_quantity == Decimal("0.5")
        await gateway.aclose()


class TestTheEngineWiring:
    """The defect itself: a LIVE session's router yields no fills on a bar.

    ``SimulatedOrderRouter.process_candle`` hands back ``(order, fill)`` pairs and the
    engine applies them. ``LiveOrderRouter.process_candle`` returns ``()`` — correctly, a
    bar does not fill anything on a real venue — and that was the whole of the fill path.
    These prove the engine now has a second one.
    """

    @pytest.mark.asyncio
    async def test_a_live_session_adopts_persists_and_closes_venue_positions(self) -> None:
        from quantflow.core.clock import FrozenClock
        from quantflow.core.config import TradingMode
        from quantflow.domain.enums import Timeframe
        from quantflow.execution.router import LiveOrderRouter
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine
        from tests.conftest import make_candle

        bars = [
            make_candle(
                BTC,
                open_time=NOW - timedelta(minutes=15 * (10 - index)),
                close=60000,
                timeframe=Timeframe.parse("15m"),
            )
            for index in range(10)
        ]
        gateway = LiveFakeGateway(bars)
        gateway.positions = [venue_position(BTC, quantity="0.5", entry="60000", stop="59000")]
        gateway.wallet = Decimal("49915")
        database = RecordingDatabase()

        engine = PaperTradingEngine(
            HoldStrategy(),  # type: ignore[arg-type]
            PaperConfig(
                symbols=(BTC,),
                timeframe=Timeframe.parse("15m"),
                starting_equity=Decimal("49915"),
                market_type=MarketType.FUTURE,
                mode=TradingMode.LIVE,
                session_id="test-live",
            ),
            instruments={BTC: instrument()},
            database=database,  # type: ignore[arg-type]
            clock=FrozenClock(NOW),
            router=LiveOrderRouter(gateway),
        )

        await engine.prepare(gateway)  # type: ignore[arg-type]

        # Startup: the venue's position is in the local book, on disk, and in the risk view.
        position = engine.portfolio.position_for(BTC)
        assert position is not None
        assert position.quantity == Decimal("0.5")
        assert position.stop_loss_price == Decimal("59000")
        assert database.written["positions"]
        assert engine.portfolio.snapshot(NOW).position_count == 1
        # Cash is the venue's wallet, not a figure reconstructed from a bounded replay.
        assert engine.portfolio.cash == Decimal("49915")

        # A real close, reported the way the venue reports one.
        gateway.executions[BTC] = [
            execution(
                BTC,
                fill_id="exit-1",
                order_id="venue-stop-1",
                side=OrderSide.SELL,
                quantity="0.5",
                price="61000",
                fee="18.3",
                at=NOW + timedelta(minutes=5),
                realized_pnl="500",
            )
        ]
        gateway.positions = []

        await engine.on_candle(
            make_candle(
                BTC,
                open_time=NOW,
                close=61000,
                timeframe=Timeframe.parse("15m"),
            )
        )

        assert engine.portfolio.position_for(BTC) is None
        trades = database.written["trades"]
        assert len(trades) == 1
        assert trades[0].exit_price == Decimal("61000")
        assert trades[0].fees == Decimal("18.3")
        assert engine.portfolio.realized_pnl == Decimal("500")
        # The venue-side stop had no local order; one was created so its fill has a home.
        persisted = {order.order_id: order for order in database.written["orders"]}
        assert "venue-stop-1" in persisted
        assert persisted["venue-stop-1"].fills[0].fill_id == "exit-1"
        assert engine.state.fills == 1


class TestSessionStartFloorsTheExecutionLookback:
    """A session cannot have made a fill before it existed.

    The execution query reaches back 24 hours so a fill missed during a disconnect is still
    recovered. On a *new* session that same reach adopts a whole day of the venue's history
    — every fill from whatever ran before — and books it as this session's trades.

    Observed live on 2026-08-14: a session three minutes old opened with 39 closed trades
    and −49.42 realised PnL, all of it belonging to the run it replaced, all of it with no
    strategy attribution. The equity curve then measures a 10,000 allocation against
    another session's losses.
    """

    async def test_fills_from_before_the_session_started_are_not_adopted(self) -> None:
        book = portfolio()
        gateway = FakeGateway()
        # One fill from an hour before this session began, one from after.
        gateway.executions[BTC] = [
            execution(BTC, fill_id="old-fill", order_id="o1", at=NOW - timedelta(hours=1)),
            execution(BTC, fill_id="new-fill", order_id="o2", at=NOW + timedelta(seconds=30)),
        ]
        subject = LiveReconciler(
            gateway,
            book,
            symbols=(BTC,),
            clock=lambda: NOW + timedelta(minutes=1),
            repair_confirmations=1,
            not_before=NOW,
        )

        await subject.reconcile([])

        adopted = book.applied_fill_ids
        assert "old-fill" not in adopted
        assert "new-fill" in adopted

    async def test_without_a_floor_the_lookback_is_unchanged(self) -> None:
        # The recovery behaviour a long-running session depends on must not regress.
        book = portfolio()
        gateway = FakeGateway()
        gateway.executions[BTC] = [
            execution(BTC, fill_id="old-fill", order_id="o1", at=NOW - timedelta(hours=1)),
        ]
        subject = reconciler(gateway, book)

        await subject.reconcile([])

        assert "old-fill" in book.applied_fill_ids


class TestVenueHeldSymbolsAreReported:
    """The pass must say which symbols the venue holds, not only what changed.

    Ownership synchronisation needs the venue's answer, and the local book is not a
    substitute for it: between an order filling and the fill being reconciled there is a
    window in which the venue holds a position the portfolio has not seen. Observed live on
    2026-08-14 — an ETH short opened at 18:30:13 and had its ownership released at 18:30:27
    because the local book was still empty, leaving a real position with no owning strategy.

    ``None`` rather than an empty set when the position read failed: "the venue holds
    nothing" and "the venue could not be asked" must never be the same value, because the
    first is a reason to release ownership and the second is a reason to leave it alone.
    """

    async def test_the_pass_reports_every_symbol_the_venue_holds(self) -> None:
        book = portfolio()
        gateway = FakeGateway()
        gateway.positions = [venue_position(BTC, quantity="1", entry="100")]

        outcome = await reconciler(gateway, book).reconcile([])

        assert outcome.venue_symbols == {BTC}

    async def test_an_empty_venue_reports_an_empty_set_not_none(self) -> None:
        outcome = await reconciler(FakeGateway(), portfolio()).reconcile([])

        assert outcome.venue_symbols == set()

    async def test_a_failed_position_read_reports_none(self) -> None:
        gateway = FakeGateway()
        gateway.positions_error = RuntimeError("venue timed out")

        outcome = await reconciler(gateway, portfolio()).reconcile([])

        assert outcome.venue_symbols is None


class TestStrategyAttributionSurvivesReconciliation:
    """A reconciled fill must still name the strategy that opened the trade.

    Orders carry two identities: the local ``order_id`` this process generates, and the
    ``venue_order_id`` Bybit assigns. Executions come back from ``fetch_my_trades``
    referencing the *venue's* id, and the strategy lookup was keyed only by the local one.
    The two never match, so every fill applied through reconciliation was attributed to
    nobody.

    On a live venue this is not an edge case — reconciliation is the *only* path by which
    fills reach the portfolio, so it means no trade is ever attributed. Session
    demo-10k-fresh: 13 closed trades, 13 with ``strategy_id`` NULL, making every
    per-strategy question unanswerable.
    """

    async def test_a_fill_returned_under_the_venue_id_is_still_attributed(self) -> None:
        book = portfolio()
        gateway = FakeGateway()
        order = local_order(BTC, order_id="1")  # venue id becomes "venue-1"
        gateway.orders["venue-1"] = order
        # The venue reports the execution against its own order id, not ours.
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e1", order_id="venue-1", at=NOW + timedelta(seconds=5))
        ]
        # The venue agrees the position exists, so the repair pass leaves it alone.
        gateway.positions = [venue_position(BTC, quantity="1", entry="100")]

        await reconciler(gateway, book).reconcile([order])

        position = book.position_for(BTC)
        assert position is not None
        assert position.strategy_id == "ema_cross", "the venue id must resolve to the strategy"

    async def test_the_local_id_still_resolves(self) -> None:
        # Some paths report the local id; both must work.
        book = portfolio()
        gateway = FakeGateway()
        order = local_order(BTC, order_id="2")
        gateway.orders["venue-2"] = order
        gateway.executions[BTC] = [
            execution(BTC, fill_id="e2", order_id="2", at=NOW + timedelta(seconds=5))
        ]
        gateway.positions = [venue_position(BTC, quantity="1", entry="100")]

        await reconciler(gateway, book).reconcile([order])

        position = book.position_for(BTC)
        assert position is not None
        assert position.strategy_id == "ema_cross"


class TestReconciliationPreservesLocalAttribution:
    """Re-reading an order from the venue must not erase what only we know.

    The venue knows an order's price, size and status. It has never heard of the strategy
    that produced it, or the sizing method behind it. When reconciliation rebuilds an
    ``Order`` from a venue payload and persists it under the same id, every locally-known
    field is overwritten with nothing.

    Measured on the live session: 81 orders, **zero** with a strategy, and ``meta`` empty on
    every row — even though the risk engine always sets both. The engine wrote them
    correctly and reconciliation blanked them moments later, which is why every closed trade
    reports NOT RECORDED and no per-strategy question can be answered.
    """

    async def test_the_strategy_survives_a_venue_read(self) -> None:
        book = portfolio()
        gateway = FakeGateway()
        local = local_order(BTC, order_id="1")  # strategy_id="ema_cross"
        # The venue returns the same order, filled, knowing nothing about strategies.
        gateway.orders["1"] = Order(
            order_id="1",
            client_order_id="qf-1",
            symbol=BTC,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            filled_quantity=Decimal("1"),
            status=OrderStatus.FILLED,
            created_at=NOW,
            updated_at=NOW,
            venue_order_id="venue-1",
        )

        outcome = await reconciler(gateway, book).reconcile([local])

        rebuilt = next((o for o in outcome.orders if o.order_id == "1"), None)
        assert rebuilt is not None
        assert (
            rebuilt.strategy_id == "ema_cross"
        ), "a venue read must not blank the strategy the engine recorded"
