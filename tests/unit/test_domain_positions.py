"""FIFO position accounting, including property-based invariants.

The properties here are the ones that must hold for *any* fill sequence — they are the
reason a subtle accounting regression cannot ship unnoticed.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import LiquidityRole, OrderSide, PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill
from quantflow.domain.portfolio import Balance, EquityPoint, PortfolioSnapshot, build_equity_curve
from quantflow.domain.positions import Lot, Position, gross_exposure, net_exposure
from tests.conftest import REFERENCE_TIME


def fill(
    symbol: Symbol,
    side: OrderSide,
    quantity: str,
    price: str,
    *,
    fill_id: str = "f",
    fee: str = "0",
    offset_seconds: int = 0,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id="o",
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency=symbol.quote,
        timestamp=REFERENCE_TIME + timedelta(seconds=offset_seconds),
        role=LiquidityRole.TAKER,
    )


class TestLot:
    def test_cost(self) -> None:
        lot = Lot(quantity=Decimal("2"), price=Decimal("50"), opened_at=REFERENCE_TIME)
        assert lot.cost == Decimal("100")

    def test_take_partial_splits_fee_pro_rata(self) -> None:
        lot = Lot(
            quantity=Decimal("4"), price=Decimal("10"), opened_at=REFERENCE_TIME, fee=Decimal("0.4")
        )
        taken, remainder = lot.take(Decimal("1"))
        assert taken.quantity == Decimal("1")
        assert taken.fee == Decimal("0.1")
        assert remainder is not None
        assert remainder.quantity == Decimal("3")
        assert remainder.fee == Decimal("0.3")

    def test_take_all_leaves_no_remainder(self) -> None:
        lot = Lot(quantity=Decimal("2"), price=Decimal("10"), opened_at=REFERENCE_TIME)
        taken, remainder = lot.take(Decimal("2"))
        assert taken.quantity == Decimal("2")
        assert remainder is None

    @pytest.mark.parametrize("amount", ["0", "-1", "3"])
    def test_take_rejects_invalid_amount(self, amount: str) -> None:
        lot = Lot(quantity=Decimal("2"), price=Decimal("10"), opened_at=REFERENCE_TIME)
        with pytest.raises(ValidationError, match="cannot take"):
            lot.take(Decimal(amount))

    def test_rejects_invalid_construction(self) -> None:
        with pytest.raises(ValidationError, match="quantity must be positive"):
            Lot(quantity=ZERO, price=Decimal("1"), opened_at=REFERENCE_TIME)
        with pytest.raises(ValidationError, match="price must be positive"):
            Lot(quantity=Decimal("1"), price=ZERO, opened_at=REFERENCE_TIME)


class TestOpeningAndIncreasing:
    def test_open_long(self, btc: Symbol) -> None:
        position, closed = Position(symbol=btc).apply_fill(
            fill(btc, OrderSide.BUY, "1", "100", fee="0.1")
        )
        assert position.side is PositionSide.LONG
        assert position.quantity == Decimal("1")
        assert position.average_entry_price == Decimal("100")
        assert position.fees_paid == Decimal("0.1")
        assert position.opened_at == REFERENCE_TIME
        assert closed == ()

    def test_open_short(self, btc: Symbol) -> None:
        position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.SELL, "2", "100"))
        assert position.side is PositionSide.SHORT
        assert position.quantity == Decimal("-2")
        assert position.absolute_quantity == Decimal("2")

    def test_increase_averages_entry(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.BUY, "1", "100", fill_id="a"))
        position, _ = position.apply_fill(fill(btc, OrderSide.BUY, "3", "200", fill_id="b"))
        assert position.quantity == Decimal("4")
        assert position.average_entry_price == Decimal("175")
        assert len(position.lots) == 2

    def test_rejects_wrong_symbol(self, btc: Symbol, eth: Symbol) -> None:
        with pytest.raises(ValidationError, match="cannot be applied"):
            Position(symbol=btc).apply_fill(fill(eth, OrderSide.BUY, "1", "100"))

    def test_rejects_incoherent_construction(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="does not match position size"):
            Position(symbol=btc, quantity=Decimal("5"), lots=())


class TestUnrealisedPnl:
    def test_long_pnl(self, btc: Symbol) -> None:
        position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.BUY, "2", "100"))
        assert position.unrealized_pnl(Decimal("110")) == Decimal("20")
        assert position.unrealized_pnl(Decimal("90")) == Decimal("-20")
        assert position.unrealized_pnl_pct(Decimal("110")) == Decimal("0.1")

    def test_short_pnl(self, btc: Symbol) -> None:
        position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.SELL, "2", "100"))
        assert position.unrealized_pnl(Decimal("90")) == Decimal("20")
        assert position.unrealized_pnl(Decimal("110")) == Decimal("-20")

    def test_flat_position_has_no_pnl(self, btc: Symbol) -> None:
        assert Position(symbol=btc).unrealized_pnl(Decimal("100")) == ZERO

    def test_market_value_is_signed_notional_unsigned(self, btc: Symbol) -> None:
        position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.SELL, "2", "100"))
        assert position.market_value(Decimal("100")) == Decimal("-200")
        assert position.notional(Decimal("100")) == Decimal("200")


class TestFifoRealisation:
    def test_fifo_matches_oldest_lot_first(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.BUY, "1", "100", fill_id="a"))
        position, _ = position.apply_fill(
            fill(btc, OrderSide.BUY, "1", "200", fill_id="b", offset_seconds=60)
        )
        position, closed = position.apply_fill(
            fill(btc, OrderSide.SELL, "1", "300", fill_id="c", offset_seconds=120)
        )
        assert len(closed) == 1
        assert closed[0].entry_price == Decimal("100")  # oldest lot, not the average
        assert closed[0].gross_pnl == Decimal("200")
        assert position.quantity == Decimal("1")
        assert position.average_entry_price == Decimal("200")
        assert position.realized_pnl == Decimal("200")

    def test_close_spanning_multiple_lots_produces_multiple_trades(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.BUY, "1", "100", fill_id="a"))
        position, _ = position.apply_fill(
            fill(btc, OrderSide.BUY, "1", "120", fill_id="b", offset_seconds=60)
        )
        position, closed = position.apply_fill(
            fill(btc, OrderSide.SELL, "2", "150", fill_id="c", offset_seconds=120)
        )
        assert len(closed) == 2
        assert [trade.entry_price for trade in closed] == [Decimal("100"), Decimal("120")]
        assert sum(trade.gross_pnl for trade in closed) == Decimal("80")
        assert position.is_flat
        assert position.opened_at is None

    def test_partial_close_leaves_partial_lot(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.BUY, "4", "100", fill_id="a"))
        position, closed = position.apply_fill(
            fill(btc, OrderSide.SELL, "1", "110", fill_id="b", offset_seconds=60)
        )
        assert closed[0].quantity == Decimal("1")
        assert position.quantity == Decimal("3")
        assert len(position.lots) == 1
        assert position.lots[0].quantity == Decimal("3")

    def test_short_close_realises_correct_sign(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.SELL, "1", "100", fill_id="a"))
        position, closed = position.apply_fill(
            fill(btc, OrderSide.BUY, "1", "90", fill_id="b", offset_seconds=60)
        )
        assert closed[0].side is PositionSide.SHORT
        assert closed[0].gross_pnl == Decimal("10")
        assert position.is_flat

    def test_exit_fees_are_apportioned(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(
            fill(btc, OrderSide.BUY, "2", "100", fill_id="a", fee="0.2")
        )
        position, closed = position.apply_fill(
            fill(btc, OrderSide.SELL, "1", "110", fill_id="b", fee="0.11", offset_seconds=60)
        )
        # Half the entry fee (0.1) plus the whole exit fee on that quantity (0.11).
        assert closed[0].fees == Decimal("0.21")
        assert closed[0].net_pnl == Decimal("10") - Decimal("0.21")

    def test_closed_trade_metrics(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.BUY, "1", "100", fill_id="a"))
        _, closed = position.apply_fill(
            fill(btc, OrderSide.SELL, "1", "110", fill_id="b", offset_seconds=3600)
        )
        trade = closed[0]
        assert trade.is_win
        assert trade.return_pct == Decimal("0.1")
        assert trade.holding_period == Decimal("3600")


class TestPositionFlip:
    def test_flip_long_to_short(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.BUY, "1", "100", fill_id="a"))
        position, closed = position.apply_fill(
            fill(btc, OrderSide.SELL, "3", "110", fill_id="b", offset_seconds=60)
        )
        assert len(closed) == 1
        assert closed[0].gross_pnl == Decimal("10")
        assert position.side is PositionSide.SHORT
        assert position.quantity == Decimal("-2")
        assert position.average_entry_price == Decimal("110")
        assert position.opened_at == REFERENCE_TIME + timedelta(seconds=60)

    def test_flip_short_to_long(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.SELL, "2", "100", fill_id="a"))
        position, closed = position.apply_fill(
            fill(btc, OrderSide.BUY, "5", "90", fill_id="b", offset_seconds=60)
        )
        assert closed[0].gross_pnl == Decimal("20")
        assert position.quantity == Decimal("3")
        assert position.side is PositionSide.LONG

    def test_flip_clears_protective_levels(self, btc: Symbol) -> None:
        position = Position(symbol=btc)
        position, _ = position.apply_fill(fill(btc, OrderSide.BUY, "1", "100", fill_id="a"))
        position = position.with_protection(stop_loss_price=Decimal("95"))
        position, _ = position.apply_fill(
            fill(btc, OrderSide.SELL, "2", "110", fill_id="b", offset_seconds=60)
        )
        assert position.stop_loss_price is None


class TestProtectiveLevels:
    def test_long_stop_and_target(self, btc: Symbol) -> None:
        position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.BUY, "1", "100"))
        position = position.with_protection(
            stop_loss_price=Decimal("95"), take_profit_price=Decimal("110")
        )
        assert position.is_stop_breached(Decimal("95"))
        assert position.is_stop_breached(Decimal("94"))
        assert not position.is_stop_breached(Decimal("96"))
        assert position.is_target_reached(Decimal("110"))
        assert not position.is_target_reached(Decimal("109"))

    def test_short_stop_and_target(self, btc: Symbol) -> None:
        position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.SELL, "1", "100"))
        position = position.with_protection(
            stop_loss_price=Decimal("105"), take_profit_price=Decimal("90")
        )
        assert position.is_stop_breached(Decimal("105"))
        assert not position.is_stop_breached(Decimal("104"))
        assert position.is_target_reached(Decimal("90"))

    def test_flat_position_never_breaches(self, btc: Symbol) -> None:
        position = Position(symbol=btc).with_protection(stop_loss_price=Decimal("1"))
        assert not position.is_stop_breached(Decimal("0.5"))

    def test_closing_side(self, btc: Symbol) -> None:
        long_position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.BUY, "1", "100"))
        short_position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.SELL, "1", "100"))
        assert long_position.closing_side() is OrderSide.SELL
        assert short_position.closing_side() is OrderSide.BUY
        assert Position(symbol=btc).closing_side() is None


class TestReplay:
    def test_from_fills_reconstructs_state(self, btc: Symbol) -> None:
        fills = [
            fill(btc, OrderSide.BUY, "1", "100", fill_id="a", offset_seconds=0),
            fill(btc, OrderSide.BUY, "1", "120", fill_id="b", offset_seconds=60),
            fill(btc, OrderSide.SELL, "1", "150", fill_id="c", offset_seconds=120),
        ]
        position, closed = Position.from_fills(btc, fills)
        assert position.quantity == Decimal("1")
        assert position.realized_pnl == Decimal("50")
        assert len(closed) == 1

    def test_from_fills_sorts_by_timestamp(self, btc: Symbol) -> None:
        fills = [
            fill(btc, OrderSide.SELL, "1", "150", fill_id="c", offset_seconds=120),
            fill(btc, OrderSide.BUY, "1", "100", fill_id="a", offset_seconds=0),
        ]
        position, closed = Position.from_fills(btc, fills)
        assert position.is_flat
        assert closed[0].gross_pnl == Decimal("50")


class TestExposureAggregation:
    def test_net_and_gross_exposure(self, btc: Symbol, eth: Symbol) -> None:
        long_btc, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.BUY, "1", "100"))
        short_eth, _ = Position(symbol=eth).apply_fill(fill(eth, OrderSide.SELL, "2", "50"))
        prices = {btc: Decimal("100"), eth: Decimal("50")}
        assert net_exposure([long_btc, short_eth], prices) == ZERO
        assert gross_exposure([long_btc, short_eth], prices) == Decimal("200")

    def test_missing_price_raises_rather_than_understating(self, btc: Symbol) -> None:
        position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.BUY, "1", "100"))
        with pytest.raises(ValidationError, match="missing mark price"):
            gross_exposure([position], {})

    def test_flat_positions_are_skipped(self, btc: Symbol) -> None:
        assert gross_exposure([Position(symbol=btc)], {}) == ZERO


class TestPortfolioSnapshot:
    def _snapshot(self, btc: Symbol, **overrides: object) -> PortfolioSnapshot:
        position, _ = Position(symbol=btc).apply_fill(fill(btc, OrderSide.BUY, "1", "100"))
        kwargs: dict[str, object] = {
            "timestamp": REFERENCE_TIME,
            "base_currency": "USDT",
            "cash": Decimal("900"),
            "positions": (position,),
            "mark_prices": {btc: Decimal("110")},
            "peak_equity": Decimal("1000"),
            "day_start_equity": Decimal("1000"),
        }
        kwargs.update(overrides)
        return PortfolioSnapshot(**kwargs)  # type: ignore[arg-type]

    def test_equity_and_exposure(self, btc: Symbol) -> None:
        snapshot = self._snapshot(btc)
        assert snapshot.equity == Decimal("1010")
        assert snapshot.gross_exposure == Decimal("110")
        assert snapshot.unrealized_pnl == Decimal("10")
        assert snapshot.position_count == 1
        assert snapshot.has_position(btc)

    def test_leverage_and_position_pct(self, btc: Symbol) -> None:
        snapshot = self._snapshot(btc)
        assert snapshot.leverage == Decimal("110") / Decimal("1010")
        assert snapshot.position_pct(btc) == Decimal("110") / Decimal("1010")

    def test_drawdown(self, btc: Symbol) -> None:
        snapshot = self._snapshot(btc, cash=Decimal("790"), peak_equity=Decimal("1000"))
        # equity = 790 + 110 = 900 → 10% below the 1000 peak
        assert snapshot.equity == Decimal("900")
        assert snapshot.drawdown_pct == Decimal("0.1")

    def test_drawdown_is_zero_at_a_new_high(self, btc: Symbol) -> None:
        assert self._snapshot(btc, peak_equity=Decimal("500")).drawdown_pct == ZERO

    def test_daily_pnl(self, btc: Symbol) -> None:
        snapshot = self._snapshot(btc)
        assert snapshot.daily_pnl == Decimal("10")
        assert snapshot.daily_pnl_pct == Decimal("0.01")

    def test_daily_pnl_without_baseline_is_zero(self, btc: Symbol) -> None:
        assert self._snapshot(btc, day_start_equity=ZERO).daily_pnl == ZERO

    def test_missing_mark_price_raises(self, btc: Symbol) -> None:
        snapshot = self._snapshot(btc, mark_prices={})
        with pytest.raises(ValidationError, match="missing mark price"):
            _ = snapshot.equity

    def test_rejects_negative_cash(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="cash cannot be negative"):
            self._snapshot(btc, cash=Decimal("-1"))

    def test_position_for_ignores_flat(self, btc: Symbol) -> None:
        snapshot = self._snapshot(btc, positions=(Position(symbol=btc),))
        assert snapshot.position_for(btc) is None
        assert snapshot.position_pct(btc) == ZERO


class TestBalance:
    def test_total(self) -> None:
        assert Balance(asset="USDT", free=Decimal("10"), locked=Decimal("5")).total == Decimal("15")

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="negative balance"):
            Balance(asset="USDT", free=Decimal("-1"))


class TestEquityCurve:
    def test_sorts_and_recomputes_drawdown(self) -> None:
        points = [
            EquityPoint(
                timestamp=REFERENCE_TIME + timedelta(hours=index),
                equity=equity,
                cash=equity,
                position_count=0,
            )
            for index, equity in enumerate(
                [Decimal("1000"), Decimal("1200"), Decimal("900"), Decimal("1300")]
            )
        ]
        curve = build_equity_curve(list(reversed(points)))
        assert [point.equity for point in curve] == [
            Decimal("1000"),
            Decimal("1200"),
            Decimal("900"),
            Decimal("1300"),
        ]
        assert curve[0].drawdown_pct == ZERO
        assert curve[1].drawdown_pct == ZERO
        assert curve[2].drawdown_pct == Decimal("300") / Decimal("1200")
        assert curve[3].drawdown_pct == ZERO

    def test_empty_curve(self) -> None:
        assert build_equity_curve([]) == ()


# --------------------------------------------------------------------------- #
# Property-based invariants
# --------------------------------------------------------------------------- #
quantities = st.decimals(
    min_value=Decimal("0.001"),
    max_value=Decimal("100"),
    places=3,
    allow_nan=False,
    allow_infinity=False,
)
prices = st.decimals(
    min_value=Decimal("1"),
    max_value=Decimal("100000"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)
sides = st.sampled_from([OrderSide.BUY, OrderSide.SELL])


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.tuples(sides, quantities, prices), min_size=1, max_size=12))
def test_lots_always_reconcile_with_quantity(
    entries: list[tuple[OrderSide, Decimal, Decimal]],
) -> None:
    """The sum of lot quantities always equals the absolute position size."""
    symbol = Symbol(base="BTC", quote="USDT")
    position = Position(symbol=symbol)
    for index, (side, quantity, price) in enumerate(entries):
        position, _ = position.apply_fill(
            fill(symbol, side, str(quantity), str(price), fill_id=f"f{index}", offset_seconds=index)
        )
        lot_total = sum((lot.quantity for lot in position.lots), ZERO)
        assert lot_total == position.absolute_quantity


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.tuples(sides, quantities, prices), min_size=1, max_size=12))
def test_signed_quantity_equals_sum_of_signed_fills(
    entries: list[tuple[OrderSide, Decimal, Decimal]],
) -> None:
    """Position quantity is exactly the running sum of signed fill quantities."""
    symbol = Symbol(base="BTC", quote="USDT")
    position = Position(symbol=symbol)
    expected = ZERO
    for index, (side, quantity, price) in enumerate(entries):
        expected += quantity * side.sign
        position, _ = position.apply_fill(
            fill(symbol, side, str(quantity), str(price), fill_id=f"f{index}", offset_seconds=index)
        )
        assert position.quantity == expected


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(quantity=quantities, entry=prices, exit_price=prices)
def test_round_trip_pnl_matches_price_difference(
    quantity: Decimal, entry: Decimal, exit_price: Decimal
) -> None:
    """A full round trip realises exactly (exit - entry) * quantity, direction-adjusted."""
    symbol = Symbol(base="BTC", quote="USDT")
    for side in (OrderSide.BUY, OrderSide.SELL):
        position = Position(symbol=symbol)
        position, _ = position.apply_fill(
            fill(symbol, side, str(quantity), str(entry), fill_id="in")
        )
        position, closed = position.apply_fill(
            fill(
                symbol,
                side.opposite,
                str(quantity),
                str(exit_price),
                fill_id="out",
                offset_seconds=1,
            )
        )
        direction = Decimal(1) if side is OrderSide.BUY else Decimal(-1)
        assert position.is_flat
        assert position.realized_pnl == (exit_price - entry) * quantity * direction
        assert sum(trade.gross_pnl for trade in closed) == position.realized_pnl


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(quantity=quantities, entry=prices, mark=prices)
def test_closing_realises_exactly_the_prior_unrealised(
    quantity: Decimal, entry: Decimal, mark: Decimal
) -> None:
    """Unrealised PnL at a mark price becomes realised PnL when closed at that price."""
    symbol = Symbol(base="BTC", quote="USDT")
    position = Position(symbol=symbol)
    position, _ = position.apply_fill(fill(symbol, OrderSide.BUY, str(quantity), str(entry)))
    unrealised = position.unrealized_pnl(mark)
    closed_position, _ = position.apply_fill(
        fill(symbol, OrderSide.SELL, str(quantity), str(mark), fill_id="out", offset_seconds=1)
    )
    assert closed_position.realized_pnl == unrealised


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    first=st.tuples(quantities, prices),
    second=st.tuples(quantities, prices),
    exit_price=prices,
)
def test_fifo_never_uses_average_cost(
    first: tuple[Decimal, Decimal], second: tuple[Decimal, Decimal], exit_price: Decimal
) -> None:
    """Closing the first lot realises against the *first* entry price, not the average."""
    first_quantity, first_price = first
    second_quantity, second_price = second
    assume(first_price != second_price)
    symbol = Symbol(base="BTC", quote="USDT")
    position = Position(symbol=symbol)
    position, _ = position.apply_fill(
        fill(symbol, OrderSide.BUY, str(first_quantity), str(first_price), fill_id="a")
    )
    position, _ = position.apply_fill(
        fill(
            symbol,
            OrderSide.BUY,
            str(second_quantity),
            str(second_price),
            fill_id="b",
            offset_seconds=1,
        )
    )
    _, closed = position.apply_fill(
        fill(
            symbol,
            OrderSide.SELL,
            str(first_quantity),
            str(exit_price),
            fill_id="c",
            offset_seconds=2,
        )
    )
    assert closed[0].entry_price == first_price
