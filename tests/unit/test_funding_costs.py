"""Phase 7: perpetual funding is charged on the 8h schedule, with the correct sign.

Nothing charged funding before. Backtest and paper both reported PnL as though holding a
perp were free, flattering every result in proportion to holding time.

The sign tests are the point of this file. A sign error does not merely mis-state a cost —
it turns a systematic drain into a systematic credit, and makes an unprofitable strategy
look profitable for as long as funding stays one-sided.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from quantflow.core.config import MarketType
from quantflow.domain.enums import OrderSide, PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.orders import Fill
from quantflow.portfolio.funding import (
    FundingSchedule,
    borrow_cost,
    funding_amount,
    funding_stamps,
    total_funding,
)
from quantflow.portfolio.manager import PortfolioManager

BTC = Symbol.parse("BTC/USDT")
START = Decimal("100000")


def at(hour: int, day: int = 1) -> datetime:
    return datetime(2026, 1, day, hour, 0, tzinfo=UTC)


class TestFundingSign:
    """Positive rate: longs PAY, shorts RECEIVE. Negative: reversed."""

    def test_a_long_pays_when_the_rate_is_positive(self) -> None:
        amount = funding_amount(
            side=PositionSide.LONG,
            quantity=Decimal("1"),
            price=Decimal("50000"),
            rate=Decimal("0.0001"),
        )
        assert amount == Decimal("-5"), "a long must PAY on a positive rate"
        assert amount < 0

    def test_a_short_receives_when_the_rate_is_positive(self) -> None:
        amount = funding_amount(
            side=PositionSide.SHORT,
            quantity=Decimal("1"),
            price=Decimal("50000"),
            rate=Decimal("0.0001"),
        )
        assert amount == Decimal("5"), "a short must RECEIVE on a positive rate"
        assert amount > 0

    def test_a_long_receives_when_the_rate_is_negative(self) -> None:
        amount = funding_amount(
            side=PositionSide.LONG,
            quantity=Decimal("1"),
            price=Decimal("50000"),
            rate=Decimal("-0.0001"),
        )
        assert amount == Decimal("5")
        assert amount > 0

    def test_a_short_pays_when_the_rate_is_negative(self) -> None:
        amount = funding_amount(
            side=PositionSide.SHORT,
            quantity=Decimal("1"),
            price=Decimal("50000"),
            rate=Decimal("-0.0001"),
        )
        assert amount == Decimal("-5")
        assert amount < 0

    def test_long_and_short_are_exact_mirrors(self) -> None:
        """The two sides settle against each other; they cannot both pay."""
        kwargs = {"quantity": Decimal("2"), "price": Decimal("30000"), "rate": Decimal("0.0002")}
        long = funding_amount(side=PositionSide.LONG, **kwargs)
        short = funding_amount(side=PositionSide.SHORT, **kwargs)
        assert long == -short

    def test_a_zero_rate_costs_nothing(self) -> None:
        assert funding_amount(
            side=PositionSide.LONG,
            quantity=Decimal("1"),
            price=Decimal("50000"),
            rate=Decimal("0"),
        ) == Decimal("0")

    def test_the_amount_stays_decimal(self) -> None:
        amount = funding_amount(
            side=PositionSide.LONG,
            quantity=Decimal("1"),
            price=Decimal("50000"),
            rate=Decimal("0.0001"),
        )
        assert isinstance(amount, Decimal)


class TestFundingSchedule8h:
    def test_stamps_land_on_the_eight_hour_boundaries(self) -> None:
        stamps = funding_stamps(at(1), at(20))
        assert stamps == [at(8), at(16)]

    def test_a_position_opened_exactly_on_a_stamp_is_not_charged_for_it(self) -> None:
        """Exclusive of start: it did not hold through the period that just settled."""
        assert at(8) not in funding_stamps(at(8), at(15))

    def test_a_position_closed_exactly_on_a_stamp_is_charged(self) -> None:
        """Inclusive of end: it did hold through that period."""
        assert funding_stamps(at(1), at(8)) == [at(8)]

    def test_a_holding_shorter_than_a_period_pays_nothing(self) -> None:
        assert funding_stamps(at(9), at(15)) == []

    def test_three_stamps_a_day(self) -> None:
        assert len(funding_stamps(at(0), at(0, day=2))) == 3

    def test_an_inverted_window_yields_nothing(self) -> None:
        assert funding_stamps(at(20), at(1)) == []


class TestNStampsAreCharged:
    """Acceptance: N stamps held == N x rate x notional, correctly signed."""

    def _long_book(self) -> PortfolioManager:
        book = PortfolioManager(
            starting_equity=START, market_type=MarketType.FUTURE, leverage=Decimal("1")
        )
        book.apply_fill(
            Fill(
                fill_id="f1",
                order_id="o1",
                symbol=BTC,
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                price=Decimal("50000"),
                fee=Decimal("0"),
                fee_currency="USDT",
                timestamp=at(1),
            )
        )
        book.update_mark_price(BTC, Decimal("50000"))
        return book

    def test_a_long_held_across_two_stamps_pays_twice(self) -> None:
        book = self._long_book()
        book.settle_funding(at(1), rate_for=lambda _symbol, _stamp: Decimal("0.0001"))  # baseline
        charges = book.settle_funding(at(20), rate_for=lambda _symbol, _stamp: Decimal("0.0001"))

        assert len(charges) == 2, "08:00 and 16:00"
        assert total_funding(charges) == Decimal("-10"), "2 x 0.0001 x 50000, paid"
        assert book.funding_paid == Decimal("-10")

    def test_funding_leaves_the_wallet(self) -> None:
        """It is a real cash movement, not a bookkeeping entry."""
        book = self._long_book()
        book.settle_funding(at(1), rate_for=lambda _symbol, _stamp: Decimal("0.0001"))
        book.settle_funding(at(20), rate_for=lambda _symbol, _stamp: Decimal("0.0001"))
        assert book.cash == START - Decimal("10")

    def test_a_short_is_credited_across_the_same_stamps(self) -> None:
        book = PortfolioManager(
            starting_equity=START, market_type=MarketType.FUTURE, leverage=Decimal("1")
        )
        book.apply_fill(
            Fill(
                fill_id="f1",
                order_id="o1",
                symbol=BTC,
                side=OrderSide.SELL,
                quantity=Decimal("1"),
                price=Decimal("50000"),
                fee=Decimal("0"),
                fee_currency="USDT",
                timestamp=at(1),
            )
        )
        book.update_mark_price(BTC, Decimal("50000"))
        book.settle_funding(at(1), rate_for=lambda _symbol, _stamp: Decimal("0.0001"))
        book.settle_funding(at(20), rate_for=lambda _symbol, _stamp: Decimal("0.0001"))
        assert book.cash == START + Decimal("10")

    def test_an_unknown_rate_charges_nothing(self) -> None:
        """Guessing a rate would fabricate a cost."""
        book = self._long_book()
        book.settle_funding(at(1), rate_for=lambda _symbol, _stamp: None)
        assert book.settle_funding(at(20), rate_for=lambda _symbol, _stamp: None) == ()
        assert book.cash == START

    def test_spot_is_never_charged_funding(self) -> None:
        book = PortfolioManager(starting_equity=START, market_type=MarketType.SPOT)
        assert book.settle_funding(at(20), rate_for=lambda _symbol, _stamp: Decimal("0.01")) == ()

    def test_a_flat_book_is_charged_nothing(self) -> None:
        book = PortfolioManager(starting_equity=START, market_type=MarketType.FUTURE)
        book.settle_funding(at(1), rate_for=lambda _symbol, _stamp: Decimal("0.0001"))
        assert book.settle_funding(at(20), rate_for=lambda _symbol, _stamp: Decimal("0.0001")) == ()


class TestHistoricalSchedule:
    def test_the_backtest_uses_the_rate_that_actually_applied(self) -> None:
        schedule = FundingSchedule([(at(8), Decimal("0.0003")), (at(16), Decimal("-0.0001"))])
        assert schedule.rate_at(at(8)) == Decimal("0.0003")
        assert schedule.rate_at(at(16)) == Decimal("-0.0001")

    def test_an_unrecorded_stamp_returns_none(self) -> None:
        assert FundingSchedule([(at(8), Decimal("0.0003"))]).rate_at(at(16)) is None

    def test_varying_rates_are_applied_per_stamp(self) -> None:
        """Not an average: each settlement uses its own rate."""
        schedule = FundingSchedule([(at(8), Decimal("0.0002")), (at(16), Decimal("-0.0001"))])
        book = PortfolioManager(
            starting_equity=START, market_type=MarketType.FUTURE, leverage=Decimal("1")
        )
        book.apply_fill(
            Fill(
                fill_id="f1",
                order_id="o1",
                symbol=BTC,
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                price=Decimal("50000"),
                fee=Decimal("0"),
                fee_currency="USDT",
                timestamp=at(1),
            )
        )
        book.update_mark_price(BTC, Decimal("50000"))
        book.settle_funding(at(1), rate_for=lambda _symbol, stamp: schedule.rate_at(stamp))
        book.settle_funding(at(20), rate_for=lambda _symbol, stamp: schedule.rate_at(stamp))
        # -10 at 08:00 (paid), +5 at 16:00 (received on a negative rate).
        assert book.funding_paid == Decimal("-5")


class TestBorrowHook:
    def test_one_x_borrows_nothing(self) -> None:
        """At no leverage there is nothing borrowed, so no cost is fabricated."""
        assert borrow_cost(notional=Decimal("50000"), leverage=Decimal("1")) == Decimal("0")

    def test_the_hook_charges_only_the_borrowed_portion(self) -> None:
        cost = borrow_cost(
            notional=Decimal("50000"), leverage=Decimal("2"), rate_per_period=Decimal("0.0001")
        )
        assert cost == Decimal("-2.5"), "half is borrowed at 2x"
        assert cost < 0

    def test_a_zero_rate_costs_nothing_even_when_levered(self) -> None:
        assert borrow_cost(notional=Decimal("50000"), leverage=Decimal("5")) == Decimal("0")


class TestBacktestAndPaperShareTheLogic:
    def test_both_engines_settle_through_the_portfolio(self) -> None:
        """One implementation, or the two describe different accounts."""
        import inspect

        from quantflow.backtest.engine import BacktestEngine
        from quantflow.paper.engine import PaperTradingEngine

        assert "settle_funding" in inspect.getsource(BacktestEngine)
        assert "settle_funding" in inspect.getsource(PaperTradingEngine.on_candle)

    def test_the_backtest_leverage_is_pinned_to_one_x(self) -> None:
        from quantflow.backtest.engine import BacktestConfig
        from quantflow.domain.enums import Timeframe

        # A slots dataclass exposes a descriptor on the class, so read a real instance.
        config = BacktestConfig(symbols=(BTC,), timeframe=Timeframe.H1)
        assert config.leverage == Decimal("1")
