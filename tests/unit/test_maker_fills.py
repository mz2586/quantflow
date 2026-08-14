"""Maker-first entries, modelled without the lie that makes them look good.

Maker is 0.01% against taker 0.06%, so entering passively cuts a round trip from ~0.24% to
~0.04% and turns a large class of rejected setups economic. That is the entire reason to do
it — and it is exactly why the fill model has to be pessimistic, because an optimistic one
would manufacture the result the change is supposed to be tested for.

Three things must hold or the backtest is fiction:

**A limit only fills if price traded through it.** Touching your price is not being filled:
you are behind a queue you cannot see. Requiring the bar to trade strictly past the limit
is the standard conservative proxy for queue position.

**Post-only must not silently become taker.** A buy limit placed at the touch and then
gapped through by the next bar's open would, in reality, be rejected by the exchange for
crossing the spread — not filled at a better price. Modelling it as a fill would hand the
backtest free price improvement precisely on the bars that move fastest.

**Unfilled means missed, not pending.** A resting entry that never fills has to be
cancelled and the setup abandoned. Leaving it working means it can fill many bars later on
a signal that has long since expired, which is a slow-motion form of look-ahead.

The adverse selection this leaves in place is real and intended: a buy limit fills exactly
when price is falling toward it, so passive entries are systematically filled just as the
market moves against them. That cost belongs in the result.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from quantflow.domain.enums import (
    LiquidityRole,
    OrderSide,
    OrderType,
    Timeframe,
    TimeInForce,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle
from quantflow.domain.orders import OrderRequest
from quantflow.exchange.simulator import SimulatedBroker
from tests.conftest import REFERENCE_TIME

BTC = Symbol.parse("BTC/USDT")


def instrument() -> Instrument:
    return Instrument(
        symbol=BTC,
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.001"),
        min_quantity=Decimal("0.001"),
        min_notional=Decimal("5"),
        maker_fee=Decimal("0.0001"),
        taker_fee=Decimal("0.0006"),
    )


def broker() -> SimulatedBroker:
    return SimulatedBroker(instruments={BTC: instrument()})


def candle(*, open_: str, high: str, low: str, close: str, minutes: int = 0) -> Candle:
    start = REFERENCE_TIME + timedelta(minutes=minutes)
    return Candle(
        symbol=BTC,
        timeframe=Timeframe.parse("15m"),
        open_time=start,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def buy_limit(price: str, *, post_only: bool = True) -> OrderRequest:
    return OrderRequest(
        symbol=BTC,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.5"),
        price=Decimal(price),
        time_in_force=TimeInForce.GTC,
        post_only=post_only,
    )


class TestLimitOnlyFillsWhenPriceTradesThrough:
    def test_a_bar_that_trades_through_fills(self) -> None:
        book = broker()
        book.submit(buy_limit("100"), now=REFERENCE_TIME)

        fills = book.process_candle(candle(open_="101", high="102", low="99", close="101"))

        assert len(fills) == 1

    def test_a_bar_that_only_touches_does_not_fill(self) -> None:
        """Touching is not filling: the queue ahead of you is invisible and real."""
        book = broker()
        book.submit(buy_limit("100"), now=REFERENCE_TIME)

        fills = book.process_candle(candle(open_="101", high="102", low="100", close="101"))

        assert not fills

    def test_a_fill_is_at_the_limit_not_better(self) -> None:
        book = broker()
        book.submit(buy_limit("100"), now=REFERENCE_TIME)

        _, fill = book.process_candle(candle(open_="101", high="102", low="95", close="101"))[0]

        assert fill.price == Decimal("100")

    def test_a_passive_fill_is_charged_the_maker_fee(self) -> None:
        book = broker()
        book.submit(buy_limit("100"), now=REFERENCE_TIME)

        _, fill = book.process_candle(candle(open_="101", high="102", low="99", close="101"))[0]

        assert fill.role is LiquidityRole.MAKER


class TestPostOnlyNeverCrosses:
    def test_a_crossing_post_only_order_is_rejected_not_filled(self) -> None:
        """The bar opens below a buy limit: in reality the exchange rejects, it does not
        hand you a better price."""
        book = broker()
        book.submit(buy_limit("100", post_only=True), now=REFERENCE_TIME)

        results = book.process_candle(candle(open_="99", high="101", low="98", close="100"))

        assert not [fill for _, fill in results if fill.quantity > 0]

    def test_a_non_post_only_limit_may_still_fill_on_a_gap(self) -> None:
        """The restriction belongs to post-only, not to limit orders in general."""
        book = broker()
        book.submit(buy_limit("100", post_only=False), now=REFERENCE_TIME)

        results = book.process_candle(candle(open_="99", high="101", low="98", close="100"))

        assert [fill for _, fill in results if fill.quantity > 0]

    def test_post_only_still_fills_when_price_comes_to_it(self) -> None:
        """Rejection is only for crossing. A normal passive fill must still work."""
        book = broker()
        book.submit(buy_limit("100", post_only=True), now=REFERENCE_TIME)

        results = book.process_candle(candle(open_="101", high="102", low="99", close="101"))

        assert [fill for _, fill in results if fill.quantity > 0]


class TestUnfilledEntriesExpire:
    def test_an_order_past_its_lifetime_is_cancelled(self) -> None:
        """A setup that never filled is missed, not pending indefinitely."""
        book = broker()
        book.submit(buy_limit("50"), now=REFERENCE_TIME, max_bars=2)

        for minute in (0, 15, 30):
            book.process_candle(
                candle(open_="101", high="102", low="99", close="101", minutes=minute)
            )

        assert not book.open_orders

    def test_it_survives_within_its_lifetime(self) -> None:
        book = broker()
        book.submit(buy_limit("50"), now=REFERENCE_TIME, max_bars=5)

        book.process_candle(candle(open_="101", high="102", low="99", close="101"))

        assert book.open_orders

    def test_an_expired_order_produces_no_fill(self) -> None:
        book = broker()
        book.submit(buy_limit("50"), now=REFERENCE_TIME, max_bars=1)

        book.process_candle(candle(open_="101", high="102", low="99", close="101"))
        late = book.process_candle(candle(open_="60", high="61", low="40", close="60", minutes=15))

        assert not [fill for _, fill in late if fill.quantity > 0]
