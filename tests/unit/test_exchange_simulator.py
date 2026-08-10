"""Fee, slippage and matching semantics of the simulated venue.

These assumptions decide whether a backtest tells the truth, so each one is pinned
explicitly — especially the pessimistic ones.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import (
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    Timeframe,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, Ticker
from quantflow.domain.orders import Order, OrderRequest
from quantflow.exchange.simulator import (
    FeeModel,
    FixedSlippage,
    SimulatedBroker,
    SpreadSlippage,
    VolumeShareSlippage,
    match_against_candle,
)
from tests.conftest import REFERENCE_TIME


def bar(
    symbol: Symbol,
    *,
    open_price: str,
    high: str,
    low: str,
    close: str,
    volume: str = "1000",
    index: int = 0,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=Timeframe.H1,
        open_time=REFERENCE_TIME + timedelta(hours=index),
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        quote_volume=Decimal(volume) * Decimal(close),
    )


def working_order(symbol: Symbol, **overrides: object) -> Order:
    kwargs: dict[str, object] = {
        "symbol": symbol,
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": Decimal("1"),
    }
    kwargs.update(overrides)
    request = OrderRequest(**kwargs)  # type: ignore[arg-type]
    return Order.from_request(request, now=REFERENCE_TIME).acknowledge("v1", now=REFERENCE_TIME)


class TestSlippageModels:
    def test_fixed_slippage_always_costs_the_trader(self) -> None:
        model = FixedSlippage(rate=Decimal("0.001"))
        buy = model.apply(reference_price=Decimal("100"), side=OrderSide.BUY, quantity=Decimal("1"))
        sell = model.apply(
            reference_price=Decimal("100"), side=OrderSide.SELL, quantity=Decimal("1")
        )
        assert buy == Decimal("100.1")
        assert sell == Decimal("99.9")

    def test_volume_share_slippage_scales_with_order_size(self, btc: Symbol) -> None:
        model = VolumeShareSlippage(base_rate=Decimal("0.0002"), impact_coefficient=Decimal("0.1"))
        candle = bar(btc, open_price="100", high="101", low="99", close="100", volume="1000")
        small = model.apply(
            reference_price=Decimal("100"),
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            candle=candle,
        )
        large = model.apply(
            reference_price=Decimal("100"),
            side=OrderSide.BUY,
            quantity=Decimal("500"),
            candle=candle,
        )
        # A model that ignored size would fill both at the same price, which is how a
        # backtest comes to claim capacity it does not have.
        assert large > small > Decimal("100")

    def test_volume_share_without_a_candle_charges_only_the_base_rate(self) -> None:
        model = VolumeShareSlippage(base_rate=Decimal("0.001"))
        assert model.apply(
            reference_price=Decimal("100"), side=OrderSide.BUY, quantity=Decimal("1")
        ) == Decimal("100.1")

    def test_oversized_detection(self, btc: Symbol) -> None:
        model = VolumeShareSlippage(max_volume_share=Decimal("0.1"))
        candle = bar(btc, open_price="100", high="100", low="100", close="100", volume="100")
        assert not model.is_oversized(Decimal("5"), candle)
        assert model.is_oversized(Decimal("50"), candle)

    def test_zero_volume_bar_is_always_oversized(self, btc: Symbol) -> None:
        model = VolumeShareSlippage()
        candle = bar(btc, open_price="100", high="100", low="100", close="100", volume="0")
        assert model.is_oversized(Decimal("0.0001"), candle)

    def test_spread_slippage_crosses_the_quote(self, btc: Symbol) -> None:
        ticker = Ticker(
            symbol=btc,
            timestamp=REFERENCE_TIME,
            bid=Decimal("99"),
            ask=Decimal("101"),
            last=Decimal("100"),
        )
        model = SpreadSlippage()
        assert model.apply_with_ticker(ticker=ticker, side=OrderSide.BUY) == Decimal("101")
        assert model.apply_with_ticker(ticker=ticker, side=OrderSide.SELL) == Decimal("99")


class TestFeeModel:
    def test_uses_instrument_rates_by_default(self, btc_instrument: Instrument) -> None:
        fees = FeeModel()
        assert fees.compute(
            btc_instrument, quantity=Decimal("2"), price=Decimal("50000"), role=LiquidityRole.TAKER
        ) == Decimal("100")

    def test_overrides_win(self, btc_instrument: Instrument) -> None:
        fees = FeeModel(maker_rate=Decimal("0"), taker_rate=Decimal("0.002"))
        assert fees.compute(
            btc_instrument, quantity=Decimal("1"), price=Decimal("1000"), role=LiquidityRole.MAKER
        ) == Decimal("0")
        assert fees.compute(
            btc_instrument, quantity=Decimal("1"), price=Decimal("1000"), role=LiquidityRole.TAKER
        ) == Decimal("2")


class TestMatching:
    def test_market_order_fills_at_the_bar_open(self, btc: Symbol) -> None:
        # Not the close: a decision made on the previous close can only execute at the
        # next open. Filling at the close is textbook look-ahead bias.
        order = working_order(btc)
        candle = bar(btc, open_price="100", high="120", low="90", close="110")
        decision = match_against_candle(order, candle, slippage=FixedSlippage(Decimal("0")))
        assert decision.filled
        assert decision.price == Decimal("100")
        assert decision.role is LiquidityRole.TAKER

    def test_market_order_pays_slippage(self, btc: Symbol) -> None:
        order = working_order(btc)
        candle = bar(btc, open_price="100", high="120", low="90", close="110")
        decision = match_against_candle(order, candle, slippage=FixedSlippage(Decimal("0.01")))
        assert decision.price == Decimal("101")

    def test_limit_buy_needs_price_to_trade_strictly_through(self, btc: Symbol) -> None:
        order = working_order(
            btc, order_type=OrderType.LIMIT, price=Decimal("100"), side=OrderSide.BUY
        )
        touched = bar(btc, open_price="105", high="106", low="100", close="105")
        through = bar(btc, open_price="105", high="106", low="99", close="105")
        # Touching a price does not guarantee a fill at it — you need to be at the front
        # of a queue this simulator does not model.
        assert not match_against_candle(order, touched, slippage=FixedSlippage()).filled
        assert match_against_candle(order, through, slippage=FixedSlippage()).filled

    def test_limit_fill_is_at_the_limit_and_earns_maker(self, btc: Symbol) -> None:
        order = working_order(btc, order_type=OrderType.LIMIT, price=Decimal("100"))
        candle = bar(btc, open_price="105", high="106", low="95", close="105")
        decision = match_against_candle(order, candle, slippage=FixedSlippage(Decimal("0.05")))
        assert decision.price == Decimal("100")  # no slippage on a resting limit
        assert decision.role is LiquidityRole.MAKER

    def test_limit_sell(self, btc: Symbol) -> None:
        order = working_order(
            btc, order_type=OrderType.LIMIT, side=OrderSide.SELL, price=Decimal("110")
        )
        below = bar(btc, open_price="100", high="110", low="99", close="105")
        above = bar(btc, open_price="100", high="111", low="99", close="105")
        assert not match_against_candle(order, below, slippage=FixedSlippage()).filled
        assert match_against_candle(order, above, slippage=FixedSlippage()).filled

    def test_stop_triggers_on_range_not_close(self, btc: Symbol) -> None:
        order = working_order(
            btc,
            order_type=OrderType.STOP_MARKET,
            side=OrderSide.SELL,
            trigger_price=Decimal("95"),
        )
        # The bar dipped to 94 intrabar even though it closed at 105 — the stop fired.
        candle = bar(btc, open_price="100", high="106", low="94", close="105")
        assert match_against_candle(order, candle, slippage=FixedSlippage()).filled

    def test_stop_gap_fills_at_the_worse_price(self, btc: Symbol) -> None:
        # Price gapped from 100 straight to 90; a sell stop at 95 cannot fill at 95.
        order = working_order(
            btc,
            order_type=OrderType.STOP_MARKET,
            side=OrderSide.SELL,
            trigger_price=Decimal("95"),
        )
        candle = bar(btc, open_price="90", high="92", low="88", close="91")
        decision = match_against_candle(order, candle, slippage=FixedSlippage(Decimal("0")))
        assert decision.filled
        assert decision.price == Decimal("90")  # the open, not the stop

    def test_buy_stop_gap_also_fills_worse(self, btc: Symbol) -> None:
        order = working_order(
            btc,
            order_type=OrderType.STOP_MARKET,
            side=OrderSide.BUY,
            trigger_price=Decimal("105"),
        )
        candle = bar(btc, open_price="112", high="115", low="110", close="113")
        decision = match_against_candle(order, candle, slippage=FixedSlippage(Decimal("0")))
        assert decision.price == Decimal("112")

    def test_untriggered_stop_does_not_fill(self, btc: Symbol) -> None:
        order = working_order(
            btc,
            order_type=OrderType.STOP_MARKET,
            side=OrderSide.SELL,
            trigger_price=Decimal("90"),
        )
        candle = bar(btc, open_price="100", high="106", low="95", close="105")
        decision = match_against_candle(order, candle, slippage=FixedSlippage())
        assert not decision.filled
        assert "trigger" in decision.reason

    def test_mismatched_symbol_is_rejected(self, btc: Symbol, eth: Symbol) -> None:
        order = working_order(btc)
        with pytest.raises(ValidationError, match="cannot fill an order"):
            match_against_candle(
                order,
                bar(eth, open_price="1", high="1", low="1", close="1"),
                slippage=FixedSlippage(),
            )


class TestSimulatedBroker:
    def _broker(self, btc_instrument: Instrument, **overrides: object) -> SimulatedBroker:
        return SimulatedBroker(
            instruments={btc_instrument.symbol: btc_instrument},
            slippage=FixedSlippage(Decimal("0")),
            fees=FeeModel(),
            **overrides,  # type: ignore[arg-type]
        )

    def _request(self, btc: Symbol, **overrides: object) -> OrderRequest:
        kwargs: dict[str, object] = {
            "symbol": btc,
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("1"),
        }
        kwargs.update(overrides)
        return OrderRequest(**kwargs)  # type: ignore[arg-type]

    def test_submit_and_fill(self, btc: Symbol, btc_instrument: Instrument) -> None:
        broker = self._broker(btc_instrument)
        broker.submit(self._request(btc), now=REFERENCE_TIME, reference_price=Decimal("100"))
        assert len(broker.open_orders) == 1

        results = broker.process_candle(
            bar(btc, open_price="100", high="110", low="90", close="105")
        )
        assert len(results) == 1
        order, fill = results[0]
        assert order.status is OrderStatus.FILLED
        assert fill.price == Decimal("100")
        assert fill.fee == Decimal("0.1")  # 100 notional * 0.001 taker
        assert len(broker.open_orders) == 0

    def test_submit_enforces_venue_rules(self, btc: Symbol, btc_instrument: Instrument) -> None:
        # Skipping this would let a backtest fill orders the venue would have rejected.
        broker = self._broker(btc_instrument)
        with pytest.raises(ValidationError, match=r"min_notional|below minimum"):
            broker.submit(
                self._request(btc, quantity=Decimal("0.00001")),
                now=REFERENCE_TIME,
                reference_price=Decimal("1"),
            )

    def test_submit_without_a_reference_price_skips_validation(
        self, btc: Symbol, btc_instrument: Instrument
    ) -> None:
        broker = self._broker(btc_instrument)
        order = broker.submit(self._request(btc, quantity=Decimal("0.00001")), now=REFERENCE_TIME)
        assert order.status is OrderStatus.NEW

    def test_unknown_symbol_is_rejected(
        self, btc: Symbol, eth: Symbol, btc_instrument: Instrument
    ) -> None:
        broker = self._broker(btc_instrument)
        with pytest.raises(ValidationError, match="no instrument"):
            broker.submit(self._request(eth), now=REFERENCE_TIME)

    def test_resting_limit_survives_a_non_matching_bar(
        self, btc: Symbol, btc_instrument: Instrument
    ) -> None:
        broker = self._broker(btc_instrument)
        broker.submit(
            self._request(btc, order_type=OrderType.LIMIT, price=Decimal("90.00")),
            now=REFERENCE_TIME,
        )
        broker.process_candle(bar(btc, open_price="100", high="105", low="95", close="102"))
        assert len(broker.open_orders) == 1
        results = broker.process_candle(
            bar(btc, open_price="100", high="105", low="85", close="102", index=1)
        )
        assert results[0][1].price == Decimal("90.00")
        assert len(broker.open_orders) == 0

    def test_cancel(self, btc: Symbol, btc_instrument: Instrument) -> None:
        broker = self._broker(btc_instrument)
        order = broker.submit(
            self._request(btc, order_type=OrderType.LIMIT, price=Decimal("50.00")),
            now=REFERENCE_TIME,
        )
        cancelled = broker.cancel(order.order_id, now=REFERENCE_TIME)
        assert cancelled.status is OrderStatus.CANCELLED
        assert len(broker.open_orders) == 0

    def test_cancel_unknown_order_raises(self, btc_instrument: Instrument) -> None:
        with pytest.raises(ValidationError, match="not working"):
            self._broker(btc_instrument).cancel("nope", now=REFERENCE_TIME)

    def test_cancel_all_by_symbol(self, btc: Symbol, btc_instrument: Instrument) -> None:
        broker = self._broker(btc_instrument)
        for price in ("50.00", "51.00"):
            broker.submit(
                self._request(btc, order_type=OrderType.LIMIT, price=Decimal(price)),
                now=REFERENCE_TIME,
            )
        assert len(broker.cancel_all(symbol=btc, now=REFERENCE_TIME)) == 2
        assert len(broker.open_orders) == 0

    def test_orders_for_other_symbols_are_untouched(
        self, btc: Symbol, btc_instrument: Instrument, eth: Symbol
    ) -> None:
        eth_instrument = Instrument(symbol=eth, min_notional=Decimal("1"))
        broker = SimulatedBroker(
            instruments={btc: btc_instrument, eth: eth_instrument},
            slippage=FixedSlippage(Decimal("0")),
        )
        broker.submit(self._request(btc), now=REFERENCE_TIME)
        broker.submit(self._request(eth), now=REFERENCE_TIME)
        broker.process_candle(bar(btc, open_price="100", high="110", low="90", close="105"))
        assert len(broker.open_orders) == 1
        assert broker.open_orders[0].symbol == eth

    def test_oversized_order_is_rejected(self, btc: Symbol, btc_instrument: Instrument) -> None:
        broker = SimulatedBroker(
            instruments={btc: btc_instrument},
            slippage=VolumeShareSlippage(max_volume_share=Decimal("0.1")),
            reject_oversized=True,
        )
        broker.submit(self._request(btc, quantity=Decimal("100")), now=REFERENCE_TIME)
        results = broker.process_candle(
            bar(btc, open_price="100", high="110", low="90", close="105", volume="10")
        )
        assert results[0][0].status is OrderStatus.REJECTED
        assert len(broker.open_orders) == 0

    def test_oversized_order_can_be_allowed(self, btc: Symbol, btc_instrument: Instrument) -> None:
        broker = SimulatedBroker(
            instruments={btc: btc_instrument},
            slippage=VolumeShareSlippage(max_volume_share=Decimal("0.1")),
            reject_oversized=False,
        )
        broker.submit(self._request(btc, quantity=Decimal("100")), now=REFERENCE_TIME)
        results = broker.process_candle(
            bar(btc, open_price="100", high="110", low="90", close="105", volume="10")
        )
        assert results[0][0].status is OrderStatus.FILLED

    def test_fill_at_market_uses_the_live_quote(
        self, btc: Symbol, btc_instrument: Instrument
    ) -> None:
        broker = self._broker(btc_instrument)
        ticker = Ticker(
            symbol=btc,
            timestamp=REFERENCE_TIME,
            bid=Decimal("99"),
            ask=Decimal("101"),
            last=Decimal("100"),
        )
        order, fill = broker.fill_at_market(self._request(btc), ticker=ticker, now=REFERENCE_TIME)
        assert order.status is OrderStatus.FILLED
        assert fill.price == Decimal("101")  # a buy pays the offer
        assert fill.role is LiquidityRole.TAKER

    def test_fill_ids_are_unique(self, btc: Symbol, btc_instrument: Instrument) -> None:
        broker = self._broker(btc_instrument)
        for index in range(3):
            broker.submit(self._request(btc), now=REFERENCE_TIME)
            broker.process_candle(
                bar(btc, open_price="100", high="110", low="90", close="105", index=index)
            )
        # Duplicate fill ids would be silently swallowed by Order.apply_fill's idempotency.
        assert broker._fill_sequence == 3
