"""Phase 8: protective exits pay gap and slippage, not the exact level.

The defect: `_check_protective_exits` filled at the literal stop or target price. That
assumes the venue always gives you your level. A bar that gaps straight through a stop
fills at the open, and the difference is real loss the backtest never charged.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.domain.enums import OrderSide, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.exchange.simulator import FixedSlippage
from quantflow.paper.engine import PaperConfig, PaperTradingEngine
from quantflow.strategy.registry import load_builtin_strategies
from tests.conftest import REFERENCE_TIME, make_candle

BTC = Symbol.parse("BTC/USDT")


def engine(slippage_rate: str = "0") -> PaperTradingEngine:
    return PaperTradingEngine(
        load_builtin_strategies().create("ema_cross"),
        PaperConfig(
            symbols=(BTC,),
            timeframe=Timeframe.H1,
            persist=False,
            slippage=FixedSlippage(rate=Decimal(slippage_rate)),
        ),
        instruments={},
    )


def candle(*, open_price: str, high: str, low: str, close: str):
    return make_candle(
        BTC,
        open_time=REFERENCE_TIME,
        open_price=open_price,
        high=high,
        low=low,
        close=close,
        volume="100",
    )


class TestStopGapsFillWorse:
    def test_a_gapping_bar_fills_a_long_stop_below_the_trigger(self) -> None:
        """The headline case: price opened under the stop, so that is the fill."""
        bar = candle(open_price="48000", high="48500", low="47000", close="47500")
        price = engine()._protective_exit_price(
            Decimal("49000"),
            candle=bar,
            side=OrderSide.SELL,  # closing a long
            quantity=Decimal("0.01"),
            is_stop=True,
        )
        assert price == Decimal("48000"), "must fill at the gapped open, not the stop"
        assert price < Decimal("49000")

    def test_a_short_stop_gaps_upward_against_the_trader(self) -> None:
        bar = candle(open_price="52000", high="52500", low="51500", close="52200")
        price = engine()._protective_exit_price(
            Decimal("51000"),
            candle=bar,
            side=OrderSide.BUY,  # closing a short
            quantity=Decimal("0.01"),
            is_stop=True,
        )
        assert price == Decimal("52000")
        assert price > Decimal("51000")

    def test_no_gap_fills_at_the_stop(self) -> None:
        """When the bar opens above the stop, the stop level itself is reachable."""
        bar = candle(open_price="50000", high="50100", low="48500", close="49500")
        price = engine()._protective_exit_price(
            Decimal("49000"),
            candle=bar,
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            is_stop=True,
        )
        assert price == Decimal("49000")

    def test_the_gap_is_never_applied_in_the_traders_favour(self) -> None:
        """A long stop must never fill above its level just because the bar opened high."""
        bar = candle(open_price="51000", high="51200", low="48000", close="48500")
        price = engine()._protective_exit_price(
            Decimal("49000"),
            candle=bar,
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            is_stop=True,
        )
        assert price == Decimal("49000"), "min(level, open) - not the favourable open"


class TestSlippageIsCharged:
    def test_slippage_moves_a_stop_exit_against_the_trader(self) -> None:
        bar = candle(open_price="50000", high="50100", low="48500", close="49500")
        clean = engine("0")._protective_exit_price(
            Decimal("49000"),
            candle=bar,
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            is_stop=True,
        )
        slipped = engine("0.001")._protective_exit_price(
            Decimal("49000"),
            candle=bar,
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            is_stop=True,
        )
        assert slipped < clean, "selling should realise less after slippage"

    def test_targets_also_pay_slippage(self) -> None:
        """A target is not a free fill either."""
        bar = candle(open_price="50000", high="52500", low="49900", close="52000")
        clean = engine("0")._protective_exit_price(
            Decimal("52000"),
            candle=bar,
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            is_stop=False,
        )
        slipped = engine("0.001")._protective_exit_price(
            Decimal("52000"),
            candle=bar,
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            is_stop=False,
        )
        assert slipped < clean

    def test_a_target_is_not_gapped_in_the_traders_favour(self) -> None:
        """Assuming a better-than-asked fill is exactly the optimism being removed."""
        bar = candle(open_price="53000", high="53500", low="51900", close="53200")
        price = engine("0")._protective_exit_price(
            Decimal("52000"),
            candle=bar,
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            is_stop=False,
        )
        assert price == Decimal("52000"), "the target level, not the better open"
