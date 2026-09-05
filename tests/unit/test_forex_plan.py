"""The LONG / SHORT / NO-TRADE gate and intrabar exit modelling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import OrderSide
from quantflow.forex.costs import ForexCostModel
from quantflow.forex.exits import (
    IntrabarOutcome,
    evaluate_intrabar_exit,
    evaluate_position_exit,
)
from quantflow.forex.instruments import ForexInstrument, TradeMode
from quantflow.forex.plan import PlanRejection, TradeDirection, plan_trade
from quantflow.forex.protocol import ForexBar, ForexPosition, ForexTick, ForexTimeframe
from quantflow.forex.sessions import SessionClock, TradingSession

MONDAY_NOON = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
SATURDAY_NOON = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
FRIDAY_LATE = datetime(2026, 8, 14, 20, 45, tzinfo=UTC)

ENTRY = Decimal("1.10000")
LONG_STOP = Decimal("1.09800")
SHORT_STOP = Decimal("1.10200")


def make_instrument(**overrides: object) -> ForexInstrument:
    kwargs: dict[str, object] = {
        "symbol": "EURUSD+",
        "base": "EUR",
        "quote": "USD",
        "contract_size": Decimal("100000"),
        "min_lot": Decimal("0.01"),
        "max_lot": Decimal("50"),
        "lot_step": Decimal("0.01"),
        "digits": 5,
        "point": Decimal("0.00001"),
        "tick_size": Decimal("0.00001"),
        "tick_value": Decimal("1"),
        "spread_points": Decimal("12"),
    }
    kwargs.update(overrides)
    return ForexInstrument(**kwargs)  # type: ignore[arg-type]


def make_bar(high: str, low: str, open_: str = "1.10000", close: str = "1.10000") -> ForexBar:
    return ForexBar(
        symbol="EURUSD+",
        timeframe=ForexTimeframe.M15,
        open_time=MONDAY_NOON,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
    )


@pytest.fixture
def clock() -> SessionClock:
    return SessionClock()


class TestDirection:
    def test_buy_produces_a_long_plan(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
        )
        assert plan.direction is TradeDirection.LONG
        assert plan.lots == Decimal("0.50")
        assert plan.session is TradingSession.LONDON_NEW_YORK_OVERLAP
        assert bool(plan)

    def test_sell_produces_a_short_plan(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.SELL,
            entry_price=ENTRY,
            stop_loss=SHORT_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
        )
        assert plan.direction is TradeDirection.SHORT
        assert plan.lots == Decimal("0.50")

    def test_long_and_short_size_identically(self, clock: SessionClock) -> None:
        common: dict[str, object] = {
            "instrument": make_instrument(),
            "entry_price": ENTRY,
            "account_risk": Decimal("100"),
            "clock": clock,
            "now": MONDAY_NOON,
        }
        long_plan = plan_trade(side=OrderSide.BUY, stop_loss=LONG_STOP, **common)  # type: ignore[arg-type]
        short_plan = plan_trade(side=OrderSide.SELL, stop_loss=SHORT_STOP, **common)  # type: ignore[arg-type]
        assert long_plan.lots == short_plan.lots


class TestGates:
    def test_weekend_is_no_trade(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=SATURDAY_NOON,
        )
        assert plan.direction is TradeDirection.NO_TRADE
        assert plan.reason is PlanRejection.MARKET_CLOSED
        assert plan.session is TradingSession.CLOSED
        assert not bool(plan)

    def test_weekly_close_run_up_is_no_trade(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=FRIDAY_LATE,
        )
        assert plan.reason is PlanRejection.WEEKLY_CLOSE_APPROACHING

    def test_weekly_close_gate_can_be_disabled(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=FRIDAY_LATE,
            block_before_weekly_close=False,
        )
        assert plan.direction is TradeDirection.LONG

    def test_stale_quote_is_no_trade(self, clock: SessionClock) -> None:
        stale = ForexTick(
            symbol="EURUSD+",
            bid=Decimal("1.09990"),
            ask=Decimal("1.10002"),
            timestamp=MONDAY_NOON - timedelta(minutes=10),
        )
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
            tick=stale,
        )
        assert plan.reason is PlanRejection.STALE_QUOTE

    def test_fresh_quote_passes(self, clock: SessionClock) -> None:
        fresh = ForexTick(
            symbol="EURUSD+",
            bid=Decimal("1.09990"),
            ask=Decimal("1.10002"),
            timestamp=MONDAY_NOON - timedelta(seconds=1),
        )
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
            tick=fresh,
        )
        assert plan.direction is TradeDirection.LONG

    def test_sub_minimum_size_is_no_trade_with_the_sizing_reason(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("0.01"),
            clock=clock,
            now=MONDAY_NOON,
        )
        assert plan.reason is PlanRejection.SIZING_REJECTED
        assert plan.sizing is not None
        assert not plan.sizing.accepted

    def test_side_the_venue_forbids_is_no_trade(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(trade_mode=TradeMode.LONG_ONLY),
            side=OrderSide.SELL,
            entry_price=ENTRY,
            stop_loss=SHORT_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
        )
        assert plan.reason is PlanRejection.SIZING_REJECTED

    def test_venue_session_gap_is_no_trade(self, clock: SessionClock) -> None:
        from datetime import time

        from quantflow.forex.sessions import SessionWindow

        instrument = make_instrument(
            sessions=(SessionWindow(weekday=0, start=time(0, 0), end=time(6, 0)),)
        )
        plan = plan_trade(
            instrument=instrument,
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
        )
        assert plan.reason is PlanRejection.MARKET_CLOSED


class TestCostsAndEdge:
    def test_costs_are_attached_when_a_model_is_supplied(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
            cost_model=ForexCostModel(commission_per_lot_round_turn=Decimal("6")),
        )
        assert plan.costs is not None
        assert plan.costs.commission == Decimal("3.00")
        assert plan.costs.spread == Decimal("6.00")

    def test_no_costs_without_a_model(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
        )
        assert plan.costs is None
        assert plan.net_edge is None

    def test_edge_that_does_not_survive_costs_is_refused(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
            cost_model=ForexCostModel(commission_per_lot_round_turn=Decimal("6")),
            gross_edge=Decimal("5"),
        )
        assert plan.reason is PlanRejection.NEGATIVE_NET_EDGE
        assert plan.net_edge is not None
        assert plan.net_edge < Decimal("0")

    def test_edge_that_survives_costs_is_accepted(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
            cost_model=ForexCostModel(commission_per_lot_round_turn=Decimal("6")),
            gross_edge=Decimal("200"),
        )
        assert plan.direction is TradeDirection.LONG
        assert plan.net_edge == Decimal("191.00")

    def test_swap_is_priced_when_an_expected_close_is_given(self, clock: SessionClock) -> None:
        plan = plan_trade(
            instrument=make_instrument(swap_long=Decimal("-2.5")),
            side=OrderSide.BUY,
            entry_price=ENTRY,
            stop_loss=LONG_STOP,
            account_risk=Decimal("100"),
            clock=clock,
            now=MONDAY_NOON,
            cost_model=ForexCostModel(),
            expected_close=MONDAY_NOON + timedelta(days=1),
        )
        assert plan.costs is not None
        assert plan.costs.swap == Decimal("1.25")


class TestIntrabarExits:
    def test_long_stop_hit(self) -> None:
        result = evaluate_intrabar_exit(
            make_bar(high="1.10100", low="1.09700"), OrderSide.BUY, stop_loss=LONG_STOP
        )
        assert result.outcome is IntrabarOutcome.STOP_LOSS
        assert result.price == LONG_STOP
        assert not result.gapped

    def test_long_target_hit(self) -> None:
        result = evaluate_intrabar_exit(
            make_bar(high="1.10500", low="1.09900"),
            OrderSide.BUY,
            take_profit=Decimal("1.10400"),
        )
        assert result.outcome is IntrabarOutcome.TAKE_PROFIT
        assert result.price == Decimal("1.10400")

    def test_short_stop_hit(self) -> None:
        result = evaluate_intrabar_exit(
            make_bar(high="1.10300", low="1.09900"), OrderSide.SELL, stop_loss=SHORT_STOP
        )
        assert result.outcome is IntrabarOutcome.STOP_LOSS
        assert result.price == SHORT_STOP

    def test_short_target_hit(self) -> None:
        result = evaluate_intrabar_exit(
            make_bar(high="1.10050", low="1.09500"),
            OrderSide.SELL,
            take_profit=Decimal("1.09600"),
        )
        assert result.outcome is IntrabarOutcome.TAKE_PROFIT

    def test_neither_touched(self) -> None:
        result = evaluate_intrabar_exit(
            make_bar(high="1.10050", low="1.09950"),
            OrderSide.BUY,
            stop_loss=LONG_STOP,
            take_profit=Decimal("1.10400"),
        )
        assert result.outcome is IntrabarOutcome.NONE
        assert not bool(result)

    def test_both_touched_resolves_against_the_trade_and_is_flagged(self) -> None:
        result = evaluate_intrabar_exit(
            make_bar(high="1.10500", low="1.09700"),
            OrderSide.BUY,
            stop_loss=LONG_STOP,
            take_profit=Decimal("1.10400"),
        )
        assert result.outcome is IntrabarOutcome.STOP_LOSS
        assert result.ambiguous

    def test_unambiguous_stop_is_not_flagged(self) -> None:
        result = evaluate_intrabar_exit(
            make_bar(high="1.10050", low="1.09700"),
            OrderSide.BUY,
            stop_loss=LONG_STOP,
            take_profit=Decimal("1.10400"),
        )
        assert not result.ambiguous

    def test_gap_through_a_long_stop_fills_at_the_open(self) -> None:
        bar = make_bar(high="1.09750", low="1.09500", open_="1.09700", close="1.09600")
        result = evaluate_intrabar_exit(bar, OrderSide.BUY, stop_loss=LONG_STOP)
        assert result.gapped
        assert result.price == Decimal("1.09700")

    def test_gap_through_a_short_stop_fills_at_the_open(self) -> None:
        bar = make_bar(high="1.10500", low="1.10250", open_="1.10300", close="1.10400")
        result = evaluate_intrabar_exit(bar, OrderSide.SELL, stop_loss=SHORT_STOP)
        assert result.gapped
        assert result.price == Decimal("1.10300")

    def test_no_levels_means_no_exit(self) -> None:
        result = evaluate_intrabar_exit(make_bar(high="1.2", low="1.0"), OrderSide.BUY)
        assert result.outcome is IntrabarOutcome.NONE


class TestPositionExits:
    def position(self, **overrides: object) -> ForexPosition:
        kwargs: dict[str, object] = {
            "ticket": 1,
            "symbol": "EURUSD+",
            "side": OrderSide.BUY,
            "lots": Decimal("0.5"),
            "entry_price": ENTRY,
            "current_price": ENTRY,
            "opened_at": MONDAY_NOON,
            "stop_loss": LONG_STOP,
        }
        kwargs.update(overrides)
        return ForexPosition(**kwargs)  # type: ignore[arg-type]

    def test_uses_the_positions_own_levels(self) -> None:
        result = evaluate_position_exit(make_bar(high="1.10100", low="1.09700"), self.position())
        assert result.outcome is IntrabarOutcome.STOP_LOSS

    def test_symbol_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            evaluate_position_exit(
                make_bar(high="1.10100", low="1.09700"), self.position(symbol="GBPUSD+")
            )
