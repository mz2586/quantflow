"""FX cost model: spread, commission, slippage, swap and the triple-swap day."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import OrderSide
from quantflow.forex.costs import (
    BYBIT_TIGHT_SPREAD_COMMISSION_PER_LOT_ROUND_TURN,
    ForexCostModel,
    TradeCosts,
    expected_net_edge,
    swap_nights,
)
from quantflow.forex.instruments import ForexInstrument

MONDAY = 10
WEDNESDAY = 12
THURSDAY = 13
FRIDAY = 14
NEXT_MONDAY = 17


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=UTC)


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
        "swap_long": Decimal("-2.5"),
        "swap_short": Decimal("0.8"),
    }
    kwargs.update(overrides)
    return ForexInstrument(**kwargs)  # type: ignore[arg-type]


@pytest.fixture
def model() -> ForexCostModel:
    return ForexCostModel(
        commission_per_lot_round_turn=Decimal("6"),
        slippage_points=Decimal("5"),
    )


class TestDocumentedCommission:
    def test_bybit_figure_is_exposed_as_a_reference_not_a_default(self) -> None:
        assert Decimal("6") == BYBIT_TIGHT_SPREAD_COMMISSION_PER_LOT_ROUND_TURN
        assert ForexCostModel().commission_per_lot_round_turn == Decimal("0")

    def test_commission_is_configurable(self) -> None:
        model = ForexCostModel(commission_per_lot_round_turn=Decimal("3.5"))
        assert model.commission_cost(Decimal("2")) == Decimal("7.0")

    def test_negative_commission_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForexCostModel(commission_per_lot_round_turn=Decimal("-1"))


class TestSpread:
    def test_spread_cost_from_instrument_points(self, model: ForexCostModel) -> None:
        assert model.spread_cost(make_instrument(), Decimal("1")) == Decimal("12")

    def test_spread_scales_with_lots(self, model: ForexCostModel) -> None:
        assert model.spread_cost(make_instrument(), Decimal("2.5")) == Decimal("30")

    def test_spread_override_wins(self) -> None:
        model = ForexCostModel(spread_points_override=Decimal("30"))
        assert model.spread_cost(make_instrument(), Decimal("1")) == Decimal("30")

    def test_jpy_pair_spread_uses_its_own_point_value(self, model: ForexCostModel) -> None:
        jpy = make_instrument(
            symbol="USDJPY+",
            quote="JPY",
            digits=3,
            point=Decimal("0.001"),
            tick_size=Decimal("0.001"),
            tick_value=Decimal("0.65"),
            spread_points=Decimal("20"),
        )
        assert model.spread_cost(jpy, Decimal("1")) == Decimal("13.00")


class TestSlippage:
    def test_slippage_cost(self, model: ForexCostModel) -> None:
        assert model.slippage_cost(make_instrument(), Decimal("2")) == Decimal("10")

    def test_zero_by_default(self) -> None:
        assert ForexCostModel().slippage_cost(make_instrument(), Decimal("2")) == Decimal("0")


class TestSwapNights:
    def test_intraday_position_crosses_no_rollover(self) -> None:
        assert swap_nights(at(MONDAY, 10), at(MONDAY, 20), triple_swap_weekday=2) == Decimal("0")

    def test_one_overnight(self) -> None:
        assert swap_nights(at(MONDAY, 10), at(MONDAY + 1, 10), triple_swap_weekday=2) == Decimal(
            "1"
        )

    def test_wednesday_rollover_counts_triple(self) -> None:
        assert swap_nights(at(WEDNESDAY, 10), at(THURSDAY, 10), triple_swap_weekday=2) == Decimal(
            "3"
        )

    def test_monday_to_thursday_includes_the_triple_day(self) -> None:
        assert swap_nights(at(MONDAY, 10), at(THURSDAY, 10), triple_swap_weekday=2) == Decimal("5")

    def test_weekend_is_not_charged_again(self) -> None:
        assert swap_nights(at(FRIDAY, 10), at(NEXT_MONDAY, 10), triple_swap_weekday=2) == Decimal(
            "0"
        )

    def test_configurable_triple_day(self) -> None:
        assert swap_nights(at(FRIDAY, 10), at(NEXT_MONDAY, 10), triple_swap_weekday=4) == Decimal(
            "3"
        )

    def test_rollover_instant_is_inclusive_at_close(self) -> None:
        assert swap_nights(at(MONDAY, 10), at(MONDAY, 21), triple_swap_weekday=2) == Decimal("1")

    def test_open_exactly_on_rollover_is_not_charged_for_it(self) -> None:
        assert swap_nights(at(MONDAY, 21), at(MONDAY + 1, 20), triple_swap_weekday=2) == Decimal(
            "0"
        )

    def test_close_before_open_rejected(self) -> None:
        with pytest.raises(ValidationError):
            swap_nights(at(THURSDAY, 10), at(MONDAY, 10), triple_swap_weekday=2)

    def test_naive_datetimes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            swap_nights(
                datetime(2026, 8, 10, 10),  # noqa: DTZ001
                at(THURSDAY, 10),
                triple_swap_weekday=2,
            )


class TestSwapCost:
    def test_negative_swap_rate_is_a_cost(self, model: ForexCostModel) -> None:
        cost = model.swap_cost(
            make_instrument(), Decimal("1"), OrderSide.BUY, at(MONDAY, 10), at(MONDAY + 1, 10)
        )
        assert cost == Decimal("2.5")

    def test_positive_swap_rate_is_a_credit(self, model: ForexCostModel) -> None:
        cost = model.swap_cost(
            make_instrument(), Decimal("1"), OrderSide.SELL, at(MONDAY, 10), at(MONDAY + 1, 10)
        )
        assert cost == Decimal("-0.8")

    def test_triple_swap_day_triples_the_charge(self, model: ForexCostModel) -> None:
        cost = model.swap_cost(
            make_instrument(), Decimal("1"), OrderSide.BUY, at(WEDNESDAY, 10), at(THURSDAY, 10)
        )
        assert cost == Decimal("7.5")

    def test_swap_scales_with_lots(self, model: ForexCostModel) -> None:
        cost = model.swap_cost(
            make_instrument(), Decimal("4"), OrderSide.BUY, at(MONDAY, 10), at(MONDAY + 1, 10)
        )
        assert cost == Decimal("10.0")

    def test_swap_can_be_disabled(self) -> None:
        model = ForexCostModel(include_swap=False)
        cost = model.swap_cost(
            make_instrument(), Decimal("1"), OrderSide.BUY, at(MONDAY, 10), at(THURSDAY, 10)
        )
        assert cost == Decimal("0")

    def test_uses_the_instruments_own_triple_swap_weekday(self, model: ForexCostModel) -> None:
        instrument = make_instrument(triple_swap_weekday=4)
        cost = model.swap_cost(
            instrument, Decimal("1"), OrderSide.BUY, at(FRIDAY, 10), at(NEXT_MONDAY, 10)
        )
        assert cost == Decimal("7.5")


class TestEstimate:
    def test_full_breakdown_for_an_overnight_long(self, model: ForexCostModel) -> None:
        costs = model.estimate(
            make_instrument(),
            Decimal("1"),
            OrderSide.BUY,
            opened_at=at(MONDAY, 10),
            closed_at=at(MONDAY + 1, 10),
        )
        assert costs.spread == Decimal("12")
        assert costs.commission == Decimal("6")
        assert costs.slippage == Decimal("5")
        assert costs.swap == Decimal("2.5")
        assert costs.total == Decimal("25.5")

    def test_intraday_trade_has_no_swap(self, model: ForexCostModel) -> None:
        costs = model.estimate(
            make_instrument(),
            Decimal("1"),
            OrderSide.BUY,
            opened_at=at(MONDAY, 10),
            closed_at=at(MONDAY, 16),
        )
        assert costs.swap == Decimal("0")
        assert costs.total == Decimal("23")

    def test_holding_dates_are_optional(self, model: ForexCostModel) -> None:
        costs = model.estimate(make_instrument(), Decimal("1"), OrderSide.BUY)
        assert costs.swap == Decimal("0")
        assert costs.total == Decimal("23")

    def test_costs_scale_with_size(self, model: ForexCostModel) -> None:
        one = model.estimate(make_instrument(), Decimal("1"), OrderSide.BUY)
        two = model.estimate(make_instrument(), Decimal("2"), OrderSide.BUY)
        assert two.total == one.total * 2

    def test_rejects_non_positive_lots(self, model: ForexCostModel) -> None:
        with pytest.raises(ValidationError):
            model.estimate(make_instrument(), Decimal("0"), OrderSide.BUY)


class TestNetEdge:
    def test_net_edge_is_gross_minus_all_costs(self, model: ForexCostModel) -> None:
        costs = model.estimate(make_instrument(), Decimal("1"), OrderSide.BUY)
        assert costs.net_edge(Decimal("100")) == Decimal("77")

    def test_edge_can_go_negative_after_costs(self, model: ForexCostModel) -> None:
        costs = model.estimate(make_instrument(), Decimal("1"), OrderSide.BUY)
        assert costs.net_edge(Decimal("10")) == Decimal("-13")

    def test_module_level_helper_matches(self, model: ForexCostModel) -> None:
        costs = model.estimate(make_instrument(), Decimal("1"), OrderSide.BUY)
        assert expected_net_edge(Decimal("100"), costs) == costs.net_edge(Decimal("100"))

    def test_swap_credit_improves_the_edge(self, model: ForexCostModel) -> None:
        held = model.estimate(
            make_instrument(),
            Decimal("1"),
            OrderSide.SELL,
            opened_at=at(MONDAY, 10),
            closed_at=at(MONDAY + 1, 10),
        )
        assert held.swap == Decimal("-0.8")
        assert held.net_edge(Decimal("100")) == Decimal("77.8")

    def test_breakdown_totals_are_self_consistent(self) -> None:
        costs = TradeCosts(
            spread=Decimal("1"),
            commission=Decimal("2"),
            slippage=Decimal("3"),
            swap=Decimal("4"),
        )
        assert costs.total == Decimal("10")
        assert costs.net_edge(Decimal("10")) == Decimal("0")
