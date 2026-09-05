"""FX lot sizing — the piece that decides how much money is actually at risk."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import OrderSide
from quantflow.forex.instruments import ForexInstrument, TradeMode
from quantflow.forex.sizing import (
    LotSizingResult,
    SizingRejection,
    lots_for_risk,
    lots_for_risk_from_prices,
    margin_required,
    pip_value,
    risk_for_lots,
    stop_distance_points,
    value_per_point,
)


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
    }
    kwargs.update(overrides)
    return ForexInstrument(**kwargs)  # type: ignore[arg-type]


def make_jpy(**overrides: object) -> ForexInstrument:
    kwargs: dict[str, object] = {
        "symbol": "USDJPY+",
        "base": "USD",
        "quote": "JPY",
        "digits": 3,
        "point": Decimal("0.001"),
        "tick_size": Decimal("0.001"),
        "tick_value": Decimal("0.65"),
    }
    kwargs.update(overrides)
    return make_instrument(**kwargs)


def size(
    instrument: ForexInstrument,
    risk: str,
    stop_points: str,
) -> LotSizingResult:
    return lots_for_risk(
        Decimal(risk),
        Decimal(stop_points),
        instrument.tick_value,
        instrument.contract_size,
        instrument,
    )


class TestValuePerPoint:
    def test_standard_five_digit_pair(self) -> None:
        assert value_per_point(make_instrument()) == Decimal("1")

    def test_scales_with_tick_size(self) -> None:
        instrument = make_instrument(tick_size=Decimal("0.00005"), tick_value=Decimal("5"))
        assert value_per_point(instrument) == Decimal("1")

    def test_explicit_overrides_win(self) -> None:
        instrument = make_instrument()
        assert value_per_point(instrument, tick_value=Decimal("2")) == Decimal("2")

    def test_falls_back_to_contract_size_when_tick_value_missing(self) -> None:
        instrument = make_instrument(tick_value=Decimal("0"))
        assert value_per_point(instrument, contract_size=Decimal("100000")) == Decimal("1")


class TestLongSizing:
    def test_basic_case(self) -> None:
        result = size(make_instrument(), "100", "200")
        assert result.accepted
        assert result.lots == Decimal("0.50")
        assert result.reason is None

    def test_projected_risk_matches_the_budget(self) -> None:
        result = size(make_instrument(), "100", "200")
        assert result.projected_risk == Decimal("100.00")

    def test_projected_risk_never_exceeds_the_budget(self) -> None:
        for risk in ("37", "101.37", "9.99", "5000"):
            result = size(make_instrument(), risk, "173")
            if result.accepted:
                assert result.projected_risk <= Decimal(risk)

    def test_result_is_truthy_when_accepted(self) -> None:
        assert bool(size(make_instrument(), "100", "200")) is True

    def test_bigger_stop_means_smaller_size(self) -> None:
        tight = size(make_instrument(), "100", "100")
        wide = size(make_instrument(), "100", "400")
        assert tight.lots > wide.lots


class TestShortSizingSymmetry:
    def test_short_and_long_size_identically_for_the_same_stop_distance(self) -> None:
        instrument = make_instrument()
        entry = Decimal("1.10000")
        long_result = lots_for_risk_from_prices(
            Decimal("100"), entry, Decimal("1.09800"), instrument, OrderSide.BUY
        )
        short_result = lots_for_risk_from_prices(
            Decimal("100"), entry, Decimal("1.10200"), instrument, OrderSide.SELL
        )
        assert long_result.accepted
        assert short_result.accepted
        assert long_result.lots == short_result.lots == Decimal("0.50")

    def test_long_stop_above_entry_is_rejected(self) -> None:
        result = lots_for_risk_from_prices(
            Decimal("100"),
            Decimal("1.10000"),
            Decimal("1.10200"),
            make_instrument(),
            OrderSide.BUY,
        )
        assert not result.accepted
        assert result.reason is SizingRejection.STOP_WRONG_SIDE

    def test_short_stop_below_entry_is_rejected(self) -> None:
        result = lots_for_risk_from_prices(
            Decimal("100"),
            Decimal("1.10000"),
            Decimal("1.09800"),
            make_instrument(),
            OrderSide.SELL,
        )
        assert not result.accepted
        assert result.reason is SizingRejection.STOP_WRONG_SIDE

    def test_short_blocked_on_long_only_instrument(self) -> None:
        instrument = make_instrument(trade_mode=TradeMode.LONG_ONLY)
        result = lots_for_risk_from_prices(
            Decimal("100"),
            Decimal("1.10000"),
            Decimal("1.10200"),
            instrument,
            OrderSide.SELL,
        )
        assert not result.accepted
        assert result.reason is SizingRejection.SIDE_NOT_ALLOWED


class TestJpyPairs:
    def test_jpy_pair_uses_its_own_point_scale(self) -> None:
        result = size(make_jpy(), "100", "300")
        assert result.accepted
        assert result.value_per_point == Decimal("0.65")
        assert result.lots == Decimal("0.51")

    def test_stop_distance_in_points_differs_from_a_five_digit_pair(self) -> None:
        jpy = make_jpy()
        eur = make_instrument()
        assert stop_distance_points(Decimal("157.500"), Decimal("157.200"), jpy) == Decimal("300")
        assert stop_distance_points(Decimal("1.10000"), Decimal("1.09700"), eur) == Decimal("300")

    def test_thirty_pips_on_a_jpy_pair(self) -> None:
        jpy = make_jpy()
        distance = stop_distance_points(Decimal("157.500"), Decimal("157.200"), jpy)
        assert distance / jpy.points_per_pip == Decimal("30")

    def test_pip_value_for_jpy_pair(self) -> None:
        assert pip_value(make_jpy(), Decimal("1")) == Decimal("6.50")


class TestLotStepRounding:
    def test_rounds_down_to_the_step(self) -> None:
        instrument = make_instrument(lot_step=Decimal("0.10"), min_lot=Decimal("0.10"))
        result = size(instrument, "100", "220")
        assert result.raw_lots > Decimal("0.4")
        assert result.lots == Decimal("0.40")

    def test_honours_a_coarse_step(self) -> None:
        instrument = make_instrument(min_lot=Decimal("1"), lot_step=Decimal("1"))
        result = size(instrument, "1000", "300")
        assert result.lots == Decimal("3")

    def test_offset_grid_anchored_on_min_lot(self) -> None:
        instrument = make_instrument(min_lot=Decimal("0.03"), lot_step=Decimal("0.02"))
        result = size(instrument, "100", "1000")
        # raw = 0.10 -> grid is 0.03, 0.05, 0.07, 0.09 -> 0.09
        assert result.lots == Decimal("0.09")

    def test_rounding_is_never_upward(self) -> None:
        instrument = make_instrument()
        result = size(instrument, "100", "199")
        assert result.lots <= result.raw_lots


class TestClamps:
    def test_clamped_to_max_lot(self) -> None:
        instrument = make_instrument(max_lot=Decimal("2"))
        result = size(instrument, "100000", "100")
        assert result.accepted
        assert result.lots == Decimal("2")
        assert result.clamped_to_max

    def test_not_flagged_as_clamped_when_under_max(self) -> None:
        assert not size(make_instrument(), "100", "200").clamped_to_max

    def test_result_sits_on_the_lot_grid_after_clamping(self) -> None:
        instrument = make_instrument(
            min_lot=Decimal("0.03"), lot_step=Decimal("0.02"), max_lot=Decimal("0.10")
        )
        result = size(instrument, "100000", "100")
        assert result.lots == Decimal("0.09")


class TestSubMinimumRejection:
    def test_sub_minimum_is_rejected_with_a_reason(self) -> None:
        result = size(make_instrument(), "1", "1000")
        assert not result.accepted
        assert result.lots == Decimal("0")
        assert result.reason is SizingRejection.BELOW_MIN_LOT

    def test_rejection_message_names_the_shortfall(self) -> None:
        result = size(make_instrument(), "1", "1000")
        assert result.message is not None
        assert "0.01" in result.message

    def test_rejected_result_is_falsy(self) -> None:
        assert bool(size(make_instrument(), "1", "1000")) is False

    def test_zero_risk_is_rejected(self) -> None:
        result = size(make_instrument(), "0", "200")
        assert result.reason is SizingRejection.NON_POSITIVE_RISK

    def test_negative_risk_is_rejected(self) -> None:
        result = size(make_instrument(), "-50", "200")
        assert result.reason is SizingRejection.NON_POSITIVE_RISK

    def test_zero_stop_distance_is_rejected(self) -> None:
        result = size(make_instrument(), "100", "0")
        assert result.reason is SizingRejection.NON_POSITIVE_STOP

    def test_untradable_instrument_is_rejected(self) -> None:
        result = size(make_instrument(tradable=False), "100", "200")
        assert result.reason is SizingRejection.INSTRUMENT_NOT_TRADABLE

    def test_zero_point_value_is_rejected(self) -> None:
        instrument = make_instrument(tick_value=Decimal("0"), contract_size=Decimal("100000"))
        result = lots_for_risk(
            Decimal("100"), Decimal("200"), Decimal("0"), Decimal("0"), instrument
        )
        assert result.reason is SizingRejection.ZERO_POINT_VALUE


class TestNonZeroGuarantee:
    @pytest.mark.parametrize(
        ("risk", "stop_points"),
        [
            ("10", "1000"),
            ("10.01", "1000"),
            ("10.99", "1000"),
            ("0.11", "10"),
            ("1000", "3"),
        ],
    )
    def test_any_size_at_or_above_minimum_never_collapses_to_zero(
        self, risk: str, stop_points: str
    ) -> None:
        instrument = make_instrument()
        result = size(instrument, risk, stop_points)
        assert result.accepted
        assert result.lots >= instrument.min_lot
        assert result.lots > Decimal("0")

    def test_exactly_minimum_lot_is_accepted(self) -> None:
        # risk that buys exactly 0.01 lots over a 1000-point stop
        result = size(make_instrument(), "10", "1000")
        assert result.accepted
        assert result.lots == Decimal("0.01")

    def test_a_hair_under_minimum_is_rejected_not_silently_zeroed(self) -> None:
        result = size(make_instrument(), "9.99", "1000")
        assert not result.accepted
        assert result.reason is SizingRejection.BELOW_MIN_LOT

    def test_accepted_result_always_sits_on_or_above_min_lot(self) -> None:
        instrument = make_instrument(min_lot=Decimal("0.03"), lot_step=Decimal("0.02"))
        result = size(instrument, "31", "1000")
        assert result.accepted
        assert result.lots >= instrument.min_lot


class TestRiskForLots:
    def test_round_trips_with_sizing(self) -> None:
        instrument = make_instrument()
        result = size(instrument, "100", "200")
        assert risk_for_lots(result.lots, Decimal("200"), instrument) == Decimal("100.00")

    def test_rejects_negative_lots(self) -> None:
        with pytest.raises(ValidationError):
            risk_for_lots(Decimal("-1"), Decimal("200"), make_instrument())


class TestMargin:
    def test_margin_required_from_margin_rate(self) -> None:
        instrument = make_instrument(margin_rate=Decimal("0.02"))
        assert margin_required(instrument, Decimal("1"), Decimal("1.10")) == Decimal("2200.00")

    def test_margin_unknown_without_a_rate(self) -> None:
        assert margin_required(make_instrument(), Decimal("1"), Decimal("1.10")) is None


class TestNotACryptoFormula:
    def test_sizing_does_not_use_notional_divided_by_price(self) -> None:
        # The crypto formula (risk / (price * stop_pct)) would give a wildly different
        # answer; FX sizing must go through value-per-point, not notional/price.
        instrument = make_instrument()
        result = size(instrument, "100", "200")
        crypto_style = Decimal("100") / (Decimal("1.10") * Decimal("0.002"))
        assert result.lots != crypto_style
        assert result.lots == Decimal("0.50")
