"""FX instrument metadata: pip/point maths, trade modes, majors prioritisation."""

from __future__ import annotations

from datetime import time
from decimal import Decimal

import pytest

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import OrderSide
from quantflow.forex.instruments import (
    MAJORS,
    ForexInstrument,
    TradeMode,
    mt5_weekday_to_python,
    normalise_symbol,
    prioritise_symbols,
    python_weekday_to_mt5,
)
from quantflow.forex.sessions import SessionWindow


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
        "tick_value": Decimal("0.68"),
    }
    kwargs.update(overrides)
    return make_instrument(**kwargs)


class TestValidation:
    def test_rejects_non_positive_contract_size(self) -> None:
        with pytest.raises(ValidationError):
            make_instrument(contract_size=Decimal("0"))

    def test_rejects_min_above_max(self) -> None:
        with pytest.raises(ValidationError):
            make_instrument(min_lot=Decimal("10"), max_lot=Decimal("1"))

    def test_rejects_non_positive_lot_step(self) -> None:
        with pytest.raises(ValidationError):
            make_instrument(lot_step=Decimal("0"))

    def test_rejects_negative_digits(self) -> None:
        with pytest.raises(ValidationError):
            make_instrument(digits=-1)

    def test_rejects_empty_symbol(self) -> None:
        with pytest.raises(ValidationError):
            make_instrument(symbol="  ")

    def test_is_frozen(self) -> None:
        instrument = make_instrument()
        with pytest.raises(AttributeError):
            instrument.min_lot = Decimal("1")  # type: ignore[misc]


class TestPipMaths:
    def test_five_digit_pair_pip_is_ten_points(self) -> None:
        instrument = make_instrument()
        assert instrument.pip_size == Decimal("0.0001")
        assert instrument.points_per_pip == Decimal("10")
        assert not instrument.is_jpy_quoted

    def test_jpy_pair_pip_is_ten_points_at_three_digits(self) -> None:
        instrument = make_jpy()
        assert instrument.pip_size == Decimal("0.01")
        assert instrument.points_per_pip == Decimal("10")
        assert instrument.is_jpy_quoted

    def test_four_digit_pair_pip_equals_point(self) -> None:
        instrument = make_instrument(digits=4, point=Decimal("0.0001"), tick_size=Decimal("0.0001"))
        assert instrument.pip_size == Decimal("0.0001")
        assert instrument.points_per_pip == Decimal("1")

    def test_value_per_point_uses_tick_value_and_tick_size(self) -> None:
        instrument = make_instrument(tick_size=Decimal("0.00002"), tick_value=Decimal("2"))
        assert instrument.value_per_point_per_lot == Decimal("1")

    def test_value_per_point_falls_back_to_contract_size(self) -> None:
        instrument = make_instrument(tick_value=Decimal("0"))
        assert instrument.value_per_point_per_lot == Decimal("1")

    def test_pip_value_per_lot(self) -> None:
        assert make_instrument().pip_value_per_lot == Decimal("10")

    def test_price_and_point_conversions_round_trip(self) -> None:
        instrument = make_instrument()
        assert instrument.price_to_points(Decimal("0.00250")) == Decimal("250")
        assert instrument.points_to_price(Decimal("250")) == Decimal("0.00250")

    def test_price_to_points_is_absolute(self) -> None:
        instrument = make_instrument()
        assert instrument.price_to_points(Decimal("-0.00250")) == Decimal("250")

    def test_round_price_snaps_to_tick_grid(self) -> None:
        instrument = make_jpy()
        assert instrument.round_price(Decimal("157.123456")) == Decimal("157.123")


class TestContractSizeAndMargin:
    def test_notional_uses_contract_size(self) -> None:
        instrument = make_instrument()
        assert instrument.notional(Decimal("2"), Decimal("1.1")) == Decimal("220000.0")

    def test_leverage_is_none_when_margin_rate_unknown(self) -> None:
        assert make_instrument().leverage is None

    def test_leverage_derived_from_margin_rate(self) -> None:
        instrument = make_instrument(margin_rate=Decimal("0.02"))
        assert instrument.leverage == Decimal("50")

    def test_margin_rate_must_be_a_fraction(self) -> None:
        with pytest.raises(ValidationError):
            make_instrument(margin_rate=Decimal("1.5"))


class TestLotQuantisation:
    def test_snaps_down_to_lot_step_grid(self) -> None:
        instrument = make_instrument()
        assert instrument.quantise_lots(Decimal("0.1749")) == Decimal("0.17")

    def test_respects_non_multiple_min_lot(self) -> None:
        instrument = make_instrument(min_lot=Decimal("0.03"), lot_step=Decimal("0.02"))
        assert instrument.quantise_lots(Decimal("0.08")) == Decimal("0.07")

    def test_clamps_to_max_lot(self) -> None:
        instrument = make_instrument(max_lot=Decimal("2"))
        assert instrument.quantise_lots(Decimal("500")) == Decimal("2")

    def test_below_min_lot_returns_zero(self) -> None:
        instrument = make_instrument()
        assert instrument.quantise_lots(Decimal("0.004")) == Decimal("0")


class TestTradeMode:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0, TradeMode.DISABLED),
            (1, TradeMode.LONG_ONLY),
            (2, TradeMode.SHORT_ONLY),
            (3, TradeMode.CLOSE_ONLY),
            (4, TradeMode.FULL),
        ],
    )
    def test_from_mt5(self, code: int, expected: TradeMode) -> None:
        assert TradeMode.from_mt5(code) is expected

    def test_unknown_code_is_disabled(self) -> None:
        assert TradeMode.from_mt5(99) is TradeMode.DISABLED

    def test_long_only_blocks_sell(self) -> None:
        instrument = make_instrument(trade_mode=TradeMode.LONG_ONLY)
        assert instrument.can_trade(OrderSide.BUY)
        assert not instrument.can_trade(OrderSide.SELL)

    def test_short_only_blocks_buy(self) -> None:
        instrument = make_instrument(trade_mode=TradeMode.SHORT_ONLY)
        assert not instrument.can_trade(OrderSide.BUY)
        assert instrument.can_trade(OrderSide.SELL)

    def test_close_only_blocks_both(self) -> None:
        instrument = make_instrument(trade_mode=TradeMode.CLOSE_ONLY)
        assert not instrument.can_trade(OrderSide.BUY)
        assert not instrument.can_trade(OrderSide.SELL)

    def test_untradable_flag_overrides_full_mode(self) -> None:
        instrument = make_instrument(trade_mode=TradeMode.FULL, tradable=False)
        assert not instrument.can_trade(OrderSide.BUY)


class TestSessionsField:
    def test_sessions_default_to_empty(self) -> None:
        assert make_instrument().sessions == ()

    def test_sessions_are_carried_verbatim(self) -> None:
        windows = (SessionWindow(weekday=0, start=time(0, 0), end=time(23, 59)),)
        assert make_instrument(sessions=windows).sessions == windows


class TestSwapMetadata:
    def test_swap_defaults_are_zero(self) -> None:
        instrument = make_instrument()
        assert instrument.swap_long == Decimal("0")
        assert instrument.swap_short == Decimal("0")

    def test_default_triple_swap_day_is_wednesday_python_convention(self) -> None:
        assert make_instrument().triple_swap_weekday == 2

    def test_triple_swap_weekday_validated(self) -> None:
        with pytest.raises(ValidationError):
            make_instrument(triple_swap_weekday=9)


class TestWeekdayConversion:
    def test_mt5_wednesday_maps_to_python_wednesday(self) -> None:
        assert mt5_weekday_to_python(3) == 2

    def test_mt5_sunday_maps_to_python_sunday(self) -> None:
        assert mt5_weekday_to_python(0) == 6

    def test_round_trip(self) -> None:
        for day in range(7):
            assert python_weekday_to_mt5(mt5_weekday_to_python(day)) == day

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            mt5_weekday_to_python(7)


class TestMajorsPrioritisation:
    def test_majors_ordering_is_the_documented_seven(self) -> None:
        assert MAJORS == (
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
        )

    def test_prioritise_puts_majors_first_in_majors_order(self) -> None:
        discovered = ["XAUUSD+", "NZDUSD+", "EURUSD+", "USDJPY+"]
        assert prioritise_symbols(discovered) == (
            "EURUSD+",
            "USDJPY+",
            "NZDUSD+",
            "XAUUSD+",
        )

    def test_prioritise_never_invents_symbols(self) -> None:
        discovered = ["EURUSD+"]
        assert prioritise_symbols(discovered) == ("EURUSD+",)

    def test_prioritise_of_nothing_is_nothing(self) -> None:
        assert prioritise_symbols([]) == ()

    def test_non_majors_keep_a_stable_alphabetical_order(self) -> None:
        assert prioritise_symbols(["ZARJPY", "AUDNZD"]) == ("AUDNZD", "ZARJPY")

    def test_duplicates_collapse(self) -> None:
        assert prioritise_symbols(["EURUSD+", "EURUSD+"]) == ("EURUSD+",)

    def test_normalise_strips_venue_suffixes(self) -> None:
        assert normalise_symbol("EURUSD+") == "EURUSD"
        assert normalise_symbol("eurusd.raw") == "EURUSD"
        assert normalise_symbol("EUR/USD") == "EURUSD"
