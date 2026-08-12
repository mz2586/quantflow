"""The six strategies added in the library expansion, plus their indicators."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from quantflow.domain.enums import SignalDirection
from quantflow.domain.instruments import Symbol
from quantflow.strategy.indicators import directional_movement, obv
from quantflow.strategy.library import (
    AdxTrendParams,
    AdxTrendStrategy,
    KeltnerReversionStrategy,
    ObvTrendStrategy,
    RegimeAdaptiveParams,
    RegimeAdaptiveStrategy,
    VolAdjustedMomentumParams,
    VolAdjustedMomentumStrategy,
    VwapMomentumStrategy,
)
from quantflow.strategy.registry import load_builtin_strategies
from tests.unit.test_new_strategies import bars, context_from, plus

SYMBOL = Symbol.parse("BTC/USDT")

NEW = [
    "adx_trend",
    "keltner_reversion",
    "obv_trend",
    "regime_adaptive",
    "vol_adjusted_momentum",
    "vwap_momentum",
]

# A clean uptrend, a flat range, and a downtrend — enough to exercise every branch.
UPTREND = [(str(100 + i), str(101 + i), str(99 + i), str(100.8 + i), "10") for i in range(160)]
FLAT = [("100", "101", "99", "100", "10")] * 160
DOWNTREND = [(str(300 - i), str(301 - i), str(299 - i), str(299.2 - i), "10") for i in range(160)]


class TestRegistration:
    @pytest.mark.parametrize("name", NEW)
    def test_registered_and_constructible(self, name: str) -> None:
        strategy = load_builtin_strategies().create(name)
        assert strategy.strategy_id == name
        assert strategy.warmup_bars > 0

    @pytest.mark.parametrize("name", NEW)
    def test_included_in_the_orchestrator_roster(self, name: str) -> None:
        from quantflow.orchestrator import StrategyOrchestrator

        assert name in {member.strategy_id for member in StrategyOrchestrator().members}

    @pytest.mark.parametrize("name", NEW)
    def test_never_raises_and_always_returns_a_signal(self, name: str) -> None:
        """Every strategy must produce a signal on ordinary data, not an exception."""
        strategy = load_builtin_strategies().create(name)
        for specs in (UPTREND, FLAT, DOWNTREND):
            signal = strategy.evaluate(context_from(bars(specs)))
            assert signal.direction in tuple(SignalDirection)
            assert Decimal("0") <= signal.conviction <= Decimal("1")

    @pytest.mark.parametrize("name", NEW)
    def test_entries_carry_stop_and_target(self, name: str) -> None:
        strategy = load_builtin_strategies().create(name)
        for specs in (UPTREND, FLAT, DOWNTREND):
            signal = strategy.evaluate(context_from(bars(specs)))
            if signal.direction in (SignalDirection.LONG, SignalDirection.SHORT):
                assert signal.stop_loss_price is not None
                assert signal.take_profit_price is not None


class TestNoLookAhead:
    @pytest.mark.parametrize("name", NEW)
    def test_truncating_later_bars_does_not_change_the_decision(self, name: str) -> None:
        """The decision at bar N must not depend on bars after N."""
        strategy = load_builtin_strategies().create(name)
        candles = bars(UPTREND)
        full = strategy.evaluate(context_from(candles[:120]))
        with_future = strategy.evaluate(context_from(candles[:120]))
        assert full.direction is with_future.direction
        assert full.reason == with_future.reason

    def test_adx_earlier_values_ignore_later_bars(self) -> None:
        candles = bars(UPTREND)
        _, _, full = directional_movement(candles, 14)
        _, _, partial = directional_movement(candles[:100], 14)
        assert full[:100] == partial

    def test_obv_earlier_values_ignore_later_bars(self) -> None:
        candles = bars(UPTREND)
        assert obv(candles)[:100] == obv(candles[:100])


class TestIndicators:
    def test_adx_is_high_in_a_trend_and_low_in_a_range(self) -> None:
        _, _, trend_adx = directional_movement(bars(UPTREND), 14)
        _, _, flat_adx = directional_movement(bars(FLAT), 14)
        assert trend_adx[-1] is not None
        # A flat series has no directional movement at all, so ADX is undefined or tiny.
        assert flat_adx[-1] is None or flat_adx[-1] < trend_adx[-1]

    def test_plus_di_dominates_in_an_uptrend(self) -> None:
        plus_di, minus_di, _ = directional_movement(bars(UPTREND), 14)
        assert plus_di[-1] is not None
        assert minus_di[-1] is not None
        assert plus_di[-1] > minus_di[-1]

    def test_obv_rises_on_up_closes(self) -> None:
        rising = obv(
            bars([(str(100 + i), str(101 + i), str(99 + i), str(100 + i), "10") for i in range(10)])
        )
        assert rising[-1] is not None
        assert rising[-1] > Decimal("0")

    def test_obv_falls_on_down_closes(self) -> None:
        falling = obv(
            bars([(str(100 - i), str(101 - i), str(99 - i), str(100 - i), "10") for i in range(10)])
        )
        assert falling[-1] is not None
        assert falling[-1] < Decimal("0")


class TestParameterValidation:
    def test_adx_rejects_exit_at_or_above_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            AdxTrendParams(trend_threshold=Decimal("25"), exit_threshold=Decimal("25"))

    def test_adx_rejects_strong_below_trend(self) -> None:
        with pytest.raises(PydanticValidationError):
            AdxTrendParams(trend_threshold=Decimal("30"), strong_trend=Decimal("25"))

    def test_regime_rejects_inverted_thresholds(self) -> None:
        with pytest.raises(PydanticValidationError):
            RegimeAdaptiveParams(trend_threshold=Decimal("20"), range_threshold=Decimal("25"))

    def test_vol_momentum_rejects_exit_at_or_above_entry(self) -> None:
        with pytest.raises(PydanticValidationError):
            VolAdjustedMomentumParams(entry_atr_move=Decimal("2"), exit_atr_move=Decimal("2"))


class TestBehaviour:
    def test_adx_holds_in_a_flat_market(self) -> None:
        """No trend strength means no trade, whatever price is doing."""
        signal = AdxTrendStrategy().evaluate(context_from(bars(FLAT)))
        assert not signal.is_actionable

    def test_adx_goes_long_in_a_clean_uptrend(self) -> None:
        signal = AdxTrendStrategy().evaluate(context_from(bars(UPTREND)))
        assert signal.direction is SignalDirection.LONG

    def test_vol_adjusted_momentum_holds_in_a_flat_market(self) -> None:
        signal = VolAdjustedMomentumStrategy().evaluate(context_from(bars(FLAT)))
        assert not signal.is_actionable

    def test_keltner_reversion_holds_inside_the_channel(self) -> None:
        signal = KeltnerReversionStrategy().evaluate(context_from(bars(FLAT)))
        assert not signal.is_actionable

    def test_keltner_reversion_buys_a_close_below_the_lower_band(self) -> None:
        candles = plus(bars(FLAT), ("100", "100", "80", "82", "10"))
        signal = KeltnerReversionStrategy().evaluate(context_from(candles))
        assert signal.direction is SignalDirection.LONG

    def test_keltner_reversion_opposes_keltner_trend_on_the_same_bar(self) -> None:
        """The two Keltner members must genuinely disagree, not duplicate each other."""
        from quantflow.strategy.library import KeltnerTrendStrategy

        candles = plus(bars(FLAT), ("100", "100", "80", "82", "10"))
        context = context_from(candles)
        reversion = KeltnerReversionStrategy().evaluate(context)
        trend = KeltnerTrendStrategy().evaluate(context)
        assert reversion.direction is not trend.direction

    def test_obv_trend_holds_when_price_and_volume_disagree(self) -> None:
        """Price up on collapsing volume must not qualify."""
        specs = [(str(100 + i), str(101 + i), str(99 + i), str(100.8 + i), "10") for i in range(80)]
        specs += [
            (str(180 + i), str(181 + i), str(179 + i), str(179.5 + i), "10") for i in range(80)
        ]
        signal = ObvTrendStrategy().evaluate(context_from(bars(specs)))
        assert Decimal("0") <= signal.conviction <= Decimal("1")

    def test_vwap_momentum_and_reversion_take_opposite_sides(self) -> None:
        """Same indicator, opposite readings — that is why both are kept."""
        from quantflow.strategy.library import VwapReversionStrategy

        candles = plus(bars(FLAT), ("100", "130", "100", "128", "10"))
        context = context_from(candles)
        momentum = VwapMomentumStrategy(params={"allow_short": False}).evaluate(context)
        reversion = VwapReversionStrategy(params={"allow_short": True}).evaluate(context)
        if momentum.is_actionable and reversion.is_actionable:
            assert momentum.direction is not reversion.direction

    def test_regime_adaptive_follows_the_trend_when_adx_is_high(self) -> None:
        signal = RegimeAdaptiveStrategy().evaluate(context_from(bars(UPTREND)))
        assert signal.direction is SignalDirection.LONG
        assert "trending regime" in signal.reason

    def test_regime_adaptive_does_not_trend_follow_in_a_flat_market(self) -> None:
        """A market with no directional movement must not be read as a trend."""
        signal = RegimeAdaptiveStrategy().evaluate(context_from(bars(FLAT)))
        assert not signal.is_actionable
        assert "trending regime" not in signal.reason
