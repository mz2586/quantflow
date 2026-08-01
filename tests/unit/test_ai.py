"""AI engine: regime detection and the risk-reducing-only decision contract.

The class :class:`TestAICannotIncreaseRisk` is the most important in this file. The AI is
allowed to veto a trade or make it smaller; it must be structurally incapable of making one
larger, creating one, or removing its protection. A model that can only reduce exposure has
a bounded worst case — the worst it can do is stop the system trading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal

import pytest

from quantflow.ai.decision import (
    AIAdvice,
    AIDecisionEngine,
    RegimeAdvisor,
    assert_risk_reducing,
    build_engine,
)
from quantflow.ai.regime import (
    HIGH_VOLATILITY_THRESHOLD,
    MIN_BARS,
    GaussianMixtureRegimeDetector,
    RuleBasedRegimeDetector,
    build_detector,
    extract_features,
    regime_history,
    summarise,
)
from quantflow.ai.strategy import AIAugmentedStrategy, wrap
from quantflow.core.errors import InsufficientDataError, ValidationError
from quantflow.core.precision import ONE, ZERO
from quantflow.domain.enums import MarketRegime, SignalDirection, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.signals import Signal
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from tests.conftest import REFERENCE_TIME, make_candles


def series(symbol: Symbol, closes: Sequence[float | int]) -> list[Candle]:
    return make_candles(symbol, [Decimal(str(value)) for value in closes])


def volatile_series(symbol: Symbol, count: int = 120) -> list[Candle]:
    """Bars with large intrabar ranges, so normalised ATR clears the volatility threshold."""
    candles: list[Candle] = []
    price = Decimal("100")
    for index in range(count):
        swing = price * Decimal("0.08")
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=Timeframe.H1,
                open_time=REFERENCE_TIME + Timeframe.H1.delta * index,
                open=price,
                high=price + swing,
                low=price - swing,
                close=price,
                volume=Decimal("100"),
                quote_volume=Decimal("10000"),
            )
        )
    return candles


def context_for(symbol: Symbol, candles: list[Candle]) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        timeframe=Timeframe.H1,
        history=CandleSeries(candles),
        now=candles[-1].close_time,
        portfolio=PortfolioSnapshot(
            timestamp=candles[-1].close_time,
            base_currency="USDT",
            cash=Decimal("10000"),
            mark_prices={symbol: candles[-1].close},
        ),
    )


def signal_for(symbol: Symbol, direction: SignalDirection, conviction: str = "1") -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        timestamp=REFERENCE_TIME,
        strategy_id="test",
        conviction=Decimal(conviction),
        reference_price=Decimal("100"),
        stop_loss_price=Decimal("98") if direction is SignalDirection.LONG else Decimal("102"),
        reason="base signal",
    )


class TestFeatureExtraction:
    def test_requires_enough_history(self, btc: Symbol) -> None:
        with pytest.raises(InsufficientDataError, match="needs 60 bars"):
            extract_features(series(btc, [100.0] * 10))

    def test_uptrend_has_positive_trend_strength(self, btc: Symbol) -> None:
        features = extract_features(series(btc, [100 + index for index in range(120)]))
        assert features.trend_strength > ZERO
        assert features.directional_share > Decimal("0.9")

    def test_downtrend_has_negative_trend_strength(self, btc: Symbol) -> None:
        features = extract_features(series(btc, [200 - index for index in range(120)]))
        assert features.trend_strength < ZERO

    def test_flat_market_has_near_zero_trend(self, btc: Symbol) -> None:
        features = extract_features(series(btc, [100.0] * 120))
        assert abs(features.trend_strength) < Decimal("0.001")

    def test_feature_vector_shape(self, btc: Symbol) -> None:
        vector = extract_features(series(btc, [100 + index for index in range(120)])).as_vector()
        assert len(vector) == 5
        assert all(isinstance(value, float) for value in vector)


class TestRuleBasedDetector:
    def test_detects_a_bull_trend(self, btc: Symbol) -> None:
        detector = RuleBasedRegimeDetector()
        observation = detector.detect(series(btc, [100 * (1.01**index) for index in range(120)]))
        assert observation.regime is MarketRegime.BULL_TREND
        assert observation.confidence > ZERO
        assert observation.reason

    def test_detects_a_bear_trend(self, btc: Symbol) -> None:
        detector = RuleBasedRegimeDetector()
        observation = detector.detect(series(btc, [100 * (0.99**index) for index in range(120)]))
        assert observation.regime is MarketRegime.BEAR_TREND

    def test_detects_a_range(self, btc: Symbol) -> None:
        detector = RuleBasedRegimeDetector()
        closes = [100 + (index % 5) * 0.2 for index in range(120)]
        observation = detector.detect(series(btc, closes))
        assert observation.regime is MarketRegime.RANGE

    def test_volatility_dominates_the_trend_label(self, btc: Symbol) -> None:
        # In a violent market the trend label is unreliable and sizing should shrink
        # regardless of direction, so volatility is checked first.
        detector = RuleBasedRegimeDetector()
        observation = detector.detect(volatile_series(btc))
        assert observation.regime is MarketRegime.HIGH_VOLATILITY
        assert observation.features.normalized_volatility >= HIGH_VOLATILITY_THRESHOLD

    def test_confidence_is_bounded(self, btc: Symbol) -> None:
        detector = RuleBasedRegimeDetector()
        for closes in (
            [100 * (1.02**index) for index in range(120)],
            [100.0] * 120,
            [100 + (index % 7) for index in range(120)],
        ):
            observation = detector.detect(series(btc, closes))
            assert ZERO <= observation.confidence <= ONE

    def test_is_deterministic(self, btc: Symbol) -> None:
        detector = RuleBasedRegimeDetector()
        candles = series(btc, [100 + index * 0.5 for index in range(120)])
        first = detector.detect(candles)
        second = detector.detect(candles)
        assert first.regime is second.regime
        assert first.confidence == second.confidence


class TestGaussianMixtureDetector:
    def test_falls_back_before_fitting(self, btc: Symbol) -> None:
        # A missing model must degrade to the rule-based detector, never crash.
        detector = GaussianMixtureRegimeDetector()
        assert not detector.is_fitted
        observation = detector.detect(series(btc, [100 + index for index in range(120)]))
        assert observation.detector == "rule_based"

    def test_fit_refuses_too_little_data(self, btc: Symbol) -> None:
        detector = GaussianMixtureRegimeDetector(n_components=4)
        assert detector.fit(series(btc, [100.0] * 80)) is False

    def test_fits_and_classifies(self, btc: Symbol) -> None:
        closes = (
            [100 + index for index in range(150)]
            + [250 - index for index in range(150)]
            + [100 + (index % 6) for index in range(150)]
        )
        detector = GaussianMixtureRegimeDetector(n_components=3)
        if not detector.fit(series(btc, closes), step=3):
            pytest.skip("scikit-learn unavailable")
        observation = detector.detect(series(btc, closes))
        assert observation.detector == "gaussian_mixture"
        assert ZERO <= observation.confidence <= ONE

    def test_build_detector_by_name(self) -> None:
        assert build_detector("rule_based").name == "rule_based"
        assert build_detector("gaussian_mixture").name == "gaussian_mixture"
        with pytest.raises(ValidationError, match="unknown regime detector"):
            build_detector("crystal_ball")


class TestRegimeHistory:
    def test_classifies_a_series(self, btc: Symbol) -> None:
        observations = regime_history(
            series(btc, [100 + index for index in range(120)]),
            RuleBasedRegimeDetector(),
            step=10,
        )
        assert observations
        described = summarise(observations)
        assert described["total"] == len(observations)
        assert "distribution" in described

    def test_summarise_empty(self) -> None:
        assert summarise([])["total"] == 0


# --------------------------------------------------------------------------- #
# The safety contract
# --------------------------------------------------------------------------- #
class TestAICannotIncreaseRisk:
    """The AI may only ever reduce exposure."""

    def test_a_multiplier_above_one_is_rejected_at_construction(self) -> None:
        # Not merely discouraged — structurally impossible to express.
        with pytest.raises(ValidationError, match="may only reduce risk"):
            AIAdvice(conviction_multiplier=Decimal("1.5"))

    def test_a_negative_multiplier_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            AIAdvice(conviction_multiplier=Decimal("-0.1"))

    def test_conviction_only_ever_shrinks(self, btc: Symbol) -> None:
        engine = build_engine()
        candles = volatile_series(btc)
        original = signal_for(btc, SignalDirection.LONG, conviction="1")
        adjusted, advice = engine.apply(original, candles)
        assert adjusted is not None
        assert adjusted.conviction <= original.conviction
        assert advice.conviction_multiplier <= ONE

    def test_the_guard_catches_an_increase(self, btc: Symbol) -> None:
        # Prove the assertion is not vacuous: hand it a deliberately enlarged signal.
        original = signal_for(btc, SignalDirection.LONG, conviction="0.5")
        enlarged = replace(original, conviction=Decimal("1"))
        with pytest.raises(ValidationError, match="increased conviction"):
            assert_risk_reducing(original, enlarged)

    def test_the_guard_catches_a_direction_flip(self, btc: Symbol) -> None:
        original = signal_for(btc, SignalDirection.LONG)
        flipped = replace(original, direction=SignalDirection.SHORT)
        with pytest.raises(ValidationError, match="changed the signal direction"):
            assert_risk_reducing(original, flipped)

    def test_the_guard_catches_a_moved_stop(self, btc: Symbol) -> None:
        original = signal_for(btc, SignalDirection.LONG)
        widened = replace(original, stop_loss_price=Decimal("90"))
        with pytest.raises(ValidationError, match="modified the stop loss"):
            assert_risk_reducing(original, widened)

    def test_a_veto_is_permitted(self, btc: Symbol) -> None:
        assert_risk_reducing(signal_for(btc, SignalDirection.LONG), None)

    def test_advice_has_no_field_that_can_enlarge_a_trade(self) -> None:
        # A structural check: if someone adds such a field, this fails and asks why.
        fields = set(AIAdvice.__dataclass_fields__)
        assert fields == {
            "veto",
            "conviction_multiplier",
            "regime",
            "regime_confidence",
            "reasons",
            "features",
        }


class TestRegimeAdvisor:
    def test_halves_conviction_in_high_volatility(self, btc: Symbol) -> None:
        advisor = RegimeAdvisor()
        advice = advisor.advise(signal_for(btc, SignalDirection.LONG), volatile_series(btc))
        assert advice.regime is MarketRegime.HIGH_VOLATILITY
        assert advice.conviction_multiplier == Decimal("0.5")
        assert not advice.veto

    def test_discounts_a_counter_trend_entry(self, btc: Symbol) -> None:
        advisor = RegimeAdvisor()
        downtrend = series(btc, [100 * (0.99**index) for index in range(120)])
        advice = advisor.advise(signal_for(btc, SignalDirection.LONG), downtrend)
        assert advice.regime is MarketRegime.BEAR_TREND
        assert advice.conviction_multiplier < ONE
        # Discounted, not forbidden: a strategy that can never trade against the trend
        # cannot catch a reversal.
        assert not advice.veto

    def test_can_be_configured_to_veto_counter_trend(self, btc: Symbol) -> None:
        advisor = RegimeAdvisor(veto_counter_trend=True)
        downtrend = series(btc, [100 * (0.99**index) for index in range(120)])
        advice = advisor.advise(signal_for(btc, SignalDirection.LONG), downtrend)
        assert advice.veto

    def test_leaves_an_aligned_signal_alone(self, btc: Symbol) -> None:
        advisor = RegimeAdvisor()
        uptrend = series(btc, [100 * (1.01**index) for index in range(120)])
        advice = advisor.advise(signal_for(btc, SignalDirection.LONG), uptrend)
        assert advice.regime is MarketRegime.BULL_TREND
        assert advice.conviction_multiplier == ONE

    def test_stays_silent_without_enough_history(self, btc: Symbol) -> None:
        advisor = RegimeAdvisor()
        advice = advisor.advise(signal_for(btc, SignalDirection.LONG), series(btc, [100.0] * 5))
        assert advice.is_neutral
        assert "insufficient history" in advice.reasons[0]

    def test_low_confidence_regimes_are_not_acted_on(self, btc: Symbol) -> None:
        # A low-confidence label is worse than no label: it invites a decision.
        advisor = RegimeAdvisor()
        closes = [100 + (index % 11) * 0.9 for index in range(120)]
        advice = advisor.advise(signal_for(btc, SignalDirection.LONG), series(btc, closes))
        if advice.regime_confidence < Decimal("0.55"):
            assert advice.conviction_multiplier == ONE


class TestDecisionEngine:
    def test_disabled_engine_is_a_passthrough(self, btc: Symbol) -> None:
        engine = AIDecisionEngine(advisors=(RegimeAdvisor(),), enabled=False)
        advice = engine.advise(signal_for(btc, SignalDirection.LONG), volatile_series(btc))
        assert advice.is_neutral

    def test_hold_signals_are_ignored(self, btc: Symbol) -> None:
        engine = build_engine()
        advice = engine.advise(Signal.hold(btc, REFERENCE_TIME, "test"), volatile_series(btc))
        assert advice.is_neutral

    def test_multipliers_compound_rather_than_average(self, btc: Symbol) -> None:
        # Averaging would let one optimistic advisor dilute another's warning.
        class Halver:
            name = "halver"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                return AIAdvice(conviction_multiplier=Decimal("0.5"))

        engine = AIDecisionEngine(advisors=(Halver(), Halver()))
        advice = engine.advise(signal_for(btc, SignalDirection.LONG), volatile_series(btc))
        assert advice.conviction_multiplier == Decimal("0.25")

    def test_any_veto_wins(self, btc: Symbol) -> None:
        class Vetoer:
            name = "vetoer"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                return AIAdvice(veto=True, reasons=("nope",))

        class Permissive:
            name = "permissive"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                return AIAdvice()

        engine = AIDecisionEngine(advisors=(Permissive(), Vetoer()))
        advice = engine.advise(signal_for(btc, SignalDirection.LONG), volatile_series(btc))
        assert advice.veto

    def test_multiplier_has_a_floor(self, btc: Symbol) -> None:
        # Without a floor, repeated halvings shrink the position below the venue minimum,
        # which reads as "the AI broke it" rather than "the AI was cautious".
        class Crusher:
            name = "crusher"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                return AIAdvice(conviction_multiplier=Decimal("0.01"))

        engine = AIDecisionEngine(advisors=(Crusher(), Crusher(), Crusher()))
        advice = engine.advise(signal_for(btc, SignalDirection.LONG), volatile_series(btc))
        assert advice.conviction_multiplier == Decimal("0.1")

    def test_a_broken_advisor_is_ignored_not_fatal(self, btc: Symbol) -> None:
        # Failing open is right *because* advisors can only reduce risk: losing one is a
        # lost safety check, not a lost trading decision.
        class Exploding:
            name = "exploding"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                raise RuntimeError("model crashed")

        engine = AIDecisionEngine(advisors=(Exploding(),))
        advice = engine.advise(signal_for(btc, SignalDirection.LONG), volatile_series(btc))
        assert advice.is_neutral

    def test_apply_returns_none_on_veto(self, btc: Symbol) -> None:
        class Vetoer:
            name = "vetoer"

            def advise(self, signal: Signal, candles: object) -> AIAdvice:
                return AIAdvice(veto=True, reasons=("test veto",))

        engine = AIDecisionEngine(advisors=(Vetoer(),))
        adjusted, advice = engine.apply(signal_for(btc, SignalDirection.LONG), volatile_series(btc))
        assert adjusted is None
        assert advice.veto

    def test_apply_preserves_everything_except_conviction(self, btc: Symbol) -> None:
        engine = build_engine()
        original = signal_for(btc, SignalDirection.LONG)
        adjusted, _ = engine.apply(original, volatile_series(btc))
        assert adjusted is not None
        assert adjusted.symbol == original.symbol
        assert adjusted.direction is original.direction
        assert adjusted.stop_loss_price == original.stop_loss_price
        assert adjusted.take_profit_price == original.take_profit_price

    def test_describe_states_it_cannot_increase_risk(self) -> None:
        assert build_engine().describe()["can_increase_risk"] is False


class TestAIAugmentedStrategy:
    class AlwaysLong(Strategy):
        strategy_id = "always_long"
        params_model = StrategyParams

        @property
        def warmup_bars(self) -> int:
            return 5

        def generate(self, context: StrategyContext) -> Signal:
            return Signal(
                symbol=context.symbol,
                direction=SignalDirection.LONG,
                timestamp=context.now,
                strategy_id=self.strategy_id,
                reference_price=context.price,
                stop_loss_price=context.price * Decimal("0.98"),
                reason="always long",
            )

    def test_warmup_covers_the_ai_window(self, btc: Symbol) -> None:
        # Trading before the AI can classify would mean the first trades run with no
        # oversight at all, silently.
        wrapped = AIAugmentedStrategy(self.AlwaysLong(), build_engine())
        assert wrapped.warmup_bars >= MIN_BARS

    def test_attribution_stays_with_the_wrapped_strategy(self, btc: Symbol) -> None:
        wrapped = AIAugmentedStrategy(self.AlwaysLong(), build_engine())
        signal = wrapped.evaluate(context_for(btc, volatile_series(btc)))
        assert signal.strategy_id == "always_long"

    def test_scales_conviction_in_a_hostile_regime(self, btc: Symbol) -> None:
        wrapped = AIAugmentedStrategy(self.AlwaysLong(), build_engine())
        signal = wrapped.evaluate(context_for(btc, volatile_series(btc)))
        assert signal.conviction < ONE
        assert "AI scaled conviction" in signal.reason

    def test_a_veto_becomes_a_hold(self, btc: Symbol) -> None:
        wrapped = AIAugmentedStrategy(self.AlwaysLong(), build_engine(veto_counter_trend=True))
        downtrend = series(btc, [100 * (0.99**index) for index in range(120)])
        signal = wrapped.evaluate(context_for(btc, downtrend))
        assert signal.direction is SignalDirection.HOLD
        assert "AI veto" in signal.reason

    def test_warmup_produces_a_hold(self, btc: Symbol) -> None:
        wrapped = AIAugmentedStrategy(self.AlwaysLong(), build_engine())
        signal = wrapped.evaluate(context_for(btc, series(btc, [100.0] * 10)))
        assert signal.direction is SignalDirection.HOLD
        assert "warming up" in signal.reason

    def test_advice_summary_makes_the_effect_measurable(self, btc: Symbol) -> None:
        # An AI layer that vetoed nothing and scaled nothing is not earning its complexity.
        wrapped = AIAugmentedStrategy(self.AlwaysLong(), build_engine())
        wrapped.evaluate(context_for(btc, volatile_series(btc)))
        summary = wrapped.advice_summary()
        assert summary["signals_seen"] == 1
        assert summary["scaled"] == 1

    def test_failure_inside_the_ai_is_contained(self, btc: Symbol) -> None:
        class Exploding(AIDecisionEngine):
            def apply(self, signal, candles):
                raise RuntimeError("engine exploded")

        wrapped = AIAugmentedStrategy(self.AlwaysLong(), Exploding())
        signal = wrapped.evaluate(context_for(btc, volatile_series(btc)))
        assert signal.direction is SignalDirection.HOLD
        assert "AI strategy error" in signal.reason

    def test_wrap_disabled_returns_the_original(self) -> None:
        original = self.AlwaysLong()
        assert wrap(original, enabled=False) is original

    def test_wrap_enabled_returns_a_wrapper(self) -> None:
        assert isinstance(wrap(self.AlwaysLong()), AIAugmentedStrategy)

    def test_describe_exposes_both_layers(self) -> None:
        described = AIAugmentedStrategy(self.AlwaysLong(), build_engine()).describe()
        assert described["strategy_id"] == "always_long"
        assert described["ai"]["can_increase_risk"] is False
