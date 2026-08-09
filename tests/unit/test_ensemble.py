"""Tests for ensemble trading."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.backtest.metrics import PerformanceMetrics
from quantflow.core.errors import ValidationError
from quantflow.domain.enums import SignalDirection, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.domain.signals import Signal
from quantflow.ensemble import (
    MAX_WEIGHT,
    EnsembleStrategy,
    WeightSet,
    compute_weights,
    equal_weights,
    score_of,
)
from quantflow.ensemble.weights import StrategyWeight
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams

BTC = Symbol(base="BTC", quote="USDT")
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def metrics(
    *, sharpe: str = "1.0", drawdown: str = "0.10", trades: int = 100
) -> PerformanceMetrics:
    """Metrics with only the weighting inputs set meaningfully."""
    return PerformanceMetrics(
        starting_equity=Decimal("10000"),
        final_equity=Decimal("11000"),
        total_return_pct=Decimal("0.10"),
        cagr=Decimal("0.1"),
        max_drawdown_pct=Decimal(drawdown),
        max_drawdown_duration_days=Decimal("5"),
        volatility_annual=Decimal("0.3"),
        downside_volatility_annual=Decimal("0.2"),
        sharpe_ratio=Decimal(sharpe),
        sortino_ratio=Decimal("1"),
        calmar_ratio=Decimal("1"),
        trade_count=trades,
        win_count=trades // 2,
        loss_count=trades - trades // 2,
        win_rate=Decimal("0.5"),
        profit_factor=Decimal("1.2"),
        expectancy=Decimal("1"),
        average_win=Decimal("10"),
        average_loss=Decimal("-8"),
        largest_win=Decimal("50"),
        largest_loss=Decimal("-40"),
        average_holding_hours=Decimal("12"),
        total_fees=Decimal("50"),
        turnover=Decimal("5"),
        exposure_pct=Decimal("0.5"),
        duration_days=Decimal("365"),
        bars=1000,
    )


class TestWeighting:
    """A weight is a claim that one strategy beats another. It must be earned."""

    def test_better_risk_adjusted_performance_earns_more(self) -> None:
        weights = compute_weights({"good": metrics(sharpe="2.0"), "poor": metrics(sharpe="0.6")})
        assert weights.weight_for("good") > weights.weight_for("poor")

    def test_drawdown_is_penalised_not_just_return(self) -> None:
        # Raw return would hand the book to whichever strategy took the most risk in the
        # sample, which is the one most likely to blow up outside it.
        assert score_of(metrics(sharpe="1", drawdown="0.05")) > score_of(
            metrics(sharpe="1", drawdown="0.50")
        )

    def test_a_losing_strategy_gets_nothing(self) -> None:
        # Not a token allocation: a small weight on a known loser is still a decision to
        # lose money slowly.
        weights = compute_weights({"loser": metrics(sharpe="-1.0")})
        assert weights.weight_for("loser") == Decimal("0")

    def test_a_thin_record_cannot_justify_a_weight(self) -> None:
        weights = compute_weights({"new": metrics(trades=5)})
        assert weights.weight_for("new") == Decimal("0")
        assert "below the" in weights.weights[0].reason

    def test_no_qualifying_strategy_yields_an_empty_allocation(self) -> None:
        # Which the ensemble must read as "do not trade", never as "use equal weights".
        weights = compute_weights({"a": metrics(sharpe="-1"), "b": metrics(trades=2)})
        assert weights.active == ()
        assert weights.total == Decimal("0")

    def test_weights_sum_to_one_when_anything_qualifies(self) -> None:
        weights = compute_weights({name: metrics(sharpe="1.0") for name in ("a", "b", "c", "d")})
        assert weights.total == pytest.approx(Decimal("1"), abs=Decimal("0.0001"))

    def test_no_member_may_exceed_the_cap(self) -> None:
        # An ensemble whose weights collapse onto one member is not an ensemble.
        weights = compute_weights(
            {
                "dominant": metrics(sharpe="10.0"),
                "b": metrics(sharpe="0.5"),
                "c": metrics(sharpe="0.5"),
                "d": metrics(sharpe="0.5"),
            }
        )
        assert weights.weight_for("dominant") <= MAX_WEIGHT + Decimal("0.0001")

    def test_capping_redistributes_rather_than_discarding(self) -> None:
        weights = compute_weights(
            {
                "dominant": metrics(sharpe="10.0"),
                "b": metrics(sharpe="0.5"),
                "c": metrics(sharpe="0.5"),
            }
        )
        assert weights.total == pytest.approx(Decimal("1"), abs=Decimal("0.001"))

    def test_equal_weights_are_labelled_as_unearned(self) -> None:
        weights = equal_weights(["a", "b"])
        assert weights.weight_for("a") == weights.weight_for("b")
        assert all(not item.reliable for item in weights.weights)

    def test_an_empty_input_is_an_empty_allocation(self) -> None:
        assert compute_weights({}).weights == ()


class Scripted(Strategy):
    """A member that always votes the same way."""

    params_model = StrategyParams

    def __init__(self, identifier: str, direction: SignalDirection) -> None:
        self.strategy_id = identifier  # type: ignore[misc]
        self.description = f"always {direction.value}"  # type: ignore[misc]
        super().__init__(None)
        self._direction = direction

    @property
    def warmup_bars(self) -> int:
        return 1

    def generate(self, context: StrategyContext) -> Signal:
        if self._direction is SignalDirection.HOLD:
            return context.hold("abstaining", self.strategy_id)
        return Signal(
            symbol=context.symbol,
            direction=self._direction,
            timestamp=context.now,
            strategy_id=self.strategy_id,
            reference_price=context.price,
            stop_loss_price=context.price * Decimal("0.98"),
            reason="scripted",
        )


def context() -> StrategyContext:
    """A minimal decision context."""
    candles = [
        Candle(
            symbol=BTC,
            timeframe=Timeframe.H1,
            open_time=BASE + timedelta(hours=i),
            open=Decimal("1000"),
            high=Decimal("1010"),
            low=Decimal("990"),
            close=Decimal("1000"),
            volume=Decimal("10"),
            quote_volume=Decimal("10000"),
            trades=5,
        )
        for i in range(10)
    ]
    return StrategyContext(
        symbol=BTC,
        timeframe=Timeframe.H1,
        history=CandleSeries(candles),
        now=candles[-1].close_time,
        portfolio=PortfolioSnapshot(
            timestamp=candles[-1].close_time, base_currency="USDT", cash=Decimal("10000")
        ),
    )


def weights_for(*pairs: tuple[str, str]) -> WeightSet:
    """A WeightSet from (strategy_id, weight) pairs."""
    return WeightSet(
        weights=tuple(
            StrategyWeight(
                strategy_id=name,
                weight=Decimal(value),
                score=Decimal("1"),
                trade_count=100,
                reliable=True,
                reason="test",
            )
            for name, value in pairs
        )
    )


class TestEnsembleDecision:
    """The ensemble must be willing to do nothing."""

    def test_agreement_produces_a_trade(self) -> None:
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.LONG), Scripted("b", SignalDirection.LONG)],
            weights_for(("a", "0.5"), ("b", "0.5")),
        )
        decision = ensemble.decide(context())
        assert decision.direction is SignalDirection.LONG
        assert decision.confidence == Decimal("1")

    def test_a_split_decision_produces_no_trade(self) -> None:
        # There is no such thing as being slightly right about direction, so an even
        # split must not average into a small position.
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.LONG), Scripted("b", SignalDirection.SHORT)],
            weights_for(("a", "0.5"), ("b", "0.5")),
        )
        decision = ensemble.decide(context())
        assert not decision.is_actionable
        assert "below the" in decision.reason

    def test_one_voice_is_not_a_consensus(self) -> None:
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.LONG), Scripted("b", SignalDirection.HOLD)],
            weights_for(("a", "0.9"), ("b", "0.1")),
        )
        decision = ensemble.decide(context())
        assert not decision.is_actionable
        assert "required" in decision.reason

    def test_silence_from_everyone_is_no_trade(self) -> None:
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.HOLD), Scripted("b", SignalDirection.HOLD)],
            weights_for(("a", "0.5"), ("b", "0.5")),
        )
        decision = ensemble.decide(context())
        assert not decision.is_actionable
        assert decision.participating == 0

    def test_zero_weighted_members_do_not_vote(self) -> None:
        ensemble = EnsembleStrategy(
            [
                Scripted("a", SignalDirection.LONG),
                Scripted("b", SignalDirection.LONG),
                Scripted("excluded", SignalDirection.SHORT),
            ],
            weights_for(("a", "0.5"), ("b", "0.5"), ("excluded", "0")),
        )
        decision = ensemble.decide(context())
        assert decision.direction is SignalDirection.LONG
        assert all(vote.strategy_id != "excluded" for vote in decision.votes)

    def test_dissent_can_be_made_disqualifying(self) -> None:
        ensemble = EnsembleStrategy(
            [
                Scripted("a", SignalDirection.LONG),
                Scripted("b", SignalDirection.LONG),
                Scripted("c", SignalDirection.SHORT),
            ],
            weights_for(("a", "0.4"), ("b", "0.4"), ("c", "0.2")),
            {"require_no_dissent": True},
        )
        decision = ensemble.decide(context())
        assert not decision.is_actionable
        assert "disagree" in decision.reason

    def test_a_decision_explains_itself(self) -> None:
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.LONG), Scripted("b", SignalDirection.LONG)],
            weights_for(("a", "0.5"), ("b", "0.5")),
        )
        assert "confidence" in ensemble.decide(context()).explain()


class TestEnsembleStrategy:
    """The ensemble is a Strategy and gets no shortcut around the risk engine."""

    def test_it_emits_a_signal_carrying_confidence_as_conviction(self) -> None:
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.LONG), Scripted("b", SignalDirection.LONG)],
            weights_for(("a", "0.5"), ("b", "0.5")),
        )
        signal = ensemble.generate(context())
        assert signal.direction is SignalDirection.LONG
        assert signal.conviction == Decimal("1")

    def test_it_inherits_protective_levels_from_the_leading_member(self) -> None:
        # Averaging stop prices across members would invent a level none of them chose.
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.LONG), Scripted("b", SignalDirection.LONG)],
            weights_for(("a", "0.6"), ("b", "0.4")),
        )
        signal = ensemble.generate(context())
        assert signal.stop_loss_price == Decimal("1000") * Decimal("0.98")

    def test_a_no_trade_decision_becomes_a_hold_signal(self) -> None:
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.LONG), Scripted("b", SignalDirection.SHORT)],
            weights_for(("a", "0.5"), ("b", "0.5")),
        )
        assert ensemble.generate(context()).direction is SignalDirection.HOLD

    def test_warmup_is_the_slowest_member(self) -> None:
        ensemble = EnsembleStrategy([Scripted("a", SignalDirection.LONG)], weights_for(("a", "1")))
        assert ensemble.warmup_bars == 1

    def test_an_empty_ensemble_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            EnsembleStrategy([], WeightSet(weights=()))

    def test_attribution_is_to_the_ensemble_not_a_member(self) -> None:
        # The base class rejects a signal whose strategy_id does not match its emitter,
        # so this also proves the ensemble satisfies the Strategy contract.
        ensemble = EnsembleStrategy(
            [Scripted("a", SignalDirection.LONG), Scripted("b", SignalDirection.LONG)],
            weights_for(("a", "0.5"), ("b", "0.5")),
        )
        assert ensemble.evaluate(context()).strategy_id == "ensemble"


class TestCapSatisfiability:
    """A cap that cannot be satisfied must not be applied."""

    def test_two_members_keep_their_ranking(self) -> None:
        # With n=2 and a 40% ceiling there is no allocation summing to one in which both
        # are under the cap. Applying it anyway drove both to 0.40 and erased the very
        # ranking the weighting exists to express.
        weights = compute_weights({"good": metrics(sharpe="2.0"), "poor": metrics(sharpe="0.6")})
        assert weights.weight_for("good") > weights.weight_for("poor")
        assert weights.total == pytest.approx(Decimal("1"), abs=Decimal("0.0001"))

    def test_a_two_member_ensemble_is_reported_as_concentrated(self) -> None:
        # The concentration is real and is surfaced rather than hidden behind a cap that
        # could not be honoured.
        weights = compute_weights({"good": metrics(sharpe="5.0"), "poor": metrics(sharpe="0.5")})
        assert weights.is_concentrated

    def test_the_cap_binds_once_it_is_satisfiable(self) -> None:
        weights = compute_weights(
            {
                name: metrics(sharpe=s)
                for name, s in (("a", "10.0"), ("b", "0.5"), ("c", "0.5"), ("d", "0.5"))
            }
        )
        assert weights.weight_for("a") <= MAX_WEIGHT + Decimal("0.0001")
        assert weights.total == pytest.approx(Decimal("1"), abs=Decimal("0.001"))
