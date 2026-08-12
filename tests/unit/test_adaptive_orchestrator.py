"""Regime detection, performance memory, economic gates and adaptive selection."""

from __future__ import annotations

from decimal import Decimal

from quantflow.domain.enums import MarketRegime, PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.positions import ClosedTrade
from quantflow.orchestrator import StrategyOrchestrator
from quantflow.orchestrator.performance import (
    MIN_TRADES_FOR_EVIDENCE,
    RECENCY_HALF_LIFE,
    PerformanceMemory,
    evidence_score,
    summarise,
)
from quantflow.orchestrator.scoring import (
    MAX_POSITIONS_PER_STRATEGY,
    MIN_RISK_REWARD,
    gate_candidate,
)
from tests.conftest import REFERENCE_TIME
from tests.unit.test_new_strategies import bars, context_from
from tests.unit.test_orchestrator import StubStrategy, candidate_from, entry

SYMBOL = Symbol.parse("BTC/USDT")
COST = Decimal("0.002")

UPTREND = [(str(100 + i), str(101 + i), str(99 + i), str(100.8 + i), "10") for i in range(160)]
FLAT = [("100", "101", "99", "100", "10")] * 160


def trade(pnl: str, *, strategy: str = "s", symbol: Symbol = SYMBOL) -> ClosedTrade:
    value = Decimal(pnl)
    return ClosedTrade(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        exit_price=Decimal("100") + value,
        entry_time=REFERENCE_TIME,
        exit_time=REFERENCE_TIME,
        gross_pnl=value,
        fees=Decimal("0"),
        strategy_id=strategy,
    )


class TestPerformanceMemory:
    def test_buckets_stay_separate_per_strategy(self) -> None:
        """One strategy's result must never be credited to another."""
        memory = PerformanceMemory()
        memory.record(trade("10", strategy="good"))
        memory.record(trade("-10", strategy="bad"))
        assert memory.overall("good").net_pnl == Decimal("10")
        assert memory.overall("bad").net_pnl == Decimal("-10")

    def test_symbol_and_regime_slices_are_separate(self) -> None:
        memory = PerformanceMemory()
        memory.record(trade("5", symbol=Symbol.parse("ETH/USDT")))
        assert memory.for_symbol("s", Symbol.parse("ETH/USDT")).trades == 1
        assert memory.for_symbol("s", SYMBOL).trades == 0

    def test_recency_weighting_favours_newer_results(self) -> None:
        """Two records with the same raw PnL but opposite order must differ."""
        old_wins = summarise("a", [trade("10")] * 5 + [trade("-10")] * 5)
        new_wins = summarise("b", [trade("-10")] * 5 + [trade("10")] * 5)
        assert new_wins.weighted_pnl > old_wins.weighted_pnl
        assert old_wins.net_pnl == new_wins.net_pnl

    def test_half_life_halves_the_weight(self) -> None:
        recent = summarise("r", [trade("1")])
        older = summarise("o", [trade("1")] + [trade("0")] * RECENCY_HALF_LIFE)
        assert older.weighted_pnl < recent.weighted_pnl

    def test_loss_streak_and_drawdown_are_tracked(self) -> None:
        record = summarise("s", [trade("5"), trade("-1"), trade("-2"), trade("-3")])
        assert record.loss_streak == 3
        assert record.max_drawdown == Decimal("6")

    def test_profit_factor_is_none_without_losses(self) -> None:
        assert summarise("s", [trade("1"), trade("2")]).profit_factor is None


class TestEvidenceScore:
    def test_small_sample_is_neutral(self) -> None:
        """Two lucky trades must not promote a strategy."""
        assert evidence_score(summarise("s", [trade("100")] * 2)) == Decimal("0.5")

    def test_meaningful_losing_record_scores_below_neutral(self) -> None:
        losing = summarise("s", [trade("-1")] * MIN_TRADES_FOR_EVIDENCE)
        assert evidence_score(losing) < Decimal("0.5")

    def test_meaningful_winning_record_scores_above_neutral(self) -> None:
        winning = summarise("s", [trade("3"), trade("-1")] * MIN_TRADES_FOR_EVIDENCE)
        assert evidence_score(winning) > Decimal("0.5")

    def test_a_strategy_can_recover_its_weight(self) -> None:
        """A penalty must be a function of current evidence, never a latch."""
        losing = [trade("-1")] * MIN_TRADES_FOR_EVIDENCE
        before = evidence_score(summarise("s", losing))
        after = evidence_score(summarise("s", [*losing, *([trade("5")] * 20)]))
        assert after > before

    def test_active_loss_streak_is_penalised(self) -> None:
        base = [trade("3"), trade("-1")] * MIN_TRADES_FOR_EVIDENCE
        streaking = [*base, *([trade("-1")] * 5)]
        assert evidence_score(summarise("s", streaking)) < evidence_score(summarise("s", base))

    def test_score_always_within_unit_interval(self) -> None:
        for sample in ([trade("-99")] * 40, [trade("99")] * 40, []):
            assert Decimal("0") <= evidence_score(summarise("s", sample)) <= Decimal("1")


class TestEconomicGates:
    def test_poor_reward_risk_is_rejected(self) -> None:
        """The payoff structure that produced a 0.959 median R:R must not pass."""
        candidate = candidate_from(entry("s", stop="90", target="105"))  # 1.5:10 -> 0.5
        reason = gate_candidate(candidate, cost_rate=COST)
        assert reason is not None
        assert "reward:risk" in reason

    def test_adequate_reward_risk_passes(self) -> None:
        candidate = candidate_from(entry("s", stop="95", target="120"))  # 4:1
        assert gate_candidate(candidate, cost_rate=COST) is None

    def test_missing_stop_is_rejected(self) -> None:
        candidate = candidate_from(entry("s", stop=None))
        reason = gate_candidate(candidate, cost_rate=COST)
        assert reason is not None
        assert "reward:risk" in reason

    def test_edge_consumed_by_costs_is_rejected(self) -> None:
        """A target barely beyond the fee is not an opportunity."""
        candidate = candidate_from(entry("s", stop="99.8", target="100.3"))
        reason = gate_candidate(candidate, cost_rate=COST)
        assert reason is not None
        assert "edge" in reason or "cost" in reason

    def test_higher_costs_reject_what_lower_costs_allow(self) -> None:
        candidate = candidate_from(entry("s", stop="99", target="101.6"))
        assert gate_candidate(candidate, cost_rate=Decimal("0.0005")) is None
        assert gate_candidate(candidate, cost_rate=Decimal("0.02")) is not None

    def test_strategy_concentration_is_capped(self) -> None:
        """The book must not fill with one idea expressed many times."""
        candidate = candidate_from(entry("s", stop="95", target="120"))
        counts = {"s": MAX_POSITIONS_PER_STRATEGY}
        reason = gate_candidate(candidate, cost_rate=COST, strategy_position_counts=counts)
        assert reason is not None
        assert "already holds" in reason

    def test_min_risk_reward_is_above_one(self) -> None:
        """A 1:1 payoff cannot work at the win rates these strategies achieve."""
        assert Decimal("1") < MIN_RISK_REWARD


class TestRegimeDetection:
    def test_regime_is_not_permanently_unknown(self) -> None:
        orchestrator = StrategyOrchestrator(members=[StubStrategy("s")])
        orchestrator.evaluate(context_from(bars(UPTREND)))
        decision = orchestrator.last_decision
        assert decision is not None
        assert decision.regime is not MarketRegime.UNKNOWN

    def test_a_flat_market_is_not_classified_as_a_trend(self) -> None:
        orchestrator = StrategyOrchestrator(members=[StubStrategy("s")])
        orchestrator.evaluate(context_from(bars(FLAT)))
        decision = orchestrator.last_decision
        assert decision is not None
        assert decision.regime not in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND)

    def test_classification_ignores_later_bars(self) -> None:
        """The regime at bar N must not depend on bars after N."""
        candles = bars(UPTREND)
        first = StrategyOrchestrator(members=[StubStrategy("s")])
        first.evaluate(context_from(candles[:120]))
        second = StrategyOrchestrator(members=[StubStrategy("s")])
        second.evaluate(context_from(candles))
        assert first.last_decision is not None
        assert second.last_decision is not None
        # Re-running the truncated series must reproduce its own label exactly.
        third = StrategyOrchestrator(members=[StubStrategy("s")])
        third.evaluate(context_from(candles[:120]))
        assert third.last_decision is not None
        assert third.last_decision.regime is first.last_decision.regime


class TestAdaptiveSelection:
    def test_gated_candidates_never_reach_ranking(self) -> None:
        weak = StubStrategy("weak", entry("weak", stop="90", target="105"))
        orchestrator = StrategyOrchestrator(members=[weak])
        signal = orchestrator.evaluate(context_from(bars(UPTREND)))
        assert not signal.is_actionable
        decision = orchestrator.last_decision
        assert decision is not None
        assert decision.gated

    def test_holding_cash_is_a_valid_outcome(self) -> None:
        """Nothing tradable means no trade, not the least bad trade."""
        members = [StubStrategy(f"s{i}", entry(f"s{i}", stop="90", target="105")) for i in range(5)]
        orchestrator = StrategyOrchestrator(members=members)
        assert not orchestrator.evaluate(context_from(bars(UPTREND))).is_actionable

    def test_a_viable_candidate_is_still_selected(self) -> None:
        good = StubStrategy("good", entry("good", stop="95", target="120"))
        orchestrator = StrategyOrchestrator(members=[good])
        signal = orchestrator.evaluate(context_from(bars(UPTREND)))
        assert signal.strategy_id == "good"

    def test_recorded_trades_reach_the_memory(self) -> None:
        orchestrator = StrategyOrchestrator(members=[StubStrategy("s")])
        for _ in range(3):
            orchestrator.on_trade_closed(trade("-1"))
        assert orchestrator.memory.overall("s").trades == 3

    def test_a_proven_loser_is_refused_entry_outright(self) -> None:
        """Down-weighting is not enough: if every rival is weak the loser still wins."""
        loser = StubStrategy("loser", entry("loser", stop="95", target="120"))
        orchestrator = StrategyOrchestrator(members=[loser])
        for _ in range(MIN_TRADES_FOR_EVIDENCE + 5):
            orchestrator.on_trade_closed(trade("-2", strategy="loser"))

        signal = orchestrator.evaluate(context_from(bars(UPTREND)))
        assert not signal.is_actionable
        decision = orchestrator.last_decision
        assert decision is not None
        assert any("negative expectancy" in reason for _, reason in decision.gated)

    def test_a_blocked_strategy_recovers_when_results_improve(self) -> None:
        """The block is evidence, not a latch."""
        loser = StubStrategy("loser", entry("loser", stop="95", target="120"))
        orchestrator = StrategyOrchestrator(members=[loser])
        for _ in range(MIN_TRADES_FOR_EVIDENCE + 5):
            orchestrator.on_trade_closed(trade("-2", strategy="loser"))
        assert not orchestrator.evaluate(context_from(bars(UPTREND))).is_actionable

        for _ in range(40):
            orchestrator.on_trade_closed(trade("6", strategy="loser"))
        assert orchestrator.evaluate(context_from(bars(UPTREND))).is_actionable

    def test_a_small_losing_sample_does_not_block(self) -> None:
        """Two bad trades must not disqualify a strategy."""
        loser = StubStrategy("loser", entry("loser", stop="95", target="120"))
        orchestrator = StrategyOrchestrator(members=[loser])
        for _ in range(2):
            orchestrator.on_trade_closed(trade("-2", strategy="loser"))
        assert orchestrator.evaluate(context_from(bars(UPTREND))).is_actionable

    def test_a_losing_strategy_loses_score_against_an_identical_rival(self) -> None:
        """Same signal, different history: the proven loser must rank lower."""
        good = StubStrategy("good", entry("good", stop="95", target="120"))
        bad = StubStrategy("bad", entry("bad", stop="95", target="120"))
        orchestrator = StrategyOrchestrator(members=[good, bad])
        for _ in range(MIN_TRADES_FOR_EVIDENCE + 5):
            orchestrator.on_trade_closed(trade("-2", strategy="bad"))
            orchestrator.on_trade_closed(trade("2", strategy="good"))

        orchestrator.evaluate(context_from(bars(UPTREND)))
        decision = orchestrator.last_decision
        assert decision is not None
        assert decision.selected is not None
        assert decision.selected.strategy_id == "good"
