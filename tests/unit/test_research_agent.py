"""Tests for the AI Research Agent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantflow.ai.research_agent import (
    MIN_TRADES_FOR_ANALYSIS,
    STREAK_THRESHOLD,
    FindingKind,
    ResearchAgent,
    Severity,
)
from quantflow.domain.enums import PositionSide, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle
from quantflow.domain.positions import ClosedTrade
from quantflow.intelligence.regime import classify
from quantflow.lab.attribution import RegimeBreakdown, RegimePerformance
from quantflow.lab.diagnosis import Diagnosis, FailureCause

BTC = Symbol(base="BTC", quote="USDT")
ETH = Symbol(base="ETH", quote="USDT")
NOW = datetime(2026, 6, 1, tzinfo=UTC)


def trade(
    index: int,
    net: str,
    *,
    symbol: Symbol = BTC,
    gross: str | None = None,
    fees: str = "1",
) -> ClosedTrade:
    """A round-trip with an explicit net result."""
    net_value = Decimal(net)
    fee_value = Decimal(fees)
    gross_value = Decimal(gross) if gross is not None else net_value + fee_value
    return ClosedTrade(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=Decimal("1"),
        entry_price=Decimal("1000"),
        exit_price=Decimal("1000") + gross_value,
        entry_time=NOW + timedelta(hours=index),
        exit_time=NOW + timedelta(hours=index + 1),
        gross_pnl=gross_value,
        fees=fee_value,
    )


def winners(count: int) -> list[ClosedTrade]:
    """A run of profitable trades."""
    return [trade(i, "10") for i in range(count)]


class TestLossAnalysis:
    """Findings must rest on evidence a reader can check."""

    def test_a_thin_sample_produces_no_conclusions(self) -> None:
        report = ResearchAgent().analyse_losses([trade(0, "-5")])
        assert report[0].kind is FindingKind.INSUFFICIENT_DATA
        assert report[0].severity is Severity.INFO

    def test_no_losses_produces_no_findings(self) -> None:
        assert ResearchAgent().analyse_losses(winners(MIN_TRADES_FOR_ANALYSIS)) == []

    def test_a_losing_streak_is_reported(self) -> None:
        trades = winners(MIN_TRADES_FOR_ANALYSIS) + [
            trade(100 + i, "-5") for i in range(STREAK_THRESHOLD)
        ]
        findings = ResearchAgent().analyse_losses(trades)
        streaks = [item for item in findings if item.kind is FindingKind.STREAK]
        assert streaks
        assert streaks[0].evidence["streak"] == str(STREAK_THRESHOLD)

    def test_a_long_streak_is_urgent(self) -> None:
        trades = winners(MIN_TRADES_FOR_ANALYSIS) + [
            trade(100 + i, "-5") for i in range(STREAK_THRESHOLD * 2)
        ]
        streaks = [
            item
            for item in ResearchAgent().analyse_losses(trades)
            if item.kind is FindingKind.STREAK
        ]
        assert streaks[0].severity is Severity.URGENT

    def test_loss_concentration_in_one_symbol_is_named(self) -> None:
        trades = (
            winners(MIN_TRADES_FOR_ANALYSIS)
            + [trade(100 + i, "-20", symbol=BTC) for i in range(5)]
            + [trade(200 + i, "-1", symbol=ETH) for i in range(2)]
        )
        findings = [
            item
            for item in ResearchAgent().analyse_losses(trades)
            if item.kind is FindingKind.CONCENTRATION
        ]
        assert findings
        assert findings[0].evidence["symbol"] == "BTC/USDT"

    def test_concentration_is_not_claimed_for_a_single_symbol(self) -> None:
        # With one symbol traded, 100% of losses come from it by definition. Reporting
        # that as a finding would be noise dressed as insight.
        trades = winners(MIN_TRADES_FOR_ANALYSIS) + [trade(100 + i, "-20") for i in range(5)]
        assert not [
            item
            for item in ResearchAgent().analyse_losses(trades)
            if item.kind is FindingKind.CONCENTRATION
        ]

    def test_directionally_correct_trades_lost_to_fees_are_separated(self) -> None:
        # The most actionable loss analysis there is: these trades called direction
        # right, so the fix is execution, not entry logic.
        trades = winners(MIN_TRADES_FOR_ANALYSIS) + [
            trade(100 + i, "-0.5", gross="0.5", fees="1") for i in range(6)
        ]
        findings = [
            item
            for item in ResearchAgent().analyse_losses(trades)
            if item.kind is FindingKind.COST_DRAG
        ]
        assert findings
        assert "execution or" in findings[0].recommendation

    def test_every_finding_carries_evidence(self) -> None:
        trades = winners(MIN_TRADES_FOR_ANALYSIS) + [
            trade(100 + i, "-5") for i in range(STREAK_THRESHOLD)
        ]
        for finding in ResearchAgent().analyse_losses(trades):
            assert finding.evidence, f"{finding.kind} has no evidence"
            assert finding.recommendation


def series(count: int, *, rising: bool, spread: str = "2") -> list[Candle]:
    """A trending or choppy candle series."""
    out: list[Candle] = []
    for i in range(count):
        price = Decimal("1000") + (Decimal(i) if rising else Decimal(i % 2) * Decimal("10"))
        out.append(
            Candle(
                symbol=BTC,
                timeframe=Timeframe.H1,
                open_time=NOW + timedelta(hours=i),
                open=price,
                high=price + Decimal(spread),
                low=price - Decimal(spread),
                close=price,
                volume=Decimal("100"),
                quote_volume=Decimal("100000"),
                trades=10,
            )
        )
    return out


class TestRegimeChange:
    """A change must say which axis moved, not merely that something did."""

    def test_identical_regimes_produce_nothing(self) -> None:
        profile = classify(series(200, rising=True))
        assert profile is not None
        assert ResearchAgent().detect_regime_change(profile, profile) == []

    def test_a_change_names_the_axis(self) -> None:
        trending = classify(series(200, rising=True))
        ranging = classify(series(200, rising=False))
        assert trending is not None
        assert ranging is not None
        findings = ResearchAgent().detect_regime_change(trending, ranging)
        assert findings
        assert any(key in findings[0].evidence for key in ("direction", "structure", "volatility"))

    def test_a_missing_observation_produces_nothing(self) -> None:
        # Two observations are needed to compare; inventing a baseline would be worse
        # than staying quiet.
        profile = classify(series(200, rising=True))
        assert ResearchAgent().detect_regime_change(None, profile) == []


class TestRegimeFit:
    """A regime-dependent strategy should be gated, not discarded."""

    def test_a_uniform_strategy_produces_no_finding(self) -> None:
        breakdown = RegimeBreakdown(
            by_regime=(
                RegimePerformance(
                    "bull/trending/normal", 50, Decimal("100"), Decimal("120"), Decimal("20"), 30
                ),
            )
        )
        assert ResearchAgent().assess_regime_fit(breakdown, strategy_id="x") == []

    def test_a_split_strategy_is_recommended_for_gating(self) -> None:
        breakdown = RegimeBreakdown(
            by_regime=(
                RegimePerformance(
                    "bull/trending/normal", 50, Decimal("400"), Decimal("450"), Decimal("50"), 30
                ),
                RegimePerformance(
                    "sideways/ranging/low", 40, Decimal("-350"), Decimal("-300"), Decimal("50"), 10
                ),
            )
        )
        findings = ResearchAgent().assess_regime_fit(breakdown, strategy_id="trend_follower")
        assert findings
        assert findings[0].kind is FindingKind.REGIME_MISMATCH
        assert "Gate trend_follower" in findings[0].recommendation
        # The blended figure is shown precisely because it is the misleading one.
        assert "blended_would_hide" in findings[0].evidence


class TestDiagnosisInterpretation:
    """Only execution-fixable diagnoses become recommendations."""

    def test_a_cost_problem_becomes_a_finding(self) -> None:
        diagnosis = Diagnosis(
            cause=FailureCause.COSTS,
            explanation="costs took 80%",
            recommendation="Try maker-only entries.",
            frictionless_return=Decimal("0.40"),
            cost_share=Decimal("0.80"),
        )
        findings = ResearchAgent().interpret_diagnosis(diagnosis, strategy_id="x")
        assert findings
        assert findings[0].evidence["return_without_costs"] == "40.00%"

    def test_a_worthless_signal_produces_no_recommendation(self) -> None:
        # There is nothing to recommend. Manufacturing advice here would imply the idea
        # is salvageable when the measurement says it is not.
        diagnosis = Diagnosis(
            cause=FailureCause.NO_SIGNAL,
            explanation="lost money for free",
            recommendation="Discard.",
        )
        assert ResearchAgent().interpret_diagnosis(diagnosis, strategy_id="x") == []


class TestReport:
    """The report must say what it declined to conclude, and why."""

    def test_missing_inputs_are_recorded_as_withheld(self) -> None:
        report = ResearchAgent().report(winners(30), now=NOW, strategy_id="x")
        assert report.withheld
        assert any("regime change" in item for item in report.withheld)

    def test_urgent_findings_sort_first(self) -> None:
        trades = winners(MIN_TRADES_FOR_ANALYSIS) + [
            trade(100 + i, "-5") for i in range(STREAK_THRESHOLD * 2)
        ]
        report = ResearchAgent().report(trades, now=NOW)
        assert report.findings[0].severity is Severity.URGENT

    def test_the_report_serialises(self) -> None:
        payload = ResearchAgent().report(winners(30), now=NOW).to_dict()
        assert payload["trades_analysed"] == 30
        assert "summary" in payload

    def test_the_agent_exposes_no_trading_surface(self) -> None:
        # The structural guarantee: there is no method here whose return value an
        # execution path would accept.
        agent = ResearchAgent()
        public = {name for name in dir(agent) if not name.startswith("_")}
        forbidden = {"generate", "evaluate", "place_order", "submit", "execute", "trade"}
        assert not (public & forbidden)

    def test_findings_are_deterministic(self) -> None:
        # A recommendation that changes between runs on identical data cannot be
        # reasoned about.
        trades = winners(MIN_TRADES_FOR_ANALYSIS) + [trade(100 + i, "-5") for i in range(5)]
        first = ResearchAgent().report(trades, now=NOW).to_dict()
        second = ResearchAgent().report(trades, now=NOW).to_dict()
        assert first == second
