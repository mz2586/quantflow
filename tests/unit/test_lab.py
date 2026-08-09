"""Tests for the Strategy Laboratory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quantflow.backtest.metrics import PerformanceMetrics
from quantflow.domain.enums import PositionSide, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle
from quantflow.domain.positions import ClosedTrade
from quantflow.lab.attribution import (
    MIN_TRADES_PER_REGIME,
    RegimeTimeline,
    attribute,
    merge,
)
from quantflow.lab.diagnosis import FailureCause, diagnose

BTC = Symbol(base="BTC", quote="USDT")
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def metrics(
    *,
    total_return: str = "0.20",
    trades: int = 100,
    fees: str = "50",
    drawdown: str = "0.10",
    starting: str = "10000",
) -> PerformanceMetrics:
    """Metrics with only the fields the diagnosis reads set meaningfully."""
    start = Decimal(starting)
    return PerformanceMetrics(
        starting_equity=start,
        final_equity=start * (Decimal("1") + Decimal(total_return)),
        total_return_pct=Decimal(total_return),
        cagr=Decimal("0.1"),
        max_drawdown_pct=Decimal(drawdown),
        max_drawdown_duration_days=Decimal("10"),
        volatility_annual=Decimal("0.3"),
        downside_volatility_annual=Decimal("0.2"),
        sharpe_ratio=Decimal("1"),
        sortino_ratio=Decimal("1"),
        calmar_ratio=Decimal("1"),
        trade_count=trades,
        win_count=trades // 2,
        loss_count=trades - trades // 2,
        win_rate=Decimal("0.5"),
        profit_factor=Decimal("1.1"),
        expectancy=Decimal("1"),
        average_win=Decimal("10"),
        average_loss=Decimal("-8"),
        largest_win=Decimal("50"),
        largest_loss=Decimal("-40"),
        average_holding_hours=Decimal("12"),
        total_fees=Decimal(fees),
        turnover=Decimal("5"),
        exposure_pct=Decimal("0.5"),
        duration_days=Decimal("365"),
        bars=10_000,
    )


class TestDiagnosis:
    """The cause must be actionable, not a restatement of the symptom."""

    def test_a_thin_sample_is_not_a_verdict(self) -> None:
        result = diagnose(metrics(trades=5))
        assert result.cause is FailureCause.INSUFFICIENT_SAMPLE
        assert not result.is_fixable_by_execution

    def test_a_signal_that_loses_for_free_is_worthless(self) -> None:
        # The decisive comparison. No execution work rescues this.
        result = diagnose(
            metrics(total_return="-0.08"),
            frictionless=metrics(total_return="-0.05"),
        )
        assert result.cause is FailureCause.NO_SIGNAL
        assert not result.is_fixable_by_execution
        assert "no edge" in result.explanation

    def test_costs_are_named_when_the_signal_worked_before_them(self) -> None:
        # Net 2% on 4,000 of fees against a frictionless 40%: the venue took it.
        result = diagnose(
            metrics(total_return="0.02", fees="4000", trades=100),
            frictionless=metrics(total_return="0.40", trades=100),
        )
        assert result.cause is FailureCause.COSTS
        assert result.is_fixable_by_execution
        assert "maker-only" in result.recommendation

    def test_over_trading_is_distinguished_from_costs(self) -> None:
        # Positive before costs, but the per-trade edge is smaller than any friction.
        result = diagnose(
            metrics(total_return="-0.05", trades=3000, fees="900"),
            frictionless=metrics(total_return="0.02", trades=3000),
        )
        assert result.cause is FailureCause.OVER_TRADING
        assert result.is_fixable_by_execution
        assert "slower timeframe" in result.recommendation

    def test_a_profitable_but_unholdable_strategy_is_flagged_for_sizing(self) -> None:
        # Returns are not the binding constraint here, so this outranks signal quality.
        result = diagnose(metrics(total_return="0.50", drawdown="0.80"))
        assert result.cause is FailureCause.RISK_OF_RUIN
        assert "position size" in result.recommendation

    def test_without_a_cost_free_run_the_diagnosis_says_so(self) -> None:
        result = diagnose(metrics(total_return="-0.05"))
        assert "cannot be separated" in result.explanation
        assert "zero_cost" in result.recommendation

    def test_the_cost_share_is_reported(self) -> None:
        result = diagnose(
            metrics(total_return="0.02", fees="4000"),
            frictionless=metrics(total_return="0.40"),
        )
        assert result.cost_share is not None
        assert result.cost_share > Decimal("0.9")

    def test_the_payload_round_trips(self) -> None:
        payload = diagnose(metrics(trades=5)).to_dict()
        assert payload["cause"] == "insufficient_sample"
        assert payload["fixable_by_execution"] is False


def candles(count: int, *, rising: bool = True) -> list[Candle]:
    """A trending or choppy series."""
    out: list[Candle] = []
    for i in range(count):
        price = Decimal("1000") + (Decimal(i) if rising else Decimal(i % 2 * 10))
        out.append(
            Candle(
                symbol=BTC,
                timeframe=Timeframe.H1,
                open_time=BASE + timedelta(hours=i),
                open=price,
                high=price + Decimal("2"),
                low=price - Decimal("2"),
                close=price,
                volume=Decimal("100"),
                quote_volume=Decimal("100000"),
                trades=10,
            )
        )
    return out


def trade(hour: int, net: str) -> ClosedTrade:
    """A round-trip opened at ``hour`` with the given net PnL."""
    return ClosedTrade(
        symbol=BTC,
        side=PositionSide.LONG,
        quantity=Decimal("1"),
        entry_price=Decimal("1000"),
        exit_price=Decimal("1000") + Decimal(net),
        entry_time=BASE + timedelta(hours=hour),
        exit_time=BASE + timedelta(hours=hour + 1),
        gross_pnl=Decimal(net),
        fees=Decimal("0.5"),
    )


class TestRegimeTimeline:
    """Regime lookup must never read a label from the future."""

    def test_it_classifies_a_trending_series(self) -> None:
        timeline = RegimeTimeline.build(candles(300), stride=24)
        assert len(timeline) > 0
        assert any("trending" in label for label in timeline.labels)

    def test_lookup_before_the_first_classification_is_none(self) -> None:
        timeline = RegimeTimeline.build(candles(300), stride=24)
        assert timeline.at(BASE - timedelta(days=1)) is None

    def test_lookup_returns_the_most_recent_past_label(self) -> None:
        # Looking forward would be a look-ahead: a regime is only known once its bars
        # have closed.
        timeline = RegimeTimeline.build(candles(300), stride=24)
        label = timeline.at(BASE + timedelta(hours=299))
        assert label is not None

    def test_an_unclassifiable_series_yields_an_empty_timeline(self) -> None:
        assert len(RegimeTimeline.build(candles(5))) == 0


class TestAttribution:
    """Per-regime breakdown separates 'does not work' from 'works, sometimes'."""

    def test_trades_are_bucketed_by_entry_regime(self) -> None:
        timeline = RegimeTimeline.build(candles(300), stride=24)
        trades = [trade(hour, "10") for hour in range(100, 160)]
        breakdown = attribute(trades, timeline)
        assert breakdown.by_regime
        assert sum(item.trade_count for item in breakdown.by_regime) == len(trades)

    def test_trades_before_any_classification_are_marked_unclassified(self) -> None:
        # Silently dropping them would make the PnL in the breakdown disagree with the
        # PnL in the leaderboard, with no indication why.
        timeline = RegimeTimeline.build(candles(300), stride=24)
        breakdown = attribute([trade(0, "10")], timeline)
        assert breakdown.by_regime[0].regime == "unclassified"

    def test_a_thin_regime_sample_is_flagged_unreliable(self) -> None:
        timeline = RegimeTimeline.build(candles(300), stride=24)
        breakdown = attribute([trade(200, "10")], timeline)
        assert not breakdown.by_regime[0].is_reliable

    def test_regime_dependence_requires_a_reliable_winner_and_loser(self) -> None:
        timeline = RegimeTimeline.build(candles(300), stride=24)
        breakdown = attribute([trade(200, "10")] * MIN_TRADES_PER_REGIME, timeline)
        # One profitable regime and no reliable losing one is not regime dependence.
        assert not breakdown.is_regime_dependent

    def test_merge_sums_the_same_regime_across_symbols(self) -> None:
        timeline = RegimeTimeline.build(candles(300), stride=24)
        first = attribute([trade(200, "10") for _ in range(12)], timeline)
        second = attribute([trade(200, "10") for _ in range(12)], timeline)
        combined = merge([first, second])
        assert sum(item.trade_count for item in combined.by_regime) == 24

    def test_an_empty_breakdown_summarises_honestly(self) -> None:
        timeline = RegimeTimeline.build(candles(300), stride=24)
        assert attribute([], timeline).summary() == "no trades to attribute"
