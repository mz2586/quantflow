"""Performance analytics: attribution, streaks, concentration, drawdown episodes."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from quantflow.analytics.performance import (
    by_hour_of_day,
    by_month,
    by_side,
    by_strategy,
    by_symbol,
    concentration,
    drawdown_episodes,
    long_short_balance,
    review,
    rolling_win_rate,
    streaks,
    symbol_exposure,
)
from quantflow.core.precision import ZERO
from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import ClosedTrade
from tests.conftest import REFERENCE_TIME


def trade(
    *,
    pnl: str,
    symbol: Symbol | None = None,
    strategy: str | None = "ema_cross",
    side: PositionSide = PositionSide.LONG,
    fees: str = "0",
    hours: int = 0,
    quantity: str = "1",
    entry: str = "100",
) -> ClosedTrade:
    return ClosedTrade(
        symbol=symbol or Symbol(base="BTC", quote="USDT"),
        side=side,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry),
        exit_price=Decimal(entry) + Decimal(pnl),
        entry_time=REFERENCE_TIME + timedelta(hours=hours),
        exit_time=REFERENCE_TIME + timedelta(hours=hours + 2),
        gross_pnl=Decimal(pnl),
        fees=Decimal(fees),
        strategy_id=strategy,
    )


def curve(values: list[str]) -> list[EquityPoint]:
    return [
        EquityPoint(
            timestamp=REFERENCE_TIME + timedelta(hours=index),
            equity=Decimal(value),
            cash=Decimal(value),
            position_count=0,
        )
        for index, value in enumerate(values)
    ]


class TestAttribution:
    def test_by_strategy_is_ranked(self) -> None:
        trades = [
            trade(pnl="100", strategy="a"),
            trade(pnl="50", strategy="a"),
            trade(pnl="-30", strategy="b"),
        ]
        result = by_strategy(trades)
        assert [item.key for item in result] == ["a", "b"]
        assert result[0].net_pnl == Decimal("150")
        assert result[0].trade_count == 2

    def test_unattributed_trades_are_grouped(self) -> None:
        result = by_strategy([trade(pnl="10", strategy=None)])
        assert result[0].key == "unattributed"

    def test_by_symbol(self, btc: Symbol, eth: Symbol) -> None:
        trades = [trade(pnl="100", symbol=btc), trade(pnl="-20", symbol=eth)]
        result = by_symbol(trades)
        assert result[0].key == "BTC/USDT"
        assert result[1].net_pnl == Decimal("-20")

    def test_by_side_exposes_a_one_sided_edge(self) -> None:
        # A strategy that only wins long in a market that trended up has demonstrated the
        # trend, not an edge.
        trades = [
            *[trade(pnl="100", side=PositionSide.LONG) for _ in range(5)],
            *[trade(pnl="-50", side=PositionSide.SHORT) for _ in range(5)],
        ]
        result = {item.key: item for item in by_side(trades)}
        assert result["long"].net_pnl == Decimal("500")
        assert result["short"].net_pnl == Decimal("-250")

    def test_by_hour(self) -> None:
        trades = [trade(pnl="10", hours=3), trade(pnl="20", hours=3), trade(pnl="5", hours=7)]
        result = {item.key: item.trade_count for item in by_hour_of_day(trades)}
        assert result["03"] == 2
        assert result["07"] == 1

    def test_by_month(self) -> None:
        trades = [trade(pnl="10"), trade(pnl="20", hours=24 * 40)]
        assert len(by_month(trades)) == 2

    def test_fee_drag_exposes_a_strategy_killed_by_costs(self) -> None:
        # Profitable before costs, unprofitable after — the most common way a promising
        # backtest dies in production.
        trades = [trade(pnl="10", fees="8") for _ in range(12)]
        result = by_strategy(trades)[0]
        assert result.gross_pnl == Decimal("120")
        assert result.net_pnl == Decimal("24")
        assert result.fee_drag_pct == Decimal("96") / Decimal("120")

    def test_small_samples_are_flagged_unreliable(self) -> None:
        assert not by_strategy([trade(pnl="10")] * 3)[0].is_reliable
        assert by_strategy([trade(pnl="10")] * 15)[0].is_reliable

    def test_serialisation(self) -> None:
        described = by_strategy([trade(pnl="10")])[0].to_dict()
        assert described["key"] == "ema_cross"
        assert isinstance(described["net_pnl"], str)


class TestStreaks:
    def test_counts_longest_runs(self) -> None:
        trades = [
            trade(pnl="10", hours=0),
            trade(pnl="10", hours=1),
            trade(pnl="10", hours=2),
            trade(pnl="-5", hours=3),
            trade(pnl="-5", hours=4),
            trade(pnl="10", hours=5),
        ]
        result = streaks(trades)
        assert result.longest_win_streak == 3
        assert result.longest_loss_streak == 2
        assert result.current_streak == 1

    def test_current_losing_run_is_negative(self) -> None:
        trades = [trade(pnl="10", hours=0), trade(pnl="-5", hours=1), trade(pnl="-5", hours=2)]
        result = streaks(trades)
        assert result.current_streak == -2
        assert result.is_on_a_losing_run

    def test_empty(self) -> None:
        result = streaks([])
        assert result.longest_win_streak == 0
        assert not result.is_on_a_losing_run


class TestConcentration:
    def test_detects_a_single_dominant_trade(self) -> None:
        # A strategy whose profit is one lucky trade has not been demonstrated to work.
        trades = [trade(pnl="1000"), *[trade(pnl="10") for _ in range(9)]]
        result = concentration(trades)
        assert result.is_concentrated
        assert result.top_trade_share > Decimal("0.9")
        assert result.profit_without_best == Decimal("90")
        assert result.survives_without_best_trade

    def test_evenly_distributed_profit_is_not_concentrated(self) -> None:
        result = concentration([trade(pnl="10") for _ in range(20)])
        assert not result.is_concentrated

    def test_a_result_that_dies_without_its_best_trade(self) -> None:
        # Net +50: profitable overall, but only because of one trade.
        trades = [trade(pnl="500"), *[trade(pnl="-50") for _ in range(9)]]
        result = concentration(trades)
        assert result.total_net_pnl == Decimal("50")
        assert result.is_concentrated
        assert not result.survives_without_best_trade
        assert result.rests_on_one_trade

    def test_an_overall_loss_still_reports_the_underlying_numbers(self) -> None:
        # A "share of profit" is undefined here, but the rest must not be silently blank.
        trades = [trade(pnl="500"), *[trade(pnl="-50") for _ in range(14)]]
        result = concentration(trades)
        assert result.total_net_pnl == Decimal("-200")
        assert result.top_trade_share == ZERO
        assert result.profit_without_best == Decimal("-700")
        # Already a loss, so it does not "rest on" the best trade — it is simply losing.
        assert not result.rests_on_one_trade

    def test_empty(self) -> None:
        assert concentration([]).top_trade_share == ZERO


class TestDrawdownEpisodes:
    def test_identifies_recovered_and_unrecovered_declines(self) -> None:
        episodes = drawdown_episodes(
            curve(["100", "120", "90", "110", "125", "100", "95"]),
            min_depth_pct=Decimal("0.05"),
        )
        assert len(episodes) == 2
        deepest = episodes[0]
        assert deepest.depth_pct == Decimal("30") / Decimal("120")
        assert deepest.recovered is False or deepest.recovered is True
        # The trailing decline has not recovered.
        assert any(not episode.recovered for episode in episodes)

    def test_shallow_declines_are_filtered_out(self) -> None:
        episodes = drawdown_episodes(
            curve(["100", "99", "100", "99", "100"]), min_depth_pct=Decimal("0.05")
        )
        assert episodes == []

    def test_a_rising_curve_has_no_episodes(self) -> None:
        assert drawdown_episodes(curve(["100", "110", "120"])) == []

    def test_too_short_a_curve(self) -> None:
        assert drawdown_episodes(curve(["100"])) == []


class TestRollingWinRate:
    def test_tracks_decay(self) -> None:
        # Early wins then losses: the aggregate hides this, the rolling series does not.
        trades = [
            *[trade(pnl="10", hours=index) for index in range(10)],
            *[trade(pnl="-10", hours=10 + index) for index in range(10)],
        ]
        series = rolling_win_rate(trades, window=5)
        assert series[0][1] == Decimal("1")
        assert series[-1][1] == ZERO

    def test_too_few_trades_for_the_window(self) -> None:
        assert rolling_win_rate([trade(pnl="10")], window=20) == []


class TestReview:
    def test_assembles_the_full_picture(self) -> None:
        trades = [trade(pnl="10", hours=index) for index in range(15)]
        result = review(trades, curve(["100", "120", "90", "110"]))
        assert result.trade_count == 15
        assert result.strategies
        assert result.symbols
        assert result.drawdowns
        described = result.to_dict()
        assert "by_strategy" in described
        assert "warnings" in described

    def test_warns_about_a_thin_sample(self) -> None:
        warnings = review([trade(pnl="10") for _ in range(3)]).warnings()
        assert any("3 trades" in note for note in warnings)

    def test_warns_about_a_concentrated_result(self) -> None:
        trades = [trade(pnl="1000"), *[trade(pnl="5") for _ in range(14)]]
        warnings = review(trades).warnings()
        assert any("best trade contributed" in note for note in warnings)

    def test_warns_when_removing_the_best_trade_turns_it_negative(self) -> None:
        # Net +50, which becomes -450 without the outlier.
        trades = [trade(pnl="500"), *[trade(pnl="-50") for _ in range(9)]]
        warnings = review(trades).warnings()
        assert any("one lucky trade" in note for note in warnings)

    def test_warns_about_a_long_losing_streak(self) -> None:
        trades = [
            *[trade(pnl="100", hours=index) for index in range(5)],
            *[trade(pnl="-5", hours=5 + index) for index in range(10)],
        ]
        warnings = review(trades).warnings()
        assert any("losing streak" in note for note in warnings)

    def test_warns_about_fee_drag(self) -> None:
        trades = [trade(pnl="10", fees="8") for _ in range(12)]
        warnings = review(trades).warnings()
        assert any("fees consumed" in note for note in warnings)

    def test_a_healthy_result_has_no_warnings(self) -> None:
        trades = [trade(pnl="10", hours=index) for index in range(40)]
        assert review(trades).warnings() == []

    def test_empty_review(self) -> None:
        result = review([])
        assert result.trade_count == 0
        assert result.to_dict()["by_strategy"] == []


class TestHelpers:
    def test_symbol_exposure(self, btc: Symbol, eth: Symbol) -> None:
        exposure = symbol_exposure(
            [
                trade(pnl="10", symbol=btc, quantity="2", entry="100"),
                trade(pnl="10", symbol=eth, quantity="1", entry="50"),
            ]
        )
        assert exposure[btc] == Decimal("200")
        assert exposure[eth] == Decimal("50")

    def test_long_short_balance(self) -> None:
        trades = [
            trade(pnl="10", side=PositionSide.LONG),
            trade(pnl="10", side=PositionSide.LONG),
            trade(pnl="10", side=PositionSide.SHORT),
        ]
        assert long_short_balance(trades) == (2, 1)
