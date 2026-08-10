"""Tests for the strategy research framework."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.backtest.metrics import PerformanceMetrics
from quantflow.core.errors import ValidationError
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle
from quantflow.research.benchmark import buy_and_hold_metrics
from quantflow.research.costs import build_cost_model, pessimistic, realistic, zero_cost
from quantflow.research.leaderboard import METRICS, LeaderboardEntry, rank
from quantflow.research.report import build_html, build_json, build_markdown
from quantflow.research.runner import (
    MIN_HISTORY_BARS,
    ResearchConfig,
    ResearchOutcome,
    StrategyRun,
    history_window_for,
)
from quantflow.research.thresholds import (
    DEFAULT_THRESHOLDS,
    AcceptanceThresholds,
    RejectionCode,
    screen,
)
from quantflow.strategy.registry import load_builtin_strategies


def metrics(
    *,
    total_return: str = "0.50",
    profit_factor: str = "2.0",
    sharpe: str = "1.5",
    drawdown: str = "0.10",
    win_rate: str = "0.55",
    trades: int = 100,
    fees: str = "10",
    starting: str = "10000",
    final: str | None = None,
) -> PerformanceMetrics:
    """A metrics object that passes every threshold unless a field is overridden."""
    start = Decimal(starting)
    return PerformanceMetrics(
        starting_equity=start,
        final_equity=Decimal(final) if final else start * (Decimal("1") + Decimal(total_return)),
        total_return_pct=Decimal(total_return),
        cagr=Decimal("0.2"),
        max_drawdown_pct=Decimal(drawdown),
        max_drawdown_duration_days=Decimal("10"),
        volatility_annual=Decimal("0.3"),
        downside_volatility_annual=Decimal("0.2"),
        sharpe_ratio=Decimal(sharpe),
        sortino_ratio=Decimal("2.0"),
        calmar_ratio=Decimal("2.0"),
        trade_count=trades,
        win_count=int(trades * float(win_rate)),
        loss_count=trades - int(trades * float(win_rate)),
        win_rate=Decimal(win_rate),
        profit_factor=Decimal(profit_factor),
        expectancy=Decimal("5"),
        average_win=Decimal("20"),
        average_loss=Decimal("-10"),
        largest_win=Decimal("100"),
        largest_loss=Decimal("-50"),
        average_holding_hours=Decimal("12"),
        total_fees=Decimal(fees),
        turnover=Decimal("5"),
        exposure_pct=Decimal("0.5"),
        duration_days=Decimal("365"),
        bars=10_000,
    )


class TestThresholds:
    """The gate must reject mechanically, and say exactly why."""

    def test_a_good_result_passes(self) -> None:
        assert screen(metrics()).accepted

    def test_a_loss_is_rejected_as_a_loss(self) -> None:
        # Not merely "below minimum": losing money is a different failure from
        # underperforming, and collapsing the two hides which one happened.
        result = screen(metrics(total_return="-0.20", final="8000"))
        assert not result.accepted
        assert RejectionCode.NEGATIVE_RETURN in {r.code for r in result.rejections}

    def test_a_small_profit_is_rejected_as_insufficient(self) -> None:
        result = screen(metrics(total_return="0.01"))
        assert RejectionCode.INSUFFICIENT_RETURN in {r.code for r in result.rejections}

    @pytest.mark.parametrize(
        ("override", "code"),
        [
            ({"profit_factor": "1.0"}, RejectionCode.LOW_PROFIT_FACTOR),
            ({"sharpe": "0.1"}, RejectionCode.LOW_SHARPE),
            ({"drawdown": "0.60"}, RejectionCode.EXCESSIVE_DRAWDOWN),
            ({"trades": 5}, RejectionCode.TOO_FEW_TRADES),
            ({"trades": 90_000}, RejectionCode.TOO_MANY_TRADES),
            ({"win_rate": "0.05"}, RejectionCode.LOW_WIN_RATE),
        ],
    )
    def test_each_criterion_is_enforced(
        self, override: dict[str, object], code: RejectionCode
    ) -> None:
        result = screen(metrics(**override))  # type: ignore[arg-type]
        assert code in {r.code for r in result.rejections}

    def test_every_failure_is_reported_not_just_the_first(self) -> None:
        # One miss is a tuning problem; five is a dead end. Stopping at the first
        # failure would hide that difference.
        result = screen(metrics(profit_factor="0.5", sharpe="-1", drawdown="0.9", trades=3))
        assert len(result.rejections) >= 4

    def test_a_rejection_carries_the_number_that_caused_it(self) -> None:
        result = screen(metrics(sharpe="0.1"))
        rejection = next(r for r in result.rejections if r.code is RejectionCode.LOW_SHARPE)
        assert rejection.observed == Decimal("0.1")
        assert rejection.threshold == DEFAULT_THRESHOLDS.min_sharpe

    def test_losing_to_the_benchmark_is_a_rejection(self) -> None:
        result = screen(metrics(total_return="0.20"), benchmark_return=Decimal("0.50"))
        assert RejectionCode.LOST_TO_BENCHMARK in {r.code for r in result.rejections}

    def test_beating_the_benchmark_passes(self) -> None:
        assert screen(metrics(total_return="0.60"), benchmark_return=Decimal("0.50")).accepted

    def test_fee_dominated_results_are_rejected(self) -> None:
        # Net 100 on 900 of fees is 1000 gross with 90% given to the venue.
        result = screen(metrics(total_return="0.01", final="10100", fees="900"))
        assert RejectionCode.FEE_DOMINATED in {r.code for r in result.rejections}

    def test_the_benchmark_rule_can_be_switched_off(self) -> None:
        thresholds = AcceptanceThresholds(must_beat_benchmark=False)
        assert screen(metrics(), thresholds, benchmark_return=Decimal("9.0")).accepted

    def test_thresholds_describe_themselves(self) -> None:
        described = DEFAULT_THRESHOLDS.describe()
        assert "net return" in described
        assert described["beats buy-and-hold"] == "required"


class TestCosts:
    """Cost presets must be explicit and pessimistic."""

    def test_realistic_charges_bybit_base_tier_on_both_legs(self) -> None:
        assert realistic().round_trip_cost_pct() == Decimal("0.002")

    def test_pessimistic_is_strictly_worse_than_realistic(self) -> None:
        assert pessimistic().round_trip_cost_pct() > realistic().round_trip_cost_pct()

    def test_zero_cost_is_free(self) -> None:
        assert zero_cost().round_trip_cost_pct() == Decimal("0")

    def test_every_preset_explains_itself(self) -> None:
        # The summary is carried into the report; a reader must never have to guess
        # which assumptions produced the numbers.
        for name in ("realistic", "pessimistic", "zero_cost"):
            assert build_cost_model(name).summary

    def test_an_unknown_preset_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            build_cost_model("free_money")


class TestHistoryWindow:
    """The research window must scale with what a strategy declared it needs."""

    def test_the_window_is_a_multiple_of_the_warmup(self) -> None:
        assert history_window_for(1000) == 3000

    def test_a_short_warmup_still_gets_the_floor(self) -> None:
        assert history_window_for(5) == MIN_HISTORY_BARS

    def test_every_builtin_strategy_gets_more_than_it_needs(self) -> None:
        registry = load_builtin_strategies()
        for name in registry.names():
            strategy = registry.create(name, {})
            assert history_window_for(strategy.warmup_bars) > strategy.warmup_bars


def candles(count: int, *, start: str = "100", step: str = "1") -> list[Candle]:
    """A rising series, one hour apart."""
    symbol = Symbol(base="BTC", quote="USDT")
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out: list[Candle] = []
    price = Decimal(start)
    for index in range(count):
        out.append(
            Candle(
                symbol=symbol,
                timeframe=Timeframe.H1,
                open_time=base + timedelta(hours=index),
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price + Decimal(step),
                volume=Decimal("100"),
                quote_volume=Decimal("10000"),
                trades=10,
            )
        )
        price += Decimal(step)
    return out


class TestBenchmark:
    """Buy-and-hold must actually hold, and must still pay its costs."""

    def test_it_takes_exactly_one_trade(self) -> None:
        # The whole reason it is computed rather than traded: routed through the risk
        # engine it gets stopped out and re-enters, which is not buy-and-hold.
        result = buy_and_hold_metrics(
            Symbol(base="BTC", quote="USDT"),
            candles(200),
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
            costs=zero_cost(),
        )
        assert result.trade_count == 1

    def test_a_rising_market_produces_a_profit(self) -> None:
        result = buy_and_hold_metrics(
            Symbol(base="BTC", quote="USDT"),
            candles(200),
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
            costs=zero_cost(),
        )
        assert result.total_return_pct > 0

    def test_costs_reduce_the_return(self) -> None:
        series = candles(200)
        free = buy_and_hold_metrics(
            Symbol(base="BTC", quote="USDT"),
            series,
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
            costs=zero_cost(),
        )
        charged = buy_and_hold_metrics(
            Symbol(base="BTC", quote="USDT"),
            series,
            starting_equity=Decimal("10000"),
            timeframe=Timeframe.H1,
            costs=realistic(),
        )
        assert charged.total_return_pct < free.total_return_pct
        assert charged.total_fees > 0

    def test_an_empty_series_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            buy_and_hold_metrics(
                Symbol(base="BTC", quote="USDT"),
                [],
                starting_equity=Decimal("10000"),
                timeframe=Timeframe.H1,
                costs=realistic(),
            )


def entry(
    strategy_id: str,
    *,
    net_return: str = "0.5",
    sharpe: str = "1.0",
    drawdown: str = "0.1",
    win_rate: str = "0.5",
    profit_factor: str = "2.0",
    trades: int = 100,
    accepted: bool = True,
) -> LeaderboardEntry:
    """A leaderboard entry for ranking tests."""
    return LeaderboardEntry(
        strategy_id=strategy_id,
        symbols_tested=1,
        symbols_accepted=1 if accepted else 0,
        net_return=Decimal(net_return),
        profit_factor=Decimal(profit_factor),
        sharpe_ratio=Decimal(sharpe),
        max_drawdown=Decimal(drawdown),
        win_rate=Decimal(win_rate),
        trade_count=trades,
        total_fees=Decimal("10"),
        excess_return=Decimal("0.1"),
        worst_symbol_return=Decimal(net_return),
        accepted=accepted,
        rejection_summary="" if accepted else "failed",
        runs=(),
    )


class TestLeaderboard:
    """Ranking must be multi-metric and must not be gamed by one dimension."""

    def test_it_ranks_every_required_metric(self) -> None:
        assert {m.key for m in METRICS} == {
            "net_return",
            "profit_factor",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "trade_count",
        }

    def test_lower_drawdown_ranks_better(self) -> None:
        # The only metric where smaller wins; getting its direction wrong would invert
        # the safest strategy to the bottom of the table.
        board = rank([entry("deep", drawdown="0.4"), entry("shallow", drawdown="0.05")])
        ranks = {item.entry.strategy_id: item.ranks["max_drawdown"] for item in board}
        assert ranks["shallow"] < ranks["deep"]

    def test_rejected_strategies_sort_below_accepted_ones(self) -> None:
        # Even when the rejected one scores better on every metric: the gate is the
        # decision, and the composite only orders within it.
        board = rank(
            [
                entry("brilliant_but_rejected", net_return="9.0", sharpe="9.0", accepted=False),
                entry("modest_but_accepted", net_return="0.2", sharpe="0.6"),
            ]
        )
        assert board[0].entry.strategy_id == "modest_but_accepted"

    def test_one_huge_metric_cannot_dominate_the_composite(self) -> None:
        # Ranks are ordinal precisely so an unbounded metric cannot drown five bounded
        # ones. A strategy that wins only on return must not top the table.
        board = rank(
            [
                entry(
                    "one_trick",
                    net_return="99.0",
                    sharpe="0.1",
                    drawdown="0.9",
                    win_rate="0.1",
                    profit_factor="0.2",
                    trades=1,
                ),
                entry(
                    "balanced",
                    net_return="0.5",
                    sharpe="2.0",
                    drawdown="0.05",
                    win_rate="0.6",
                    profit_factor="3.0",
                    trades=200,
                ),
            ]
        )
        assert board[0].entry.strategy_id == "balanced"

    def test_a_flawless_record_marks_profit_factor_undefined(self) -> None:
        # With no losing trades the ratio is undefined, not enormous. compute_metrics
        # reports gross profit so the value sorts, but printing ~162,000 under a column
        # headed "profit factor" reads as a broken calculation - which is exactly how
        # buy-and-hold rendered before this flag existed.
        from quantflow.research.leaderboard import aggregate

        flawless = metrics(trades=4, win_rate="1.0", profit_factor="162203.47")
        run = StrategyRun(
            strategy_id="buy_and_hold",
            symbol=Symbol(base="BTC", quote="USDT"),
            metrics=flawless,
            screen=screen(flawless),
            params={},
            bars=1000,
            signals=1,
            orders=1,
            rejected_signals=0,
            duration_seconds=0.0,
        )
        entry_built = aggregate(outcome_with((run,)))[0]
        assert entry_built.profit_factor_undefined

    def test_a_record_with_losses_reports_a_real_profit_factor(self) -> None:
        from quantflow.research.leaderboard import aggregate

        entry_built = aggregate(outcome_with((run_for("ema_cross", accepted=True),)))[0]
        assert not entry_built.profit_factor_undefined

    def test_ranking_an_empty_board_is_not_an_error(self) -> None:
        assert rank([]) == ()

    def test_positions_are_dense_and_one_based(self) -> None:
        board = rank([entry("a"), entry("b"), entry("c")])
        assert [item.position for item in board] == [1, 2, 3]


def outcome_with(runs: tuple[StrategyRun, ...] = ()) -> ResearchOutcome:
    """A minimal outcome for report tests."""
    return ResearchOutcome(
        config=ResearchConfig(symbols=(Symbol(base="BTC", quote="USDT"),)),
        runs=runs,
        failures=(),
        bars_per_symbol={"BTC/USDT": 1000},
        period_start="2021-01-01T00:00:00+00:00",
        period_end="2026-01-01T00:00:00+00:00",
        duration_seconds=1.0,
    )


def run_for(strategy_id: str, *, accepted: bool) -> StrategyRun:
    """A completed run for report tests."""
    result = screen(metrics()) if accepted else screen(metrics(sharpe="-1"))
    return StrategyRun(
        strategy_id=strategy_id,
        symbol=Symbol(base="BTC", quote="USDT"),
        metrics=metrics() if accepted else metrics(sharpe="-1"),
        screen=result,
        params={},
        bars=1000,
        signals=10,
        orders=10,
        rejected_signals=0,
        duration_seconds=0.1,
    )


class TestReport:
    """Reports must state their assumptions and survive being opened offline."""

    def test_markdown_states_the_period_costs_and_thresholds(self) -> None:
        text = build_markdown(outcome_with((run_for("ema_cross", accepted=True),)))
        assert "2021-01-01" in text
        assert "taker" in text
        assert "Acceptance thresholds" in text

    def test_markdown_lists_rejection_reasons(self) -> None:
        text = build_markdown(outcome_with((run_for("ema_cross", accepted=False),)))
        assert "low_sharpe" in text

    def test_html_is_self_contained(self) -> None:
        # A report that needs a CDN stops rendering the moment it is opened somewhere
        # without one, and a research record that decays is not a record.
        page = build_html(outcome_with((run_for("ema_cross", accepted=True),)))
        assert "<style>" in page
        assert "http://" not in page
        assert "https://" not in page

    def test_an_undefined_profit_factor_is_named_not_printed(self) -> None:
        from quantflow.research.leaderboard import aggregate, rank

        flawless = metrics(trades=4, win_rate="1.0", profit_factor="162203.47")
        run = StrategyRun(
            strategy_id="buy_and_hold",
            symbol=Symbol(base="BTC", quote="USDT"),
            metrics=flawless,
            screen=screen(flawless),
            params={},
            bars=1000,
            signals=1,
            orders=1,
            rejected_signals=0,
            duration_seconds=0.0,
        )
        outcome = outcome_with((run,))
        assert "no losses" in build_markdown(outcome)
        assert "162203" not in build_markdown(outcome)
        assert rank(aggregate(outcome))[0].entry.profit_factor_undefined

    def test_json_round_trips(self) -> None:
        import json

        payload = json.loads(build_json(outcome_with((run_for("ema_cross", accepted=True),))))
        assert payload["leaderboard"][0]["strategy_id"] == "ema_cross"
        assert payload["symbols"] == ["BTC/USDT"]

    def test_reports_handle_an_empty_sweep(self) -> None:
        empty = outcome_with()
        assert build_markdown(empty)
        assert build_html(empty)
        assert build_json(empty)


class TestStrategyLibrary:
    """The library must be broad enough for the leaderboard to mean something."""

    def test_there_are_at_least_ten_strategies(self) -> None:
        assert len(load_builtin_strategies()) >= 10

    def test_every_strategy_declares_an_id_and_description(self) -> None:
        registry = load_builtin_strategies()
        for description in registry.describe_all():
            assert description["strategy_id"]
            assert description["description"]

    def test_strategy_ids_are_unique(self) -> None:
        names = load_builtin_strategies().names()
        assert len(names) == len(set(names))

    def test_every_strategy_has_a_positive_warmup(self) -> None:
        registry = load_builtin_strategies()
        for name in registry.names():
            assert registry.create(name, {}).warmup_bars >= 1


class TestWorkerSizing:
    """Pool size must account for memory, not just cores."""

    def test_an_explicit_request_is_honoured(self) -> None:
        from quantflow.research.runner import workers_for

        assert workers_for(1_000_000, requested=2) == 2

    def test_a_request_of_zero_still_yields_one_worker(self) -> None:
        from quantflow.research.runner import workers_for

        assert workers_for(1000, requested=0) == 1

    def test_a_huge_dataset_reduces_the_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The failure this guards against: sizing by CPU alone, every worker receiving
        # its own copy of the dataset under spawn, and the pool never starting. That
        # presents as a hang, not an error - it stalled a run for five hours with 112 MB
        # free and zero worker children.
        from quantflow.research import runner

        monkeypatch.setattr(runner, "available_memory_bytes", lambda: 100 * 1024 * 1024)
        assert runner.workers_for(10_000_000) == 1

    def test_a_small_dataset_uses_the_cpu_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quantflow.research import runner

        monkeypatch.setattr(runner, "available_memory_bytes", lambda: 64 * 1024**3)
        assert runner.workers_for(100) == runner.default_worker_count()

    def test_unknown_memory_falls_back_to_the_cpu_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Guessing low would serialise every sweep on platforms we cannot measure.
        from quantflow.research import runner

        monkeypatch.setattr(runner, "available_memory_bytes", lambda: None)
        assert runner.workers_for(1_000_000) == runner.default_worker_count()
