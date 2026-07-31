"""Walk-forward window construction and HTML report generation.

The property that matters here: **train and test windows never overlap**. An overlap of
even one bar leaks the answer into the exam, and walk-forward's only purpose is to prevent
exactly that.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from quantflow.backtest.engine import BacktestConfig, BacktestEngine
from quantflow.backtest.metrics import compute_metrics
from quantflow.backtest.report import (
    cumulative_trades_figure,
    equity_figure,
    render_html,
    render_summary_json,
    trade_distribution_figure,
    write_report,
)
from quantflow.backtest.walkforward import (
    WalkForwardReport,
    Window,
    WindowResult,
    anchored_windows,
    describe_windows,
    rolling_windows,
    windows_from_bars,
)
from quantflow.core.errors import InsufficientDataError, ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import PositionSide, SignalDirection, Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import ClosedTrade
from tests.conftest import REFERENCE_TIME
from tests.unit.test_backtest import (
    ScriptedStrategy,
    flat_candles,
    instrument,
    permissive_risk,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)


class TestRollingWindows:
    def test_builds_consecutive_windows(self) -> None:
        windows = rolling_windows(START, END, train=timedelta(days=90), test=timedelta(days=30))
        assert len(windows) > 1
        assert windows[0].train_start == START
        assert windows[0].train_days == Decimal("90")
        assert windows[0].test_days == Decimal("30")

    def test_train_and_test_never_overlap(self) -> None:
        # The single property walk-forward exists to guarantee.
        for window in rolling_windows(
            START, END, train=timedelta(days=60), test=timedelta(days=20)
        ):
            assert window.train_end <= window.test_start
            assert window.train_start < window.train_end
            assert window.test_start < window.test_end

    def test_windows_advance_by_the_test_length_by_default(self) -> None:
        windows = rolling_windows(START, END, train=timedelta(days=90), test=timedelta(days=30))
        for first, second in pairwise(windows):
            assert second.train_start - first.train_start == timedelta(days=30)

    def test_custom_step(self) -> None:
        windows = rolling_windows(
            START,
            END,
            train=timedelta(days=90),
            test=timedelta(days=30),
            step=timedelta(days=60),
        )
        for first, second in pairwise(windows):
            assert second.train_start - first.train_start == timedelta(days=60)

    def test_never_runs_past_the_end(self) -> None:
        for window in rolling_windows(
            START, END, train=timedelta(days=90), test=timedelta(days=30)
        ):
            assert window.test_end <= END

    def test_too_short_a_range_raises(self) -> None:
        with pytest.raises(InsufficientDataError, match="too short"):
            rolling_windows(
                START,
                START + timedelta(days=10),
                train=timedelta(days=90),
                test=timedelta(days=30),
            )

    def test_invalid_windows_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            rolling_windows(START, END, train=timedelta(0), test=timedelta(days=30))
        with pytest.raises(ValidationError, match="end must be after start"):
            rolling_windows(END, START, train=timedelta(days=1), test=timedelta(days=1))


class TestAnchoredWindows:
    def test_train_start_is_fixed_and_the_window_grows(self) -> None:
        windows = anchored_windows(
            START, END, initial_train=timedelta(days=90), test=timedelta(days=30)
        )
        assert all(window.train_start == START for window in windows)
        for first, second in pairwise(windows):
            assert second.train_days > first.train_days

    def test_no_overlap(self) -> None:
        for window in anchored_windows(
            START, END, initial_train=timedelta(days=60), test=timedelta(days=30)
        ):
            assert window.train_end <= window.test_start


class TestBarWindows:
    def test_splits_by_bar_count(self, btc: Symbol) -> None:
        # More robust than duration when the data has gaps.
        candles = flat_candles(btc, ["100"] * 500)
        windows = windows_from_bars(candles, train_bars=200, test_bars=50)
        assert len(windows) == 6
        for window in windows:
            train, test = window.split(candles)
            assert len(train) == 200
            assert len(test) == 50
            assert not {c.open_time for c in train} & {c.open_time for c in test}

    def test_too_few_bars_raises(self, btc: Symbol) -> None:
        with pytest.raises(InsufficientDataError, match="need at least"):
            windows_from_bars(flat_candles(btc, ["100"] * 10), train_bars=200, test_bars=50)

    def test_invalid_sizes_are_rejected(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="at least 1"):
            windows_from_bars(flat_candles(btc, ["100"] * 100), train_bars=0, test_bars=5)


class TestWindowValidation:
    def test_overlapping_windows_are_rejected(self) -> None:
        # Constructing an overlapping window must be impossible, not merely discouraged.
        with pytest.raises(ValidationError, match="leak"):
            Window(
                index=0,
                train_start=START,
                train_end=START + timedelta(days=60),
                test_start=START + timedelta(days=30),  # overlaps the train window
                test_end=START + timedelta(days=90),
            )

    def test_empty_ranges_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty train"):
            Window(
                index=0,
                train_start=START,
                train_end=START,
                test_start=START,
                test_end=START + timedelta(days=1),
            )

    def test_split_is_half_open(self, btc: Symbol) -> None:
        candles = flat_candles(btc, ["100"] * 100)
        window = Window(
            index=0,
            train_start=REFERENCE_TIME,
            train_end=REFERENCE_TIME + timedelta(hours=50),
            test_start=REFERENCE_TIME + timedelta(hours=50),
            test_end=REFERENCE_TIME + timedelta(hours=80),
        )
        train, test = window.split(candles)
        assert len(train) == 50
        assert len(test) == 30
        assert train[-1].open_time < test[0].open_time

    def test_to_dict(self) -> None:
        window = rolling_windows(START, END, train=timedelta(days=90), test=timedelta(days=30))[0]
        described = window.to_dict()
        assert described["index"] == 0
        assert described["train_days"] == 90.0


def metrics_for(equities: list[str], pnls: list[str]):
    curve = [
        EquityPoint(
            timestamp=REFERENCE_TIME + timedelta(hours=index),
            equity=Decimal(value),
            cash=Decimal(value),
            position_count=1,
        )
        for index, value in enumerate(equities)
    ]
    trades = [
        ClosedTrade(
            symbol=Symbol(base="BTC", quote="USDT"),
            side=PositionSide.LONG,
            quantity=Decimal("1"),
            entry_price=Decimal("100"),
            exit_price=Decimal("100") + Decimal(pnl),
            entry_time=REFERENCE_TIME,
            exit_time=REFERENCE_TIME + timedelta(hours=1),
            gross_pnl=Decimal(pnl),
            fees=ZERO,
        )
        for pnl in pnls
    ]
    return compute_metrics(
        curve=curve,
        trades=trades,
        starting_equity=Decimal(equities[0]),
        timeframe=Timeframe.H1,
    )


class TestWalkForwardReport:
    def _result(self, index: int, *, generalises: bool) -> WindowResult:
        rising = ["10000", "10200", "10150", "10400", "10600", "10550", "11000"]
        falling = ["10000", "9900", "9950", "9700", "9500", "9550", "9000"]
        return WindowResult(
            window=Window(
                index=index,
                train_start=START + timedelta(days=index * 30),
                train_end=START + timedelta(days=index * 30 + 60),
                test_start=START + timedelta(days=index * 30 + 60),
                test_end=START + timedelta(days=index * 30 + 90),
            ),
            params={"fast": 12},
            in_sample=metrics_for(rising, ["100"] * 30),
            out_of_sample=metrics_for(
                rising if generalises else falling,
                ["100"] * 30 if generalises else ["-100"] * 30,
            ),
        )

    def test_generalisation_rate(self) -> None:
        report = WalkForwardReport(
            results=[
                self._result(0, generalises=True),
                self._result(1, generalises=True),
                self._result(2, generalises=False),
                self._result(3, generalises=False),
            ]
        )
        assert report.window_count == 4
        assert report.generalisation_rate == Decimal("0.5")

    def test_a_strategy_that_fails_out_of_sample_is_not_robust(self) -> None:
        # 1 in 4 windows surviving is curve-fitting, whatever the aggregate return says.
        report = WalkForwardReport(
            results=[
                self._result(0, generalises=True),
                self._result(1, generalises=False),
                self._result(2, generalises=False),
                self._result(3, generalises=False),
            ]
        )
        assert not report.is_robust

    def test_a_consistently_generalising_strategy_is_robust(self) -> None:
        report = WalkForwardReport(
            results=[self._result(index, generalises=True) for index in range(5)]
        )
        assert report.is_robust
        assert report.mean_out_of_sample_sharpe > ZERO

    def test_too_few_windows_is_never_robust(self) -> None:
        # Two windows cannot distinguish a real edge from luck.
        report = WalkForwardReport(
            results=[self._result(index, generalises=True) for index in range(2)]
        )
        assert not report.is_robust

    def test_empty_report(self) -> None:
        report = WalkForwardReport()
        assert report.generalisation_rate == ZERO
        assert report.mean_degradation == ZERO
        assert not report.is_robust

    def test_serialisation(self) -> None:
        report = WalkForwardReport(results=[self._result(0, generalises=True)])
        described = report.to_dict()
        assert "summary" in described
        assert len(described["windows"]) == 1
        assert "degradation" in described["windows"][0]

    def test_describe_windows(self) -> None:
        windows = rolling_windows(START, END, train=timedelta(days=60), test=timedelta(days=20))
        text = describe_windows(windows, Timeframe.H1)
        assert "walk-forward windows" in text
        assert describe_windows([], Timeframe.H1) == "no windows"


class TestHtmlReport:
    async def _result(self, btc: Symbol):
        candles = flat_candles(btc, [str(100 + (index % 13)) for index in range(60)])
        strategy = ScriptedStrategy(
            dict.fromkeys((5, 25, 45), SignalDirection.LONG)
            | dict.fromkeys((15, 35, 55), SignalDirection.CLOSE),
            warmup=1,
        )
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.H1, risk=permissive_risk())
        return await BacktestEngine(strategy, config, {btc: instrument(btc)}).run({btc: candles})

    async def test_renders_self_contained_html(self, btc: Symbol) -> None:
        html = render_html(await self._result(btc))
        assert html.startswith("<!doctype html>")
        assert "</html>" in html
        # No external stylesheet or asset: the report must render offline years later.
        assert "<style>" in html
        assert 'rel="stylesheet"' not in html

    async def test_leads_with_drawdown_not_return(self, btc: Symbol) -> None:
        # A large return from four trades is noise; leading with it invites that misread.
        html = render_html(await self._result(btc))
        assert html.index("Max drawdown") < html.index("Total return")

    async def test_flags_a_statistically_thin_result(self, btc: Symbol) -> None:
        html = render_html(await self._result(btc))
        assert "Caveats" in html
        assert "closed trades" in html

    async def test_escapes_untrusted_text(self, btc: Symbol) -> None:
        from dataclasses import replace

        result = await self._result(btc)
        injected = replace(result, strategy_id="<script>alert(1)</script>")
        html = render_html(injected)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    async def test_writes_to_disk(self, btc: Symbol, tmp_path: Path) -> None:
        path = write_report(await self._result(btc), tmp_path)
        assert path.exists()
        assert path.suffix == ".html"
        assert path.stat().st_size > 1000

    async def test_json_summary(self, btc: Symbol) -> None:
        summary = render_summary_json(await self._result(btc))
        assert summary["strategy_id"] == "scripted"
        assert "metrics" in summary
        assert "sharpe_ratio" in summary["metrics"]
        assert summary["statistically_thin"] is True

    async def test_figures_build_without_data(self) -> None:
        # An empty run must still produce a viewable report rather than crashing.
        assert equity_figure([]) is not None
        assert trade_distribution_figure([]) is not None
        assert cumulative_trades_figure([]) is not None

    async def test_report_of_an_empty_run(self, btc: Symbol) -> None:
        candles = flat_candles(btc, ["100"] * 20)
        config = BacktestConfig(symbols=(btc,), timeframe=Timeframe.H1, risk=permissive_risk())
        result = await BacktestEngine(
            ScriptedStrategy({}, warmup=1), config, {btc: instrument(btc)}
        ).run({btc: candles})
        html = render_html(result)
        assert "No closed trades" in html
