"""Backtesting: the event-driven engine, metrics, walk-forward and reports."""

from __future__ import annotations

from quantflow.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    assert_no_lookahead,
    rejection_reasons,
    run_backtest,
    signal_summary,
)
from quantflow.backtest.metrics import PerformanceMetrics, compute_metrics

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "PerformanceMetrics",
    "assert_no_lookahead",
    "compute_metrics",
    "rejection_reasons",
    "run_backtest",
    "signal_summary",
]
