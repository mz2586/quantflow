"""Strategy research framework.

Backtests every registered strategy over the same data, under the same realistic costs,
screens each result against thresholds fixed in advance, and produces a ranked leaderboard
with the reasons for every rejection attached.

The framework's job is to *reject*. Finding a strategy that works is rare; the common and
far more valuable outcome is establishing, on evidence and before any money moves, that a
strategy does not.
"""

from __future__ import annotations

from quantflow.research.costs import CostModel, build_cost_model, pessimistic, realistic, zero_cost
from quantflow.research.leaderboard import LeaderboardEntry, RankedEntry, aggregate, leaderboard
from quantflow.research.report import build_html, build_json, build_markdown
from quantflow.research.runner import (
    BENCHMARK_STRATEGY_ID,
    FailedRun,
    ResearchConfig,
    ResearchOutcome,
    ResearchRunner,
    StrategyRun,
)
from quantflow.research.thresholds import (
    DEFAULT_THRESHOLDS,
    AcceptanceThresholds,
    Rejection,
    RejectionCode,
    ScreenResult,
    screen,
)

__all__ = [
    "BENCHMARK_STRATEGY_ID",
    "DEFAULT_THRESHOLDS",
    "AcceptanceThresholds",
    "CostModel",
    "FailedRun",
    "LeaderboardEntry",
    "RankedEntry",
    "Rejection",
    "RejectionCode",
    "ResearchConfig",
    "ResearchOutcome",
    "ResearchRunner",
    "ScreenResult",
    "StrategyRun",
    "aggregate",
    "build_cost_model",
    "build_html",
    "build_json",
    "build_markdown",
    "leaderboard",
    "pessimistic",
    "realistic",
    "screen",
    "zero_cost",
]
