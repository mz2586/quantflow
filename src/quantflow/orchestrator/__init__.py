"""Strategy orchestration.

Sits between the strategy library and the existing risk/execution stack: evaluates every
registered strategy on each completed bar and forwards the strongest valid candidate, or
nothing at all when none is strong enough.
"""

from __future__ import annotations

from quantflow.orchestrator.scoring import (
    MIN_SCORE_TO_TRADE,
    MIN_TRADES_FOR_EVIDENCE,
    WEIGHTS,
    Candidate,
    StrategyRecord,
    rank,
    score_candidate,
)
from quantflow.orchestrator.strategy import (
    DEFAULT_EXCLUDED,
    Decision,
    OrchestratorParams,
    StrategyOrchestrator,
)

__all__ = [
    "DEFAULT_EXCLUDED",
    "MIN_SCORE_TO_TRADE",
    "MIN_TRADES_FOR_EVIDENCE",
    "WEIGHTS",
    "Candidate",
    "Decision",
    "OrchestratorParams",
    "StrategyOrchestrator",
    "StrategyRecord",
    "rank",
    "score_candidate",
]
