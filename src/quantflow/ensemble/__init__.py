"""Ensemble trading: several strategies together, or nothing at all.

Members vote, votes are weighted by historical risk-adjusted performance, and the
ensemble acts only when weighted agreement clears a floor. Below it the ensemble holds —
"the members disagree" is information, and acting on a split decision means taking a
position nobody actually believed in.

Disagreement is never averaged away. One member long and another short does not make a
small long; it makes no trade, because there is no such thing as being slightly right
about direction.

The ensemble is a `Strategy` like any other and routes through the risk engine unchanged.
Combining strategies does not earn a shortcut around the gate, and structurally cannot
get one.
"""

from __future__ import annotations

from quantflow.ensemble.strategy import (
    EnsembleDecision,
    EnsembleParams,
    EnsembleStrategy,
    Vote,
)
from quantflow.ensemble.weights import (
    MAX_WEIGHT,
    MIN_TRADES_FOR_WEIGHTING,
    StrategyWeight,
    WeightSet,
    compute_weights,
    equal_weights,
    score_of,
)

__all__ = [
    "MAX_WEIGHT",
    "MIN_TRADES_FOR_WEIGHTING",
    "EnsembleDecision",
    "EnsembleParams",
    "EnsembleStrategy",
    "StrategyWeight",
    "Vote",
    "WeightSet",
    "compute_weights",
    "equal_weights",
    "score_of",
]
