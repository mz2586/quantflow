"""AI engine.

The AI **advises**; it never trades. Every decision it touches still flows through the
strategy contract and then through the risk engine, unchanged. Its interface is
deliberately asymmetric: it can veto a signal or shrink its conviction, and there is no
field anywhere in it that can create a position, flip a side, widen a stop or raise
conviction.
"""

from __future__ import annotations

from quantflow.ai.decision import (
    AIAdvice,
    AIDecisionEngine,
    RegimeAdvisor,
    assert_risk_reducing,
    build_engine,
)
from quantflow.ai.regime import (
    GaussianMixtureRegimeDetector,
    RegimeDetector,
    RegimeFeatures,
    RegimeObservation,
    RuleBasedRegimeDetector,
    build_detector,
    extract_features,
    regime_history,
)
from quantflow.ai.research_agent import (
    Finding,
    FindingKind,
    ResearchAgent,
    ResearchReport,
    Severity,
)
from quantflow.ai.strategy import AIAugmentedStrategy, wrap

__all__ = [
    "AIAdvice",
    "AIAugmentedStrategy",
    "AIDecisionEngine",
    "Finding",
    "FindingKind",
    "GaussianMixtureRegimeDetector",
    "RegimeAdvisor",
    "RegimeDetector",
    "RegimeFeatures",
    "RegimeObservation",
    "ResearchAgent",
    "ResearchReport",
    "RuleBasedRegimeDetector",
    "Severity",
    "assert_risk_reducing",
    "build_detector",
    "build_engine",
    "extract_features",
    "regime_history",
    "wrap",
]
