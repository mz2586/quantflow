"""The Strategy Laboratory.

Evaluates every strategy under every market regime, rejects the failures, and — the part
that makes it a laboratory rather than a scoreboard — explains *why* each one failed.

Every strategy is run twice: once under realistic costs and once with costs removed. That
second run is the whole point. "Lost 8%" is a symptom with at least three incompatible
causes — a worthless signal, a good signal handed to the venue, or a good signal traded
too often — and they lead to opposite decisions. Running the same strategy on the same
bars for free is the only measurement that tells them apart, and it costs one extra
backtest.
"""

from __future__ import annotations

from quantflow.lab.attribution import (
    RegimeBreakdown,
    RegimePerformance,
    RegimeTimeline,
    attribute,
    build_timelines,
    merge,
)
from quantflow.lab.diagnosis import Diagnosis, FailureCause, diagnose
from quantflow.lab.laboratory import LabReport, LabResult, StrategyLaboratory

__all__ = [
    "Diagnosis",
    "FailureCause",
    "LabReport",
    "LabResult",
    "RegimeBreakdown",
    "RegimePerformance",
    "RegimeTimeline",
    "StrategyLaboratory",
    "attribute",
    "build_timelines",
    "diagnose",
    "merge",
]
