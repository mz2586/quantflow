"""Risk engine: position sizing, hard limits and the emergency kill switch.

Every order in the system passes through `RiskEngine.approve`. There is no other path from
a strategy signal to an exchange.
"""

from __future__ import annotations

from quantflow.risk.engine import (
    RiskDecision,
    RiskEngine,
    assert_protected,
    summarise_headroom,
)
from quantflow.risk.killswitch import KillSwitch, KillSwitchState
from quantflow.risk.rules import RiskContext, RiskRule, RiskVerdict, build_default_rules
from quantflow.risk.sizing import (
    FixedFractionalSizer,
    FixedNotionalSizer,
    PositionSizer,
    SizingRequest,
    SizingResult,
    VolatilityTargetSizer,
    build_sizer,
)

__all__ = [
    "FixedFractionalSizer",
    "FixedNotionalSizer",
    "KillSwitch",
    "KillSwitchState",
    "PositionSizer",
    "RiskContext",
    "RiskDecision",
    "RiskEngine",
    "RiskRule",
    "RiskVerdict",
    "SizingRequest",
    "SizingResult",
    "VolatilityTargetSizer",
    "assert_protected",
    "build_default_rules",
    "build_sizer",
    "summarise_headroom",
]
