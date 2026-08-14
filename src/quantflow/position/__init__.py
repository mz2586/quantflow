"""Position management between candle closes.

The strategy layer only speaks when a bar completes. Everything in this package exists to
answer the question the strategy layer cannot: what should happen to an *already open*
position in the minutes between one close and the next.
"""

from quantflow.position.intrabar import (
    DEFAULT_STAGES,
    PRIORITY_EXCHANGE_STOP,
    PRIORITY_INTRABAR,
    PRIORITY_NONE,
    PRIORITY_RISK_FLATTEN,
    PRIORITY_STRATEGY_EXIT,
    PRIORITY_TIME_EXIT,
    ActionKind,
    IntrabarConfig,
    ManagementAction,
    PositionState,
    ProfitStage,
    StageAction,
    is_stale,
    on_price,
    r_multiple,
    ratchet_stop,
    resolve_actions,
    trail_distance,
    unrealized_pct,
)

__all__ = [
    "DEFAULT_STAGES",
    "PRIORITY_EXCHANGE_STOP",
    "PRIORITY_INTRABAR",
    "PRIORITY_NONE",
    "PRIORITY_RISK_FLATTEN",
    "PRIORITY_STRATEGY_EXIT",
    "PRIORITY_TIME_EXIT",
    "ActionKind",
    "IntrabarConfig",
    "ManagementAction",
    "PositionState",
    "ProfitStage",
    "StageAction",
    "is_stale",
    "on_price",
    "r_multiple",
    "ratchet_stop",
    "resolve_actions",
    "trail_distance",
    "unrealized_pct",
]
