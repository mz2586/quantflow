"""Strategy engine: the pure decision contract, indicators and the registry."""

from __future__ import annotations

from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.registry import (
    StrategyRegistry,
    load_builtin_strategies,
    register_strategy,
    registry,
)

__all__ = [
    "Strategy",
    "StrategyContext",
    "StrategyParams",
    "StrategyRegistry",
    "load_builtin_strategies",
    "register_strategy",
    "registry",
]
