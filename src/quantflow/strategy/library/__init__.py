"""Built-in strategy library.

Importing this package registers every bundled strategy with the global registry.
"""

from __future__ import annotations

from quantflow.strategy.library.donchian_breakout import (
    DonchianBreakoutParams,
    DonchianBreakoutStrategy,
)
from quantflow.strategy.library.ema_cross import EmaCrossParams, EmaCrossStrategy
from quantflow.strategy.library.rsi_reversion import RsiReversionParams, RsiReversionStrategy

__all__ = [
    "DonchianBreakoutParams",
    "DonchianBreakoutStrategy",
    "EmaCrossParams",
    "EmaCrossStrategy",
    "RsiReversionParams",
    "RsiReversionStrategy",
]
