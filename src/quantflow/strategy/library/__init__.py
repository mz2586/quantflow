"""Built-in strategy library.

Importing this package registers every bundled strategy with the global registry.

The library deliberately spans distinct *families* — trend, breakout, mean reversion,
volatility, volume, calendar — plus a buy-and-hold benchmark. A leaderboard populated only
with variations on one idea would rank parameter choices while appearing to rank ideas,
and would be silent on the question that actually matters: which kind of edge, if any,
exists in this market.
"""

from __future__ import annotations

from quantflow.strategy.library.bollinger_reversion import (
    BollingerReversionParams,
    BollingerReversionStrategy,
)
from quantflow.strategy.library.bollinger_squeeze import (
    BollingerSqueezeParams,
    BollingerSqueezeStrategy,
)
from quantflow.strategy.library.buy_and_hold import BuyAndHoldParams, BuyAndHoldStrategy
from quantflow.strategy.library.donchian_breakout import (
    DonchianBreakoutParams,
    DonchianBreakoutStrategy,
)
from quantflow.strategy.library.dual_thrust import DualThrustParams, DualThrustStrategy
from quantflow.strategy.library.ema_cross import EmaCrossParams, EmaCrossStrategy
from quantflow.strategy.library.keltner_trend import KeltnerTrendParams, KeltnerTrendStrategy
from quantflow.strategy.library.macd_trend import MacdTrendParams, MacdTrendStrategy
from quantflow.strategy.library.momentum_roc import MomentumRocParams, MomentumRocStrategy
from quantflow.strategy.library.opening_range_breakout import (
    OpeningRangeBreakoutParams,
    OpeningRangeBreakoutStrategy,
)
from quantflow.strategy.library.rsi_reversion import RsiReversionParams, RsiReversionStrategy
from quantflow.strategy.library.triple_ma import TripleMaParams, TripleMaStrategy
from quantflow.strategy.library.volume_breakout import (
    VolumeBreakoutParams,
    VolumeBreakoutStrategy,
)
from quantflow.strategy.library.zscore_reversion import (
    ZScoreReversionParams,
    ZScoreReversionStrategy,
)

__all__ = [
    "BollingerReversionParams",
    "BollingerReversionStrategy",
    "BollingerSqueezeParams",
    "BollingerSqueezeStrategy",
    "BuyAndHoldParams",
    "BuyAndHoldStrategy",
    "DonchianBreakoutParams",
    "DonchianBreakoutStrategy",
    "DualThrustParams",
    "DualThrustStrategy",
    "EmaCrossParams",
    "EmaCrossStrategy",
    "KeltnerTrendParams",
    "KeltnerTrendStrategy",
    "MacdTrendParams",
    "MacdTrendStrategy",
    "MomentumRocParams",
    "MomentumRocStrategy",
    "OpeningRangeBreakoutParams",
    "OpeningRangeBreakoutStrategy",
    "RsiReversionParams",
    "RsiReversionStrategy",
    "TripleMaParams",
    "TripleMaStrategy",
    "VolumeBreakoutParams",
    "VolumeBreakoutStrategy",
    "ZScoreReversionParams",
    "ZScoreReversionStrategy",
]
