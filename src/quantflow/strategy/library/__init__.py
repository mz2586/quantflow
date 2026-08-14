"""Built-in strategy library.

Importing this package registers every bundled strategy with the global registry.

The library deliberately spans distinct *families* — trend, breakout, mean reversion,
volatility, volume, calendar — plus a buy-and-hold benchmark. A leaderboard populated only
with variations on one idea would rank parameter choices while appearing to rank ideas,
and would be silent on the question that actually matters: which kind of edge, if any,
exists in this market.
"""

from __future__ import annotations

from quantflow.strategy.library.accumulation_distribution import (
    AccumulationDistributionParams,
    AccumulationDistributionStrategy,
)
from quantflow.strategy.library.adx_trend import AdxTrendParams, AdxTrendStrategy
from quantflow.strategy.library.atr_breakout import (
    AtrBreakoutParams,
    AtrBreakoutStrategy,
)
from quantflow.strategy.library.atr_expansion import AtrExpansionParams, AtrExpansionStrategy
from quantflow.strategy.library.bollinger_reversion import (
    BollingerReversionParams,
    BollingerReversionStrategy,
)
from quantflow.strategy.library.bollinger_squeeze import (
    BollingerSqueezeParams,
    BollingerSqueezeStrategy,
)
from quantflow.strategy.library.breakout_retest import (
    BreakoutRetestParams,
    BreakoutRetestStrategy,
)
from quantflow.strategy.library.buy_and_hold import BuyAndHoldParams, BuyAndHoldStrategy
from quantflow.strategy.library.donchian_breakout import (
    DonchianBreakoutParams,
    DonchianBreakoutStrategy,
)
from quantflow.strategy.library.dual_thrust import DualThrustParams, DualThrustStrategy
from quantflow.strategy.library.ema_cross import EmaCrossParams, EmaCrossStrategy
from quantflow.strategy.library.ichimoku_trend import (
    IchimokuTrendParams,
    IchimokuTrendStrategy,
)
from quantflow.strategy.library.keltner_reversion import (
    KeltnerReversionParams,
    KeltnerReversionStrategy,
)
from quantflow.strategy.library.keltner_trend import KeltnerTrendParams, KeltnerTrendStrategy
from quantflow.strategy.library.ma_deviation_reversion import (
    MaDeviationReversionParams,
    MaDeviationReversionStrategy,
)
from quantflow.strategy.library.macd_trend import MacdTrendParams, MacdTrendStrategy
from quantflow.strategy.library.momentum_acceleration import (
    MomentumAccelerationParams,
    MomentumAccelerationStrategy,
)
from quantflow.strategy.library.momentum_roc import MomentumRocParams, MomentumRocStrategy
from quantflow.strategy.library.money_flow_index import (
    MoneyFlowIndexParams,
    MoneyFlowIndexStrategy,
)
from quantflow.strategy.library.mtf_trend import (
    MtfTrendParams,
    MtfTrendStrategy,
)
from quantflow.strategy.library.normalized_momentum import (
    NormalizedMomentumParams,
    NormalizedMomentumStrategy,
)
from quantflow.strategy.library.obv_trend import ObvTrendParams, ObvTrendStrategy
from quantflow.strategy.library.opening_range_breakout import (
    OpeningRangeBreakoutParams,
    OpeningRangeBreakoutStrategy,
)
from quantflow.strategy.library.parabolic_sar import (
    ParabolicSarParams,
    ParabolicSarStrategy,
)
from quantflow.strategy.library.percentile_reversion import (
    PercentileReversionParams,
    PercentileReversionStrategy,
)
from quantflow.strategy.library.pullback_continuation import (
    PullbackContinuationParams,
    PullbackContinuationStrategy,
)
from quantflow.strategy.library.range_expansion import (
    RangeExpansionParams,
    RangeExpansionStrategy,
)
from quantflow.strategy.library.regime_adaptive import RegimeAdaptiveParams, RegimeAdaptiveStrategy
from quantflow.strategy.library.relative_momentum import (
    RelativeMomentumParams,
    RelativeMomentumStrategy,
)
from quantflow.strategy.library.rsi_reversion import RsiReversionParams, RsiReversionStrategy
from quantflow.strategy.library.stochastic_reversion import (
    StochasticReversionParams,
    StochasticReversionStrategy,
)
from quantflow.strategy.library.supertrend import (
    SupertrendParams,
    SupertrendStrategy,
)
from quantflow.strategy.library.support_resistance_breakout import (
    SupportResistanceBreakoutParams,
    SupportResistanceBreakoutStrategy,
)
from quantflow.strategy.library.swing_structure import (
    SwingStructureParams,
    SwingStructureStrategy,
)
from quantflow.strategy.library.triple_ma import TripleMaParams, TripleMaStrategy
from quantflow.strategy.library.vol_adjusted_momentum import (
    VolAdjustedMomentumParams,
    VolAdjustedMomentumStrategy,
)
from quantflow.strategy.library.volatility_normalized_reversion import (
    VolatilityNormalizedReversionParams,
    VolatilityNormalizedReversionStrategy,
)
from quantflow.strategy.library.volatility_regime import (
    VolatilityRegimeParams,
    VolatilityRegimeStrategy,
)
from quantflow.strategy.library.volatility_transition import (
    VolatilityTransitionParams,
    VolatilityTransitionStrategy,
)
from quantflow.strategy.library.volume_breakout import (
    VolumeBreakoutParams,
    VolumeBreakoutStrategy,
)
from quantflow.strategy.library.volume_price_divergence import (
    VolumePriceDivergenceParams,
    VolumePriceDivergenceStrategy,
)
from quantflow.strategy.library.vwap_momentum import VwapMomentumParams, VwapMomentumStrategy
from quantflow.strategy.library.vwap_reversion import (
    VwapReversionParams,
    VwapReversionStrategy,
)
from quantflow.strategy.library.zscore_reversion import (
    ZScoreReversionParams,
    ZScoreReversionStrategy,
)

__all__ = [
    "AccumulationDistributionParams",
    "AccumulationDistributionStrategy",
    "AdxTrendParams",
    "AdxTrendStrategy",
    "AtrBreakoutParams",
    "AtrBreakoutStrategy",
    "AtrExpansionParams",
    "AtrExpansionStrategy",
    "BollingerReversionParams",
    "BollingerReversionStrategy",
    "BollingerSqueezeParams",
    "BollingerSqueezeStrategy",
    "BreakoutRetestParams",
    "BreakoutRetestStrategy",
    "BuyAndHoldParams",
    "BuyAndHoldStrategy",
    "DonchianBreakoutParams",
    "DonchianBreakoutStrategy",
    "DualThrustParams",
    "DualThrustStrategy",
    "EmaCrossParams",
    "EmaCrossStrategy",
    "IchimokuTrendParams",
    "IchimokuTrendStrategy",
    "KeltnerReversionParams",
    "KeltnerReversionStrategy",
    "KeltnerTrendParams",
    "KeltnerTrendStrategy",
    "MaDeviationReversionParams",
    "MaDeviationReversionStrategy",
    "MacdTrendParams",
    "MacdTrendStrategy",
    "MomentumAccelerationParams",
    "MomentumAccelerationStrategy",
    "MomentumRocParams",
    "MomentumRocStrategy",
    "MoneyFlowIndexParams",
    "MoneyFlowIndexStrategy",
    "MtfTrendParams",
    "MtfTrendStrategy",
    "NormalizedMomentumParams",
    "NormalizedMomentumStrategy",
    "ObvTrendParams",
    "ObvTrendStrategy",
    "OpeningRangeBreakoutParams",
    "OpeningRangeBreakoutStrategy",
    "ParabolicSarParams",
    "ParabolicSarStrategy",
    "PercentileReversionParams",
    "PercentileReversionStrategy",
    "PullbackContinuationParams",
    "PullbackContinuationStrategy",
    "RangeExpansionParams",
    "RangeExpansionStrategy",
    "RegimeAdaptiveParams",
    "RegimeAdaptiveStrategy",
    "RelativeMomentumParams",
    "RelativeMomentumStrategy",
    "RsiReversionParams",
    "RsiReversionStrategy",
    "StochasticReversionParams",
    "StochasticReversionStrategy",
    "SupertrendParams",
    "SupertrendStrategy",
    "SupportResistanceBreakoutParams",
    "SupportResistanceBreakoutStrategy",
    "SwingStructureParams",
    "SwingStructureStrategy",
    "TripleMaParams",
    "TripleMaStrategy",
    "VolAdjustedMomentumParams",
    "VolAdjustedMomentumStrategy",
    "VolatilityNormalizedReversionParams",
    "VolatilityNormalizedReversionStrategy",
    "VolatilityRegimeParams",
    "VolatilityRegimeStrategy",
    "VolatilityTransitionParams",
    "VolatilityTransitionStrategy",
    "VolumeBreakoutParams",
    "VolumeBreakoutStrategy",
    "VolumePriceDivergenceParams",
    "VolumePriceDivergenceStrategy",
    "VwapMomentumParams",
    "VwapMomentumStrategy",
    "VwapReversionParams",
    "VwapReversionStrategy",
    "ZScoreReversionParams",
    "ZScoreReversionStrategy",
]
