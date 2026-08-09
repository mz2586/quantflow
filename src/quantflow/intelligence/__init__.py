"""Market Intelligence: what the market is doing, measured rather than assumed.

Eight measurements — trend strength, volatility, volume expansion, liquidity,
correlation, funding rates, open interest and market regime — behind one snapshot.

Two decisions run through the whole module:

**Absence is reported, never defaulted.** Every measure returns ``None`` when it cannot be
computed, and a snapshot lists what it could not measure. A volatility of 0.0 because
there were four bars is indistinguishable from a genuinely calm market once it leaves the
function, and a strategy gating on it would take the wrong branch with no way to tell.

**Regime is three axes, not one label.** Direction, structure and volatility are
independent and routinely co-occur; collapsing them into a single enum discards two thirds
of what was measured.
"""

from __future__ import annotations

from quantflow.intelligence.derivatives import (
    CcxtDerivativesSource,
    DerivativesSource,
    FundingSnapshot,
    OpenInterestSnapshot,
    funding_trend,
    has_perpetual,
    perpetual_symbol,
)
from quantflow.intelligence.measures import (
    LiquidityMeasure,
    TrendMeasure,
    VolatilityMeasure,
    VolumeMeasure,
    measure_liquidity,
    measure_trend,
    measure_volatility,
    measure_volume,
)
from quantflow.intelligence.regime import (
    Direction,
    RegimeProfile,
    Structure,
    VolatilityBand,
    classify,
    classify_series,
)
from quantflow.intelligence.snapshot import (
    MarketIntelligence,
    concentration_score,
    observe,
    portfolio_correlations,
)

__all__ = [
    "CcxtDerivativesSource",
    "DerivativesSource",
    "Direction",
    "FundingSnapshot",
    "LiquidityMeasure",
    "MarketIntelligence",
    "OpenInterestSnapshot",
    "RegimeProfile",
    "Structure",
    "TrendMeasure",
    "VolatilityBand",
    "VolatilityMeasure",
    "VolumeMeasure",
    "classify",
    "classify_series",
    "concentration_score",
    "funding_trend",
    "has_perpetual",
    "measure_liquidity",
    "measure_trend",
    "measure_volatility",
    "measure_volume",
    "observe",
    "perpetual_symbol",
    "portfolio_correlations",
]
