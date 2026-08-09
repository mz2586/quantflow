"""A single market-intelligence reading, combining every measurement.

One object answering "what is this market doing right now", assembled from the pure
measures, the regime profile, the cross-asset correlation and the perpetual-futures
positioning data.

Every field is optional and every absence is explicit. A snapshot taken on a symbol with
no perpetual, or with too little history for volatility, is still a valid snapshot — it
simply reports what it could not measure instead of substituting a zero that a caller
would read as a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from quantflow.core.precision import ZERO
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle
from quantflow.intelligence.derivatives import (
    DerivativesSource,
    FundingSnapshot,
    OpenInterestSnapshot,
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
from quantflow.intelligence.regime import RegimeProfile, classify
from quantflow.risk.correlation import CorrelationMatrix, aligned_returns


@dataclass(frozen=True, slots=True)
class MarketIntelligence:
    """Everything measured about one symbol at one moment."""

    symbol: Symbol
    observed_at: datetime
    bars_used: int
    trend: TrendMeasure | None = None
    volatility: VolatilityMeasure | None = None
    volume: VolumeMeasure | None = None
    liquidity: LiquidityMeasure | None = None
    regime: RegimeProfile | None = None
    funding: FundingSnapshot | None = None
    open_interest: OpenInterestSnapshot | None = None
    #: Correlation against other symbols in the same reading.
    correlations: Mapping[str, Decimal] = field(default_factory=dict)
    #: What could not be measured, and why. Never silently empty.
    unavailable: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        """Whether every requested measurement was obtained."""
        return not self.unavailable

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API, the dashboard and research reports."""
        return {
            "symbol": str(self.symbol),
            "observed_at": self.observed_at.isoformat(),
            "bars_used": self.bars_used,
            "regime": self.regime.to_dict() if self.regime else None,
            "trend": (
                {
                    "strength": float(self.trend.strength),
                    "direction": float(self.trend.direction),
                    "efficiency": float(self.trend.efficiency),
                    "persistence": float(self.trend.persistence),
                    "is_trending": self.trend.is_trending,
                }
                if self.trend
                else None
            ),
            "volatility": (
                {
                    "normalized_atr": float(self.volatility.normalized_atr),
                    "return_dispersion": float(self.volatility.return_dispersion),
                    "relative_level": float(self.volatility.relative_level),
                    "is_high": self.volatility.is_high,
                    "is_low": self.volatility.is_low,
                }
                if self.volatility
                else None
            ),
            "volume": (
                {
                    "expansion": float(self.volume.expansion),
                    "breadth": float(self.volume.breadth),
                    "is_expanding": self.volume.is_expanding,
                    "is_drying_up": self.volume.is_drying_up,
                }
                if self.volume
                else None
            ),
            "liquidity": (
                {
                    "typical_quote_volume": str(self.liquidity.typical_quote_volume),
                    "average_range_pct": float(self.liquidity.average_range_pct),
                    "depth_proxy": str(self.liquidity.depth_proxy),
                    "quote_volume_observed": self.liquidity.quote_volume_observed,
                    "note": (
                        "inferred from bar data; not an order-book depth measurement"
                        + (
                            ""
                            if self.liquidity.quote_volume_observed
                            else "; quote volume derived from base volume x typical price"
                        )
                    ),
                }
                if self.liquidity
                else None
            ),
            "funding": self.funding.to_dict() if self.funding else None,
            "open_interest": self.open_interest.to_dict() if self.open_interest else None,
            "correlations": {key: str(value) for key, value in self.correlations.items()},
            "unavailable": list(self.unavailable),
        }

    def summary(self) -> str:
        """One line an operator can read without opening anything else."""
        parts: list[str] = [str(self.symbol)]
        if self.regime:
            parts.append(self.regime.label)
        if self.volume and self.volume.is_expanding:
            parts.append(f"volume {self.volume.expansion:.1f}x")
        if self.funding and self.funding.is_crowded_long:
            parts.append(f"crowded long (funding {self.funding.annualised:.1%} annualised)")
        elif self.funding and self.funding.is_crowded_short:
            parts.append(f"crowded short (funding {self.funding.annualised:.1%} annualised)")
        if self.unavailable:
            parts.append(f"unmeasured: {', '.join(self.unavailable)}")
        return " · ".join(parts)


async def observe(
    symbol: Symbol,
    candles: Sequence[Candle],
    *,
    derivatives: DerivativesSource | None = None,
    peers: Mapping[Symbol, Sequence[Candle]] | None = None,
) -> MarketIntelligence:
    """Take a full intelligence reading for one symbol.

    Derivatives are optional and failure there never fails the reading: funding is a
    nice-to-have positioning signal, and losing it must not cost the trend and volatility
    measurements that a risk decision actually depends on.
    """
    if not candles:
        return MarketIntelligence(
            symbol=symbol,
            observed_at=datetime.now(tz=UTC),
            bars_used=0,
            unavailable=("no candles supplied",),
        )

    bar_measures, missing = _measure_bars(candles)
    funding, open_interest, derivative_gaps = await _fetch_derivatives(symbol, derivatives)
    missing.extend(derivative_gaps)
    correlations = _peer_correlations(symbol, candles, peers)

    trend, volatility, volume, liquidity, regime = bar_measures
    return MarketIntelligence(
        symbol=symbol,
        observed_at=candles[-1].close_time,
        bars_used=len(candles),
        trend=trend,
        volatility=volatility,
        volume=volume,
        liquidity=liquidity,
        regime=regime,
        funding=funding,
        open_interest=open_interest,
        correlations=correlations,
        unavailable=tuple(missing),
    )


def _measure_bars(
    candles: Sequence[Candle],
) -> tuple[
    tuple[
        TrendMeasure | None,
        VolatilityMeasure | None,
        VolumeMeasure | None,
        LiquidityMeasure | None,
        RegimeProfile | None,
    ],
    list[str],
]:
    """Every bar-derived measurement, plus the names of those that could not be taken."""
    trend = measure_trend(candles)
    volatility = measure_volatility(candles)
    volume = measure_volume(candles)
    liquidity = measure_liquidity(candles)
    regime = classify(candles)

    missing = [
        f"{name} (insufficient history)"
        for name, value in (
            ("trend", trend),
            ("volatility", volatility),
            ("volume", volume),
            ("liquidity", liquidity),
            ("regime", regime),
        )
        if value is None
    ]
    return (trend, volatility, volume, liquidity, regime), missing


async def _fetch_derivatives(
    symbol: Symbol, source: DerivativesSource | None
) -> tuple[FundingSnapshot | None, OpenInterestSnapshot | None, list[str]]:
    """Funding and open interest, and what could not be obtained.

    A derivatives failure is never allowed to fail the whole reading: funding is a
    positioning nicety, and losing it must not cost the trend and volatility numbers a
    risk decision actually depends on.
    """
    if source is None:
        return None, None, ["funding and open interest (no derivatives source configured)"]

    missing: list[str] = []
    funding = await source.fetch_funding_rate(str(symbol))
    if funding is None:
        missing.append("funding (no perpetual, or venue unavailable)")
    open_interest = await source.fetch_open_interest(str(symbol))
    if open_interest is None:
        missing.append("open interest (no perpetual, or venue unavailable)")
    return funding, open_interest, missing


def _peer_correlations(
    symbol: Symbol,
    candles: Sequence[Candle],
    peers: Mapping[Symbol, Sequence[Candle]] | None,
) -> dict[str, Decimal]:
    """Return correlation against each peer that could be measured."""
    if not peers:
        return {}
    # Aligned on shared timestamps, not on position: live ingestion updates some symbols
    # before others, and correlating the last N bars of each would compare different
    # windows without saying so.
    points = {symbol: [(c.close_time, c.close) for c in candles]}
    for peer, peer_candles in peers.items():
        if peer != symbol:
            points[peer] = [(c.close_time, c.close) for c in peer_candles]

    matrix = CorrelationMatrix.from_returns(aligned_returns(points))
    out: dict[str, Decimal] = {}
    for peer in points:
        if peer == symbol:
            continue
        value = matrix.between(symbol, peer)
        if value is not None:
            out[str(peer)] = value
    return out


def portfolio_correlations(
    series: Mapping[Symbol, Sequence[Candle]],
) -> CorrelationMatrix:
    """Correlation matrix across a whole watchlist.

    Built once and handed to the risk engine, which needs it to enforce the
    correlated-position cap but must not acquire market data of its own.
    """
    points = {
        symbol: [(candle.close_time, candle.close) for candle in candles]
        for symbol, candles in series.items()
    }
    return CorrelationMatrix.from_returns(aligned_returns(points))


def concentration_score(correlations: Mapping[str, Decimal]) -> Decimal:
    """Mean absolute correlation against peers, 0 (independent) to 1 (one bet).

    Absolute because an inverted bet on the same driver is still a bet on that driver.
    """
    if not correlations:
        return ZERO
    return sum((abs(value) for value in correlations.values()), ZERO) / Decimal(len(correlations))


__all__ = [
    "MarketIntelligence",
    "concentration_score",
    "observe",
    "portfolio_correlations",
]
