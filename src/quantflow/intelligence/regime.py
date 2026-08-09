"""Market regime as three orthogonal axes rather than one label.

The requested taxonomy — Trending, Ranging, High Volatility, Low Volatility, Bull, Bear,
Sideways — is not a single set of mutually exclusive states. "Bull" and "Trending"
describe different things and routinely co-occur; "High Volatility" can qualify any of
the others. Forcing them into one enum means every classification silently discards two
thirds of what was measured, and a strategy that only trades quiet uptrends cannot
express that against a single label.

So a regime here is a **profile** with three independent axes:

* direction  — bull / bear / sideways
* structure  — trending / ranging
* volatility — high / normal / low

A single-label view is still available for the existing `MarketRegime` enum and for
display, but it is a projection of the profile, never the source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from quantflow.domain.enums import MarketRegime
from quantflow.domain.market import Candle
from quantflow.intelligence.measures import (
    TrendMeasure,
    VolatilityMeasure,
    VolumeMeasure,
    measure_trend,
    measure_volatility,
    measure_volume,
)

#: Fractional fast-slow gap beyond which a market counts as directional.
DIRECTION_THRESHOLD = Decimal("0.005")


class Direction(StrEnum):
    """Which way the market is going."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class Structure(StrEnum):
    """Whether the market is making progress or oscillating."""

    TRENDING = "trending"
    RANGING = "ranging"


class VolatilityBand(StrEnum):
    """How violent the market is relative to its own history."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class RegimeProfile:
    """A market's regime across three independent axes."""

    direction: Direction
    structure: Structure
    volatility: VolatilityBand
    timestamp: datetime
    #: The measurements behind the classification, so a surprising label can be
    #: explained rather than merely observed.
    trend: TrendMeasure
    volatility_measure: VolatilityMeasure
    volume: VolumeMeasure | None = None

    @property
    def label(self) -> str:
        """Human-readable composite, e.g. ``bull/trending/high``."""
        return f"{self.direction}/{self.structure}/{self.volatility}"

    @property
    def as_market_regime(self) -> MarketRegime:
        """Projection onto the single-label enum the rest of the platform uses.

        Lossy by construction. Volatility wins when extreme because it changes how a
        position should be sized regardless of direction — that is the one axis whose
        answer alters behaviour on its own.
        """
        if self.volatility is VolatilityBand.HIGH:
            return MarketRegime.HIGH_VOLATILITY
        if self.structure is Structure.RANGING:
            return MarketRegime.RANGE
        if self.direction is Direction.BULL:
            return MarketRegime.BULL_TREND
        if self.direction is Direction.BEAR:
            return MarketRegime.BEAR_TREND
        return MarketRegime.RANGE

    def matches(self, other: RegimeProfile) -> bool:
        """Whether two profiles describe the same regime on every axis."""
        return (
            self.direction is other.direction
            and self.structure is other.structure
            and self.volatility is other.volatility
        )

    def to_dict(self) -> dict[str, str | float]:
        """Serialise for persistence, the API and reports."""
        return {
            "direction": str(self.direction),
            "structure": str(self.structure),
            "volatility": str(self.volatility),
            "label": self.label,
            "timestamp": self.timestamp.isoformat(),
            "trend_strength": float(self.trend.strength),
            "trend_direction": float(self.trend.direction),
            "normalized_atr": float(self.volatility_measure.normalized_atr),
            "volatility_relative": float(self.volatility_measure.relative_level),
            "volume_expansion": float(self.volume.expansion) if self.volume else 0.0,
        }

    def explain(self) -> str:
        """One sentence describing why this classification was made."""
        return (
            f"{self.label}: trend strength {self.trend.strength:.2f} "
            f"(direction {self.trend.direction:+.4f}), volatility "
            f"{self.volatility_measure.relative_level:.2f}x its own baseline at "
            f"{self.volatility_measure.normalized_atr:.2%} ATR"
            + (f", volume {self.volume.expansion:.2f}x baseline" if self.volume is not None else "")
        )


def classify(candles: Sequence[Candle]) -> RegimeProfile | None:
    """Classify the most recent bar's regime, or ``None`` if unmeasurable.

    Returns ``None`` rather than an ``UNKNOWN`` profile: a caller that gates on regime
    must distinguish "the market is calm" from "I could not tell", and an enum member
    named unknown gets treated as the former the first time somebody writes an
    `if regime.volatility is not HIGH` check.
    """
    trend = measure_trend(candles)
    volatility = measure_volatility(candles)
    if trend is None or volatility is None:
        return None

    if not trend.is_trending:
        direction = Direction.SIDEWAYS
    elif trend.direction > DIRECTION_THRESHOLD:
        direction = Direction.BULL
    elif trend.direction < -DIRECTION_THRESHOLD:
        direction = Direction.BEAR
    else:
        direction = Direction.SIDEWAYS

    structure = Structure.TRENDING if trend.is_trending else Structure.RANGING

    if volatility.is_high:
        band = VolatilityBand.HIGH
    elif volatility.is_low:
        band = VolatilityBand.LOW
    else:
        band = VolatilityBand.NORMAL

    return RegimeProfile(
        direction=direction,
        structure=structure,
        volatility=band,
        timestamp=candles[-1].close_time,
        trend=trend,
        volatility_measure=volatility,
        volume=measure_volume(candles),
    )


def classify_series(candles: Sequence[Candle], *, step: int = 1) -> tuple[RegimeProfile, ...]:
    """Classify every bar from the first measurable one onward.

    Used by the Strategy Laboratory to attribute each trade to the regime it was taken
    in. ``step`` subsamples for speed on long histories, where consecutive bars almost
    always share a regime anyway.
    """
    out: list[RegimeProfile] = []
    for end in range(len(candles), 0, -step):
        profile = classify(candles[:end])
        if profile is None:
            break
        out.append(profile)
    return tuple(reversed(out))


__all__ = [
    "DIRECTION_THRESHOLD",
    "Direction",
    "RegimeProfile",
    "Structure",
    "VolatilityBand",
    "classify",
    "classify_series",
]
