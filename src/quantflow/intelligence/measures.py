"""Market measurements: trend, volatility, volume, liquidity.

Pure functions over candles. No IO, no state, no opinions about what to do with the
numbers — that separation is what lets the same measurement feed a regime classifier, a
research report and a risk decision without three subtly different implementations
drifting apart.

Every measure returns ``None`` rather than a default when it cannot be computed. A
volatility of "0.0" because there were four bars is indistinguishable from a genuinely
calm market once it leaves the function, and a strategy that gates on it would take the
wrong branch with no way to tell.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from quantflow.core.precision import ZERO
from quantflow.domain.market import Candle
from quantflow.strategy.indicators import atr, ema, sma, stdev

#: Bars below which a measurement is refused as unreliable.
MIN_BARS = 30

ONE = Decimal("1")


@dataclass(frozen=True, slots=True)
class TrendMeasure:
    """How strongly and in which direction price is trending."""

    #: Signed fractional gap between a fast and slow moving average. Positive is up.
    direction: Decimal
    #: 0..1. How much of the period's movement was net progress rather than churn —
    #: net displacement divided by total path length. A market that travels 100 points
    #: up and 100 back scores near zero however violent it looked.
    efficiency: Decimal
    #: Fraction of bars closing in the dominant direction, 0.5 is a coin flip.
    persistence: Decimal

    @property
    def strength(self) -> Decimal:
        """Overall trend strength, 0..1, ignoring direction.

        Efficiency and persistence are combined because either alone is misleading: a
        single enormous gap gives perfect efficiency with no persistence, and a slow
        grind gives high persistence with poor efficiency.
        """
        return (self.efficiency + _rescale_persistence(self.persistence)) / Decimal("2")

    @property
    def is_trending(self) -> bool:
        """Whether the market is making net progress rather than oscillating."""
        return self.strength >= Decimal("0.35")


@dataclass(frozen=True, slots=True)
class VolatilityMeasure:
    """How violently price is moving, and whether that is unusual for it."""

    #: ATR as a fraction of price.
    normalized_atr: Decimal
    #: Standard deviation of bar returns.
    return_dispersion: Decimal
    #: Current dispersion divided by its own longer-run average. 1.0 is normal for this
    #: market, which is the only useful frame — 2% daily moves are calm for an alt and
    #: extreme for a major.
    relative_level: Decimal

    @property
    def is_high(self) -> bool:
        """Whether volatility is elevated relative to this market's own history."""
        return self.relative_level >= Decimal("1.5")

    @property
    def is_low(self) -> bool:
        """Whether volatility is compressed relative to this market's own history."""
        return self.relative_level <= Decimal("0.7")


@dataclass(frozen=True, slots=True)
class VolumeMeasure:
    """Whether participation is expanding or drying up."""

    #: Recent average volume divided by the longer-run average.
    expansion: Decimal
    #: Fraction of recent bars trading above the longer-run average.
    breadth: Decimal

    @property
    def is_expanding(self) -> bool:
        """Whether participation is meaningfully above its own baseline."""
        return self.expansion >= Decimal("1.3")

    @property
    def is_drying_up(self) -> bool:
        """Whether participation has fallen away."""
        return self.expansion <= Decimal("0.7")


@dataclass(frozen=True, slots=True)
class LiquidityMeasure:
    """How much size the market can absorb, inferred from bar data.

    Inferred, not observed: without an order book this is a proxy built from traded
    volume and intrabar range. It is honest about being a proxy — a real depth
    measurement needs `fetch_order_book`, and calling this "depth" would overstate it.
    """

    #: Median quote-currency volume per bar.
    typical_quote_volume: Decimal
    #: Intrabar range as a fraction of price, averaged. A wide-ranging bar on thin
    #: volume is the signature of a market that moves when you touch it.
    average_range_pct: Decimal
    #: Quote volume per unit of range. Higher means more size absorbed per unit moved.
    depth_proxy: Decimal
    #: True when quote volume came from the venue; False when it was derived from base
    #: volume x typical price. CCXT's `fetch_ohlcv` returns six fields and quote volume
    #: is not one of them, so every backfilled candle has it as zero — and a liquidity
    #: measure that reads that zero literally reports every market as untradeable.
    quote_volume_observed: bool = True

    def can_absorb(self, notional: Decimal, *, share_of_bar: Decimal = Decimal("0.01")) -> bool:
        """Whether ``notional`` is small enough relative to a typical bar.

        The default of 1% of a bar's traded value is deliberately conservative: an order
        that is a large share of the bar it trades in *is* the bar, and the fill will
        reflect that.
        """
        if self.typical_quote_volume <= ZERO:
            return False
        return notional <= self.typical_quote_volume * share_of_bar


def measure_trend(
    candles: Sequence[Candle], *, fast: int = 20, slow: int = 50
) -> TrendMeasure | None:
    """Trend direction, efficiency and persistence, or ``None`` if unmeasurable."""
    if len(candles) < max(MIN_BARS, slow + 1):
        return None
    closes = [candle.close for candle in candles]
    index = len(closes) - 1

    fast_value = ema(closes, fast)[index]
    slow_value = ema(closes, slow)[index]
    price = closes[index]
    if fast_value is None or slow_value is None or price <= ZERO:
        return None
    direction = (fast_value - slow_value) / price

    window = closes[-slow:]
    displacement = abs(window[-1] - window[0])
    path = sum((abs(b - a) for a, b in pairwise(window)), ZERO)
    efficiency = displacement / path if path > ZERO else ZERO

    ups = sum(1 for a, b in pairwise(window) if b > a)
    persistence = Decimal(ups) / Decimal(len(window) - 1)
    if direction < ZERO:
        persistence = ONE - persistence

    return TrendMeasure(
        direction=direction,
        efficiency=min(efficiency, ONE),
        persistence=persistence,
    )


def measure_volatility(
    candles: Sequence[Candle], *, period: int = 14, baseline: int = 100
) -> VolatilityMeasure | None:
    """Absolute and relative volatility, or ``None`` if unmeasurable."""
    if len(candles) < max(MIN_BARS, period + 1):
        return None
    closes = [candle.close for candle in candles]
    index = len(closes) - 1
    price = closes[index]
    if price <= ZERO:
        return None

    atr_value = atr(candles, period)[index]
    if atr_value is None:
        return None

    returns = _returns(closes)
    if len(returns) < period:
        return None
    recent = stdev(returns, period)[len(returns) - 1]
    if recent is None:
        return None

    # Compared against its own history rather than an absolute threshold: 2% bars are
    # calm for an alt and extreme for a major, and one constant cannot serve both.
    span = min(baseline, len(returns))
    longrun = stdev(returns, span)[len(returns) - 1] if span >= period else None
    relative = recent / longrun if longrun and longrun > ZERO else ONE

    return VolatilityMeasure(
        normalized_atr=atr_value / price,
        return_dispersion=recent,
        relative_level=relative,
    )


def measure_volume(
    candles: Sequence[Candle], *, recent: int = 20, baseline: int = 100
) -> VolumeMeasure | None:
    """Volume expansion and breadth, or ``None`` if unmeasurable."""
    if len(candles) < max(MIN_BARS, recent + 1):
        return None
    volumes = [candle.volume for candle in candles]
    index = len(volumes) - 1

    recent_average = sma(volumes, recent)[index]
    span = min(baseline, len(volumes))
    long_average = sma(volumes, span)[index]
    if recent_average is None or long_average is None or long_average <= ZERO:
        return None

    window = volumes[-recent:]
    above = sum(1 for value in window if value > long_average)
    return VolumeMeasure(
        expansion=recent_average / long_average,
        breadth=Decimal(above) / Decimal(len(window)),
    )


def measure_liquidity(candles: Sequence[Candle], *, window: int = 50) -> LiquidityMeasure | None:
    """Liquidity proxies from bar data, or ``None`` if unmeasurable."""
    if len(candles) < MIN_BARS:
        return None
    recent = candles[-window:]

    # Derive quote volume when the venue did not supply it. base x typical price is the
    # standard approximation and is close enough for a liquidity proxy; silently using
    # the zero would claim the market has no volume at all.
    observed = any(candle.quote_volume > ZERO for candle in recent)
    quote_volumes = sorted(
        candle.quote_volume if observed else candle.volume * _typical_price(candle)
        for candle in recent
    )
    typical = quote_volumes[len(quote_volumes) // 2]

    ranges: list[Decimal] = []
    for candle in recent:
        if candle.close > ZERO:
            ranges.append((candle.high - candle.low) / candle.close)
    if not ranges:
        return None
    average_range = sum(ranges, ZERO) / Decimal(len(ranges))

    depth = typical / average_range if average_range > ZERO else ZERO
    return LiquidityMeasure(
        typical_quote_volume=typical,
        average_range_pct=average_range,
        depth_proxy=depth,
        quote_volume_observed=observed,
    )


def _typical_price(candle: Candle) -> Decimal:
    """(high + low + close) / 3, the usual stand-in for a bar's average trade price."""
    return (candle.high + candle.low + candle.close) / Decimal("3")


def _returns(closes: Sequence[Decimal]) -> list[Decimal]:
    """Period-over-period returns, skipping non-positive references."""
    out: list[Decimal] = []
    for previous, current in pairwise(closes):
        if previous > ZERO:
            out.append((current - previous) / previous)
    return out


def _rescale_persistence(persistence: Decimal) -> Decimal:
    """Map a 0.5-is-random persistence onto 0..1 where 0 is random."""
    return min(ONE, max(ZERO, (persistence - Decimal("0.5")) * Decimal("2")))


__all__ = [
    "MIN_BARS",
    "LiquidityMeasure",
    "TrendMeasure",
    "VolatilityMeasure",
    "VolumeMeasure",
    "measure_liquidity",
    "measure_trend",
    "measure_volatility",
    "measure_volume",
]
