"""Return correlation between instruments.

Crypto is not a diversified universe. Almost everything is a leveraged expression of the
same BTC beta, so holding five "different" alt positions is usually one position in five
costumes — with five times the intended size. A portfolio that looks diversified by symbol
count and is in fact concentrated by exposure is how an account takes a loss it never
sized for.

Correlation is measured from realised returns rather than assumed from a hand-maintained
list of "these are majors". A hardcoded grouping is wrong the moment a new asset decouples
or an old one starts tracking, and nobody remembers to update it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import pairwise

from quantflow.core.precision import ZERO
from quantflow.domain.instruments import Symbol

#: Minimum overlapping observations before a correlation is trusted.
#:
#: Below this the estimate is dominated by noise, and a spurious 0.9 from six shared bars
#: would block legitimate trades. Too few observations is reported as "unknown", never as
#: zero — treating an unmeasurable correlation as absence of correlation is the exact
#: failure this module exists to prevent.
MIN_OBSERVATIONS = 30


def aligned_returns(
    series: Mapping[Symbol, Sequence[tuple[datetime, Decimal]]],
) -> dict[Symbol, tuple[Decimal, ...]]:
    """Returns for every symbol, restricted to timestamps they all share.

    Correlating by *position* is silently wrong whenever two series differ in length or
    coverage — and they routinely do, because live ingestion updates some symbols before
    others. Aligning on the intersection of timestamps costs one set operation and is the
    difference between a real correlation and a comparison of two different weeks.
    """
    if not series:
        return {}

    common: set[datetime] | None = None
    for points in series.values():
        stamps = {stamp for stamp, _ in points}
        common = stamps if common is None else (common & stamps)
    if not common:
        return dict.fromkeys(series, ())

    ordered = sorted(common)
    out: dict[Symbol, tuple[Decimal, ...]] = {}
    for symbol, points in series.items():
        lookup = dict(points)
        out[symbol] = returns_from_prices([lookup[stamp] for stamp in ordered])
    return out


def returns_from_prices(prices: Sequence[Decimal]) -> tuple[Decimal, ...]:
    """Simple period-over-period returns.

    Bars with a non-positive previous price are skipped rather than producing an infinite
    return that would poison the whole correlation.
    """
    out: list[Decimal] = []
    for previous, current in pairwise(prices):
        if previous <= ZERO:
            continue
        out.append((current - previous) / previous)
    return tuple(out)


def pearson(first: Sequence[Decimal], second: Sequence[Decimal]) -> Decimal | None:
    """Pearson correlation of two return series, or ``None`` when undefined.

    Returns ``None`` — not zero — when there is too little overlap or either series is
    flat. A flat series has no variance and therefore no correlation with anything; saying
    "zero" would claim independence that was never measured.
    """
    length = min(len(first), len(second))
    if length < MIN_OBSERVATIONS:
        return None

    left = first[-length:]
    right = second[-length:]
    count = Decimal(length)
    mean_left = sum(left, ZERO) / count
    mean_right = sum(right, ZERO) / count

    covariance = ZERO
    variance_left = ZERO
    variance_right = ZERO
    for a, b in zip(left, right, strict=True):
        da = a - mean_left
        db = b - mean_right
        covariance += da * db
        variance_left += da * da
        variance_right += db * db

    if variance_left <= ZERO or variance_right <= ZERO:
        return None

    # Decimal has no sqrt on the type itself; the context method keeps this exact-ish and
    # avoids a float round trip on a value that feeds a risk decision.
    denominator = (variance_left * variance_right).sqrt()
    if denominator <= ZERO:
        return None
    return covariance / denominator


@dataclass(frozen=True, slots=True)
class CorrelationMatrix:
    """Pairwise return correlations, with unmeasurable pairs left absent."""

    values: Mapping[tuple[str, str], Decimal]

    @classmethod
    def from_returns(cls, returns: Mapping[Symbol, Sequence[Decimal]]) -> CorrelationMatrix:
        """Build the matrix from per-symbol return series."""
        symbols = sorted(returns, key=str)
        values: dict[tuple[str, str], Decimal] = {}
        for index, left in enumerate(symbols):
            for right in symbols[index + 1 :]:
                value = pearson(returns[left], returns[right])
                if value is not None:
                    values[_key(left, right)] = value
        return cls(values=values)

    def between(self, left: Symbol, right: Symbol) -> Decimal | None:
        """Correlation between two symbols, or ``None`` if not measurable."""
        if left == right:
            return Decimal("1")
        return self.values.get(_key(left, right))

    def correlated_with(
        self, candidate: Symbol, others: Sequence[Symbol], *, threshold: Decimal
    ) -> tuple[Symbol, ...]:
        """Which of ``others`` move with ``candidate`` beyond ``threshold``.

        Absolute value: a strongly *negative* correlation is also a concentrated bet, just
        an inverted one, and sizing that ignores it is equally wrong.
        """
        hits: list[Symbol] = []
        for other in others:
            if other == candidate:
                continue
            value = self.between(candidate, other)
            if value is not None and abs(value) >= threshold:
                hits.append(other)
        return tuple(hits)


def _key(left: Symbol, right: Symbol) -> tuple[str, str]:
    """Order-independent key for a symbol pair."""
    a, b = str(left), str(right)
    return (a, b) if a <= b else (b, a)


__all__ = [
    "MIN_OBSERVATIONS",
    "CorrelationMatrix",
    "aligned_returns",
    "pearson",
    "returns_from_prices",
]
