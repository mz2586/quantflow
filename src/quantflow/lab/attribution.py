"""Attribute trades to the market regime they were taken in.

A single blended number hides the thing worth knowing. A trend follower that makes 40% in
trending markets and gives back 35% in ranging ones nets 5% and looks mediocre; the truth
is that it works and is being run in the wrong conditions half the time. Only a per-regime
breakdown separates "this does not work" from "this works, sometimes".

Attribution is by **entry** regime, not exit. The decision to open was made under the
conditions prevailing then, and crediting a trade to the regime it happened to close in
would score a strategy on information it did not have.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.market import Candle
from quantflow.domain.positions import ClosedTrade
from quantflow.intelligence.regime import classify

#: Bars between regime classifications. Consecutive bars almost always share a regime,
#: and classifying all 49,000 of them costs far more than the resolution is worth.
DEFAULT_STRIDE = 24

#: Trades below which a per-regime figure is reported but flagged as unreliable.
MIN_TRADES_PER_REGIME = 10

#: Bars each classification may look back over. Comfortably above what every measure
#: needs (the slowest is volatility's 100-return baseline) while keeping the label
#: responsive to recent conditions rather than diluted by years of history.
REGIME_LOOKBACK = 500


@dataclass(frozen=True, slots=True)
class RegimePerformance:
    """How a strategy did in one regime."""

    regime: str
    trade_count: int
    net_pnl: Decimal
    gross_pnl: Decimal
    fees: Decimal
    win_count: int

    @property
    def win_rate(self) -> Decimal:
        """Fraction of trades that made money after fees."""
        return safe_divide(Decimal(self.win_count), Decimal(self.trade_count))

    @property
    def average_pnl(self) -> Decimal:
        """Mean net PnL per trade."""
        return safe_divide(self.net_pnl, Decimal(self.trade_count))

    @property
    def is_reliable(self) -> bool:
        """Whether there are enough trades for the figure to mean anything."""
        return self.trade_count >= MIN_TRADES_PER_REGIME

    @property
    def is_profitable(self) -> bool:
        """Whether the strategy made money in this regime."""
        return self.net_pnl > ZERO

    def to_dict(self) -> dict[str, object]:
        """Serialise for reports and the API."""
        return {
            "regime": self.regime,
            "trade_count": self.trade_count,
            "net_pnl": str(self.net_pnl),
            "gross_pnl": str(self.gross_pnl),
            "fees": str(self.fees),
            "win_rate": str(self.win_rate),
            "average_pnl": str(self.average_pnl),
            "reliable": self.is_reliable,
            "profitable": self.is_profitable,
        }


@dataclass(frozen=True, slots=True)
class RegimeBreakdown:
    """A strategy's performance across every regime it traded in."""

    by_regime: tuple[RegimePerformance, ...]

    @property
    def profitable_regimes(self) -> tuple[str, ...]:
        """Regimes where the strategy made money on a reliable sample."""
        return tuple(
            item.regime for item in self.by_regime if item.is_profitable and item.is_reliable
        )

    @property
    def losing_regimes(self) -> tuple[str, ...]:
        """Regimes where the strategy lost money on a reliable sample."""
        return tuple(
            item.regime for item in self.by_regime if not item.is_profitable and item.is_reliable
        )

    @property
    def is_regime_dependent(self) -> bool:
        """Whether the strategy works in some conditions and not others.

        The finding that justifies regime gating: a strategy that is profitable in one
        regime and unprofitable in another should be allowed to trade only the first,
        rather than being discarded on a blended average that describes neither.
        """
        return bool(self.profitable_regimes) and bool(self.losing_regimes)

    def best(self) -> RegimePerformance | None:
        """The regime with the highest net PnL on a reliable sample."""
        reliable = [item for item in self.by_regime if item.is_reliable]
        return max(reliable, key=lambda item: item.net_pnl, default=None)

    def worst(self) -> RegimePerformance | None:
        """The regime with the lowest net PnL on a reliable sample."""
        reliable = [item for item in self.by_regime if item.is_reliable]
        return min(reliable, key=lambda item: item.net_pnl, default=None)

    def summary(self) -> str:
        """One line naming where the strategy works and where it does not."""
        if not self.by_regime:
            return "no trades to attribute"
        if not self.is_regime_dependent:
            profitable = self.profitable_regimes
            if profitable:
                return f"profitable across {len(profitable)} regime(s), no reliable losing regime"
            return "no reliable profitable regime"
        return (
            f"works in {', '.join(self.profitable_regimes)}; "
            f"loses in {', '.join(self.losing_regimes)}"
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise for reports and the API."""
        return {
            "by_regime": [item.to_dict() for item in self.by_regime],
            "profitable_regimes": list(self.profitable_regimes),
            "losing_regimes": list(self.losing_regimes),
            "regime_dependent": self.is_regime_dependent,
            "summary": self.summary(),
        }


class RegimeTimeline:
    """Regime labels over time, queryable at any instant.

    Built once per symbol and reused across every strategy, because classifying the same
    49,000 bars once per strategy would dominate the cost of an entire laboratory run.
    """

    __slots__ = ("_labels", "_stamps")

    def __init__(self, stamps: Sequence[datetime], labels: Sequence[str]) -> None:
        self._stamps = list(stamps)
        self._labels = list(labels)

    @classmethod
    def build(
        cls,
        candles: Sequence[Candle],
        *,
        stride: int = DEFAULT_STRIDE,
        lookback: int = REGIME_LOOKBACK,
    ) -> RegimeTimeline:
        """Classify a candle series at intervals of ``stride`` bars.

        Each classification sees a bounded window ending at that bar, not the whole
        history before it. That is both the correct definition and the affordable one.

        Correct, because a regime is a *local* property. Handed the full prefix, the
        moving averages and dispersion that decide the label are dominated by however
        many years came before: a market that trended for three years and turned last
        week still classifies as trending, which is precisely backwards for a signal
        whose entire job is to notice that conditions changed.

        Affordable, because the indicators recompute over their input on every call. Over
        the full prefix that is O(bars^2 / stride) - on 20,000 bars it left a laboratory
        run single-threaded for the better part of an hour after its worker pools had
        already finished.
        """
        stamps: list[datetime] = []
        labels: list[str] = []
        for end in range(1, len(candles) + 1, stride):
            window = candles[max(0, end - lookback) : end]
            profile = classify(window)
            if profile is None:
                continue
            stamps.append(candles[end - 1].close_time)
            labels.append(profile.label)
        return cls(stamps, labels)

    def at(self, moment: datetime) -> str | None:
        """The regime in force at ``moment``, or ``None`` before the first classification.

        Looks *backwards* only. Reading the label of the next classification would be a
        look-ahead: the regime is only known once its bars have closed.
        """
        if not self._stamps:
            return None
        index = bisect_right(self._stamps, moment) - 1
        if index < 0:
            return None
        return self._labels[index]

    @property
    def labels(self) -> tuple[str, ...]:
        """Every distinct regime observed, in first-seen order."""
        return tuple(dict.fromkeys(self._labels))

    def __len__(self) -> int:
        return len(self._stamps)


def attribute(trades: Sequence[ClosedTrade], timeline: RegimeTimeline) -> RegimeBreakdown:
    """Group trades by the regime in force when each was opened."""
    buckets: dict[str, list[ClosedTrade]] = {}
    for trade in trades:
        label = timeline.at(trade.entry_time) or "unclassified"
        buckets.setdefault(label, []).append(trade)

    performances = [
        RegimePerformance(
            regime=label,
            trade_count=len(group),
            net_pnl=sum((trade.net_pnl for trade in group), ZERO),
            gross_pnl=sum((trade.gross_pnl for trade in group), ZERO),
            fees=sum((trade.fees for trade in group), ZERO),
            win_count=sum(1 for trade in group if trade.net_pnl > ZERO),
        )
        for label, group in buckets.items()
    ]
    performances.sort(key=lambda item: item.net_pnl, reverse=True)
    return RegimeBreakdown(by_regime=tuple(performances))


def merge(breakdowns: Sequence[RegimeBreakdown]) -> RegimeBreakdown:
    """Combine per-symbol breakdowns into one, summing each regime across symbols."""
    totals: dict[str, list[RegimePerformance]] = {}
    for breakdown in breakdowns:
        for item in breakdown.by_regime:
            totals.setdefault(item.regime, []).append(item)

    merged = [
        RegimePerformance(
            regime=label,
            trade_count=sum(item.trade_count for item in group),
            net_pnl=sum((item.net_pnl for item in group), ZERO),
            gross_pnl=sum((item.gross_pnl for item in group), ZERO),
            fees=sum((item.fees for item in group), ZERO),
            win_count=sum(item.win_count for item in group),
        )
        for label, group in totals.items()
    ]
    merged.sort(key=lambda item: item.net_pnl, reverse=True)
    return RegimeBreakdown(by_regime=tuple(merged))


def build_timelines(
    data: Mapping[object, Sequence[Candle]], *, stride: int = DEFAULT_STRIDE
) -> dict[object, RegimeTimeline]:
    """One timeline per symbol, built once and shared by every strategy."""
    return {
        symbol: RegimeTimeline.build(candles, stride=stride) for symbol, candles in data.items()
    }


__all__ = [
    "DEFAULT_STRIDE",
    "MIN_TRADES_PER_REGIME",
    "REGIME_LOOKBACK",
    "RegimeBreakdown",
    "RegimePerformance",
    "RegimeTimeline",
    "attribute",
    "build_timelines",
    "merge",
]
