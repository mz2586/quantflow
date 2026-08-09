"""Weighting strategies by what they actually earned.

Two failure modes to avoid, and they pull in opposite directions.

Weight purely by past return and the ensemble becomes a momentum bet on its own
backtest: whichever strategy happened to suit the sample dominates, and the moment
conditions change the ensemble is concentrated in the wrong thing. Weight everything
equally and a strategy known to lose money gets the same say as one known to work.

The middle is to weight by *risk-adjusted* performance, floor the weights so no single
strategy can own the book, and refuse to weight at all on a sample too thin to justify
the distinction. A weight is a claim that one strategy is better than another; on twelve
trades that claim is not supportable and equal weights are the honest answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from quantflow.backtest.metrics import PerformanceMetrics
from quantflow.core.precision import ZERO, safe_divide

ONE = Decimal("1")

#: Trades below which a strategy's record cannot justify a weight of its own.
MIN_TRADES_FOR_WEIGHTING = 30

#: No strategy may exceed this share of the ensemble, however good its record.
#: An ensemble whose weights collapse onto one member is not an ensemble.
MAX_WEIGHT = Decimal("0.40")

#: Strategies scoring at or below zero get nothing rather than a token allocation.
#: A small weight on a known loser is still a decision to lose money slowly.
MIN_WEIGHT = Decimal("0")


@dataclass(frozen=True, slots=True)
class StrategyWeight:
    """One strategy's share of the ensemble, and why it has it."""

    strategy_id: str
    weight: Decimal
    score: Decimal
    trade_count: int
    reliable: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialise for reports and the API."""
        return {
            "strategy_id": self.strategy_id,
            "weight": str(self.weight),
            "score": str(self.score),
            "trade_count": self.trade_count,
            "reliable": self.reliable,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class WeightSet:
    """The full allocation across an ensemble."""

    weights: tuple[StrategyWeight, ...]

    @property
    def total(self) -> Decimal:
        """Sum of all weights. One when anything was allocated, zero otherwise."""
        return sum((item.weight for item in self.weights), ZERO)

    @property
    def active(self) -> tuple[StrategyWeight, ...]:
        """Strategies with a non-zero allocation."""
        return tuple(item for item in self.weights if item.weight > ZERO)

    def weight_for(self, strategy_id: str) -> Decimal:
        """Share allocated to this strategy, or zero if it has none."""
        for item in self.weights:
            if item.strategy_id == strategy_id:
                return item.weight
        return ZERO

    @property
    def is_concentrated(self) -> bool:
        """Whether one strategy dominates despite the cap.

        Reachable when few strategies qualify: with two survivors the cap cannot bind
        below 0.5, and the ensemble is a coin flip between two ideas rather than a
        diversified book.
        """
        return bool(self.active) and max(item.weight for item in self.active) > MAX_WEIGHT

    def to_dict(self) -> dict[str, object]:
        """Serialise for reports and the API."""
        return {
            "weights": [item.to_dict() for item in self.weights],
            "active": len(self.active),
            "concentrated": self.is_concentrated,
        }


def score_of(metrics: PerformanceMetrics) -> Decimal:
    """Risk-adjusted score for weighting.

    Sharpe divided by a drawdown penalty, rather than raw return. Raw return would hand
    the book to whichever strategy took the most risk in the sample, which is precisely
    the strategy most likely to blow up out of it.
    """
    if metrics.trade_count == 0:
        return ZERO
    penalty = ONE + metrics.max_drawdown_pct * Decimal("2")
    return safe_divide(metrics.sharpe_ratio, penalty)


def compute_weights(
    performance: Mapping[str, PerformanceMetrics],
    *,
    min_trades: int = MIN_TRADES_FOR_WEIGHTING,
    max_weight: Decimal = MAX_WEIGHT,
) -> WeightSet:
    """Allocate across strategies by risk-adjusted performance.

    Strategies with a losing or unmeasurable record get zero. If nothing qualifies, the
    result is an empty allocation — which the ensemble reads as "do not trade", not as
    "fall back to equal weights".
    """
    if not performance:
        return WeightSet(weights=())

    scored: list[tuple[str, Decimal, PerformanceMetrics]] = []
    rejected: list[StrategyWeight] = []

    for strategy_id, metrics in sorted(performance.items()):
        if metrics.trade_count < min_trades:
            rejected.append(
                StrategyWeight(
                    strategy_id=strategy_id,
                    weight=ZERO,
                    score=ZERO,
                    trade_count=metrics.trade_count,
                    reliable=False,
                    reason=f"{metrics.trade_count} trades is below the {min_trades} minimum",
                )
            )
            continue

        score = score_of(metrics)
        if score <= ZERO:
            rejected.append(
                StrategyWeight(
                    strategy_id=strategy_id,
                    weight=ZERO,
                    score=score,
                    trade_count=metrics.trade_count,
                    reliable=True,
                    reason=f"risk-adjusted score {score:.3f} is not positive",
                )
            )
            continue
        scored.append((strategy_id, score, metrics))

    if not scored:
        return WeightSet(weights=tuple(rejected))

    total = sum((score for _, score, _ in scored), ZERO)
    raw = {strategy_id: safe_divide(score, total) for strategy_id, score, _ in scored}
    capped = _apply_cap(raw, max_weight)

    allocated = [
        StrategyWeight(
            strategy_id=strategy_id,
            weight=capped[strategy_id],
            score=score,
            trade_count=metrics.trade_count,
            reliable=True,
            reason=(
                f"score {score:.3f} of {total:.3f} total"
                + (" (capped)" if capped[strategy_id] < raw[strategy_id] else "")
            ),
        )
        for strategy_id, score, metrics in scored
    ]
    allocated.sort(key=lambda item: item.weight, reverse=True)
    return WeightSet(weights=tuple(allocated) + tuple(rejected))


def _apply_cap(raw: Mapping[str, Decimal], cap: Decimal) -> dict[str, Decimal]:
    """Cap each weight and redistribute the excess across the uncapped ones.

    The cap is skipped entirely when it cannot be satisfied — when ``cap * n < 1`` there
    is no allocation summing to one in which every member is under the cap, and applying
    it anyway drives every member to the cap. With two members and a 40% ceiling that
    means both land on 0.40: the weights sum to 0.80, and a strategy with three times the
    risk-adjusted score of its peer receives exactly the same allocation. A cap that
    erases the ranking it was meant to moderate is worse than no cap, so below the
    satisfiable threshold the raw weights stand and `WeightSet.is_concentrated` reports
    the concentration rather than hiding it.

    Otherwise iterative, because redistributing can push another weight over the cap.
    The loop is bounded by the number of strategies so it cannot spin.
    """
    if cap * Decimal(len(raw)) < ONE:
        return dict(raw)

    weights = dict(raw)
    for _ in range(len(weights)):
        over = {key: value for key, value in weights.items() if value > cap}
        if not over:
            break
        excess = sum((value - cap for value in over.values()), ZERO)
        under = [key for key, value in weights.items() if value < cap]
        if not under:
            # Everything is at the cap; the remainder cannot be placed and the weights
            # will sum below one. The ensemble reads that as reduced conviction, which
            # is the correct reading — there is nowhere left to put the risk.
            for key in over:
                weights[key] = cap
            break
        share = safe_divide(excess, Decimal(len(under)))
        for key in over:
            weights[key] = cap
        for key in under:
            weights[key] += share
    return weights


def equal_weights(strategy_ids: Sequence[str]) -> WeightSet:
    """Equal allocation, for when no record justifies distinguishing between them."""
    if not strategy_ids:
        return WeightSet(weights=())
    share = safe_divide(ONE, Decimal(len(strategy_ids)))
    return WeightSet(
        weights=tuple(
            StrategyWeight(
                strategy_id=strategy_id,
                weight=share,
                score=ZERO,
                trade_count=0,
                reliable=False,
                reason="equal weight: no record to justify a distinction",
            )
            for strategy_id in sorted(strategy_ids)
        )
    )


__all__ = [
    "MAX_WEIGHT",
    "MIN_TRADES_FOR_WEIGHTING",
    "MIN_WEIGHT",
    "StrategyWeight",
    "WeightSet",
    "compute_weights",
    "equal_weights",
    "score_of",
]
