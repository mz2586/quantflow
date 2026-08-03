"""Ranking strategies across the six required metrics.

Ranking on a single metric is how research goes wrong. Sort by net return and the top of
the table fills with strategies that took enormous risk; sort by Sharpe and it fills with
strategies that traded four times. So the leaderboard does two separate things and keeps
them visibly separate:

* **Per-metric ranks** — where a strategy places on each of the six criteria
  individually, so a reader can see *why* something ranks where it does.
* **A composite score** — the mean of those ranks, which is deliberately ordinal. Averaging
  the raw numbers would let one metric with a wide range (return, unbounded) drown five
  with narrow ranges (win rate, bounded at 1). A rank average cannot be dominated that way.

The composite is a reading aid, never an authority: a strategy that fails the acceptance
gate is reported as rejected however well it scores.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean

from quantflow.core.precision import ZERO
from quantflow.research.runner import BENCHMARK_STRATEGY_ID, ResearchOutcome, StrategyRun


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One ranking criterion."""

    key: str
    label: str
    #: Extract the comparable value from an aggregated entry.
    extract: Callable[[LeaderboardEntry], Decimal]
    #: True when a larger value is better.
    higher_is_better: bool
    #: How to render it.
    fmt: str = "{:.2f}"


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """One strategy's aggregated performance across every symbol it ran on."""

    strategy_id: str
    symbols_tested: int
    symbols_accepted: int
    #: Mean across symbols. Averaging keeps a strategy from ranking highly on the strength
    #: of one symbol it happened to suit.
    net_return: Decimal
    profit_factor: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    trade_count: int
    total_fees: Decimal
    excess_return: Decimal | None
    #: Worst single-symbol return. A strategy is only as deployable as its worst market.
    worst_symbol_return: Decimal
    accepted: bool
    rejection_summary: str
    runs: tuple[StrategyRun, ...]

    @property
    def is_benchmark(self) -> bool:
        """Whether this row is the buy-and-hold benchmark."""
        return self.strategy_id == BENCHMARK_STRATEGY_ID


@dataclass(frozen=True, slots=True)
class RankedEntry:
    """A leaderboard entry with its ranks attached."""

    entry: LeaderboardEntry
    #: Metric key → 1-based rank among all entries.
    ranks: dict[str, int]
    composite: float
    position: int


#: The six criteria the brief requires, in the order they are reported.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("net_return", "Net return", lambda e: e.net_return, True, "{:.2%}"),
    MetricSpec("profit_factor", "Profit factor", lambda e: e.profit_factor, True, "{:.2f}"),
    MetricSpec("sharpe_ratio", "Sharpe", lambda e: e.sharpe_ratio, True, "{:.2f}"),
    MetricSpec("max_drawdown", "Max drawdown", lambda e: e.max_drawdown, False, "{:.2%}"),
    MetricSpec("win_rate", "Win rate", lambda e: e.win_rate, True, "{:.2%}"),
    MetricSpec("trade_count", "Trades", lambda e: Decimal(e.trade_count), True, "{:.0f}"),
)


def aggregate(outcome: ResearchOutcome) -> tuple[LeaderboardEntry, ...]:
    """Collapse per-symbol runs into one entry per strategy."""
    grouped: dict[str, list[StrategyRun]] = {}
    for run in outcome.runs:
        grouped.setdefault(run.strategy_id, []).append(run)

    entries = [_entry_for(strategy_id, runs) for strategy_id, runs in grouped.items()]
    return tuple(sorted(entries, key=lambda entry: entry.strategy_id))


def _entry_for(strategy_id: str, runs: Sequence[StrategyRun]) -> LeaderboardEntry:
    """Aggregate one strategy's runs."""
    count = len(runs)
    accepted_runs = [run for run in runs if run.accepted]
    excesses = [run.excess_return for run in runs if run.excess_return is not None]

    # A strategy counts as accepted only if it passed on *every* symbol. Passing on one
    # market and failing on three is a strategy that found one favourable regime, and
    # reporting that as "accepted" is precisely the overfit this framework exists to catch.
    accepted = count > 0 and len(accepted_runs) == count

    reasons: list[str] = []
    for run in runs:
        if not run.accepted:
            reasons.append(f"{run.symbol}: {run.screen.summary}")

    return LeaderboardEntry(
        strategy_id=strategy_id,
        symbols_tested=count,
        symbols_accepted=len(accepted_runs),
        net_return=_mean(run.metrics.total_return_pct for run in runs),
        profit_factor=_mean(run.metrics.profit_factor for run in runs),
        sharpe_ratio=_mean(run.metrics.sharpe_ratio for run in runs),
        max_drawdown=_mean(run.metrics.max_drawdown_pct for run in runs),
        win_rate=_mean(run.metrics.win_rate for run in runs),
        trade_count=sum(run.metrics.trade_count for run in runs),
        total_fees=sum((run.metrics.total_fees for run in runs), ZERO),
        excess_return=_mean(excesses) if excesses else None,
        worst_symbol_return=min((run.metrics.total_return_pct for run in runs), default=ZERO),
        accepted=accepted,
        rejection_summary=" | ".join(reasons),
        runs=tuple(runs),
    )


def _mean(values: Iterable[Decimal]) -> Decimal:
    """Arithmetic mean in Decimal, or zero for an empty sequence.

    Decimal throughout rather than `statistics.fmean`: these are monetary and ratio
    quantities, and routing them through float to take an average is how a leaderboard
    ends up reporting a return that does not match the equity curve it came from.
    """
    materialised = list(values)
    if not materialised:
        return ZERO
    return sum(materialised, ZERO) / Decimal(len(materialised))


def rank(entries: Sequence[LeaderboardEntry]) -> tuple[RankedEntry, ...]:
    """Rank entries on every metric, then order by the composite.

    The benchmark is ranked alongside everything else on purpose: seeing exactly where
    buy-and-hold lands in the table is the single most informative line in the report.
    """
    if not entries:
        return ()

    ranks: dict[str, dict[str, int]] = {entry.strategy_id: {} for entry in entries}
    for metric in METRICS:
        ordered = sorted(
            entries,
            key=lambda entry, m=metric: m.extract(entry),  # type: ignore[misc]
            reverse=metric.higher_is_better,
        )
        for position, entry in enumerate(ordered, start=1):
            ranks[entry.strategy_id][metric.key] = position

    ranked = [
        RankedEntry(
            entry=entry,
            ranks=ranks[entry.strategy_id],
            composite=fmean(ranks[entry.strategy_id].values()),
            position=0,
        )
        for entry in entries
    ]
    # Accepted strategies sort above rejected ones regardless of composite: the gate is
    # the decision, and the score only orders within it.
    ranked.sort(key=lambda item: (not item.entry.accepted, item.composite))
    return tuple(
        RankedEntry(item.entry, item.ranks, item.composite, position)
        for position, item in enumerate(ranked, start=1)
    )


def leaderboard(outcome: ResearchOutcome) -> tuple[RankedEntry, ...]:
    """Aggregate and rank in one step."""
    return rank(aggregate(outcome))


__all__ = [
    "METRICS",
    "LeaderboardEntry",
    "MetricSpec",
    "RankedEntry",
    "aggregate",
    "leaderboard",
    "rank",
]
