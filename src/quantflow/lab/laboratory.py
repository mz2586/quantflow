"""The Strategy Laboratory.

Runs every strategy over every symbol **twice** — once under realistic costs and once with
costs removed — attributes each trade to the regime it was opened in, screens the result
against the acceptance gate, and diagnoses the cause of any failure.

The second run is the point of the whole module. Without it, "lost 8%" is a symptom with
at least three incompatible causes: a worthless signal, a good signal handed to the venue,
or a good signal traded too often. Those lead to *opposite* decisions — discard, change
execution, change timeframe — and no amount of staring at the net figure distinguishes
them. Running the same strategy on the same bars for free is the only measurement that
does, and it costs one extra backtest.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal

from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle
from quantflow.lab.attribution import (
    DEFAULT_STRIDE,
    RegimeBreakdown,
    RegimeTimeline,
    attribute,
    merge,
)
from quantflow.lab.diagnosis import Diagnosis, FailureCause, diagnose
from quantflow.research.costs import CostModel, realistic, zero_cost
from quantflow.research.runner import (
    ResearchConfig,
    ResearchOutcome,
    ResearchRunner,
    StrategyRun,
)
from quantflow.research.thresholds import DEFAULT_THRESHOLDS, AcceptanceThresholds

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LabResult:
    """One strategy's complete laboratory verdict."""

    strategy_id: str
    accepted: bool
    #: Per-symbol runs under realistic costs.
    runs: tuple[StrategyRun, ...]
    diagnosis: Diagnosis
    regimes: RegimeBreakdown
    net_return: Decimal
    frictionless_return: Decimal | None
    trade_count: int
    total_fees: Decimal
    rejection_reasons: tuple[str, ...] = ()

    @property
    def cost_drag(self) -> Decimal | None:
        """Return given up to costs: frictionless minus net."""
        if self.frictionless_return is None:
            return None
        return self.frictionless_return - self.net_return

    @property
    def is_regime_dependent(self) -> bool:
        """Whether it works in some conditions and not others."""
        return self.regimes.is_regime_dependent

    def to_dict(self) -> dict[str, object]:
        """Serialise for reports and the API."""
        return {
            "strategy_id": self.strategy_id,
            "accepted": self.accepted,
            "net_return": str(self.net_return),
            "frictionless_return": (
                str(self.frictionless_return) if self.frictionless_return is not None else None
            ),
            "cost_drag": str(self.cost_drag) if self.cost_drag is not None else None,
            "trade_count": self.trade_count,
            "total_fees": str(self.total_fees),
            "diagnosis": self.diagnosis.to_dict(),
            "regimes": self.regimes.to_dict(),
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True, slots=True)
class LabReport:
    """Everything the laboratory established in one run."""

    results: tuple[LabResult, ...]
    period_start: str
    period_end: str
    bars_per_symbol: dict[str, int]
    costs: str
    thresholds: AcceptanceThresholds
    duration_seconds: float
    regimes_observed: tuple[str, ...] = ()
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> tuple[LabResult, ...]:
        """Strategies that passed the gate."""
        return tuple(item for item in self.results if item.accepted)

    @property
    def execution_fixable(self) -> tuple[LabResult, ...]:
        """Rejected strategies whose signal made money before costs.

        The most valuable output of the laboratory: these are not bad ideas, they are
        ideas being executed badly, and they are the only rejections worth revisiting.
        """
        return tuple(
            item
            for item in self.results
            if not item.accepted and item.diagnosis.is_fixable_by_execution
        )

    @property
    def regime_dependent(self) -> tuple[LabResult, ...]:
        """Strategies that work in some regimes and not others."""
        return tuple(item for item in self.results if item.is_regime_dependent)

    def by_cause(self) -> dict[str, int]:
        """How many strategies failed for each reason."""
        counts: dict[str, int] = {}
        for item in self.results:
            if item.accepted:
                continue
            key = str(item.diagnosis.cause)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda pair: -pair[1]))


class StrategyLaboratory:
    """Evaluates strategies under realistic and frictionless costs, per regime."""

    def __init__(
        self,
        symbols: Sequence[Symbol],
        *,
        costs: CostModel | None = None,
        thresholds: AcceptanceThresholds = DEFAULT_THRESHOLDS,
        starting_equity: Decimal = Decimal("10000"),
        strategy_ids: Sequence[str] = (),
        regime_stride: int = DEFAULT_STRIDE,
    ) -> None:
        self._symbols = tuple(symbols)
        self._costs = costs or realistic()
        self._thresholds = thresholds
        self._starting_equity = starting_equity
        self._strategy_ids = tuple(strategy_ids)
        self._regime_stride = regime_stride

    async def run(
        self,
        data: dict[Symbol, Sequence[Candle]],
        instruments: dict[Symbol, Instrument],
    ) -> LabReport:
        """Evaluate every strategy and return the full verdict."""
        started = time.monotonic()

        base = ResearchConfig(
            symbols=self._symbols,
            starting_equity=self._starting_equity,
            costs=self._costs,
            thresholds=self._thresholds,
            strategy_ids=self._strategy_ids,
        )
        logger.info("lab.priced_run_started", costs=self._costs.name)
        priced = await ResearchRunner(base).run(data, instruments)

        # The decisive second pass. Thresholds are irrelevant here — this run exists only
        # to answer "did the signal make money before anyone was paid", so the gate is
        # switched off rather than producing rejections nobody will read.
        logger.info("lab.frictionless_run_started")
        free_config = replace(
            base,
            costs=zero_cost(),
            thresholds=AcceptanceThresholds(must_beat_benchmark=False),
        )
        frictionless = await ResearchRunner(free_config).run(data, instruments)

        timelines = {
            symbol: RegimeTimeline.build(candles, stride=self._regime_stride)
            for symbol, candles in data.items()
        }
        observed = tuple(
            dict.fromkeys(label for timeline in timelines.values() for label in timeline.labels)
        )

        results = self._assemble(priced, frictionless, timelines)
        candles = data[self._symbols[0]]
        return LabReport(
            results=results,
            period_start=candles[0].open_time.isoformat(),
            period_end=candles[-1].open_time.isoformat(),
            bars_per_symbol={str(s): len(data[s]) for s in self._symbols},
            costs=self._costs.summary,
            thresholds=self._thresholds,
            duration_seconds=time.monotonic() - started,
            regimes_observed=observed,
            failures=tuple(
                f"{failure.strategy_id} on {failure.symbol}: {failure.error}"
                for failure in priced.failures
            ),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _assemble(
        self,
        priced: ResearchOutcome,
        frictionless: ResearchOutcome,
        timelines: dict[Symbol, RegimeTimeline],
    ) -> tuple[LabResult, ...]:
        """Join the two runs, attribute regimes and diagnose each strategy."""
        priced_by_strategy: dict[str, list[StrategyRun]] = {}
        for run in priced.runs:
            priced_by_strategy.setdefault(run.strategy_id, []).append(run)

        free_by_strategy: dict[str, list[StrategyRun]] = {}
        for run in frictionless.runs:
            free_by_strategy.setdefault(run.strategy_id, []).append(run)

        results: list[LabResult] = []
        for strategy_id, runs in priced_by_strategy.items():
            free_runs = free_by_strategy.get(strategy_id, [])
            results.append(self._result_for(strategy_id, runs, free_runs, timelines))

        # Accepted first, then by how much was left on the table to costs: an execution
        # problem worth fixing sorts above an idea that was never going to work.
        results.sort(
            key=lambda item: (
                not item.accepted,
                -(item.cost_drag or ZERO),
                -item.net_return,
            )
        )
        return tuple(results)

    def _result_for(
        self,
        strategy_id: str,
        runs: Sequence[StrategyRun],
        free_runs: Sequence[StrategyRun],
        timelines: dict[Symbol, RegimeTimeline],
    ) -> LabResult:
        """Build one strategy's verdict from its priced and frictionless runs."""
        accepted = bool(runs) and all(run.accepted for run in runs)
        net_return = _mean(run.metrics.total_return_pct for run in runs)
        free_return = (
            _mean(run.metrics.total_return_pct for run in free_runs) if free_runs else None
        )

        breakdown = merge([attribute(run.trades, timelines[run.symbol]) for run in runs])

        worst = max(runs, key=lambda run: run.metrics.max_drawdown_pct, default=None)
        diagnosis = diagnose(
            worst.metrics if worst else runs[0].metrics,
            frictionless=free_runs[0].metrics if free_runs else None,
            max_drawdown=self._thresholds.max_drawdown,
        )

        reasons = tuple(
            f"{run.symbol}: {rejection.detail}"
            for run in runs
            for rejection in run.screen.rejections
        )

        return LabResult(
            strategy_id=strategy_id,
            accepted=accepted,
            runs=tuple(runs),
            diagnosis=diagnosis if not accepted else _passing_diagnosis(),
            regimes=breakdown,
            net_return=net_return,
            frictionless_return=free_return,
            trade_count=sum(run.metrics.trade_count for run in runs),
            total_fees=sum((run.metrics.total_fees for run in runs), ZERO),
            rejection_reasons=reasons,
        )


def _passing_diagnosis() -> Diagnosis:
    """The diagnosis attached to a strategy that passed."""
    return Diagnosis(
        cause=FailureCause.NONE,
        explanation="passed every threshold on every symbol",
        recommendation="Promote to walk-forward validation, not to capital.",
    )


def _mean(values: Iterable[Decimal]) -> Decimal:
    """Arithmetic mean in Decimal, zero for an empty sequence."""
    materialised = list(values)
    if not materialised:
        return ZERO
    return sum(materialised, ZERO) / Decimal(len(materialised))


__all__ = ["LabReport", "LabResult", "StrategyLaboratory"]
