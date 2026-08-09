"""The research runner: backtest every strategy over every symbol, then screen the results.

Two design decisions carry most of the weight.

**Every strategy sees identical data, identical costs and identical risk settings.** If
one strategy were tested on a different period or with a different position size, the
leaderboard would be ranking the experimental setup rather than the ideas, and the ranking
would be worthless. The grid is therefore fully crossed and the configuration is built
once and shared.

**Buy-and-hold is run as a peer, not as an afterthought.** It pays the same fees and the
same slippage as everything else, which is the only way its comparison is honest. Its
return per symbol becomes the benchmark that the acceptance gate applies to every other
strategy on that symbol.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from quantflow.backtest.engine import BacktestConfig, BacktestEngine
from quantflow.backtest.metrics import PerformanceMetrics, compute_metrics
from quantflow.core.config import RiskSettings
from quantflow.core.errors import QuantFlowError, ValidationError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle
from quantflow.domain.positions import ClosedTrade
from quantflow.research.benchmark import buy_and_hold_metrics
from quantflow.research.costs import CostModel, realistic
from quantflow.research.thresholds import (
    DEFAULT_THRESHOLDS,
    AcceptanceThresholds,
    Rejection,
    RejectionCode,
    ScreenResult,
    screen,
)
from quantflow.strategy.registry import StrategyRegistry, load_builtin_strategies

logger = get_logger(__name__)

#: The strategy whose result is the benchmark every other strategy must beat.
BENCHMARK_STRATEGY_ID = "buy_and_hold"

#: Multiple of a strategy's declared warm-up handed to it as history.
#:
#: Indicators here recompute over the whole visible window on every bar, so the cost of a
#: run is O(bars x window). The engine default of 5,000 bars makes a full sweep take days
#: while giving a strategy that declared it needs 52 bars 5,000 of them. Three times the
#: declared warm-up leaves recursive indicators (EMA, Wilder smoothing) thoroughly
#: converged — their weight decays geometrically, so a bar three warm-ups back contributes
#: far less than rounding error — while cutting the work by one to two orders of magnitude.
#:
#: `tests/unit/test_research.py` asserts the equivalence rather than assuming it.
HISTORY_WARMUP_MULTIPLE = 3

#: Floor for the history window, for strategies that declare a very short warm-up.
MIN_HISTORY_BARS = 300


def history_window_for(warmup_bars: int) -> int:
    """How much history to hand a strategy with this warm-up."""
    return max(warmup_bars * HISTORY_WARMUP_MULTIPLE, MIN_HISTORY_BARS)


def default_worker_count() -> int:
    """Processes to run backtests in.

    Leaves two cores free so the machine stays usable during a sweep that can run for
    tens of minutes. Backtests are pure, CPU-bound and independent, which makes them
    close to the ideal case for process-level parallelism — and they cannot share a
    thread pool usefully, because the work is Decimal arithmetic under the GIL.
    """
    return max(1, (os.cpu_count() or 2) - 2)


# --------------------------------------------------------------------------- #
# Worker process
# --------------------------------------------------------------------------- #
#: Per-process market data, populated once by the pool initialiser.
#:
#: Passing candles with every task would pickle tens of thousands of Decimal-bearing
#: objects 56 times over and undo most of the benefit of running in parallel. The
#: initialiser pays that cost once per worker instead.
_WORKER_DATA: dict[Symbol, Sequence[Candle]] = {}
_WORKER_INSTRUMENTS: dict[Symbol, Instrument] = {}


def _init_worker(
    data: dict[Symbol, Sequence[Candle]], instruments: dict[Symbol, Instrument]
) -> None:
    """Seed a worker process with the sweep's market data."""
    global _WORKER_DATA, _WORKER_INSTRUMENTS  # noqa: PLW0603 - per-process cache by design
    _WORKER_DATA = data
    _WORKER_INSTRUMENTS = instruments


@dataclass(frozen=True, slots=True)
class _WorkerRequest:
    """One backtest to run, small enough to pickle cheaply."""

    strategy_id: str
    symbol: Symbol
    timeframe: Timeframe
    starting_equity: Decimal
    risk: RiskSettings
    costs: CostModel
    risk_free_rate: float


@dataclass(frozen=True, slots=True)
class _WorkerResponse:
    """The outcome of one backtest, reduced to what the parent needs."""

    strategy_id: str
    symbol: Symbol
    metrics: PerformanceMetrics | None
    params: dict[str, Any]
    bars: int
    signals: int
    orders: int
    rejected_signals: int
    error: str | None = None
    trades: tuple[ClosedTrade, ...] = ()


def _run_in_worker(request: _WorkerRequest) -> _WorkerResponse:
    """Run one backtest inside a worker process.

    Returns metrics rather than the full `BacktestResult`: the result carries every
    order, fill and signal, and shipping all of that back across a process boundary
    would cost more than the backtest itself.
    """
    empty = _WorkerResponse(request.strategy_id, request.symbol, None, {}, 0, 0, 0, 0)
    try:
        registry = load_builtin_strategies()
        strategy = registry.create(request.strategy_id, {})
        config = BacktestConfig(
            symbols=(request.symbol,),
            timeframe=request.timeframe,
            starting_equity=request.starting_equity,
            risk=request.risk,
            slippage=request.costs.slippage,
            fees=request.costs.fees,
            risk_free_rate=request.risk_free_rate,
            max_history_bars=history_window_for(strategy.warmup_bars),
        )
        engine = BacktestEngine(
            strategy, config, {request.symbol: _WORKER_INSTRUMENTS[request.symbol]}
        )
        result = asyncio.run(engine.run({request.symbol: _WORKER_DATA[request.symbol]}))
    except Exception as exc:  # one bad strategy must not kill the pool
        return replace(empty, error=f"{type(exc).__name__}: {exc}")

    if not result.succeeded:
        return replace(empty, error=result.error or "run did not complete")

    metrics = compute_metrics(
        curve=result.equity_curve,
        trades=result.closed_trades,
        starting_equity=request.starting_equity,
        timeframe=request.timeframe,
        total_fees=sum((trade.fees for trade in result.closed_trades), ZERO),
        risk_free_rate=request.risk_free_rate,
    )
    return _WorkerResponse(
        strategy_id=request.strategy_id,
        symbol=request.symbol,
        metrics=metrics,
        params=result.strategy_params,
        bars=result.bars_processed,
        signals=len(result.signals),
        orders=len(result.orders),
        rejected_signals=len(result.rejected_signals),
        trades=tuple(result.closed_trades),
    )


@dataclass(frozen=True, slots=True)
class StrategyRun:
    """One strategy backtested on one symbol."""

    strategy_id: str
    symbol: Symbol
    metrics: PerformanceMetrics
    screen: ScreenResult
    params: dict[str, Any]
    bars: int
    signals: int
    orders: int
    rejected_signals: int
    duration_seconds: float
    benchmark_return: Decimal | None = None
    #: The round-trips this run produced. Needed to attribute performance to the regime
    #: each trade was opened in; without them the laboratory would have to re-run the
    #: whole backtest just to see the trades it already computed.
    trades: tuple[ClosedTrade, ...] = ()

    @property
    def accepted(self) -> bool:
        """Whether this run passed the gate."""
        return self.screen.accepted

    @property
    def excess_return(self) -> Decimal | None:
        """Return above buy-and-hold on the same symbol, when a benchmark exists."""
        if self.benchmark_return is None:
            return None
        return self.metrics.total_return_pct - self.benchmark_return


@dataclass(frozen=True, slots=True)
class FailedRun:
    """A backtest that did not complete. Recorded rather than silently dropped."""

    strategy_id: str
    symbol: Symbol
    error: str


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Everything that defines a research sweep."""

    symbols: tuple[Symbol, ...]
    timeframe: Timeframe = Timeframe.H1
    starting_equity: Decimal = Decimal("10000")
    costs: CostModel = field(default_factory=realistic)
    thresholds: AcceptanceThresholds = DEFAULT_THRESHOLDS
    risk: RiskSettings = field(default_factory=RiskSettings)
    risk_free_rate: float = 0.0
    #: Strategy ids to run. Empty means every registered strategy.
    strategy_ids: tuple[str, ...] = ()
    #: Backtest worker processes. None picks a default that leaves the machine usable.
    workers: int | None = None


@dataclass(frozen=True, slots=True)
class ResearchOutcome:
    """The complete result of a sweep."""

    config: ResearchConfig
    runs: tuple[StrategyRun, ...]
    failures: tuple[FailedRun, ...]
    bars_per_symbol: dict[str, int]
    period_start: str
    period_end: str
    duration_seconds: float

    @property
    def accepted(self) -> tuple[StrategyRun, ...]:
        """Runs that passed the gate."""
        return tuple(run for run in self.runs if run.accepted)

    @property
    def strategy_ids(self) -> tuple[str, ...]:
        """Every strategy that produced at least one run."""
        return tuple(dict.fromkeys(run.strategy_id for run in self.runs))


class ResearchRunner:
    """Runs the strategy x symbol grid and screens every result.

    Backtests execute in a process pool. On macOS and Windows that pool uses the *spawn*
    start method, which re-imports the calling module in every worker — so a script that
    calls `run()` at import time must guard its entry point::

        if __name__ == "__main__":
            asyncio.run(main())

    Without the guard each worker re-executes the script and multiprocessing aborts the
    run. The `quantflow research` CLI is already guarded; this only affects direct
    library use from a bare script.
    """

    def __init__(
        self,
        config: ResearchConfig,
        *,
        registry: StrategyRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or load_builtin_strategies()

    async def run(
        self,
        data: dict[Symbol, Sequence[Candle]],
        instruments: dict[Symbol, Instrument],
    ) -> ResearchOutcome:
        """Backtest every strategy on every symbol.

        Raises:
            ValidationError: if a requested symbol has no data, which would otherwise
                produce a silent hole in the leaderboard.

        """
        started = time.monotonic()
        missing = [symbol for symbol in self._config.symbols if not data.get(symbol)]
        if missing:
            raise ValidationError(
                f"no candles for {', '.join(str(symbol) for symbol in missing)}",
                field="symbols",
            )

        strategy_ids = self._config.strategy_ids or tuple(self._registry.names())

        # The benchmark is computed from the price series rather than traded through the
        # engine — see `research.benchmark` for why routing it through the risk engine
        # produces something that is not buy-and-hold at all. It is therefore exact,
        # instant, and needs no worker.
        benchmarks = [
            self._benchmark_response(symbol, data[symbol]) for symbol in self._config.symbols
        ]
        benchmark_returns = {
            response.symbol: response.metrics.total_return_pct
            for response in benchmarks
            if response.metrics is not None
        }

        jobs = [
            (strategy_id, symbol)
            for strategy_id in strategy_ids
            if strategy_id != BENCHMARK_STRATEGY_ID
            for symbol in self._config.symbols
        ]
        workers = self._config.workers or default_worker_count()
        loop = asyncio.get_running_loop()
        logger.info("research.sweep_started", jobs=len(jobs), workers=workers)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(data, instruments)
        ) as pool:
            responses = await self._gather(loop, pool, jobs)

        runs: list[StrategyRun] = []
        failures: list[FailedRun] = []
        for response in [*benchmarks, *responses]:
            if response.metrics is None:
                failures.append(
                    FailedRun(
                        response.strategy_id, response.symbol, response.error or "unknown error"
                    )
                )
                continue
            runs.append(self._screen(response, benchmark_returns.get(response.symbol)))

        candles = data[self._config.symbols[0]]
        return ResearchOutcome(
            config=self._config,
            runs=tuple(runs),
            failures=tuple(failures),
            bars_per_symbol={str(symbol): len(data[symbol]) for symbol in self._config.symbols},
            period_start=candles[0].open_time.isoformat(),
            period_end=candles[-1].open_time.isoformat(),
            duration_seconds=time.monotonic() - started,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _gather(
        self,
        loop: asyncio.AbstractEventLoop,
        pool: ProcessPoolExecutor,
        jobs: Sequence[tuple[str, Symbol]],
    ) -> list[_WorkerResponse]:
        """Run a batch of backtests across the pool and wait for all of them."""
        if not jobs:
            return []
        futures = [
            loop.run_in_executor(pool, _run_in_worker, self._request(strategy_id, symbol))
            for strategy_id, symbol in jobs
        ]
        return list(await asyncio.gather(*futures))

    def _benchmark_response(self, symbol: Symbol, candles: Sequence[Candle]) -> _WorkerResponse:
        """Buy-and-hold for one symbol, measured directly from the price series."""
        try:
            metrics = buy_and_hold_metrics(
                symbol,
                candles,
                starting_equity=self._config.starting_equity,
                timeframe=self._config.timeframe,
                costs=self._config.costs,
                risk_free_rate=self._config.risk_free_rate,
            )
        except QuantFlowError as exc:
            logger.warning("research.benchmark_failed", symbol=str(symbol), error=str(exc))
            return _WorkerResponse(
                BENCHMARK_STRATEGY_ID, symbol, None, {}, 0, 0, 0, 0, error=str(exc)
            )
        return _WorkerResponse(
            strategy_id=BENCHMARK_STRATEGY_ID,
            symbol=symbol,
            metrics=metrics,
            params={},
            bars=len(candles),
            signals=1,
            orders=1,
            rejected_signals=0,
        )

    def _request(self, strategy_id: str, symbol: Symbol) -> _WorkerRequest:
        """Build the picklable description of one backtest."""
        return _WorkerRequest(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=self._config.timeframe,
            starting_equity=self._config.starting_equity,
            risk=self._config.risk,
            costs=self._config.costs,
            risk_free_rate=self._config.risk_free_rate,
        )

    def _screen(self, response: _WorkerResponse, benchmark_return: Decimal | None) -> StrategyRun:
        """Apply the acceptance gate to one completed run."""
        if response.metrics is None:  # pragma: no cover - caller filters failures first
            raise ValidationError("cannot screen a run with no metrics", field="metrics")
        is_benchmark = response.strategy_id == BENCHMARK_STRATEGY_ID
        # The benchmark is not screened against itself: it would always tie, and
        # "buy-and-hold failed to beat buy-and-hold" is noise in a report.
        verdict = screen(
            response.metrics,
            self._config.thresholds,
            benchmark_return=None if is_benchmark else benchmark_return,
        )
        return StrategyRun(
            strategy_id=response.strategy_id,
            symbol=response.symbol,
            metrics=response.metrics,
            screen=verdict,
            params=response.params,
            bars=response.bars,
            signals=response.signals,
            orders=response.orders,
            rejected_signals=response.rejected_signals,
            duration_seconds=0.0,
            benchmark_return=None if is_benchmark else benchmark_return,
            trades=response.trades,
        )


def failed_run_as_rejection(failure: FailedRun) -> Rejection:
    """Present a crashed run as a rejection, so the report has one uniform shape."""
    return Rejection(RejectionCode.RUN_FAILED, f"run failed: {failure.error}")


__all__ = [
    "BENCHMARK_STRATEGY_ID",
    "HISTORY_WARMUP_MULTIPLE",
    "MIN_HISTORY_BARS",
    "FailedRun",
    "ResearchConfig",
    "ResearchOutcome",
    "ResearchRunner",
    "StrategyRun",
    "failed_run_as_rejection",
    "history_window_for",
]
