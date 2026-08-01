"""Strategy catalogue and backtest endpoints."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Query

from quantflow.api.deps import AuthDep, DatabaseDep, RegistryDep, StateDep
from quantflow.api.schemas import (
    BacktestMetricsResponse,
    BacktestRequest,
    BacktestResponse,
    StrategyDescription,
)
from quantflow.backtest.engine import BacktestConfig, BacktestEngine, rejection_reasons
from quantflow.backtest.metrics import is_statistically_thin
from quantflow.backtest.report import write_report
from quantflow.core.errors import InsufficientDataError, ValidationError
from quantflow.core.logging import get_logger
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle
from quantflow.persistence.repositories import CandleRepository, InstrumentRepository

logger = get_logger(__name__)

router = APIRouter(tags=["strategies"])

#: Upper bound on a synchronous backtest. Anything larger belongs on the worker queue:
#: holding an HTTP connection open for a multi-year 1m run is how a request times out
#: halfway through and leaves the client with no result and no run id.
MAX_SYNC_BARS = 200_000


@router.get("/strategies", response_model=list[StrategyDescription], summary="List strategies")
async def list_strategies(registry: RegistryDep) -> list[StrategyDescription]:
    """Every registered strategy with its parameter schema and defaults."""
    return [
        StrategyDescription(
            strategy_id=entry["strategy_id"],
            description=entry["description"],
            warmup_bars=entry["warmup_bars"],
            defaults=entry["defaults"],
            parameter_schema=entry["schema"],
        )
        for entry in registry.describe_all()
    ]


@router.get(
    "/strategies/{strategy_id}",
    response_model=StrategyDescription,
    summary="Describe one strategy",
)
async def describe_strategy(strategy_id: str, registry: RegistryDep) -> StrategyDescription:
    """One strategy's schema, defaults and warm-up requirement."""
    entry = registry.describe(strategy_id)
    return StrategyDescription(
        strategy_id=entry["strategy_id"],
        description=entry["description"],
        warmup_bars=entry["warmup_bars"],
        defaults=entry["defaults"],
        parameter_schema=entry["schema"],
    )


@router.post("/backtest", response_model=BacktestResponse, summary="Run a backtest synchronously")
async def run_backtest(
    request: BacktestRequest,
    state: StateDep,
    database: DatabaseDep,
    registry: RegistryDep,
    _auth: AuthDep,
) -> BacktestResponse:
    """Run a backtest over stored candles and return its metrics.

    Synchronous, and bounded: a run larger than :data:`MAX_SYNC_BARS` is refused with a
    clear message rather than silently timing out mid-run.
    """
    if request.end <= request.start:
        raise ValidationError("end must be after start")

    strategy = registry.create(request.strategy_id, request.params)
    symbols = [Symbol.parse(raw) for raw in request.symbols]

    data: dict[Symbol, Sequence[Candle]] = {}
    instruments: dict[Symbol, Instrument] = {}
    total_bars = 0

    async with database.read_session() as session:
        candle_repo = CandleRepository(session)
        instrument_repo = InstrumentRepository(session)
        for symbol in symbols:
            assert isinstance(symbol, Symbol)
            candles = await candle_repo.fetch(
                symbol, request.timeframe, start=request.start, end=request.end
            )
            if not candles:
                raise InsufficientDataError(
                    f"no stored candles for {symbol} {request.timeframe.value} "
                    f"between {request.start.date()} and {request.end.date()}; "
                    "download the data first",
                    symbol=symbol.slashed,
                )
            total_bars += len(candles)
            if total_bars > MAX_SYNC_BARS:
                raise ValidationError(
                    f"requested range spans more than {MAX_SYNC_BARS:,} bars; "
                    "narrow the range or run it from the CLI",
                    bars=total_bars,
                )
            data[symbol] = candles
            instrument = await instrument_repo.get(symbol)
            instruments[symbol] = instrument or Instrument(symbol=symbol)

    config = BacktestConfig(
        symbols=tuple(data),
        timeframe=request.timeframe,
        starting_equity=request.starting_equity,
        risk=state.settings.risk,
    )
    result = await BacktestEngine(strategy, config, instruments).run(data)

    metrics_response: BacktestMetricsResponse | None = None
    if result.succeeded:
        metrics = result.metrics()
        metrics_response = BacktestMetricsResponse(
            starting_equity=metrics.starting_equity,
            final_equity=metrics.final_equity,
            total_return_pct=metrics.total_return_pct,
            cagr=metrics.cagr,
            max_drawdown_pct=metrics.max_drawdown_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            calmar_ratio=metrics.calmar_ratio,
            trade_count=metrics.trade_count,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            total_fees=metrics.total_fees,
            exposure_pct=metrics.exposure_pct,
            statistically_thin=is_statistically_thin(metrics),
        )

    report_path: str | None = None
    if request.generate_report and result.succeeded:
        path = write_report(result, state.settings.storage.report_dir)
        report_path = str(path)

    return BacktestResponse(
        run_id=result.run_id,
        status=result.status,
        strategy_id=result.strategy_id,
        symbols=tuple(symbol.slashed for symbol in data),
        timeframe=request.timeframe,
        bars=result.bars_processed,
        duration_seconds=round(result.duration_seconds, 3),
        metrics=metrics_response,
        signals=len(result.signals),
        orders=len(result.orders),
        rejections=len(result.rejected_signals),
        rejection_reasons=rejection_reasons(result),
        report_path=report_path,
        error=result.error,
    )


@router.get(
    "/backtest/runs",
    response_model=list[dict[str, Any]],
    summary="List stored backtest runs",
)
async def list_runs(
    database: DatabaseDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    strategy_id: str | None = None,
) -> list[dict[str, Any]]:
    """Recent persisted backtest runs."""
    from quantflow.persistence.repositories import BacktestRepository

    async with database.read_session() as session:
        runs = await BacktestRepository(session).list_recent(limit=limit, strategy_id=strategy_id)
    return [
        {
            "run_id": run.id,
            "strategy_id": run.strategy_id,
            "symbols": run.symbols,
            "timeframe": run.timeframe.value,
            "status": run.status.value,
            "start": run.start.isoformat(),
            "end": run.end.isoformat(),
            "final_equity": str(run.final_equity) if run.final_equity else None,
            "sharpe_ratio": str(run.sharpe_ratio) if run.sharpe_ratio else None,
            "max_drawdown_pct": str(run.max_drawdown_pct) if run.max_drawdown_pct else None,
            "trade_count": run.trade_count,
            "created_at": run.created_at.isoformat(),
        }
        for run in runs
    ]


def default_backtest_window() -> tuple[timedelta, timedelta]:
    """Sensible default train/test durations for the dashboard's walk-forward form."""
    return timedelta(days=180), timedelta(days=60)
