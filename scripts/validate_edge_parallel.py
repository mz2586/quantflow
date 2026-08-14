#!/usr/bin/env python
"""Parallel driver for the edge validation. Identical methodology, N processes.

The method is not reimplemented here: the split, the cost stack, the per-strategy run and
the deflated-Sharpe haircut are all imported from ``validate_edge`` and called unchanged.
This file only decides *where* each candidate runs. Backtests are pure functions of their
input bars with no shared state and no RNG, so distributing them cannot change a result —
only the order in which they finish, and the output is reassembled in registry order.

Two things differ from the serial script, neither of them methodological:

* Candidates run across ``cpu_count - 1`` worker processes instead of one.
* structlog is filtered at WARNING, so the per-fill DEBUG lines are not rendered. Measured
  at 1.01x on a 12k-bar slice — this is for a readable log, not for speed.

The real cost is structural and is untouched here: every bar hands the strategy a
MAX_HISTORY_BARS (5,000) trailing window and the indicators recompute over all of it in
Decimal. Parallelism divides that wall-clock; it does not remove it.
"""

from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog


def _quiet_logging() -> None:
    """Filter at the bound-logger level so .debug() is a no-op before any processor runs."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(io.StringIO()),
        cache_logger_on_first_use=True,
    )
    logging.getLogger().setLevel(logging.WARNING)


_quiet_logging()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio  # noqa: E402

import regime  # noqa: E402  — a priori regime thresholds, declared before any result
import validate_edge as ve  # noqa: E402  — the methodology, imported not copied

from quantflow.core.config import get_settings  # noqa: E402
from quantflow.domain.instruments import Symbol  # noqa: E402
from quantflow.portfolio.funding import FundingSchedule  # noqa: E402
from quantflow.strategy.registry import load_builtin_strategies  # noqa: E402

REPO = Path("/Users/muhammadzohaib/quantflow")
FUNDING_CACHE = REPO / "scratchpad" / "funding-cache.json"

#: Smoke-test only. Caps bars per symbol so the whole pipeline can be exercised in minutes.
#: When set, results are written to a throwaway path and never to the real report, so a
#: stray export cannot quietly produce a truncated "validation".
SMOKE_BARS = int(os.environ.get("QF_VALIDATE_SMOKE_BARS", "0") or 0)
SMOKE_OUT = str(REPO / "scratchpad" / "edge-validation-SMOKE.json")

#: Trailing history handed to a strategy each bar. The engine's own default is 5000; 1000 is
#: used only when scripts/verify_history_window.py has shown the two produce byte-identical
#: trade ledgers, and the choice is recorded in the report so it is never a silent change.
MAX_HISTORY_BARS = int(os.environ.get("QF_VALIDATE_MAX_HISTORY_BARS", "5000"))

#: Per-worker state, populated once by the pool initializer.
_STATE: dict[str, Any] = {}


def _universe() -> list[str]:
    if len(sys.argv) > 1:
        return sys.argv[1].split(",")
    return [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "XRP/USDT",
        "DOGE/USDT",
        "ADA/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "DOT/USDT",
    ]


def _split(data: dict) -> tuple[dict, dict]:
    """The same chronological split as the serial script, from the same OOS_FRACTION."""
    if SMOKE_BARS:
        data = {s: rows[:SMOKE_BARS] for s, rows in data.items()}
    index = {s: int(len(rows) * (1 - ve.OOS_FRACTION)) for s, rows in data.items()}
    in_sample = {s: rows[: index[s]] for s, rows in data.items()}
    out_sample = {s: rows[index[s] :] for s, rows in data.items()}
    return in_sample, out_sample


def _load_funding_cache() -> dict[Symbol, FundingSchedule]:
    raw = json.loads(FUNDING_CACHE.read_text())
    out: dict[Symbol, FundingSchedule] = {}
    for slashed, stamps in raw.items():
        schedule = FundingSchedule()
        for iso, rate in stamps:
            schedule.add(datetime.fromisoformat(iso), Decimal(rate))
        out[Symbol.parse(slashed)] = schedule
    return out


def _init_worker(universe: list[str]) -> None:
    """Load the bars and funding once per worker, not once per candidate."""
    _quiet_logging()
    settings = get_settings()
    data, instruments = asyncio.run(ve.load_data(universe, settings))
    in_sample, out_sample = _split(data)
    _STATE.update(
        settings=settings,
        instruments=instruments,
        funding=_load_funding_cache(),
        in_sample=in_sample,
        out_sample=out_sample,
    )


async def _run_with_trades(name: str, slice_: dict) -> tuple[Any, list[dict[str, Any]]]:
    """``ve.run_one`` inlined so the closed trades survive alongside the summary.

    ve.run_one builds the engine, runs it and hands back only the reduced metrics. Regime
    segmentation needs the individual round-trips, so the same construction is repeated
    here with the same BacktestConfig - identical inputs, identical costs, and the summary
    still comes from ve.summarise so the reported numbers cannot drift from the serial run.
    """
    from quantflow.backtest.engine import BacktestConfig, BacktestEngine
    from quantflow.core.config import MarketType
    from quantflow.strategy.registry import load_builtin_strategies as _load

    registry = _load()
    engine = BacktestEngine(
        registry.create(name),
        BacktestConfig(
            symbols=tuple(slice_),
            timeframe=ve.TF,
            starting_equity=ve.EQUITY,
            risk=_STATE["settings"].risk,
            market_type=MarketType.FUTURE,
            leverage=Decimal("1"),
            funding=_STATE["funding"],
            max_history_bars=MAX_HISTORY_BARS,
        ),
        instruments=_STATE["instruments"],
    )
    result = await engine.run(slice_)
    summary = ve.summarise(name, result, engine._portfolio.funding_paid)
    rows = [
        {
            "symbol": trade.symbol,
            "entry_time": trade.entry_time,
            "net_pnl": float(trade.net_pnl),
        }
        for trade in result.closed_trades
    ]
    return summary, rows


def _run_candidate(
    name: str, trials: int
) -> tuple[str, dict[str, Any], dict[str, list[dict[str, Any]]], float]:
    """One candidate over both periods, returning summaries and raw trades."""
    started = time.perf_counter()
    row: dict[str, Any] = {}
    trades: dict[str, list[dict[str, Any]]] = {}
    for period, slice_ in (
        ("in_sample", _STATE["in_sample"]),
        ("out_of_sample", _STATE["out_sample"]),
    ):
        try:
            result, rows = asyncio.run(_run_with_trades(name, slice_))
            row[period] = result.to_dict()
            row[period]["deflated_sharpe"] = round(
                ve.deflated_sharpe(result.sharpe, trials, max(result.trades, 2)), 3
            )
            trades[period] = rows
        except Exception as exc:
            row[period] = {"error": str(exc)[:160]}
            trades[period] = []
    return name, row, trades, time.perf_counter() - started


def _fmt_eta(seconds: float) -> str:
    seconds = max(seconds, 0)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


async def _prepare() -> dict[str, Any]:
    """Parent-side: metadata and the one funding fetch, shared with workers via cache."""
    settings = get_settings()
    universe = _universe()
    print("loading candles (parent, for metadata)...", flush=True)
    data, _ = await ve.load_data(universe, settings)
    if not data:
        raise SystemExit("no data")

    lengths = {s: len(rows) for s, rows in data.items()}
    first = min(rows[0].open_time for rows in data.values())
    last = max(rows[-1].open_time for rows in data.values())
    in_sample, _ = _split(data)
    is_end = max(rows[-1].open_time for rows in in_sample.values())

    print(
        f"{len(data)} symbols, {sum(lengths.values())} bars, {first.date()} -> {last.date()}",
        flush=True,
    )
    print(f"IN-SAMPLE  : {first.date()} -> {is_end.date()}")
    print(f"OUT-SAMPLE : {is_end.date()} -> {last.date()}  (never tuned)\n", flush=True)

    if FUNDING_CACHE.exists():
        print(f"reusing funding cache {FUNDING_CACHE.name}", flush=True)
    else:
        print("loading real funding history (once, shared with workers)...", flush=True)
        from quantflow.exchange.bybit.rest import BybitGateway

        gateway = BybitGateway(settings.exchange)
        try:
            funding = await ve.load_funding(gateway, list(data), first, last)
        finally:
            await gateway.aclose()
        FUNDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FUNDING_CACHE.write_text(
            json.dumps(
                {
                    # FundingSchedule exposes no iterator; _rates is the only way to
                    # serialise it, and a cache beats re-fetching from the venue per worker.
                    sym.slashed: [
                        [stamp.isoformat(), str(rate)] for stamp, rate in schedule._rates.items()
                    ]
                    for sym, schedule in funding.items()
                }
            )
        )

    meta = {
        "universe": [s.slashed for s in data],
        "bars": sum(lengths.values()),
        "window": {"start": first.isoformat(), "end": last.isoformat()},
        "split": {"in_sample_end": is_end.isoformat(), "oos_fraction": ve.OOS_FRACTION},
    }
    return meta, data


def main() -> int:
    meta, data = asyncio.run(_prepare())

    # Regime labels are built once, in the parent, from the same bars the workers replay.
    # Volatility thresholds come from in-sample bars only (see scripts/regime.py).
    print("classifying regimes (ADX 25 trend/chop, in-sample median vol split)...", flush=True)
    regime_start = time.perf_counter()
    in_sample_counts = {s: int(len(rows) * (1 - ve.OOS_FRACTION)) for s, rows in data.items()}
    regimes = regime.build_regimes(data, in_sample_counts)
    for symbol, table in regimes.items():
        print(f"  {symbol.slashed:<10} vol median {table.vol_threshold:.6f}", flush=True)
    print(f"regimes ready in {_fmt_eta(time.perf_counter() - regime_start)}\n", flush=True)
    del data  # 291MB the parent no longer needs while workers run.

    registry = load_builtin_strategies()
    candidates = [n for n in registry.names() if n not in {"buy_and_hold"}]
    trials = len(candidates)

    workers = max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(candidates))
    print(
        f"validating {trials} candidates (orchestrator included) "
        f"across {workers} worker processes\n",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    durations: list[float] = []
    wall_start = time.perf_counter()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(_universe(),),
    ) as pool:
        futures = {pool.submit(_run_candidate, name, trials): name for name in candidates}
        for future in as_completed(futures):
            # A candidate that dies outright — a worker killed, a pool broken — must cost
            # only that candidate. A partial report beats no report at all, and the
            # failure is recorded rather than quietly leaving a gap in the table.
            try:
                name, row, trades, took = future.result()
            except Exception as exc:
                failed = futures[future]
                failures[failed] = f"{type(exc).__name__}: {exc}"[:300]
                print(f"!! {failed} FAILED: {failures[failed]}", flush=True)
                results[failed] = {
                    "in_sample": {"error": failures[failed]},
                    "out_of_sample": {"error": failures[failed]},
                }
                continue
            # Bucket this candidate's round-trips by the regime at entry.
            for period, rows in trades.items():
                if isinstance(row.get(period), dict) and "error" not in row[period]:
                    row[period]["regimes"] = regime.segment(rows, regimes)
            results[name] = row
            durations.append(took)
            done = len(results)

            is_row = row.get("in_sample", {})
            oos_row = row.get("out_of_sample", {})
            print(
                f"{name:<24} IS net={is_row.get('net_pnl', 0):>9.2f} "
                f"n={is_row.get('trades', 0):>4}  "
                f"| OOS net={oos_row.get('net_pnl', 0):>9.2f} "
                f"n={oos_row.get('trades', 0):>4} "
                f"sharpe={oos_row.get('sharpe', 0):>6.2f} "
                f"dsr={oos_row.get('deflated_sharpe', 0):>6.2f}",
                flush=True,
            )

            # ETA from measured per-candidate cost, spread over the pool.
            mean_cost = sum(durations) / len(durations)
            remaining = len(candidates) - done
            eta = (mean_cost * remaining) / workers
            elapsed = time.perf_counter() - wall_start
            print(
                f"  [{done}/{len(candidates)}] this candidate {_fmt_eta(took)} | "
                f"mean {_fmt_eta(mean_cost)} | elapsed {_fmt_eta(elapsed)} | "
                f"ETA {_fmt_eta(eta)} (~{(datetime.now().timestamp() + eta):.0f})",
                flush=True,
            )
            if done == 1:
                total_est = (mean_cost * len(candidates)) / workers
                print(
                    f"  FIRST CANDIDATE DONE in {_fmt_eta(took)} -> "
                    f"projected full run {_fmt_eta(total_est)} "
                    f"({len(candidates)} candidates / {workers} workers)",
                    flush=True,
                )

    # Reassemble in registry order so the JSON is byte-identical regardless of finish order.
    report = {name: results[name] for name in candidates if name in results}

    destination = SMOKE_OUT if SMOKE_BARS else ve.OUT
    with Path(destination).open("w") as fh:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "timeframe": ve.TF.value,
                "universe": meta["universe"],
                "bars": meta["bars"],
                "window": meta["window"],
                "split": meta["split"],
                "costs": {
                    "market_type": "future",
                    "leverage": "1",
                    "funding": "real published Bybit rates",
                    "fees": "taker",
                    "slippage": "volume-share + gap on protective exits",
                },
                "candidates_tested": trials,
                "max_history_bars": MAX_HISTORY_BARS,
                "failures": failures,
                "regime_definition": {
                    "trend_vs_chop": (
                        f"Wilder ADX({regime.REGIME_PERIOD}) at entry bar; "
                        f">= {regime.ADX_TREND_THRESHOLD} trend, below chop "
                        "(Wilder's published threshold, not fitted)"
                    ),
                    "vol_split": (
                        f"normalized ATR({regime.REGIME_PERIOD}) at entry bar vs the "
                        "per-symbol median of the IN-SAMPLE bars only, applied unchanged "
                        "to the holdout (a median is parameter-free)"
                    ),
                    "labelled_at": "trade entry, not exit",
                },
                "results": report,
            },
            fh,
            indent=1,
        )
    print(f"\nwritten to {destination}")
    if SMOKE_BARS:
        print(
            f"*** SMOKE MODE ({SMOKE_BARS} bars/symbol) — NOT a validation, "
            "real report untouched ***"
        )
    print(f"total wall clock {_fmt_eta(time.perf_counter() - wall_start)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
