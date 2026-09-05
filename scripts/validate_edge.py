#!/usr/bin/env python
"""Full edge validation: in-sample vs a never-tuned out-of-sample holdout.

Honest costs throughout — futures margin accounting (Phase 6), real published funding
(Phase 7), gap-and-slippage on protective exits (Phase 8), taker fees and volume-share
slippage. No parameter is fitted anywhere in this script: every strategy runs at its
declared defaults, so there is nothing that could have been tuned to either period.

The out-of-sample window is the last 30% of history and is used ONLY for reporting. It is
never consulted to choose a strategy, a parameter or a threshold.

Testing ~22 candidates means some will look good by chance, so a raw Sharpe is not evidence.
The deflated Sharpe subtracts the Sharpe you would expect the *best* of N random strategies
to show, which is the correct null for a search this wide.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from quantflow.backtest.engine import BacktestConfig, BacktestEngine
from quantflow.core.config import MarketType, get_settings
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.exchange.bybit.rest import BybitGateway
from quantflow.persistence.database import Database
from quantflow.persistence.repositories import CandleRepository, InstrumentRepository
from quantflow.portfolio.funding import FundingSchedule
from quantflow.strategy.registry import load_builtin_strategies

TF = Timeframe.parse("15m")
EQUITY = Decimal("10000")
OOS_FRACTION = 0.30
#: Repo root, derived from this file's location so the script works from any checkout.
REPO = Path(__file__).resolve().parent.parent
OUT = str(REPO / "reports" / "edge-validation.json")

#: Bars per year at 15m, for annualising.
BARS_PER_YEAR = 365 * 24 * 4


@dataclass
class Result:
    """One strategy over one period."""

    label: str
    trades: int = 0
    net_pnl: Decimal = Decimal("0")
    gross_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    funding: Decimal = Decimal("0")
    win_rate: float = 0.0
    max_drawdown: Decimal = Decimal("0")
    sharpe: float = 0.0
    fee_drag_pct: float = 0.0
    equity_points: list[Decimal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "net_pnl": float(self.net_pnl),
            "gross_pnl": float(self.gross_pnl),
            "fees": float(self.fees),
            "funding": float(self.funding),
            "win_rate": round(self.win_rate, 2),
            "max_drawdown": float(self.max_drawdown),
            "sharpe": round(self.sharpe, 3),
            "fee_drag_pct": round(self.fee_drag_pct, 2),
        }


def summarise(label: str, result: Any, funding_paid: Decimal) -> Result:
    """Reduce a backtest result to the reported metrics."""
    trades = list(result.closed_trades)
    out = Result(label=label, trades=len(trades), funding=funding_paid)
    if not trades:
        return out

    out.net_pnl = sum((t.net_pnl for t in trades), Decimal("0"))
    out.gross_pnl = sum((t.gross_pnl for t in trades), Decimal("0"))
    out.fees = sum((t.fees for t in trades), Decimal("0"))
    wins = [t for t in trades if t.net_pnl > 0]
    out.win_rate = 100 * len(wins) / len(trades)
    if out.gross_pnl > 0:
        out.fee_drag_pct = float(out.fees / out.gross_pnl * 100)

    # Drawdown and Sharpe from the trade-by-trade equity path.
    running = Decimal("0")
    peak = Decimal("0")
    returns: list[float] = []
    for trade in trades:
        running += trade.net_pnl
        peak = max(peak, running)
        out.max_drawdown = max(out.max_drawdown, peak - running)
        returns.append(float(trade.net_pnl / EQUITY))

    if len(returns) > 1:
        mean = statistics.fmean(returns)
        stdev = statistics.stdev(returns)
        if stdev > 0:
            # Annualised on trade count over the period's length.
            out.sharpe = (mean / stdev) * math.sqrt(len(returns))
    return out


def deflated_sharpe(sharpe: float, trials: int, samples: int) -> float:
    """Sharpe haircut for having searched ``trials`` candidates.

    Subtracts the Sharpe the *best* of N independent random strategies would be expected to
    produce under the null of no skill (Bailey & Lopez de Prado). A raw Sharpe from a search
    over 22 candidates is not evidence; this is what is left after paying for the search.
    """
    if trials <= 1 or samples <= 1:
        return sharpe
    euler = 0.5772156649
    # Expected maximum of N standard normals.
    expected_max = (1 - euler) * _inv_norm(1 - 1 / trials) + euler * _inv_norm(
        1 - 1 / (trials * math.e)
    )
    # Standard error of a Sharpe estimate on this many observations. The haircut is
    # expected_max x SE - multiplying by sqrt(samples) again would cancel the 1/sqrt(n)
    # already inside SE and produce an absurd penalty.
    standard_error = math.sqrt((1 + 0.5 * sharpe**2) / max(samples - 1, 1))
    return sharpe - expected_max * standard_error


def _inv_norm(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation)."""
    if not 0 < p < 1:
        return 0.0
    a = [-39.696830, 220.946098, -275.928510, 138.357751, -30.664798, 2.506628]
    b = [-54.476098, 161.585836, -155.698979, 66.801311, -13.280681]
    c = [-0.007784894002, -0.32239645, -2.400758, -2.549732, 4.374664, 2.938163]
    d = [0.007784695709, 0.32246712, 2.445134, 3.754408]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


async def load_data(
    symbols: list[str], settings: Any
) -> tuple[dict[Symbol, list], dict[Symbol, Instrument]]:
    db = Database.from_settings(settings)
    data: dict[Symbol, list] = {}
    instruments: dict[Symbol, Instrument] = {}
    async with db.read_session() as session:
        candles = CandleRepository(session)
        insts = InstrumentRepository(session)
        for raw in symbols:
            symbol = Symbol.parse(raw)
            rows = await candles.fetch(symbol, TF)
            if rows:
                data[symbol] = rows
            found = await insts.get(symbol)
            instruments[symbol] = found or Instrument(
                symbol=symbol,
                price_tick=Decimal("0.0001"),
                quantity_step=Decimal("0.001"),
                min_quantity=Decimal("0.001"),
                min_notional=Decimal("5"),
            )
    return data, instruments


async def load_funding(
    gateway: BybitGateway, symbols: list[Symbol], start: datetime, end: datetime
) -> dict[Symbol, FundingSchedule]:
    """Real published funding for the whole window."""
    out: dict[Symbol, FundingSchedule] = {}
    for symbol in symbols:
        schedule = FundingSchedule()
        cursor = start
        while cursor < end:
            try:
                rows = await gateway.fetch_funding_history(symbol, since=cursor, limit=200)
            except Exception:
                break
            if not rows:
                break
            for stamp, rate in rows:
                if start <= stamp <= end:
                    schedule.add(stamp, rate)
            if rows[-1][0] <= cursor:
                break
            cursor = rows[-1][0]
        out[symbol] = schedule
        print(f"  funding {symbol.slashed}: {len(schedule)} stamps", flush=True)
    return out


async def run_one(
    strategy_name: str,
    data: dict[Symbol, list],
    instruments: dict[Symbol, Instrument],
    funding: dict[Symbol, FundingSchedule],
    settings: Any,
    label: str,
) -> Result:
    """Backtest one strategy (or the orchestrator) over one slice."""
    registry = load_builtin_strategies()
    strategy = registry.create(strategy_name)
    engine = BacktestEngine(
        strategy,
        BacktestConfig(
            symbols=tuple(data),
            timeframe=TF,
            starting_equity=EQUITY,
            risk=settings.risk,
            market_type=MarketType.FUTURE,
            leverage=Decimal("1"),
            funding=funding,
        ),
        instruments=instruments,
    )
    result = await engine.run(data)
    return summarise(label, result, engine._portfolio.funding_paid)


async def main() -> int:
    universe = (
        sys.argv[1].split(",")
        if len(sys.argv) > 1
        else [
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
    )
    settings = get_settings()
    print("loading candles...", flush=True)
    data, instruments = await load_data(universe, settings)
    if not data:
        print("no data")
        return 1

    lengths = {s: len(rows) for s, rows in data.items()}
    first = min(rows[0].open_time for rows in data.values())
    last = max(rows[-1].open_time for rows in data.values())
    print(f"{len(data)} symbols, {sum(lengths.values())} bars, {first.date()} -> {last.date()}")

    # Chronological split. The holdout is never used to choose anything.
    split_index = {s: int(len(rows) * (1 - OOS_FRACTION)) for s, rows in data.items()}
    in_sample = {s: rows[: split_index[s]] for s, rows in data.items()}
    out_sample = {s: rows[split_index[s] :] for s, rows in data.items()}
    is_end = max(rows[-1].open_time for rows in in_sample.values())
    print(f"IN-SAMPLE  : {first.date()} -> {is_end.date()}")
    print(f"OUT-SAMPLE : {is_end.date()} -> {last.date()}  (never tuned)\n")

    print("loading real funding history...", flush=True)
    gateway = BybitGateway(settings.exchange)
    try:
        funding = await load_funding(gateway, list(data), first, last)
    finally:
        await gateway.aclose()

    registry = load_builtin_strategies()
    candidates = [n for n in registry.names() if n not in {"buy_and_hold"}]
    print(f"\nvalidating {len(candidates)} candidates (orchestrator included)\n", flush=True)

    report: dict[str, dict[str, Any]] = {}
    for name in candidates:
        row: dict[str, Any] = {}
        for period, slice_ in (("in_sample", in_sample), ("out_of_sample", out_sample)):
            try:
                result = await run_one(name, slice_, instruments, funding, settings, name)
                row[period] = result.to_dict()
                row[period]["deflated_sharpe"] = round(
                    deflated_sharpe(result.sharpe, len(candidates), max(result.trades, 2)), 3
                )
            except Exception as exc:
                row[period] = {"error": str(exc)[:160]}
        report[name] = row
        is_row = row.get("in_sample", {})
        oos_row = row.get("out_of_sample", {})
        print(
            f"{name:<24} IS net={is_row.get('net_pnl', 0):>9.2f} n={is_row.get('trades', 0):>4}  "
            f"| OOS net={oos_row.get('net_pnl', 0):>9.2f} n={oos_row.get('trades', 0):>4} "
            f"sharpe={oos_row.get('sharpe', 0):>6.2f} dsr={oos_row.get('deflated_sharpe', 0):>6.2f}",
            flush=True,
        )

    with Path(OUT).open("w") as fh:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "timeframe": TF.value,
                "universe": [s.slashed for s in data],
                "bars": sum(lengths.values()),
                "window": {"start": first.isoformat(), "end": last.isoformat()},
                "split": {"in_sample_end": is_end.isoformat(), "oos_fraction": OOS_FRACTION},
                "costs": {
                    "market_type": "future",
                    "leverage": "1",
                    "funding": "real published Bybit rates",
                    "fees": "taker",
                    "slippage": "volume-share + gap on protective exits",
                },
                "candidates_tested": len(candidates),
                "results": report,
            },
            fh,
            indent=1,
        )
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
