#!/usr/bin/env python
"""Is MAX_HISTORY_BARS 5000 -> 1000 free? Decide by diffing, not by reasoning about it.

The window cap is not a cosmetic setting. SMA, ROC, Donchian and the like need only their
own period of bars and are unaffected. But EMA, MACD and every Wilder-smoothed indicator
(ATR, ADX, RSI) are *recursive*: their value at a bar depends on where the series started,
so handing them 1000 bars instead of 5000 can shift a value slightly, and a shifted value
can flip a crossover and change a trade. The residual decays like (1 - alpha)^overlap,
which is nothing for a 14-period ATR and distinctly not nothing for a 200-period EMA.

So this compares full trade ledgers - every entry, exit, price, quantity and PnL - not just
totals. Two runs can post the same net PnL from different trades; that would not be "free".

Strategies tested are the worst cases by design: the longest lookbacks in the library
(momentum_roc at 720, and the 200-period trend filters), plus the orchestrator, which is
the actual product.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    logger_factory=structlog.PrintLoggerFactory(io.StringIO()),
    cache_logger_on_first_use=True,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantflow.backtest.engine import BacktestConfig, BacktestEngine  # noqa: E402
from quantflow.core.config import MarketType, get_settings  # noqa: E402
from quantflow.domain.enums import Timeframe  # noqa: E402
from quantflow.domain.instruments import Instrument, Symbol  # noqa: E402
from quantflow.persistence.database import Database  # noqa: E402
from quantflow.persistence.repositories import CandleRepository  # noqa: E402
from quantflow.strategy.registry import load_builtin_strategies  # noqa: E402

TF = Timeframe.parse("15m")
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

#: Long enough that the 1000-bar cap binds for thousands of bars, and that a 720-bar
#: lookback is exercised repeatedly rather than once.
BARS = 5000

CANDIDATES = [
    "momentum_roc",  # 720-bar lookback, the longest in the library
    "triple_ma",  # 200-period slow MA
    "rsi_reversion",  # 200-period trend filter + recursive RSI
    "bollinger_reversion",  # 200-period trend filter
    "ema_cross",  # recursive EMA pair
    "macd_trend",  # recursive MACD
    "adx_trend",  # Wilder-smoothed ADX
    "orchestrator",  # the product
]

_DATA: dict[str, Any] = {}


async def _load() -> tuple[dict, dict]:
    settings = get_settings()
    db = Database.from_settings(settings)
    data, insts = {}, {}
    async with db.read_session() as session:
        repo = CandleRepository(session)
        for raw in SYMBOLS:
            sym = Symbol.parse(raw)
            rows = await repo.fetch(sym, TF)
            data[sym] = rows[:BARS]
            insts[sym] = Instrument(
                symbol=sym,
                price_tick=Decimal("0.0001"),
                quantity_step=Decimal("0.001"),
                min_quantity=Decimal("0.001"),
                min_notional=Decimal("5"),
            )
    return data, insts


def _init() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(io.StringIO()),
        cache_logger_on_first_use=True,
    )
    data, insts = asyncio.run(_load())
    _DATA.update(data=data, insts=insts, settings=get_settings())


async def _run(name: str, window: int) -> list[tuple]:
    registry = load_builtin_strategies()
    engine = BacktestEngine(
        registry.create(name),
        BacktestConfig(
            symbols=tuple(_DATA["data"]),
            timeframe=TF,
            starting_equity=Decimal("10000"),
            risk=_DATA["settings"].risk,
            market_type=MarketType.FUTURE,
            leverage=Decimal("1"),
            funding={},
            max_history_bars=window,
        ),
        instruments=_DATA["insts"],
    )
    result = await engine.run(_DATA["data"])
    # The full ledger: identical totals from different trades is not "identical".
    return [
        (
            str(t.symbol),
            t.entry_time.isoformat(),
            t.exit_time.isoformat(),
            str(t.entry_price),
            str(t.exit_price),
            str(t.quantity),
            str(t.net_pnl),
        )
        for t in result.closed_trades
    ]


def _compare(name: str) -> dict[str, Any]:
    wide = asyncio.run(_run(name, 5000))
    narrow = asyncio.run(_run(name, 1000))
    same = wide == narrow
    first_diff = None
    if not same:
        for i, (a, b) in enumerate(zip(wide, narrow, strict=False)):
            if a != b:
                first_diff = {"index": i, "at_5000": a, "at_1000": b}
                break
        if first_diff is None:
            first_diff = {"index": min(len(wide), len(narrow)), "note": "differing lengths"}
    return {
        "name": name,
        "identical": same,
        "trades_5000": len(wide),
        "trades_1000": len(narrow),
        "net_5000": round(sum(float(t[6]) for t in wide), 6),
        "net_1000": round(sum(float(t[6]) for t in narrow), 6),
        "first_diff": first_diff,
    }


def main() -> int:
    print(f"comparing MAX_HISTORY_BARS 5000 vs 1000 on {len(SYMBOLS)} symbols x {BARS} bars")
    print(f"({BARS - 1000} bars run with the 1000-cap binding)\n", flush=True)

    with ProcessPoolExecutor(max_workers=min(7, len(CANDIDATES)), initializer=_init) as pool:
        results = list(pool.map(_compare, CANDIDATES))

    print(
        f"{'strategy':<22}{'identical':<11}{'trades 5000/1000':<19}{'net 5000':>12}{'net 1000':>12}"
    )
    print("-" * 76)
    all_same = True
    for row in results:
        all_same &= row["identical"]
        mark = "YES" if row["identical"] else "NO"
        print(
            f"{row['name']:<22}{mark:<11}"
            f"{str(row['trades_5000']) + '/' + str(row['trades_1000']):<19}"
            f"{row['net_5000']:>12.2f}{row['net_1000']:>12.2f}"
        )
    print("-" * 76)

    if all_same:
        print("\nVERDICT: IDENTICAL ledgers on every candidate — the 1000-bar cap is free here.")
    else:
        print("\nVERDICT: NOT identical. Keep 5000. First divergences:")
        for row in results:
            if not row["identical"]:
                print(f"  {row['name']}: {row['first_diff']}")
    return 0 if all_same else 1


if __name__ == "__main__":
    sys.exit(main())
