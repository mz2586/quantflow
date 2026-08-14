"""Regime classification for the edge validation.

Every threshold here is declared before any result is seen, and none is chosen because it
made a number look better. That is the whole point: a regime split you tune is a second
place to overfit, and it would quietly invalidate the holdout it is meant to explain.

**Trend vs chop** — Wilder's ADX(14) at the bar the trade was entered on. ADX >= 25 is
trend, below is chop. 25 is Wilder's own published threshold, not a value fitted here.
ADX measures trend *strength* without direction, which is what makes it a regime label
rather than a signal.

**High vs low volatility** — normalized ATR(14) (ATR as a fraction of price, so symbols are
comparable) at the entry bar, against the *median* of that measure for the same symbol.
A median is parameter-free: it splits the sample in half by construction, so there is no
threshold to tune. The median is computed on the IN-SAMPLE bars only and then applied
unchanged to the holdout — deriving it from the full history would let out-of-sample data
define the buckets its own results are reported in.

A trade is labelled by the regime prevailing when it was *entered*. A trend strategy that
opens in a trend and closes after it dies is still a trend-regime trade; classifying by
exit would credit it to the regime it failed in.
"""

from __future__ import annotations

import statistics
from bisect import bisect_right
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.strategy.indicators import directional_movement, normalized_atr

#: Wilder's published ADX threshold for a trending market. Not fitted.
ADX_TREND_THRESHOLD = Decimal("25")

#: Indicator period, the standard 14 used throughout the strategy library.
REGIME_PERIOD = 14

TREND, CHOP = "trend", "chop"
HIGH_VOL, LOW_VOL = "high_vol", "low_vol"

#: The four buckets every period reports, always in this order.
BUCKETS = (TREND, CHOP, HIGH_VOL, LOW_VOL)


class SymbolRegimes:
    """Per-bar regime labels for one symbol, queryable by timestamp."""

    __slots__ = ("_adx", "_natr", "_times", "_vol_threshold")

    def __init__(
        self,
        candles: list[Any],
        vol_threshold: Decimal | None = None,
        in_sample_count: int | None = None,
    ) -> None:
        _, _, adx = directional_movement(candles, REGIME_PERIOD)
        natr = normalized_atr(candles, REGIME_PERIOD)
        self._times = [c.open_time for c in candles]
        self._adx = adx
        self._natr = natr
        if vol_threshold is not None:
            self._vol_threshold = vol_threshold
        else:
            # Median of the in-sample portion only, so the holdout never defines its buckets.
            limit = in_sample_count if in_sample_count is not None else len(candles)
            sample = [v for v in natr[:limit] if v is not None]
            self._vol_threshold = (
                Decimal(str(statistics.median(sample))) if sample else Decimal("0")
            )

    @property
    def vol_threshold(self) -> Decimal:
        return self._vol_threshold

    def _index_at(self, moment: datetime) -> int | None:
        """Index of the last bar at or before ``moment``."""
        position = bisect_right(self._times, moment) - 1
        return position if position >= 0 else None

    def labels_at(self, moment: datetime) -> tuple[str | None, str | None]:
        """``(trend_label, vol_label)`` at a timestamp; ``None`` where undefined."""
        index = self._index_at(moment)
        if index is None:
            return None, None
        adx = self._adx[index] if index < len(self._adx) else None
        natr = self._natr[index] if index < len(self._natr) else None
        trend = None if adx is None else (TREND if adx >= ADX_TREND_THRESHOLD else CHOP)
        vol = None if natr is None else (HIGH_VOL if natr > self._vol_threshold else LOW_VOL)
        return trend, vol


def build_regimes(
    data: dict[Any, list[Any]], in_sample_counts: dict[Any, int]
) -> dict[Any, SymbolRegimes]:
    """Regime labels for every symbol, vol thresholds fixed on in-sample bars."""
    return {
        symbol: SymbolRegimes(candles, in_sample_count=in_sample_counts.get(symbol))
        for symbol, candles in data.items()
    }


def segment(trades: list[dict[str, Any]], regimes: dict[Any, SymbolRegimes]) -> dict[str, Any]:
    """Aggregate per-trade rows into the four regime buckets.

    ``trades`` rows are plain dicts (``symbol``, ``entry_time``, ``net_pnl``) so they can
    cross a process boundary without dragging domain objects along.
    """
    out: dict[str, dict[str, Any]] = {
        bucket: {"trades": 0, "net_pnl": 0.0, "wins": 0} for bucket in BUCKETS
    }
    unclassified = 0

    for row in trades:
        symbol = row["symbol"]
        table = regimes.get(symbol)
        if table is None:
            unclassified += 1
            continue
        trend, vol = table.labels_at(row["entry_time"])
        if trend is None and vol is None:
            unclassified += 1
            continue
        net = float(row["net_pnl"])
        for bucket in (trend, vol):
            if bucket is None:
                continue
            out[bucket]["trades"] += 1
            out[bucket]["net_pnl"] += net
            if net > 0:
                out[bucket]["wins"] += 1

    result: dict[str, Any] = {}
    for bucket, agg in out.items():
        count = agg["trades"]
        result[bucket] = {
            "trades": count,
            "net_pnl": round(agg["net_pnl"], 2),
            "win_rate": round(100 * agg["wins"] / count, 1) if count else 0.0,
        }
    result["unclassified"] = unclassified
    return result
