"""Columnar candle store and dataframe conversion.

Postgres is the system of record; this module is the analytical mirror. Backtests and the
AI engine scan millions of bars, and a Hive-partitioned Parquet dataset read through Polars
is dramatically faster for that than a row-oriented query — while remaining a plain
directory of files that can be copied, versioned or handed to another tool.

Prices cross into ``float64`` here and only here. That is a considered trade: vectorised
statistics need it, and these values feed indicators and metrics, never order quantities or
cash balances. Anything that becomes an order goes back through ``Decimal``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from quantflow.core.errors import InsufficientDataError, MarketDataError
from quantflow.core.logging import get_logger
from quantflow.core.precision import to_decimal
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, CandleSeries

logger = get_logger(__name__)

PolarsDataType = DataType | DataTypeClass

SCHEMA_FIELDS: Mapping[str, PolarsDataType] = {
    "open_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "quote_volume": pl.Float64,
    "trades": pl.Int64,
}

CANDLE_SCHEMA = pl.Schema(SCHEMA_FIELDS)


def candles_to_frame(candles: Iterable[Candle]) -> pl.DataFrame:
    """Convert candles into a typed Polars frame, sorted by open time."""
    rows = list(candles)
    if not rows:
        return pl.DataFrame(schema=CANDLE_SCHEMA)

    frame = pl.DataFrame(
        {
            "open_time": [candle.open_time for candle in rows],
            "open": [float(candle.open) for candle in rows],
            "high": [float(candle.high) for candle in rows],
            "low": [float(candle.low) for candle in rows],
            "close": [float(candle.close) for candle in rows],
            "volume": [float(candle.volume) for candle in rows],
            "quote_volume": [float(candle.quote_volume) for candle in rows],
            "trades": [candle.trades for candle in rows],
        },
        schema=CANDLE_SCHEMA,
    )
    return frame.sort("open_time")


def frame_to_candles(frame: pl.DataFrame, symbol: Symbol, timeframe: Timeframe) -> list[Candle]:
    """Convert a frame back into domain candles.

    Floats are routed through ``repr`` by :func:`to_decimal`, so ``50000.1`` comes back as
    ``Decimal("50000.1")`` rather than its full binary expansion.
    """
    missing = set(CANDLE_SCHEMA) - set(frame.columns)
    if missing:
        raise MarketDataError(f"frame is missing columns: {sorted(missing)}")

    candles: list[Candle] = []
    for row in frame.sort("open_time").iter_rows(named=True):
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                open_time=row["open_time"],
                open=to_decimal(row["open"]),
                high=to_decimal(row["high"]),
                low=to_decimal(row["low"]),
                close=to_decimal(row["close"]),
                volume=to_decimal(row["volume"]),
                quote_volume=to_decimal(row["quote_volume"]),
                trades=int(row["trades"]),
            )
        )
    return candles


class ParquetCandleStore:
    """Hive-partitioned Parquet dataset: ``<root>/symbol=BTC-USDT/timeframe=1h/data.parquet``.

    The symbol's slash is replaced with a hyphen because a slash in a partition value would
    create a spurious directory level.
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Dataset root directory."""
        return self._root

    def partition_path(self, symbol: Symbol, timeframe: Timeframe) -> Path:
        """Path to one series' partition directory."""
        safe_symbol = symbol.slashed.replace("/", "-")
        return self._root / f"symbol={safe_symbol}" / f"timeframe={timeframe.value}"

    def _file_path(self, symbol: Symbol, timeframe: Timeframe) -> Path:
        return self.partition_path(symbol, timeframe) / "candles.parquet"

    def exists(self, symbol: Symbol, timeframe: Timeframe) -> bool:
        """Whether a series has been written."""
        return self._file_path(symbol, timeframe).is_file()

    def write(
        self, symbol: Symbol, timeframe: Timeframe, candles: Sequence[Candle], *, merge: bool = True
    ) -> int:
        """Write a series, optionally merging with what is already stored.

        Merging de-duplicates on ``open_time`` keeping the newest value, so re-writing an
        overlapping range corrects rather than duplicates.

        Returns:
            The total number of rows in the resulting file.

        """
        frame = candles_to_frame(candles)
        if frame.is_empty() and not merge:
            return 0

        path = self._file_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        if merge and path.is_file():
            existing = pl.read_parquet(path)
            frame = (
                pl.concat([existing, frame], how="vertical_relaxed")
                .unique(subset=["open_time"], keep="last")
                .sort("open_time")
            )

        frame.write_parquet(path, compression="zstd", statistics=True)
        logger.debug(
            "store.written",
            symbol=str(symbol),
            timeframe=timeframe.value,
            rows=frame.height,
            path=str(path),
        )
        return frame.height

    def read(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Read a series as a frame, filtered to ``[start, end)``.

        Uses Polars' lazy scan so the predicate is pushed into the Parquet reader and only
        the matching row groups are decoded.
        """
        path = self._file_path(symbol, timeframe)
        if not path.is_file():
            raise InsufficientDataError(
                f"no stored data for {symbol} {timeframe.value}",
                symbol=str(symbol),
                timeframe=timeframe.value,
            )
        lazy = pl.scan_parquet(path)
        if start is not None:
            lazy = lazy.filter(pl.col("open_time") >= start)
        if end is not None:
            lazy = lazy.filter(pl.col("open_time") < end)
        return lazy.sort("open_time").collect()

    def read_candles(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        """Read a series back as domain candles."""
        return frame_to_candles(
            self.read(symbol, timeframe, start=start, end=end), symbol, timeframe
        )

    def read_series(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> CandleSeries:
        """Read a series as a validated :class:`CandleSeries`."""
        return CandleSeries(self.read_candles(symbol, timeframe, start=start, end=end))

    def delete(self, symbol: Symbol, timeframe: Timeframe) -> bool:
        """Delete a stored series. Returns whether anything was removed."""
        path = self._file_path(symbol, timeframe)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def list_series(self) -> list[tuple[Symbol, Timeframe]]:
        """Every stored ``(symbol, timeframe)`` pair."""
        if not self._root.is_dir():
            return []
        found: list[tuple[Symbol, Timeframe]] = []
        for symbol_dir in sorted(self._root.glob("symbol=*")):
            raw_symbol = symbol_dir.name.removeprefix("symbol=").replace("-", "/")
            for timeframe_dir in sorted(symbol_dir.glob("timeframe=*")):
                raw_timeframe = timeframe_dir.name.removeprefix("timeframe=")
                if not (timeframe_dir / "candles.parquet").is_file():
                    continue
                try:
                    parsed = Symbol.parse(raw_symbol)
                    assert isinstance(parsed, Symbol)
                    found.append((parsed, Timeframe.parse(raw_timeframe)))
                except Exception as exc:
                    logger.debug(
                        "store.skipped_partition", path=str(timeframe_dir), reason=str(exc)
                    )
                    continue
        return found

    def stats(self, symbol: Symbol, timeframe: Timeframe) -> dict[str, object]:
        """Row count and time bounds for a stored series, without loading the values."""
        frame = self.read(symbol, timeframe).select(
            pl.len().alias("rows"),
            pl.col("open_time").min().alias("start"),
            pl.col("open_time").max().alias("end"),
        )
        row = frame.to_dicts()[0]
        return {
            "symbol": symbol.slashed,
            "timeframe": timeframe.value,
            "rows": int(row["rows"]),
            "start": row["start"],
            "end": row["end"],
        }


def returns(frame: pl.DataFrame, *, column: str = "close", log: bool = False) -> pl.Series:
    """Per-bar returns.

    Log returns are additive across time, which is what the risk and regime models want;
    simple returns are what a PnL statement shows. Both are offered explicitly rather than
    picking one and hoping the caller agrees.
    """
    prices = frame[column]
    if log:
        return (prices / prices.shift(1)).log().fill_null(0.0)
    return prices.pct_change().fill_null(0.0)


def to_decimal_series(frame: pl.DataFrame, column: str) -> list[Decimal]:
    """Extract a column as exact ``Decimal`` values, for anything order-facing."""
    return [to_decimal(value) for value in frame[column].to_list()]
