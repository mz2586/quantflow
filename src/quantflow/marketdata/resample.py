"""Timeframe resampling.

Aggregating 1m bars into 1h locally avoids re-downloading the same history at every
interval a strategy might want. It also guarantees the higher timeframe is *consistent*
with the lower one, which matters for multi-timeframe strategies: a 4h bar fetched from the
venue and a 4h bar built from 1m data can disagree at the edges, and a strategy that mixes
them will produce signals no backtest reproduces.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from quantflow.core.clock import floor_to_interval
from quantflow.core.errors import MarketDataError, ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, CandleSeries

#: Gap filling needs at least a pair of bars to have an interval between them.
MIN_CANDLES_FOR_GAP_FILL = 2


def can_resample(source: Timeframe, target: Timeframe) -> bool:
    """Whether ``source`` bars aggregate cleanly into ``target`` bars.

    The target must be a whole multiple of the source. ``1h -> 4h`` works; ``1h -> 3d``
    does not divide evenly against the epoch grid for weekly-style intervals, and
    ``4h -> 1h`` is not aggregation at all.
    """
    if target.seconds <= source.seconds:
        return False
    return target.seconds % source.seconds == 0


def resample(
    candles: Sequence[Candle],
    target: Timeframe,
    *,
    drop_incomplete: bool = True,
) -> list[Candle]:
    """Aggregate candles into ``target`` bars.

    OHLCV aggregation is: open of the first bar, max high, min low, close of the last bar,
    summed volumes, summed trade counts.

    Args:
        candles: Source bars; must share one symbol and timeframe.
        target: Desired interval.
        drop_incomplete: Discard a trailing bucket that has fewer source bars than a full
            bar requires. On by default — a partial bucket has a "close" that is really a
            mid-bar price, and feeding that to a strategy is look-ahead bias in reverse.

    Raises:
        MarketDataError: if the source is empty or mixes symbols/timeframes.
        ValidationError: if the target interval does not divide the source.

    """
    rows = list(candles)
    if not rows:
        raise MarketDataError("cannot resample an empty candle sequence")

    symbol: Symbol = rows[0].symbol
    source: Timeframe = rows[0].timeframe
    if any(candle.symbol != symbol for candle in rows):
        raise MarketDataError("cannot resample candles from multiple symbols")
    if any(candle.timeframe != source for candle in rows):
        raise MarketDataError("cannot resample candles of mixed timeframes")
    if source is target:
        return sorted(rows, key=lambda candle: candle.open_time)
    if not can_resample(source, target):
        raise ValidationError(
            f"cannot resample {source.value} into {target.value}: "
            "the target must be a whole multiple of the source",
            source=source.value,
            target=target.value,
        )

    bars_per_bucket = target.seconds // source.seconds
    ordered = sorted(rows, key=lambda candle: candle.open_time)

    buckets: dict[datetime, list[Candle]] = {}
    for candle in ordered:
        key = floor_to_interval(candle.open_time, target.delta)
        buckets.setdefault(key, []).append(candle)

    aggregated: list[Candle] = []
    for open_time in sorted(buckets):
        group = buckets[open_time]
        if drop_incomplete and len(group) < bars_per_bucket:
            continue
        aggregated.append(_aggregate(symbol, target, open_time, group))
    return aggregated


def _aggregate(
    symbol: Symbol, timeframe: Timeframe, open_time: datetime, group: list[Candle]
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open_time=open_time,
        open=group[0].open,
        high=max(candle.high for candle in group),
        low=min(candle.low for candle in group),
        close=group[-1].close,
        volume=sum((candle.volume for candle in group), ZERO),
        quote_volume=sum((candle.quote_volume for candle in group), ZERO),
        trades=sum(candle.trades for candle in group),
    )


def resample_series(series: CandleSeries, target: Timeframe) -> CandleSeries:
    """Resample a :class:`CandleSeries`, returning a validated series."""
    return CandleSeries(resample(series.candles, target))


def align_series(
    primary: CandleSeries, secondary: CandleSeries
) -> tuple[CandleSeries, CandleSeries]:
    """Trim two series to their overlapping time range.

    Multi-timeframe and cross-asset strategies must not compare a window of one symbol
    against a differently-dated window of another; this makes the overlap explicit rather
    than leaving it to index arithmetic.

    Raises:
        MarketDataError: if the two series do not overlap at all.

    """
    start = max(primary.start, secondary.start)
    end = min(primary.end, secondary.end)
    if start > end:
        raise MarketDataError(f"{primary.symbol} and {secondary.symbol} have no overlapping range")
    step = max(primary.timeframe.delta, secondary.timeframe.delta)
    return (
        primary.slice(start, end + step),
        secondary.slice(start, end + step),
    )


def forward_fill_gaps(candles: Sequence[Candle]) -> list[Candle]:
    """Synthesise flat bars across gaps.

    Each filled bar carries the previous close as its full OHLC and zero volume, marking it
    unambiguously as synthetic. Use only where a strategy requires an unbroken index; for
    performance measurement, prefer leaving the gap visible so it is not mistaken for a
    genuine period of no price movement.
    """
    rows = sorted(candles, key=lambda candle: candle.open_time)
    if len(rows) < MIN_CANDLES_FOR_GAP_FILL:
        return rows

    timeframe = rows[0].timeframe
    step = timeframe.delta
    filled: list[Candle] = [rows[0]]

    for candle in rows[1:]:
        previous = filled[-1]
        expected = previous.open_time + step
        while expected < candle.open_time:
            filled.append(
                Candle(
                    symbol=candle.symbol,
                    timeframe=timeframe,
                    open_time=expected,
                    open=previous.close,
                    high=previous.close,
                    low=previous.close,
                    close=previous.close,
                    volume=ZERO,
                    quote_volume=ZERO,
                    trades=0,
                )
            )
            expected += step
        filled.append(candle)
    return filled
