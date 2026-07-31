"""Resampling, Parquet storage and dataframe conversion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import polars as pl
import pytest

from quantflow.core.errors import InsufficientDataError, MarketDataError, ValidationError
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, CandleSeries
from quantflow.marketdata.downloader import chunk_range, estimate_requests, expected_bar_count
from quantflow.marketdata.resample import (
    align_series,
    can_resample,
    forward_fill_gaps,
    resample,
    resample_series,
)
from quantflow.marketdata.store import (
    ParquetCandleStore,
    candles_to_frame,
    frame_to_candles,
    returns,
)
from tests.conftest import REFERENCE_TIME


def hourly(
    symbol: Symbol, index: int, *, o: str, h: str, low: str, c: str, v: str = "10"
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=Timeframe.H1,
        open_time=REFERENCE_TIME + timedelta(hours=index),
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(low),
        close=Decimal(c),
        volume=Decimal(v),
        quote_volume=Decimal(v) * Decimal(c),
        trades=5,
    )


class TestCanResample:
    @pytest.mark.parametrize(
        ("source", "target", "expected"),
        [
            (Timeframe.M1, Timeframe.H1, True),
            (Timeframe.H1, Timeframe.H4, True),
            (Timeframe.H1, Timeframe.D1, True),
            (Timeframe.M5, Timeframe.M15, True),
            (Timeframe.H4, Timeframe.H1, False),  # not aggregation
            (Timeframe.H1, Timeframe.H1, False),  # not a change
            (Timeframe.H2, Timeframe.H6, True),
        ],
    )
    def test_divisibility(self, source: Timeframe, target: Timeframe, expected: bool) -> None:
        assert can_resample(source, target) is expected


class TestResample:
    def test_ohlcv_aggregation(self, btc: Symbol) -> None:
        candles = [
            hourly(btc, 0, o="100", h="110", low="95", c="105", v="10"),
            hourly(btc, 1, o="105", h="120", low="100", c="115", v="20"),
            hourly(btc, 2, o="115", h="118", low="90", c="95", v="30"),
            hourly(btc, 3, o="95", h="100", low="94", c="99", v="40"),
        ]
        aggregated = resample(candles, Timeframe.H4)
        assert len(aggregated) == 1
        bar = aggregated[0]
        assert bar.open == Decimal("100")  # first open
        assert bar.high == Decimal("120")  # max high
        assert bar.low == Decimal("90")  # min low
        assert bar.close == Decimal("99")  # last close
        assert bar.volume == Decimal("100")  # summed
        assert bar.trades == 20
        assert bar.timeframe is Timeframe.H4
        assert bar.open_time == REFERENCE_TIME

    def test_drops_a_partial_trailing_bucket_by_default(self, btc: Symbol) -> None:
        # A partial bucket's "close" is a mid-bar price; handing it to a strategy would
        # let it act on information the completed bar has not produced yet.
        candles = [hourly(btc, index, o="100", h="101", low="99", c="100") for index in range(6)]
        assert len(resample(candles, Timeframe.H4)) == 1

    def test_can_keep_a_partial_bucket(self, btc: Symbol) -> None:
        candles = [hourly(btc, index, o="100", h="101", low="99", c="100") for index in range(6)]
        assert len(resample(candles, Timeframe.H4, drop_incomplete=False)) == 2

    def test_buckets_align_to_the_epoch_grid(self, btc: Symbol) -> None:
        # Starting at 02:00, the first complete 4h bucket is 04:00-08:00.
        candles = [
            Candle(
                symbol=btc,
                timeframe=Timeframe.H1,
                open_time=datetime(2026, 1, 1, 2, tzinfo=UTC) + timedelta(hours=index),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("1"),
            )
            for index in range(8)
        ]
        aggregated = resample(candles, Timeframe.H4)
        assert [bar.open_time for bar in aggregated] == [datetime(2026, 1, 1, 4, tzinfo=UTC)]

    def test_same_timeframe_is_a_sorted_passthrough(self, btc: Symbol) -> None:
        candles = [hourly(btc, index, o="100", h="101", low="99", c="100") for index in (2, 0, 1)]
        assert [c.open_time for c in resample(candles, Timeframe.H1)] == sorted(
            c.open_time for c in candles
        )

    def test_rejects_an_indivisible_target(self, btc: Symbol) -> None:
        candles = [hourly(btc, 0, o="1", h="1", low="1", c="1")]
        with pytest.raises(ValidationError, match="whole multiple"):
            resample(candles, Timeframe.M5)

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(MarketDataError, match="empty"):
            resample([], Timeframe.H4)

    def test_rejects_mixed_symbols(self, btc: Symbol, eth: Symbol) -> None:
        candles = [
            hourly(btc, 0, o="1", h="1", low="1", c="1"),
            hourly(eth, 1, o="1", h="1", low="1", c="1"),
        ]
        with pytest.raises(MarketDataError, match="multiple symbols"):
            resample(candles, Timeframe.H4)

    def test_series_helper_validates_the_result(self, btc: Symbol) -> None:
        series = CandleSeries(
            [hourly(btc, index, o="100", h="101", low="99", c="100") for index in range(8)]
        )
        resampled = resample_series(series, Timeframe.H4)
        assert len(resampled) == 2
        assert resampled.timeframe is Timeframe.H4
        assert resampled.is_contiguous


class TestAlignSeries:
    def test_trims_to_the_overlap(self, btc: Symbol, eth: Symbol) -> None:
        first = CandleSeries(
            [hourly(btc, index, o="1", h="1", low="1", c="1") for index in range(10)]
        )
        second = CandleSeries(
            [hourly(eth, index, o="1", h="1", low="1", c="1") for index in range(4, 14)]
        )
        aligned_first, aligned_second = align_series(first, second)
        assert aligned_first.start == aligned_second.start
        assert aligned_first.end == aligned_second.end
        assert len(aligned_first) == len(aligned_second) == 6

    def test_disjoint_series_raise(self, btc: Symbol, eth: Symbol) -> None:
        first = CandleSeries(
            [hourly(btc, index, o="1", h="1", low="1", c="1") for index in range(3)]
        )
        second = CandleSeries(
            [hourly(eth, index, o="1", h="1", low="1", c="1") for index in range(50, 53)]
        )
        with pytest.raises(MarketDataError, match="no overlapping range"):
            align_series(first, second)


class TestForwardFill:
    def test_synthesises_flat_zero_volume_bars(self, btc: Symbol) -> None:
        candles = [
            hourly(btc, 0, o="100", h="110", low="90", c="105"),
            hourly(btc, 4, o="105", h="115", low="95", c="110"),
        ]
        filled = forward_fill_gaps(candles)
        assert len(filled) == 5
        synthetic = filled[1:4]
        assert all(bar.volume == Decimal("0") for bar in synthetic)
        # Zero volume and zero range mark them unambiguously as synthetic.
        assert all(bar.open == bar.close == Decimal("105") for bar in synthetic)

    def test_contiguous_input_is_unchanged(self, btc: Symbol) -> None:
        candles = [hourly(btc, index, o="100", h="101", low="99", c="100") for index in range(4)]
        assert len(forward_fill_gaps(candles)) == 4

    def test_single_candle_is_unchanged(self, btc: Symbol) -> None:
        assert len(forward_fill_gaps([hourly(btc, 0, o="1", h="1", low="1", c="1")])) == 1


class TestFrameConversion:
    def test_round_trip_preserves_values(self, btc: Symbol) -> None:
        candles = [
            hourly(btc, index, o="50000.5", h="50100.25", low="49900.75", c="50050.125")
            for index in range(3)
        ]
        frame = candles_to_frame(candles)
        assert frame.height == 3
        restored = frame_to_candles(frame, btc, Timeframe.H1)
        assert [c.close for c in restored] == [Decimal("50050.125")] * 3
        assert [c.open_time for c in restored] == [c.open_time for c in candles]

    def test_empty_input_produces_a_typed_empty_frame(self) -> None:
        frame = candles_to_frame([])
        assert frame.height == 0
        assert "close" in frame.columns

    def test_frame_is_sorted(self, btc: Symbol) -> None:
        candles = [hourly(btc, index, o="1", h="1", low="1", c="1") for index in (2, 0, 1)]
        frame = candles_to_frame(candles)
        assert frame["open_time"].is_sorted()

    def test_missing_columns_are_rejected(self, btc: Symbol) -> None:
        with pytest.raises(MarketDataError, match="missing columns"):
            frame_to_candles(pl.DataFrame({"open_time": []}), btc, Timeframe.H1)

    def test_returns_simple_and_log(self, btc: Symbol) -> None:
        candles = [
            hourly(btc, 0, o="100", h="100", low="100", c="100"),
            hourly(btc, 1, o="110", h="110", low="110", c="110"),
        ]
        frame = candles_to_frame(candles)
        simple = returns(frame)
        assert simple[0] == 0.0
        assert simple[1] == pytest.approx(0.1)
        assert returns(frame, log=True)[1] == pytest.approx(0.09531, abs=1e-4)


class TestParquetStore:
    def test_write_and_read(self, tmp_path: Path, btc: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        candles = [hourly(btc, index, o="100", h="101", low="99", c="100") for index in range(10)]
        assert store.write(btc, Timeframe.H1, candles) == 10
        assert store.exists(btc, Timeframe.H1)
        assert len(store.read_candles(btc, Timeframe.H1)) == 10

    def test_read_filters_the_range(self, tmp_path: Path, btc: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        store.write(
            btc,
            Timeframe.H1,
            [hourly(btc, index, o="100", h="101", low="99", c="100") for index in range(10)],
        )
        filtered = store.read(
            btc,
            Timeframe.H1,
            start=REFERENCE_TIME + timedelta(hours=3),
            end=REFERENCE_TIME + timedelta(hours=6),
        )
        assert filtered.height == 3

    def test_merge_deduplicates_and_corrects(self, tmp_path: Path, btc: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        store.write(btc, Timeframe.H1, [hourly(btc, 0, o="100", h="101", low="99", c="100")])
        # A corrected bar for the same open time must replace, not duplicate.
        store.write(btc, Timeframe.H1, [hourly(btc, 0, o="100", h="205", low="99", c="200")])
        stored = store.read_candles(btc, Timeframe.H1)
        assert len(stored) == 1
        assert stored[0].close == Decimal("200")

    def test_merge_extends_an_existing_series(self, tmp_path: Path, btc: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        store.write(
            btc, Timeframe.H1, [hourly(btc, i, o="1", h="1", low="1", c="1") for i in range(5)]
        )
        total = store.write(
            btc, Timeframe.H1, [hourly(btc, i, o="1", h="1", low="1", c="1") for i in range(3, 8)]
        )
        assert total == 8

    def test_read_missing_series_raises(self, tmp_path: Path, btc: Symbol) -> None:
        with pytest.raises(InsufficientDataError, match="no stored data"):
            ParquetCandleStore(tmp_path).read(btc, Timeframe.H1)

    def test_partition_layout(self, tmp_path: Path, btc: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        path = store.partition_path(btc, Timeframe.H1)
        # A slash in the symbol would otherwise create a spurious directory level.
        assert path == tmp_path / "symbol=BTC-USDT" / "timeframe=1h"

    def test_list_series(self, tmp_path: Path, btc: Symbol, eth: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        store.write(btc, Timeframe.H1, [hourly(btc, 0, o="1", h="1", low="1", c="1")])
        store.write(
            eth,
            Timeframe.D1,
            [
                Candle(
                    symbol=eth,
                    timeframe=Timeframe.D1,
                    open_time=REFERENCE_TIME,
                    open=Decimal("1"),
                    high=Decimal("1"),
                    low=Decimal("1"),
                    close=Decimal("1"),
                    volume=Decimal("1"),
                )
            ],
        )
        found = store.list_series()
        assert (btc, Timeframe.H1) in found
        assert (eth, Timeframe.D1) in found

    def test_list_series_on_an_empty_root(self, tmp_path: Path) -> None:
        assert ParquetCandleStore(tmp_path / "missing").list_series() == []

    def test_delete(self, tmp_path: Path, btc: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        store.write(btc, Timeframe.H1, [hourly(btc, 0, o="1", h="1", low="1", c="1")])
        assert store.delete(btc, Timeframe.H1) is True
        assert store.delete(btc, Timeframe.H1) is False

    def test_stats(self, tmp_path: Path, btc: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        store.write(
            btc, Timeframe.H1, [hourly(btc, i, o="1", h="1", low="1", c="1") for i in range(5)]
        )
        stats = store.stats(btc, Timeframe.H1)
        assert stats["rows"] == 5
        assert stats["symbol"] == "BTC/USDT"

    def test_read_series_validates(self, tmp_path: Path, btc: Symbol) -> None:
        store = ParquetCandleStore(tmp_path)
        store.write(
            btc, Timeframe.H1, [hourly(btc, i, o="1", h="1", low="1", c="1") for i in range(4)]
        )
        series = store.read_series(btc, Timeframe.H1)
        assert isinstance(series, CandleSeries)
        assert series.is_contiguous


class TestDownloadPlanning:
    def test_expected_bar_count(self) -> None:
        start = REFERENCE_TIME
        assert expected_bar_count(start, start + timedelta(days=1), Timeframe.H1) == 24
        assert expected_bar_count(start, start + timedelta(days=1), Timeframe.M1) == 1440
        assert expected_bar_count(start, start, Timeframe.H1) == 0

    def test_estimate_requests_rounds_up(self) -> None:
        start = REFERENCE_TIME
        # 1440 one-minute bars need two 1000-bar pages.
        assert estimate_requests(start, start + timedelta(days=1), Timeframe.M1) == 2
        assert estimate_requests(start, start + timedelta(hours=24), Timeframe.H1) == 1

    def test_chunk_range_covers_the_whole_span(self) -> None:
        start = REFERENCE_TIME
        end = start + timedelta(days=100)
        windows = chunk_range(start, end, Timeframe.H1, bars_per_chunk=1000)
        assert windows[0][0] == start
        assert windows[-1][1] == end
        for previous, current in pairwise(windows):
            assert previous[1] == current[0]

    def test_chunk_range_of_an_empty_span(self) -> None:
        assert chunk_range(REFERENCE_TIME, REFERENCE_TIME, Timeframe.H1) == []
