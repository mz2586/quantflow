"""Market-data pipeline: historical backfill, columnar storage and resampling."""

from __future__ import annotations

from quantflow.marketdata.downloader import (
    DownloadResult,
    HistoricalDownloader,
    estimate_requests,
    expected_bar_count,
)
from quantflow.marketdata.resample import align_series, can_resample, resample, resample_series
from quantflow.marketdata.store import (
    ParquetCandleStore,
    candles_to_frame,
    frame_to_candles,
    returns,
)

__all__ = [
    "DownloadResult",
    "HistoricalDownloader",
    "ParquetCandleStore",
    "align_series",
    "can_resample",
    "candles_to_frame",
    "estimate_requests",
    "expected_bar_count",
    "frame_to_candles",
    "resample",
    "resample_series",
    "returns",
]
