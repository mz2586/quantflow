"""Historical OHLCV backfill.

Binance returns at most 1000 bars per request, so any meaningful history has to be
paginated. The downloader is **resumable** and **idempotent**: it starts from the last
stored bar, writes through an upsert, and can be interrupted and restarted without
duplicating or losing data. It also refuses to silently hand back an incomplete dataset —
gaps are reported, because a backtest run across an undetected hole produces a
plausible-looking equity curve that is simply wrong.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from quantflow.core.clock import Clock, SystemClock, floor_to_interval
from quantflow.core.errors import MarketDataError
from quantflow.core.logging import get_logger
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, DataIntegrityReport
from quantflow.exchange.base import MAX_CANDLES_PER_REQUEST, MarketDataGateway
from quantflow.persistence.database import Database

logger = get_logger(__name__)

#: Pause between paginated requests, on top of the gateway's own rate limiter. Backfilling
#: years of 1m data is thousands of calls; a small deliberate gap keeps a long backfill from
#: monopolising the venue budget that live trading also needs.
INTER_REQUEST_DELAY_SECONDS = 0.05

#: Guards against a venue that keeps returning the same page. Without it, a misbehaving
#: endpoint would spin forever.
MAX_EMPTY_PAGES = 3


@dataclass(slots=True)
class DownloadResult:
    """Outcome of one symbol/timeframe backfill."""

    symbol: Symbol
    timeframe: Timeframe
    requested_start: datetime
    requested_end: datetime
    candles_written: int = 0
    requests_made: int = 0
    first_open_time: datetime | None = None
    last_open_time: datetime | None = None
    integrity: DataIntegrityReport | None = None
    skipped: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the backfill completed without error."""
        return self.error is None

    @property
    def is_complete(self) -> bool:
        """Whether the stored range is contiguous and error-free."""
        return self.succeeded and (self.integrity is None or self.integrity.is_clean)

    def summary(self) -> str:
        """One-line human summary."""
        if self.error:
            return f"{self.symbol} {self.timeframe.value}: FAILED — {self.error}"
        if self.skipped:
            return f"{self.symbol} {self.timeframe.value}: already up to date"
        gaps = self.integrity.missing_bar_count if self.integrity else 0
        suffix = f", {gaps} missing bars" if gaps else ""
        return (
            f"{self.symbol} {self.timeframe.value}: {self.candles_written} bars "
            f"in {self.requests_made} requests{suffix}"
        )


@dataclass(slots=True)
class HistoricalDownloader:
    """Paginated, resumable OHLCV backfill into Postgres."""

    gateway: MarketDataGateway
    database: Database
    clock: Clock = field(default_factory=SystemClock)
    request_delay_seconds: float = INTER_REQUEST_DELAY_SECONDS
    page_size: int = MAX_CANDLES_PER_REQUEST

    async def download(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        start: datetime,
        end: datetime | None = None,
        resume: bool = True,
        verify: bool = True,
    ) -> DownloadResult:
        """Backfill ``[start, end)`` for one symbol and timeframe.

        Args:
            symbol: Pair to download.
            timeframe: Bar interval.
            start: Inclusive start of the range.
            end: Exclusive end; defaults to the last *closed* bar. Never the current bar —
                storing a forming bar would persist a close price that is still changing.
            resume: Continue from the last stored bar instead of re-downloading.
            verify: Run an integrity check over the stored range afterwards.

        """
        effective_end = end or self._last_closed_open_time(timeframe)
        result = DownloadResult(
            symbol=symbol,
            timeframe=timeframe,
            requested_start=start,
            requested_end=effective_end,
        )

        if start >= effective_end:
            raise MarketDataError(
                f"empty download range for {symbol}: start {start.isoformat()} "
                f"is not before end {effective_end.isoformat()}"
            )

        cursor = start
        if resume:
            async with self.database.read_session() as session:
                from quantflow.persistence.repositories import CandleRepository

                latest = await CandleRepository(session).latest_open_time(symbol, timeframe)
            if latest is not None and latest >= start:
                cursor = latest + timeframe.delta
                logger.debug(
                    "downloader.resuming",
                    symbol=str(symbol),
                    timeframe=timeframe.value,
                    from_time=cursor.isoformat(),
                )

        if cursor >= effective_end:
            result.skipped = True
            if verify:
                result.integrity = await self._verify(symbol, timeframe, start, effective_end)
            return result

        logger.info(
            "downloader.started",
            symbol=str(symbol),
            timeframe=timeframe.value,
            start=cursor.isoformat(),
            end=effective_end.isoformat(),
        )

        empty_pages = 0
        try:
            while cursor < effective_end:
                page = await self.gateway.fetch_candles(
                    symbol, timeframe, since=cursor, limit=self.page_size
                )
                result.requests_made += 1

                usable = [candle for candle in page if cursor <= candle.open_time < effective_end]
                if not usable:
                    empty_pages += 1
                    if empty_pages >= MAX_EMPTY_PAGES:
                        logger.debug(
                            "downloader.no_more_data",
                            symbol=str(symbol),
                            at=cursor.isoformat(),
                        )
                        break
                    # Skip a genuine hole in the venue's history and keep going.
                    cursor += timeframe.delta * self.page_size
                    continue

                empty_pages = 0
                async with self.database.unit_of_work() as uow:
                    written = await uow.candles.upsert_many(usable)
                result.candles_written += written

                if result.first_open_time is None:
                    result.first_open_time = usable[0].open_time
                result.last_open_time = usable[-1].open_time
                cursor = usable[-1].open_time + timeframe.delta

                logger.debug(
                    "downloader.page",
                    symbol=str(symbol),
                    written=written,
                    total=result.candles_written,
                    next_start=cursor.isoformat(),
                )
                if self.request_delay_seconds > 0:
                    await self.clock.sleep(self.request_delay_seconds)

        except Exception as exc:
            result.error = str(exc)
            logger.exception(
                "downloader.failed",
                symbol=str(symbol),
                timeframe=timeframe.value,
                written=result.candles_written,
                error=str(exc),
            )
            return result

        if verify:
            result.integrity = await self._verify(symbol, timeframe, start, effective_end)

        logger.info(
            "downloader.finished",
            symbol=str(symbol),
            timeframe=timeframe.value,
            written=result.candles_written,
            requests=result.requests_made,
            complete=result.is_complete,
        )
        return result

    async def download_many(
        self,
        symbols: Sequence[Symbol],
        timeframe: Timeframe,
        *,
        start: datetime,
        end: datetime | None = None,
        resume: bool = True,
        concurrency: int = 3,
    ) -> list[DownloadResult]:
        """Backfill several symbols with bounded concurrency.

        Concurrency is capped because every worker draws on the same venue rate-limit
        budget; running twenty at once simply produces twenty rate-limited workers.
        """
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run(symbol: Symbol) -> DownloadResult:
            async with semaphore:
                return await self.download(symbol, timeframe, start=start, end=end, resume=resume)

        return list(await asyncio.gather(*(run(symbol) for symbol in symbols)))

    async def backfill_gaps(
        self, symbol: Symbol, timeframe: Timeframe, *, start: datetime, end: datetime
    ) -> DownloadResult:
        """Re-request only the ranges reported missing by the integrity check.

        Cheaper than a full re-download when a live stream dropped a handful of bars.
        """
        report = await self._verify(symbol, timeframe, start, end)
        result = DownloadResult(
            symbol=symbol,
            timeframe=timeframe,
            requested_start=start,
            requested_end=end,
            integrity=report,
        )
        if report.is_clean:
            result.skipped = True
            return result

        for gap_start, gap_end in report.gaps:
            page = await self.gateway.fetch_candles(
                symbol, timeframe, since=gap_start, limit=self.page_size
            )
            result.requests_made += 1
            usable = [c for c in page if gap_start <= c.open_time < gap_end]
            if not usable:
                continue
            async with self.database.unit_of_work() as uow:
                result.candles_written += await uow.candles.upsert_many(usable)
            if self.request_delay_seconds > 0:
                await self.clock.sleep(self.request_delay_seconds)

        result.integrity = await self._verify(symbol, timeframe, start, end)
        return result

    async def load(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        """Read stored candles back out of the database."""
        async with self.database.read_session() as session:
            from quantflow.persistence.repositories import CandleRepository

            return await CandleRepository(session).fetch(symbol, timeframe, start=start, end=end)

    async def _verify(
        self, symbol: Symbol, timeframe: Timeframe, start: datetime, end: datetime
    ) -> DataIntegrityReport:
        async with self.database.read_session() as session:
            from quantflow.persistence.repositories import CandleRepository

            return await CandleRepository(session).integrity_report(
                symbol, timeframe, start=start, end=end
            )

    def _last_closed_open_time(self, timeframe: Timeframe) -> datetime:
        """Open time of the most recent *completed* bar."""
        return floor_to_interval(self.clock.now(), timeframe.delta)


def expected_bar_count(start: datetime, end: datetime, timeframe: Timeframe) -> int:
    """How many bars a contiguous ``[start, end)`` range should contain."""
    if end <= start:
        return 0
    return int((end - start) / timeframe.delta)


def estimate_requests(start: datetime, end: datetime, timeframe: Timeframe) -> int:
    """Estimate the number of paginated requests a backfill will need.

    Used by the CLI to warn before a multi-thousand-request 1m backfill.
    """
    bars = expected_bar_count(start, end, timeframe)
    return -(-bars // MAX_CANDLES_PER_REQUEST)  # ceiling division


def chunk_range(
    start: datetime, end: datetime, timeframe: Timeframe, *, bars_per_chunk: int = 1000
) -> list[tuple[datetime, datetime]]:
    """Split a range into request-sized windows."""
    if end <= start:
        return []
    step: timedelta = timeframe.delta * bars_per_chunk
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        windows.append((cursor, min(cursor + step, end)))
        cursor += step
    return windows
