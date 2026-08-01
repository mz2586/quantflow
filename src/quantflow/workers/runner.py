"""Background worker.

Runs scheduled maintenance that must not sit inside a request: keeping market data current,
verifying stored series, and refreshing cached instrument metadata.

Deliberately does **not** place orders. Trading runs in its own process with its own
lifecycle, so a worker restart can never interrupt a position or duplicate an order.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import UTC, datetime, timedelta
from typing import Any

from quantflow.core.config import Settings, get_settings
from quantflow.core.logging import configure_logging, get_logger
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.persistence.database import Database

logger = get_logger(__name__)

#: How often the market-data refresh runs. One minute is frequent enough that an hourly
#: bar is stored within a minute of closing, and cheap enough to be unnoticeable.
REFRESH_INTERVAL_SECONDS = 60.0

#: How often instrument metadata is reloaded. Binance changes lot sizes and minimum
#: notionals occasionally, and a stale rule means rejected orders.
INSTRUMENT_REFRESH_SECONDS = 3600.0


class Worker:
    """Owns the scheduled background jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database.from_settings(settings)
        self._stopping = asyncio.Event()
        self._gateway: Any = None

    async def start(self) -> None:
        """Run until stopped."""
        from quantflow.exchange.binance.rest import BinanceGateway

        self._gateway = BinanceGateway(self.settings.exchange)
        try:
            await self._gateway.connect()
        except Exception as exc:
            logger.warning("worker.exchange_unavailable", error=str(exc))

        logger.info(
            "worker.started",
            symbols=list(self.settings.trading.symbols),
            timeframe=self.settings.trading.default_timeframe,
        )
        tasks = [
            asyncio.create_task(self._refresh_market_data(), name="refresh-market-data"),
            asyncio.create_task(self._refresh_instruments(), name="refresh-instruments"),
        ]
        try:
            await self._stopping.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.aclose()

    def stop(self) -> None:
        """Request a graceful shutdown."""
        self._stopping.set()

    async def aclose(self) -> None:
        """Release resources."""
        if self._gateway is not None:
            with contextlib.suppress(Exception):
                await self._gateway.aclose()
        with contextlib.suppress(Exception):
            await self.database.aclose()
        logger.info("worker.stopped")

    async def _refresh_market_data(self) -> None:
        """Keep the configured series current."""
        from quantflow.marketdata.downloader import HistoricalDownloader

        while not self._stopping.is_set():
            try:
                if self._gateway is not None:
                    downloader = HistoricalDownloader(gateway=self._gateway, database=self.database)
                    timeframe = Timeframe.parse(self.settings.trading.default_timeframe)
                    for raw in self.settings.trading.symbols:
                        symbol = Symbol.parse(raw)
                        assert isinstance(symbol, Symbol)
                        # Look back a day: re-requesting recent bars is idempotent and
                        # repairs anything a dropped websocket missed.
                        result = await downloader.download(
                            symbol,
                            timeframe,
                            start=datetime.now(UTC) - timedelta(days=1),
                            verify=False,
                        )
                        if result.candles_written:
                            logger.debug(
                                "worker.market_data_refreshed",
                                symbol=str(symbol),
                                written=result.candles_written,
                            )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("worker.refresh_failed", error=str(exc))

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=REFRESH_INTERVAL_SECONDS)

    async def _refresh_instruments(self) -> None:
        """Reload venue trading rules into the database."""
        while not self._stopping.is_set():
            try:
                if self._gateway is not None:
                    instruments = await self._gateway.load_instruments()
                    async with self.database.unit_of_work() as uow:
                        for instrument in instruments.values():
                            await uow.instruments.upsert(instrument)
                    logger.info("worker.instruments_refreshed", count=len(instruments))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("worker.instrument_refresh_failed", error=str(exc))

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=INSTRUMENT_REFRESH_SECONDS)


async def run() -> None:
    """Entry point: build the worker and wire signal handling."""
    settings = get_settings()
    configure_logging(settings, service="quantflow-worker")

    worker = Worker(settings)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, worker.stop)

    await worker.start()


def main() -> None:
    """Synchronous entry point for ``python -m quantflow.workers.runner``."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
