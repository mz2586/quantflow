"""Historical downloader against a scripted gateway and a real database.

The gateway is faked so pagination, gaps, resumption and failure can be exercised
deterministically; the database is real, because idempotent upsert behaviour is the
property under test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quantflow.core.clock import FrozenClock
from quantflow.core.errors import ExchangeConnectionError, MarketDataError
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle, OrderBook, Ticker, Trade
from quantflow.marketdata.downloader import HistoricalDownloader
from quantflow.persistence.database import Database

pytestmark = pytest.mark.integration

BTC = Symbol(base="BTC", quote="USDT")
START = datetime(2026, 1, 1, tzinfo=UTC)


def candle(index: int, *, close: str = "50000") -> Candle:
    price = Decimal(close)
    return Candle(
        symbol=BTC,
        timeframe=Timeframe.H1,
        open_time=START + timedelta(hours=index),
        open=price,
        high=price + Decimal("10"),
        low=price - Decimal("10"),
        close=price,
        volume=Decimal("1"),
        quote_volume=price,
    )


class ScriptedGateway:
    """A market-data gateway that serves a fixed candle universe with 1000-bar paging."""

    def __init__(
        self,
        candles: list[Candle],
        *,
        page_size: int = 1000,
        fail_after: int | None = None,
    ) -> None:
        self.universe = sorted(candles, key=lambda item: item.open_time)
        self.page_size = page_size
        self.fail_after = fail_after
        self.calls: list[datetime | None] = []

    async def fetch_candles(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        *,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        del symbol, timeframe
        self.calls.append(since)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise ExchangeConnectionError("venue went away")
        available = [c for c in self.universe if since is None or c.open_time >= since]
        return available[: min(limit, self.page_size)]

    # -- unused parts of the protocol ---------------------------------- #
    async def load_instruments(self) -> dict[Symbol, Instrument]:  # pragma: no cover
        return {}

    async def get_instrument(self, symbol: Symbol) -> Instrument:  # pragma: no cover
        return Instrument(symbol=symbol)

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:  # pragma: no cover
        raise NotImplementedError

    async def fetch_order_book(
        self, symbol: Symbol, *, depth: int = 20
    ) -> OrderBook:  # pragma: no cover
        raise NotImplementedError

    async def fetch_recent_trades(
        self, symbol: Symbol, *, limit: int = 100
    ) -> list[Trade]:  # pragma: no cover
        raise NotImplementedError

    async def server_time(self) -> datetime:  # pragma: no cover
        return START


@pytest.fixture
def downloader_factory(database: Database):
    def build(gateway: ScriptedGateway, *, now: datetime | None = None) -> HistoricalDownloader:
        return HistoricalDownloader(
            gateway=gateway,
            database=database,
            clock=FrozenClock(now or START + timedelta(days=365)),
            request_delay_seconds=0.0,
        )

    return build


class TestBasicDownload:
    async def test_writes_the_requested_range(self, downloader_factory) -> None:
        gateway = ScriptedGateway([candle(i) for i in range(48)])
        downloader = downloader_factory(gateway)

        result = await downloader.download(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=48)
        )

        assert result.succeeded
        assert result.candles_written == 48
        assert result.first_open_time == START
        assert result.last_open_time == START + timedelta(hours=47)
        assert result.is_complete
        assert len(await downloader.load(BTC, Timeframe.H1)) == 48

    async def test_paginates_beyond_one_page(self, downloader_factory) -> None:
        gateway = ScriptedGateway([candle(i) for i in range(2500)], page_size=1000)
        downloader = downloader_factory(gateway)

        result = await downloader.download(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=2500)
        )

        assert result.candles_written == 2500
        assert result.requests_made == 3
        assert result.is_complete

    async def test_respects_the_end_boundary(self, downloader_factory) -> None:
        gateway = ScriptedGateway([candle(i) for i in range(100)])
        downloader = downloader_factory(gateway)

        result = await downloader.download(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=10)
        )

        assert result.candles_written == 10
        stored = await downloader.load(BTC, Timeframe.H1)
        assert max(c.open_time for c in stored) == START + timedelta(hours=9)

    async def test_defaults_to_the_last_closed_bar(self, downloader_factory) -> None:
        # Never the forming bar: its close is still changing, and persisting it would
        # bake a mid-bar price into every future backtest.
        now = START + timedelta(hours=10, minutes=30)
        gateway = ScriptedGateway([candle(i) for i in range(20)])
        downloader = downloader_factory(gateway, now=now)

        result = await downloader.download(BTC, Timeframe.H1, start=START)

        assert result.requested_end == START + timedelta(hours=10)
        assert result.candles_written == 10

    async def test_empty_range_is_rejected(self, downloader_factory) -> None:
        downloader = downloader_factory(ScriptedGateway([]))
        with pytest.raises(MarketDataError, match="empty download range"):
            await downloader.download(BTC, Timeframe.H1, start=START, end=START)


class TestIdempotencyAndResume:
    async def test_rerunning_does_not_duplicate(self, downloader_factory) -> None:
        gateway = ScriptedGateway([candle(i) for i in range(24)])
        downloader = downloader_factory(gateway)
        window = {"start": START, "end": START + timedelta(hours=24)}

        await downloader.download(BTC, Timeframe.H1, **window, resume=False)
        await downloader.download(BTC, Timeframe.H1, **window, resume=False)

        assert len(await downloader.load(BTC, Timeframe.H1)) == 24

    async def test_resume_continues_from_the_last_stored_bar(self, downloader_factory) -> None:
        gateway = ScriptedGateway([candle(i) for i in range(48)])
        downloader = downloader_factory(gateway)

        await downloader.download(BTC, Timeframe.H1, start=START, end=START + timedelta(hours=24))
        gateway.calls.clear()

        second = await downloader.download(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=48)
        )

        assert second.candles_written == 24  # only the new half
        assert gateway.calls[0] == START + timedelta(hours=24)
        assert len(await downloader.load(BTC, Timeframe.H1)) == 48

    async def test_already_current_is_skipped(self, downloader_factory) -> None:
        gateway = ScriptedGateway([candle(i) for i in range(24)])
        downloader = downloader_factory(gateway)
        window = {"start": START, "end": START + timedelta(hours=24)}

        await downloader.download(BTC, Timeframe.H1, **window)
        gateway.calls.clear()
        second = await downloader.download(BTC, Timeframe.H1, **window)

        assert second.skipped
        assert gateway.calls == []

    async def test_resume_disabled_refetches(self, downloader_factory) -> None:
        gateway = ScriptedGateway([candle(i) for i in range(24)])
        downloader = downloader_factory(gateway)
        window = {"start": START, "end": START + timedelta(hours=24)}

        await downloader.download(BTC, Timeframe.H1, **window)
        gateway.calls.clear()
        second = await downloader.download(BTC, Timeframe.H1, **window, resume=False)

        assert not second.skipped
        assert gateway.calls[0] == START


class TestGapHandling:
    async def test_venue_side_gaps_are_reported_not_hidden(self, downloader_factory) -> None:
        # An illiquid pair genuinely has no bars for some periods; a backtest run across
        # that hole would look fine and be wrong, so it must surface.
        present = [candle(i) for i in range(10)] + [candle(i) for i in range(20, 30)]
        downloader = downloader_factory(ScriptedGateway(present))

        result = await downloader.download(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=30)
        )

        assert result.succeeded
        assert result.candles_written == 20
        assert not result.is_complete
        assert result.integrity is not None
        assert result.integrity.missing_bar_count == 10
        assert "missing bars" in result.summary()

    async def test_backfill_gaps_repairs_a_hole(self, downloader_factory) -> None:
        full = [candle(i) for i in range(30)]
        partial = [
            c for c in full if not (10 <= (c.open_time - START).total_seconds() // 3600 < 15)
        ]

        downloader = downloader_factory(ScriptedGateway(partial))
        await downloader.download(BTC, Timeframe.H1, start=START, end=START + timedelta(hours=30))

        repaired = downloader_factory(ScriptedGateway(full))
        result = await repaired.backfill_gaps(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=30)
        )

        assert result.candles_written >= 5
        assert result.integrity is not None
        assert result.integrity.is_clean

    async def test_backfill_on_clean_data_is_skipped(self, downloader_factory) -> None:
        downloader = downloader_factory(ScriptedGateway([candle(i) for i in range(10)]))
        await downloader.download(BTC, Timeframe.H1, start=START, end=START + timedelta(hours=10))
        result = await downloader.backfill_gaps(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=10)
        )
        assert result.skipped


class TestFailureHandling:
    async def test_partial_progress_is_kept_and_the_error_reported(
        self, downloader_factory
    ) -> None:
        gateway = ScriptedGateway([candle(i) for i in range(2500)], fail_after=1)
        downloader = downloader_factory(gateway)

        result = await downloader.download(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=2500)
        )

        assert not result.succeeded
        assert result.error is not None
        assert "venue went away" in result.error
        # The first page committed; a crash mid-backfill must not lose it.
        assert result.candles_written == 1000
        assert len(await downloader.load(BTC, Timeframe.H1)) == 1000
        assert "FAILED" in result.summary()

    async def test_a_failed_run_can_be_resumed(self, downloader_factory) -> None:
        universe = [candle(i) for i in range(2500)]
        failing = downloader_factory(ScriptedGateway(universe, fail_after=1))
        await failing.download(BTC, Timeframe.H1, start=START, end=START + timedelta(hours=2500))

        healthy = downloader_factory(ScriptedGateway(universe))
        result = await healthy.download(
            BTC, Timeframe.H1, start=START, end=START + timedelta(hours=2500)
        )

        assert result.succeeded
        assert result.is_complete
        assert len(await healthy.load(BTC, Timeframe.H1)) == 2500


class TestMultiSymbol:
    async def test_download_many(self, database: Database) -> None:
        eth = Symbol(base="ETH", quote="USDT")

        class MultiGateway(ScriptedGateway):
            async def fetch_candles(
                self,
                symbol: Symbol,
                timeframe: Timeframe,
                *,
                since: datetime | None = None,
                limit: int = 1000,
            ) -> list[Candle]:
                self.calls.append(since)
                base = [c for c in self.universe if since is None or c.open_time >= since]
                return [
                    Candle(
                        symbol=symbol,
                        timeframe=c.timeframe,
                        open_time=c.open_time,
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume=c.volume,
                        quote_volume=c.quote_volume,
                    )
                    for c in base[:limit]
                ]

        downloader = HistoricalDownloader(
            gateway=MultiGateway([candle(i) for i in range(24)]),
            database=database,
            clock=FrozenClock(START + timedelta(days=365)),
            request_delay_seconds=0.0,
        )

        results = await downloader.download_many(
            [BTC, eth], Timeframe.H1, start=START, end=START + timedelta(hours=24), concurrency=2
        )

        assert len(results) == 2
        assert all(result.candles_written == 24 for result in results)
        assert len(await downloader.load(BTC, Timeframe.H1)) == 24
        assert len(await downloader.load(eth, Timeframe.H1)) == 24
