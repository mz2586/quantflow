"""Market-data endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query

from quantflow.api.dashboard.cache import ResilientCache
from quantflow.api.deps import DatabaseDep, GatewayDep
from quantflow.api.schemas import (
    CandleResponse,
    CandlesResponse,
    InstrumentResponse,
    SymbolSummary,
    TickerResponse,
)
from quantflow.core.errors import NotFoundError
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.persistence.database import Database
from quantflow.persistence.repositories import CandleRepository, InstrumentRepository

router = APIRouter(prefix="/market", tags=["market-data"])

#: Symbols appear in URL paths as ``BTC-USDT`` or ``BTCUSDT``. The canonical ``BTC/USDT``
#: form cannot be used there — see :func:`_parse_symbol`.
SYMBOL_PATH_EXAMPLE = "BTC-USDT"

#: Upper bound on a single candle request. Without a cap a client can ask for millions of
#: rows and take the API down by accident.
MAX_CANDLES = 5_000


def _parse_symbol(raw: str) -> Symbol:
    """Parse a symbol supplied as a URL path segment.

    A slashed symbol (``BTC/USDT``) cannot travel in a path segment: percent-encoding it
    does not help, because the server decodes ``%2F`` before routing and then sees an extra
    segment. Clients must use the hyphenated (``BTC-USDT``) or concatenated (``BTCUSDT``)
    form, both of which :meth:`Symbol.parse` accepts and normalises.
    """
    parsed = Symbol.parse(raw)
    assert isinstance(parsed, Symbol)
    return parsed


#: Cache for the stored-series listing.
#:
#: This endpoint is the most expensive read in the API by a wide margin: an ungrouped
#: ``count(*)`` over every candle ever stored, followed by two more queries per series for
#: the first and last bar. On a million-row table that exceeds the database's statement
#: timeout, and because the dashboard used to poll it every five seconds, the cancelled
#: queries piled up until the API stopped answering anything at all — health probe
#: included. The catalogue of stored series changes when a download runs, not between
#: renders, so a minute of staleness costs nothing and bounds the damage.
_series_cache: ResilientCache[list[SymbolSummary]] = ResilientCache(60.0, name="market_series")


async def _load_series(database: Database) -> list[SymbolSummary]:
    """Read every stored ``(symbol, timeframe)`` pair with its extent."""
    async with database.read_session() as session:
        repository = CandleRepository(session)
        series = await repository.available_series()
        summaries: list[SymbolSummary] = []
        for symbol, timeframe, bars in series:
            summaries.append(
                SymbolSummary(
                    symbol=symbol.slashed,
                    timeframe=timeframe,
                    bars=bars,
                    start=await repository.earliest_open_time(symbol, timeframe),
                    end=await repository.latest_open_time(symbol, timeframe),
                )
            )
    return summaries


@router.get("/series", response_model=list[SymbolSummary], summary="List stored series")
async def list_series(database: DatabaseDep) -> list[SymbolSummary]:
    """Every stored ``(symbol, timeframe)`` pair and its bar count.

    Served from a short-lived cache; see :data:`_series_cache`. A failed refresh returns
    the previous listing rather than an error, because an empty series list makes the
    market-data panel look as though nothing has ever been downloaded.
    """
    cached = await _series_cache.get(lambda: _load_series(database))
    return cached.value or []


@router.get("/candles/{symbol}", response_model=CandlesResponse, summary="Fetch stored candles")
async def get_candles(
    symbol: str,
    database: DatabaseDep,
    timeframe: Timeframe = Timeframe.H1,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_CANDLES)] = 500,
) -> CandlesResponse:
    """Read candles from storage, newest ``limit`` bars by default.

    The response carries a ``gaps`` count: a series with holes will still render a chart,
    and a client that does not know it is looking at incomplete data will draw the wrong
    conclusion from it.
    """
    parsed = _parse_symbol(symbol)
    async with database.read_session() as session:
        repository = CandleRepository(session)
        candles = await repository.fetch(
            parsed, timeframe, start=start, end=end, limit=limit, newest_first=True
        )
        if not candles:
            raise NotFoundError(
                f"no stored candles for {parsed} {timeframe.value}",
                symbol=parsed.slashed,
                timeframe=timeframe.value,
            )
        report = await repository.integrity_report(
            parsed, timeframe, start=candles[0].open_time, end=candles[-1].close_time
        )

    return CandlesResponse(
        symbol=parsed.slashed,
        timeframe=timeframe,
        count=len(candles),
        candles=tuple(
            CandleResponse(
                open_time=candle.open_time,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                quote_volume=candle.quote_volume,
                trades=candle.trades,
            )
            for candle in candles
        ),
        gaps=report.missing_bar_count,
    )


@router.get("/ticker/{symbol}", response_model=TickerResponse, summary="Fetch a live ticker")
async def get_ticker(symbol: str, gateway: GatewayDep) -> TickerResponse:
    """Fetch the current best bid/ask from the venue."""
    parsed = _parse_symbol(symbol)
    ticker = await gateway.fetch_ticker(parsed)
    return TickerResponse(
        symbol=parsed.slashed,
        timestamp=ticker.timestamp,
        bid=ticker.bid,
        ask=ticker.ask,
        last=ticker.last,
        spread_pct=ticker.spread_pct,
    )


@router.get(
    "/instruments",
    response_model=list[InstrumentResponse],
    summary="List cached instruments",
)
async def list_instruments(database: DatabaseDep) -> list[InstrumentResponse]:
    """Every active instrument's cached trading rules."""
    async with database.read_session() as session:
        instruments = await InstrumentRepository(session).list_active()
    return [
        InstrumentResponse(
            symbol=instrument.symbol.slashed,
            market_type=instrument.market_type.value,
            price_tick=instrument.price_tick,
            quantity_step=instrument.quantity_step,
            min_quantity=instrument.min_quantity,
            min_notional=instrument.min_notional,
            maker_fee=instrument.maker_fee,
            taker_fee=instrument.taker_fee,
            active=instrument.active,
        )
        for instrument in instruments
    ]


@router.get(
    "/instruments/{symbol}",
    response_model=InstrumentResponse,
    summary="Fetch one instrument",
)
async def get_instrument(symbol: str, gateway: GatewayDep) -> InstrumentResponse:
    """Fetch a symbol's trading rules from the venue."""
    parsed = _parse_symbol(symbol)
    instrument = await gateway.get_instrument(parsed)
    return InstrumentResponse(
        symbol=instrument.symbol.slashed,
        market_type=instrument.market_type.value,
        price_tick=instrument.price_tick,
        quantity_step=instrument.quantity_step,
        min_quantity=instrument.min_quantity,
        min_notional=instrument.min_notional,
        maker_fee=instrument.maker_fee,
        taker_fee=instrument.taker_fee,
        active=instrument.active,
    )
