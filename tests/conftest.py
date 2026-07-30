"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quantflow.core.clock import FrozenClock
from quantflow.core.config import Environment, Settings, reset_settings_cache
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.market import Candle

REFERENCE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip ambient QF_* variables so tests never read a developer's local `.env`."""
    for key in list(os.environ):
        if key.startswith("QF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QF_ENV", "test")
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings() -> Settings:
    """Default test settings, built without reading a `.env` file."""
    return Settings(_env_file=None, env=Environment.TEST)


@pytest.fixture
def clock() -> FrozenClock:
    """A clock frozen at the reference time."""
    return FrozenClock(REFERENCE_TIME)


@pytest.fixture
def btc() -> Symbol:
    """BTC/USDT."""
    return Symbol(base="BTC", quote="USDT")


@pytest.fixture
def eth() -> Symbol:
    """ETH/USDT."""
    return Symbol(base="ETH", quote="USDT")


@pytest.fixture
def btc_instrument(btc: Symbol) -> Instrument:
    """A BTC/USDT instrument with realistic Binance spot rules."""
    return Instrument(
        symbol=btc,
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.00001"),
        min_quantity=Decimal("0.00001"),
        min_notional=Decimal("5"),
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
    )


def make_candle(
    symbol: Symbol,
    *,
    open_time: datetime,
    close: Decimal | str | int,
    open_price: Decimal | str | int | None = None,
    high: Decimal | str | int | None = None,
    low: Decimal | str | int | None = None,
    volume: Decimal | str | int = 10,
    timeframe: Timeframe = Timeframe.H1,
) -> Candle:
    """Build a valid candle, deriving any unspecified OHLC values from ``close``."""
    close_value = Decimal(str(close))
    open_value = Decimal(str(open_price)) if open_price is not None else close_value
    high_value = Decimal(str(high)) if high is not None else max(open_value, close_value)
    low_value = Decimal(str(low)) if low is not None else min(open_value, close_value)
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open_time=open_time,
        open=open_value,
        high=high_value,
        low=low_value,
        close=close_value,
        volume=Decimal(str(volume)),
        quote_volume=Decimal(str(volume)) * close_value,
    )


def make_candles(
    symbol: Symbol,
    closes: list[Decimal | str | int],
    *,
    start: datetime = REFERENCE_TIME,
    timeframe: Timeframe = Timeframe.H1,
) -> list[Candle]:
    """Build a contiguous candle series from a list of close prices."""
    return [
        make_candle(
            symbol,
            open_time=start + timeframe.delta * index,
            close=close,
            timeframe=timeframe,
        )
        for index, close in enumerate(closes)
    ]
