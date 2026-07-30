"""Time abstraction.

Trading code must never call :func:`datetime.now` directly: backtests replay historical
time, tests need determinism, and live code needs exchange-synchronised time. Everything
goes through a :class:`Clock`.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@runtime_checkable
class Clock(Protocol):
    """Source of current time and of sleeping."""

    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime."""
        ...

    def timestamp_ms(self) -> int:
        """Current time as milliseconds since the Unix epoch."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds, suitable for measuring durations."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend for ``seconds``."""
        ...


class SystemClock:
    """Wall-clock implementation backed by the operating system."""

    __slots__ = ()

    def now(self) -> datetime:
        """Current UTC time."""
        return datetime.now(UTC)

    def timestamp_ms(self) -> int:
        """Current epoch milliseconds."""
        return int(time.time() * 1000)

    def monotonic(self) -> float:
        """Monotonic seconds."""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """Await ``asyncio.sleep``."""
        if seconds > 0:
            await asyncio.sleep(seconds)


class FrozenClock:
    """Manually advanced clock for tests and for backtest replay.

    ``sleep`` advances virtual time instead of blocking, so an engine's timing logic can
    be exercised at full speed.
    """

    __slots__ = ("_monotonic", "_now")

    def __init__(self, start: datetime | None = None) -> None:
        self._now = _require_utc(start) if start is not None else EPOCH
        self._monotonic = 0.0

    def now(self) -> datetime:
        """Current virtual time."""
        return self._now

    def timestamp_ms(self) -> int:
        """Current virtual time in epoch milliseconds."""
        return to_epoch_ms(self._now)

    def monotonic(self) -> float:
        """Virtual monotonic seconds."""
        return self._monotonic

    async def sleep(self, seconds: float) -> None:
        """Advance virtual time by ``seconds`` and yield to the event loop once."""
        self.advance(seconds=seconds)
        await asyncio.sleep(0)

    def advance(self, *, seconds: float = 0.0, delta: timedelta | None = None) -> datetime:
        """Move virtual time forward. Returns the new time."""
        step = delta if delta is not None else timedelta(seconds=seconds)
        if step < timedelta(0):
            raise ValueError("FrozenClock cannot move backwards")
        self._now += step
        self._monotonic += step.total_seconds()
        return self._now

    def set(self, moment: datetime) -> None:
        """Jump virtual time to ``moment`` (must be UTC-aware and not in the past)."""
        target = _require_utc(moment)
        if target < self._now:
            raise ValueError("FrozenClock cannot move backwards")
        self._monotonic += (target - self._now).total_seconds()
        self._now = target


def _require_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("naive datetimes are not permitted; use timezone-aware UTC")
    return moment.astimezone(UTC)


def to_epoch_ms(moment: datetime) -> int:
    """Convert a timezone-aware datetime to epoch milliseconds."""
    return int(_require_utc(moment).timestamp() * 1000)


def from_epoch_ms(milliseconds: int) -> datetime:
    """Convert epoch milliseconds to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)


def utc_now() -> datetime:
    """Convenience wall-clock helper for non-trading code paths (logs, metadata)."""
    return datetime.now(UTC)


def floor_to_interval(moment: datetime, interval: timedelta) -> datetime:
    """Floor ``moment`` to the previous multiple of ``interval`` since the epoch."""
    if interval <= timedelta(0):
        raise ValueError("interval must be positive")
    elapsed = _require_utc(moment) - EPOCH
    steps = elapsed // interval
    return EPOCH + steps * interval


def start_of_utc_day(moment: datetime) -> datetime:
    """Midnight UTC on the day containing ``moment``."""
    aware = _require_utc(moment)
    return aware.replace(hour=0, minute=0, second=0, microsecond=0)
