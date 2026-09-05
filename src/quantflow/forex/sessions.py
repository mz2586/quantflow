"""FX trading-session calendar.

Crypto is 24/7; FX is not. The spot market opens at the Sydney open on Sunday and closes
at the New York close on Friday, and inside the week it moves through three regional
sessions whose overlaps carry most of the liquidity. A strategy that assumes continuous
trading will size against a stale Saturday quote and place orders into a closed book.

Two independent layers of truth are modelled here:

* the **week rule** — Sunday 21:00 UTC to Friday 21:00 UTC — which no venue overrides; and
* **venue-supplied session windows** (MT5's ``symbol_info_sessions_trade``, OANDA's
  instrument trading hours), which narrow the week further for a specific symbol.

Session boundaries are expressed in UTC on the hour. They are a *classification* aid for
strategy logic, not a settlement calendar: they deliberately ignore daylight-saving drift
of an hour, because no sizing or execution decision here depends on that hour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from typing import Final

from quantflow.core.errors import ValidationError

DAYS_IN_WEEK: Final = 7


class TradingSession(StrEnum):
    """Which regional FX session a moment falls in."""

    CLOSED = "closed"
    ASIAN = "asian"
    ASIAN_LONDON_OVERLAP = "asian_london_overlap"
    LONDON = "london"
    LONDON_NEW_YORK_OVERLAP = "london_new_york_overlap"
    NEW_YORK = "new_york"


#: Half-open ``[start_hour, end_hour)`` segments in UTC covering a full trading day.
#: The Asian session wraps midnight, so it appears twice — 21:00-24:00 and 00:00-07:00.
SESSION_SEGMENTS: Final[tuple[tuple[int, int, TradingSession], ...]] = (
    (0, 7, TradingSession.ASIAN),
    (7, 9, TradingSession.ASIAN_LONDON_OVERLAP),
    (9, 12, TradingSession.LONDON),
    (12, 16, TradingSession.LONDON_NEW_YORK_OVERLAP),
    (16, 21, TradingSession.NEW_YORK),
    (21, 24, TradingSession.ASIAN),
)


def require_utc(moment: datetime, *, field: str = "moment") -> datetime:
    """Normalise an aware datetime to UTC, rejecting naive ones.

    Raises:
        ValidationError: if ``moment`` carries no timezone.

    """
    if moment.tzinfo is None:
        raise ValidationError(f"{field} must be timezone-aware", field=field)
    return moment.astimezone(UTC)


def require_weekday(weekday: int, *, field: str = "weekday") -> int:
    """Validate a Python weekday (``0`` = Monday ... ``6`` = Sunday)."""
    if not 0 <= weekday < DAYS_IN_WEEK:
        raise ValidationError(f"{field} must be 0-6 (Monday-Sunday), got {weekday}", field=field)
    return weekday


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """One venue-declared trading window on a given weekday.

    ``end`` at or before ``start`` means the window wraps past midnight, which the venue
    still reports against the *opening* weekday — so a Monday 22:00-03:00 window contains
    both Monday 23:00 and Monday 01:00 as the venue labels them.
    """

    weekday: int
    start: time
    end: time

    def __post_init__(self) -> None:
        """Validate the weekday."""
        require_weekday(self.weekday)

    @property
    def crosses_midnight(self) -> bool:
        """Whether the window wraps past midnight."""
        return self.end <= self.start

    def contains(self, moment: datetime) -> bool:
        """Whether ``moment`` falls inside this window."""
        moment = require_utc(moment)
        if moment.weekday() != self.weekday:
            return False
        clock_time = moment.time()
        if self.crosses_midnight:
            return clock_time >= self.start or clock_time < self.end
        return self.start <= clock_time < self.end


@dataclass(frozen=True, slots=True)
class SessionClock:
    """The FX trading week, plus regional session classification.

    Defaults describe the spot FX week: open Sunday 21:00 UTC, close Friday 21:00 UTC.
    Both ends are configurable because brokers shift them by an hour across daylight-saving
    changes, and because the triple-swap day is not universally Wednesday.
    """

    week_open_weekday: int = 6
    week_open_time: time = time(21, 0)
    week_close_weekday: int = 4
    week_close_time: time = time(21, 0)
    friday_close_buffer: timedelta = timedelta(minutes=30)

    def __post_init__(self) -> None:
        """Validate the configured week boundaries."""
        require_weekday(self.week_open_weekday, field="week_open_weekday")
        require_weekday(self.week_close_weekday, field="week_close_weekday")
        if self.friday_close_buffer < timedelta(0):
            raise ValidationError("friday_close_buffer must not be negative")

    # ----------------------------------------------------------------- week rule
    def _closed_weekdays(self) -> frozenset[int]:
        """Weekdays lying wholly between the weekly close and the weekly open."""
        days: set[int] = set()
        day = (self.week_close_weekday + 1) % DAYS_IN_WEEK
        while day != self.week_open_weekday:
            days.add(day)
            day = (day + 1) % DAYS_IN_WEEK
        return frozenset(days)

    def is_weekend(self, moment: datetime) -> bool:
        """Whether the whole FX market is shut at ``moment``."""
        moment = require_utc(moment)
        weekday = moment.weekday()
        clock_time = moment.time()
        if weekday == self.week_close_weekday and clock_time >= self.week_close_time:
            return True
        if weekday == self.week_open_weekday:
            return clock_time < self.week_open_time
        return weekday in self._closed_weekdays()

    def is_open(self, moment: datetime, sessions: Sequence[SessionWindow] | None = None) -> bool:
        """Whether trading is possible at ``moment``.

        Venue-supplied ``sessions`` can only *narrow* the week — they can never reopen the
        weekend, because the underlying market is genuinely shut.
        """
        if self.is_weekend(moment):
            return False
        if not sessions:
            return True
        return any(window.contains(moment) for window in sessions)

    def next_open(self, moment: datetime) -> datetime:
        """The next instant trading is possible — ``moment`` itself if already open."""
        moment = require_utc(moment)
        if not self.is_weekend(moment):
            return moment
        for offset in range(DAYS_IN_WEEK + 1):
            candidate = datetime.combine(
                moment.date() + timedelta(days=offset), self.week_open_time, tzinfo=UTC
            )
            if candidate >= moment and candidate.weekday() == self.week_open_weekday:
                return candidate
        raise ValidationError("could not find the next weekly open")  # pragma: no cover

    def time_until_close(self, moment: datetime) -> timedelta | None:
        """Time left until the weekly close, or ``None`` if the market is already shut."""
        moment = require_utc(moment)
        if self.is_weekend(moment):
            return None
        for offset in range(DAYS_IN_WEEK + 1):
            candidate = datetime.combine(
                moment.date() + timedelta(days=offset), self.week_close_time, tzinfo=UTC
            )
            if candidate > moment and candidate.weekday() == self.week_close_weekday:
                return candidate - moment
        raise ValidationError("could not find the next weekly close")  # pragma: no cover

    def in_friday_close_window(self, moment: datetime) -> bool:
        """Whether ``moment`` is inside the run-up to the weekly close.

        Used to stop opening positions that would be carried over the weekend gap, where
        the reopening print can jump straight through a stop.
        """
        remaining = self.time_until_close(moment)
        return remaining is not None and remaining <= self.friday_close_buffer

    # -------------------------------------------------------------- classification
    def classify(self, moment: datetime) -> TradingSession:
        """Which regional session ``moment`` belongs to."""
        moment = require_utc(moment)
        if self.is_weekend(moment):
            return TradingSession.CLOSED
        hour = moment.hour
        for start_hour, end_hour, session in SESSION_SEGMENTS:
            if start_hour <= hour < end_hour:
                return session
        raise ValidationError(f"unclassifiable hour: {hour}")  # pragma: no cover

    def is_triple_swap_day(self, moment: datetime, triple_swap_weekday: int) -> bool:
        """Whether the rollover on ``moment``'s day carries the triple swap charge."""
        require_weekday(triple_swap_weekday, field="triple_swap_weekday")
        return require_utc(moment).weekday() == triple_swap_weekday
