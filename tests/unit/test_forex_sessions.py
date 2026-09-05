"""FX trading-session calendar: classification, weekend closure, Friday close."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import pytest

from quantflow.core.errors import ValidationError
from quantflow.forex.sessions import (
    SessionClock,
    SessionWindow,
    TradingSession,
)

# 2026-08-10 is a Monday; 2026-08-12 Wednesday; 2026-08-14 Friday; 2026-08-16 Sunday.
MONDAY = 10
WEDNESDAY = 12
FRIDAY = 14
SATURDAY = 15
SUNDAY = 16


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=UTC)


@pytest.fixture
def clock() -> SessionClock:
    return SessionClock()


class TestSessionWindow:
    def test_contains_within_day(self) -> None:
        window = SessionWindow(weekday=0, start=time(8, 0), end=time(17, 0))
        assert window.contains(at(MONDAY, 9))
        assert not window.contains(at(MONDAY, 18))

    def test_wrong_weekday_never_contained(self) -> None:
        window = SessionWindow(weekday=0, start=time(8, 0), end=time(17, 0))
        assert not window.contains(at(WEDNESDAY, 9))

    def test_window_crossing_midnight(self) -> None:
        window = SessionWindow(weekday=0, start=time(22, 0), end=time(3, 0))
        assert window.crosses_midnight
        assert window.contains(at(MONDAY, 23))
        assert window.contains(at(MONDAY, 1))
        assert not window.contains(at(MONDAY, 12))

    def test_rejects_bad_weekday(self) -> None:
        with pytest.raises(ValidationError):
            SessionWindow(weekday=7, start=time(8, 0), end=time(17, 0))

    def test_rejects_naive_datetime(self) -> None:
        window = SessionWindow(weekday=0, start=time(8, 0), end=time(17, 0))
        with pytest.raises(ValidationError):
            window.contains(datetime(2026, 8, 10, 9, 0))  # noqa: DTZ001


class TestClassification:
    @pytest.mark.parametrize(
        ("day", "hour", "expected"),
        [
            (MONDAY, 2, TradingSession.ASIAN),
            (MONDAY, 6, TradingSession.ASIAN),
            (MONDAY, 8, TradingSession.ASIAN_LONDON_OVERLAP),
            (MONDAY, 10, TradingSession.LONDON),
            (MONDAY, 14, TradingSession.LONDON_NEW_YORK_OVERLAP),
            (MONDAY, 18, TradingSession.NEW_YORK),
            (MONDAY, 22, TradingSession.ASIAN),
        ],
    )
    def test_each_session(
        self, clock: SessionClock, day: int, hour: int, expected: TradingSession
    ) -> None:
        assert clock.classify(at(day, hour)) is expected

    def test_every_open_hour_of_the_week_is_classified(self, clock: SessionClock) -> None:
        moment = at(SUNDAY, 21)
        for _ in range(24 * 5):
            assert clock.classify(moment) is not TradingSession.CLOSED
            moment += timedelta(hours=1)

    def test_boundaries_are_half_open(self, clock: SessionClock) -> None:
        assert clock.classify(at(MONDAY, 7)) is TradingSession.ASIAN_LONDON_OVERLAP
        assert clock.classify(at(MONDAY, 9)) is TradingSession.LONDON
        assert clock.classify(at(MONDAY, 12)) is TradingSession.LONDON_NEW_YORK_OVERLAP
        assert clock.classify(at(MONDAY, 16)) is TradingSession.NEW_YORK

    def test_naive_datetime_rejected(self, clock: SessionClock) -> None:
        with pytest.raises(ValidationError):
            clock.classify(datetime(2026, 8, 10, 9, 0))  # noqa: DTZ001


class TestWeekend:
    def test_saturday_is_closed(self, clock: SessionClock) -> None:
        assert clock.classify(at(SATURDAY, 12)) is TradingSession.CLOSED
        assert not clock.is_open(at(SATURDAY, 12))

    def test_friday_after_close_is_closed(self, clock: SessionClock) -> None:
        assert not clock.is_open(at(FRIDAY, 21, 1))
        assert clock.is_open(at(FRIDAY, 20, 59))

    def test_friday_close_instant_is_closed(self, clock: SessionClock) -> None:
        assert not clock.is_open(at(FRIDAY, 21, 0))

    def test_sunday_open_instant_is_open(self, clock: SessionClock) -> None:
        assert clock.is_open(at(SUNDAY, 21, 0))
        assert not clock.is_open(at(SUNDAY, 20, 59))

    def test_never_assumes_crypto_style_always_open(self, clock: SessionClock) -> None:
        closed = [m for m in (at(SATURDAY, h) for h in range(24)) if not clock.is_open(m)]
        assert len(closed) == 24

    def test_next_open_from_weekend(self, clock: SessionClock) -> None:
        assert clock.next_open(at(SATURDAY, 12)) == at(SUNDAY, 21)

    def test_next_open_when_already_open_is_now(self, clock: SessionClock) -> None:
        moment = at(MONDAY, 10)
        assert clock.next_open(moment) == moment

    def test_time_until_close(self, clock: SessionClock) -> None:
        assert clock.time_until_close(at(FRIDAY, 20)) == timedelta(hours=1)
        assert clock.time_until_close(at(SATURDAY, 12)) is None


class TestFridayCloseWindow:
    def test_inside_window(self, clock: SessionClock) -> None:
        assert clock.in_friday_close_window(at(FRIDAY, 20, 45))

    def test_outside_window(self, clock: SessionClock) -> None:
        assert not clock.in_friday_close_window(at(FRIDAY, 19, 0))
        assert not clock.in_friday_close_window(at(MONDAY, 20, 45))

    def test_configurable_buffer(self) -> None:
        clock = SessionClock(friday_close_buffer=timedelta(hours=4))
        assert clock.in_friday_close_window(at(FRIDAY, 18, 0))


class TestVenueSuppliedSessions:
    def test_venue_sessions_take_priority(self, clock: SessionClock) -> None:
        windows = (SessionWindow(weekday=0, start=time(8, 0), end=time(17, 0)),)
        assert clock.is_open(at(MONDAY, 9), sessions=windows)
        assert not clock.is_open(at(MONDAY, 19), sessions=windows)

    def test_venue_sessions_cannot_reopen_the_weekend(self, clock: SessionClock) -> None:
        windows = (SessionWindow(weekday=5, start=time(0, 0), end=time(23, 59)),)
        assert not clock.is_open(at(SATURDAY, 12), sessions=windows)

    def test_empty_sessions_falls_back_to_week_rule(self, clock: SessionClock) -> None:
        assert clock.is_open(at(MONDAY, 9), sessions=())


class TestTripleSwap:
    def test_wednesday_is_the_triple_swap_day_by_default(self, clock: SessionClock) -> None:
        assert clock.is_triple_swap_day(at(WEDNESDAY, 12), triple_swap_weekday=2)
        assert not clock.is_triple_swap_day(at(MONDAY, 12), triple_swap_weekday=2)

    def test_configurable_triple_swap_day(self, clock: SessionClock) -> None:
        assert clock.is_triple_swap_day(at(FRIDAY, 12), triple_swap_weekday=4)
