"""Short-lived caching for dashboard responses.

Three properties matter here, and each of them was learned from a failure:

**The dashboard polls, so the cost of a panel is paid every few seconds.** An endpoint that
re-runs an aggregate on every render multiplies that aggregate by the number of open tabs.
A time-to-live measured in seconds costs the operator nothing — the underlying data moves
on a 15-minute bar — and it bounds what the dashboard can do to the database.

**A slow venue must not become a slow API.** The exchange client retries with backoff and
has no overall deadline, so a single ``fetch_balances`` can legitimately take more than a
minute. Awaited directly from a request handler that turns into a request that never
returns, and because the browser re-polls, the pending requests accumulate until the API
is unusable. Every venue read here is therefore given a hard deadline.

**A failed refresh must not erase a good answer.** When the refresh fails the last good
value is served with its age and the error attached, so the dashboard can say "last
successful update 14:32:07" instead of going blank. Staleness is reported, never hidden.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from quantflow.core.clock import utc_now
from quantflow.core.logging import get_logger

logger = get_logger(__name__)

#: Hard deadline for any single venue read made on behalf of a dashboard request.
#: Comfortably longer than a healthy round trip, far shorter than the exchange client's
#: worst-case retry chain, and shorter than the dashboard's own fetch timeout so the
#: browser receives a real answer rather than aborting.
VENUE_DEADLINE_SECONDS = 8.0


@dataclass(frozen=True, slots=True)
class Cached[T]:
    """A cached value together with everything needed to judge how much to trust it.

    Attributes:
        value: The payload. ``None`` only when no refresh has ever succeeded.
        fetched_at: When the payload was last successfully produced.
        stale: True when the most recent refresh failed and ``value`` is a previous one.
        error: Why the most recent refresh failed, when it did.

    """

    value: T | None
    fetched_at: datetime | None
    stale: bool
    error: str | None
    #: How old the payload may be before it must be reported stale.
    max_age_seconds: float | None = None

    @property
    def age_seconds(self) -> float | None:
        """Seconds since the value was produced, or ``None`` if there is no value."""
        if self.fetched_at is None:
            return None
        return max(0.0, (utc_now() - self.fetched_at).total_seconds())


class ResilientCache[T]:
    """A value refreshed at most once per TTL, that survives a failing refresh.

    Refreshes are single-flight: concurrent callers arriving on an expired entry wait for
    one in-flight computation rather than each starting their own. Without that, the poll
    interval and the request rate multiply instead of adding.
    """

    __slots__ = (
        "_error",
        "_expires_at",
        "_fetched_at",
        "_lock",
        "_max_age",
        "_name",
        "_ttl",
        "_value",
    )

    def __init__(
        self, ttl_seconds: float, *, name: str, max_age_seconds: float | None = None
    ) -> None:
        """Create a cache holding one value for ``ttl_seconds``.

        Args:
            ttl_seconds: How long a successful value is served before a refresh.
            name: Identifier used in log lines when a refresh fails.
            max_age_seconds: How old the *payload* may be before it is reported stale,
                whatever the refresh schedule is doing. Without it, a failing dependency
                marks the value stale once and then — because each failure pushes the
                backoff window forward — every subsequent caller passes the freshness
                check and is handed the same ageing value labelled fresh. That is how a
                2h40m-old venue read came to be published beside ``stale: False``.
                Defaults to twice the TTL, so a value is never reported fresh after
                missing two refreshes.

        """
        self._ttl = ttl_seconds
        self._max_age = max_age_seconds if max_age_seconds is not None else max(1.0, self._ttl * 2)
        self._name = name
        self._value: T | None = None
        self._fetched_at: datetime | None = None
        self._expires_at = 0.0
        self._error: str | None = None
        self._lock = asyncio.Lock()

    async def get(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        deadline_seconds: float | None = None,
    ) -> Cached[T]:
        """Return the cached value, refreshing it if the TTL has elapsed.

        A refresh that raises — including one that exceeds ``deadline_seconds`` — never
        propagates. The previous value is returned marked stale, so one unreachable
        dependency degrades a panel instead of failing the request.

        Args:
            factory: Produces a fresh value.
            deadline_seconds: Hard limit on ``factory``. ``None`` means no limit, which is
                only appropriate for work that is already bounded, such as an indexed query
                under the database's own statement timeout.

        Returns:
            The value with its age, staleness and last error.

        """
        if self._is_fresh():
            return self._snapshot(stale=False)

        async with self._lock:
            # Re-checked under the lock: a caller that queued behind a refresh should use
            # its result rather than immediately starting another.
            if self._is_fresh():
                return self._snapshot(stale=False)

            try:
                value = (
                    await asyncio.wait_for(factory(), timeout=deadline_seconds)
                    if deadline_seconds is not None
                    else await factory()
                )
            except TimeoutError:
                self._error = f"timed out after {deadline_seconds:.0f}s"
                logger.warning("dashboard.cache_timeout", cache=self._name)
                # Backed off rather than retried on the very next request: a dependency
                # that is timing out is made worse by a retry storm, and the operator is
                # already being told the panel is stale.
                self._expires_at = time.monotonic() + self._ttl
                return self._snapshot(stale=True)
            except Exception as exc:
                self._error = str(exc) or exc.__class__.__name__
                logger.warning("dashboard.cache_refresh_failed", cache=self._name, error=str(exc))
                self._expires_at = time.monotonic() + self._ttl
                return self._snapshot(stale=True)

            self._value = value
            self._fetched_at = utc_now()
            self._expires_at = time.monotonic() + self._ttl
            self._error = None
            return self._snapshot(stale=False)

    def _is_fresh(self) -> bool:
        """Whether a value exists, its TTL has not elapsed, and the payload is not old.

        The age check is what a backoff window cannot be allowed to override: skipping a
        refresh is a decision about load, never evidence that the data is current.
        """
        if self._fetched_at is None or time.monotonic() >= self._expires_at:
            return False
        return not self._payload_is_old()

    def _payload_is_old(self) -> bool:
        """Whether the held value has aged past what may be presented as current."""
        if self._fetched_at is None:
            return False
        return (utc_now() - self._fetched_at).total_seconds() > self._max_age

    def _snapshot(self, *, stale: bool) -> Cached[T]:
        """Package the current state for a caller."""
        return Cached(
            value=self._value,
            fetched_at=self._fetched_at,
            # Stale if the caller says so *or* the payload has simply aged out. The second
            # term is the one that matters: it cannot be bypassed by any refresh-scheduling
            # decision, so no code path can publish an old value as current.
            stale=(stale or self._payload_is_old()) and self._value is not None,
            error=self._error,
            max_age_seconds=self._max_age,
        )


def freshness_block(cached: Cached[object], *, label: str) -> dict[str, object]:
    """Describe a cached value's provenance for inclusion in a response.

    Every dashboard payload carries one of these, because a number with no timestamp
    cannot be distinguished from a number that stopped updating an hour ago.

    Args:
        cached: The cached value to describe.
        label: What the value is, echoed back so the client can name the source.

    Returns:
        A JSON-safe mapping with the source label, timestamp, age and any error.

    """
    return {
        "source": label,
        "fetched_at": cached.fetched_at.isoformat() if cached.fetched_at else None,
        "age_seconds": cached.age_seconds,
        "stale": cached.stale,
        "max_age_seconds": cached.max_age_seconds,
        "error": cached.error,
        # Present *and* current. A payload that has aged out is not available data, it is
        # a record of what used to be true, and a panel that treats the two the same is
        # how an operator ends up reading a two-hour-old book as the live one.
        "available": cached.value is not None and not cached.stale,
    }


__all__ = [
    "VENUE_DEADLINE_SECONDS",
    "Cached",
    "ResilientCache",
    "freshness_block",
]
