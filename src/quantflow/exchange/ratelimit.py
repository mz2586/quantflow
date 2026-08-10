"""Client-side rate limiting and retry policy.

Bybit V5 publishes per-endpoint request limits and answers a breach with HTTP 403 (an IP
ban) if you keep going. Throttling on our side is therefore not politeness — it is the
difference between a working system and a banned IP mid-position.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from quantflow.core.clock import Clock, SystemClock
from quantflow.core.errors import (
    ExchangeConnectionError,
    ExchangeTimeoutError,
    RateLimitError,
)
from quantflow.core.logging import get_logger

logger = get_logger(__name__)

#: Tolerance when comparing accumulated tokens against a request. Refills are computed
#: as `elapsed * rate` in binary floating point, so a bucket that should hold exactly 1.0
#: token can hold 0.9999999999999999 instead. Without this tolerance the comparison fails,
#: the computed deficit is ~1e-16, and the acquire loop spins forever waiting for a
#: quantity it already has.
TOKEN_EPSILON = 1e-9

#: Floor on the wait between retries, so a near-zero deficit can never become a busy loop.
MIN_WAIT_SECONDS = 1e-4

#: Errors worth retrying: transient network or venue-side conditions. Business rejections
#: (insufficient funds, invalid symbol, bad precision) are deliberately absent — retrying
#: those just burns rate-limit budget and can double-submit.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    ExchangeConnectionError,
    ExchangeTimeoutError,
    RateLimitError,
)


class TokenBucket:
    """Async token bucket.

    Tokens refill continuously at ``rate`` per second up to ``capacity``, so a burst of
    requests is allowed after an idle period but the sustained rate stays bounded.
    """

    __slots__ = ("_capacity", "_clock", "_lock", "_rate", "_tokens", "_updated_at")

    def __init__(self, rate: float, capacity: int, *, clock: Clock | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._rate = rate
        self._capacity = capacity
        self._clock = clock or SystemClock()
        self._tokens = float(capacity)
        self._updated_at = self._clock.monotonic()
        self._lock = asyncio.Lock()

    @property
    def available(self) -> float:
        """Tokens available right now, without consuming any."""
        return min(self._capacity, self._tokens + self._elapsed_refill())

    def _elapsed_refill(self) -> float:
        return (self._clock.monotonic() - self._updated_at) * self._rate

    async def acquire(self, tokens: float = 1.0) -> float:
        """Consume ``tokens``, waiting if necessary.

        Returns:
            The number of seconds spent waiting.

        Raises:
            ValueError: if ``tokens`` exceeds the bucket capacity, which could never
                be satisfied and would otherwise hang forever.

        """
        if tokens > self._capacity:
            raise ValueError(
                f"cannot acquire {tokens} tokens from a bucket of capacity {self._capacity}"
            )
        waited = 0.0
        while True:
            async with self._lock:
                now = self._clock.monotonic()
                self._tokens = min(
                    float(self._capacity), self._tokens + (now - self._updated_at) * self._rate
                )
                self._updated_at = now
                if self._tokens + TOKEN_EPSILON >= tokens:
                    self._tokens = max(0.0, self._tokens - tokens)
                    return waited
                deficit = tokens - self._tokens
                delay = max(deficit / self._rate, MIN_WAIT_SECONDS)
            waited += delay
            await self._clock.sleep(delay)


def backoff_delay(attempt: int, *, base: float, cap: float = 60.0, jitter: bool = True) -> float:
    """Exponential backoff with full jitter.

    Full jitter (a uniform draw in ``[0, backoff]``) rather than a fixed exponential:
    without it, many clients that fail together retry together, and the venue sees the
    same thundering herd on every attempt.
    """
    ceiling = min(cap, base * (2**attempt))
    return random.uniform(0, ceiling) if jitter else ceiling  # noqa: S311 — not cryptographic


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 60.0,
    retryable: tuple[type[Exception], ...] = RETRYABLE_ERRORS,
    clock: Clock | None = None,
    description: str = "exchange call",
) -> T:
    """Run ``operation`` with exponential backoff on retryable errors.

    A :class:`RateLimitError` carrying ``retry_after`` honours that value instead of the
    computed backoff — the venue knows better than we do when it will accept traffic again.

    Raises:
        Exception: the last error, once retries are exhausted.

    """
    active_clock = clock or SystemClock()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except retryable as exc:
            last_error = exc
            if attempt == max_retries:
                break
            retry_after = getattr(exc, "details", {}).get("retry_after")
            delay = (
                float(retry_after)
                if retry_after is not None
                else backoff_delay(attempt, base=base_delay, cap=max_delay)
            )
            logger.warning(
                "exchange.retry",
                operation=description,
                attempt=attempt + 1,
                max_retries=max_retries,
                delay_seconds=round(delay, 3),
                error=str(exc),
            )
            await active_clock.sleep(delay)

    assert last_error is not None  # loop always assigns before breaking
    logger.error("exchange.retries_exhausted", operation=description, error=str(last_error))
    raise last_error


class RateLimiter:
    """Named token buckets, so heavyweight endpoints can be throttled independently.

    Venues weight endpoints differently (a deep order book costs far more than a
    ticker), so a single global bucket would either throttle cheap calls needlessly or fail
    to protect against expensive ones.
    """

    __slots__ = ("_buckets", "_capacity", "_clock", "_default", "_rate")

    def __init__(self, rate: float, capacity: int, *, clock: Clock | None = None) -> None:
        self._rate = rate
        self._capacity = capacity
        self._clock = clock or SystemClock()
        self._default = TokenBucket(rate, capacity, clock=self._clock)
        self._buckets: dict[str, TokenBucket] = {}

    def bucket(
        self, name: str, *, rate: float | None = None, capacity: int | None = None
    ) -> TokenBucket:
        """Get or create a named bucket."""
        existing = self._buckets.get(name)
        if existing is None:
            existing = TokenBucket(
                rate or self._rate, capacity or self._capacity, clock=self._clock
            )
            self._buckets[name] = existing
        return existing

    async def acquire(self, weight: float = 1.0, *, bucket: str | None = None) -> float:
        """Consume ``weight`` from the default or a named bucket."""
        target = self._default if bucket is None else self.bucket(bucket)
        return await target.acquire(weight)
