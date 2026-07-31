"""Token bucket and retry policy."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from quantflow.core.clock import FrozenClock
from quantflow.core.errors import (
    ExchangeConnectionError,
    InsufficientFundsError,
    RateLimitError,
)
from quantflow.exchange.ratelimit import (
    RateLimiter,
    TokenBucket,
    backoff_delay,
    retry_async,
)
from tests.conftest import REFERENCE_TIME


class TestTokenBucket:
    async def test_burst_up_to_capacity_is_immediate(self, clock: FrozenClock) -> None:
        bucket = TokenBucket(rate=1.0, capacity=5, clock=clock)
        for _ in range(5):
            assert await bucket.acquire() == 0.0
        assert clock.monotonic() == 0.0

    async def test_waits_once_the_bucket_is_empty(self, clock: FrozenClock) -> None:
        bucket = TokenBucket(rate=10.0, capacity=2, clock=clock)
        await bucket.acquire()
        await bucket.acquire()
        waited = await bucket.acquire()
        assert waited == pytest.approx(0.1)  # 1 token at 10/s
        assert clock.monotonic() == pytest.approx(0.1)

    async def test_refills_over_time(self, clock: FrozenClock) -> None:
        bucket = TokenBucket(rate=10.0, capacity=10, clock=clock)
        for _ in range(10):
            await bucket.acquire()
        clock.advance(seconds=0.5)
        assert bucket.available == pytest.approx(5.0)
        assert await bucket.acquire(5) == 0.0

    async def test_refill_is_capped_at_capacity(self, clock: FrozenClock) -> None:
        bucket = TokenBucket(rate=100.0, capacity=3, clock=clock)
        clock.advance(seconds=60)
        assert bucket.available == 3.0

    async def test_weighted_acquire(self, clock: FrozenClock) -> None:
        bucket = TokenBucket(rate=10.0, capacity=10, clock=clock)
        assert await bucket.acquire(10) == 0.0
        assert await bucket.acquire(5) == pytest.approx(0.5)

    async def test_request_larger_than_capacity_is_rejected(self, clock: FrozenClock) -> None:
        # Would otherwise wait forever, since the bucket can never hold that many tokens.
        bucket = TokenBucket(rate=1.0, capacity=5, clock=clock)
        with pytest.raises(ValueError, match="capacity"):
            await bucket.acquire(6)

    @pytest.mark.parametrize(("rate", "capacity"), [(0, 5), (-1, 5), (1, 0)])
    def test_rejects_invalid_configuration(self, rate: float, capacity: int) -> None:
        with pytest.raises(ValueError, match="must be"):
            TokenBucket(rate=rate, capacity=capacity)

    async def test_concurrent_acquisition_is_serialised(self, clock: FrozenClock) -> None:
        bucket = TokenBucket(rate=100.0, capacity=1, clock=clock)
        results = await asyncio.gather(*(bucket.acquire() for _ in range(5)))
        assert len(results) == 5
        # Only the first request is free; the other four must be spread across time at the
        # configured rate (4 tokens at 100/s = 40ms).
        assert clock.monotonic() >= 0.04 - 1e-9

    async def test_float_drift_cannot_livelock_the_loop(self, clock: FrozenClock) -> None:
        # Refills accumulate float error, so a bucket that should hold exactly 1.0 token
        # can hold 0.9999999999999999. Without an epsilon the acquire loop spins forever.
        bucket = TokenBucket(rate=100.0, capacity=1, clock=clock)
        for _ in range(50):
            await asyncio.wait_for(bucket.acquire(), timeout=2)


class TestBackoff:
    def test_grows_exponentially_without_jitter(self) -> None:
        delays = [backoff_delay(attempt, base=1.0, jitter=False) for attempt in range(4)]
        assert delays == [1.0, 2.0, 4.0, 8.0]

    def test_is_capped(self) -> None:
        assert backoff_delay(20, base=1.0, cap=30.0, jitter=False) == 30.0

    def test_jitter_stays_within_the_ceiling(self) -> None:
        # Full jitter prevents synchronised clients from retrying in lockstep.
        for attempt in range(5):
            ceiling = min(60.0, 1.0 * 2**attempt)
            for _ in range(20):
                assert 0 <= backoff_delay(attempt, base=1.0) <= ceiling


class TestRetryAsync:
    async def test_returns_on_first_success(self, clock: FrozenClock) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        assert await retry_async(operation, clock=clock) == "ok"
        assert calls == 1

    async def test_retries_then_succeeds(self, clock: FrozenClock) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ExchangeConnectionError("network blip")
            return "ok"

        assert await retry_async(operation, clock=clock, base_delay=0.1) == "ok"
        assert calls == 3

    async def test_raises_after_exhausting_retries(self, clock: FrozenClock) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise ExchangeConnectionError("still down")

        with pytest.raises(ExchangeConnectionError, match="still down"):
            await retry_async(operation, max_retries=2, clock=clock, base_delay=0.01)
        assert calls == 3  # the initial attempt plus two retries

    async def test_business_rejections_are_not_retried(self, clock: FrozenClock) -> None:
        # Retrying "insufficient funds" burns rate-limit budget and risks double-submitting.
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise InsufficientFundsError("balance too low")

        with pytest.raises(InsufficientFundsError):
            await retry_async(operation, max_retries=3, clock=clock)
        assert calls == 1

    async def test_honours_retry_after_from_the_venue(self, clock: FrozenClock) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RateLimitError("slow down", retry_after=7.5)
            return "ok"

        assert await retry_async(operation, clock=clock, base_delay=0.01) == "ok"
        # The venue's own hint wins over the computed backoff.
        assert clock.now() == REFERENCE_TIME + timedelta(seconds=7.5)

    async def test_zero_retries_propagates_immediately(self, clock: FrozenClock) -> None:
        async def operation() -> str:
            raise ExchangeConnectionError("down")

        with pytest.raises(ExchangeConnectionError):
            await retry_async(operation, max_retries=0, clock=clock)


class TestRateLimiter:
    async def test_named_buckets_are_independent(self, clock: FrozenClock) -> None:
        limiter = RateLimiter(rate=10.0, capacity=1, clock=clock)
        assert await limiter.acquire(bucket="orders") == 0.0
        # A different bucket has its own budget.
        assert await limiter.acquire(bucket="market_data") == 0.0
        assert await limiter.acquire(bucket="orders") > 0.0

    async def test_default_bucket_is_shared(self, clock: FrozenClock) -> None:
        limiter = RateLimiter(rate=10.0, capacity=1, clock=clock)
        assert await limiter.acquire() == 0.0
        assert await limiter.acquire() > 0.0

    def test_bucket_returns_the_same_instance(self, clock: FrozenClock) -> None:
        limiter = RateLimiter(rate=1.0, capacity=1, clock=clock)
        assert limiter.bucket("a") is limiter.bucket("a")
        assert limiter.bucket("a") is not limiter.bucket("b")
