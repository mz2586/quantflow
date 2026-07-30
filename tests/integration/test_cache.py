"""Cache, distributed lock and event bus against a real Redis."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quantflow.cache.redis import Cache, DistributedLock, EventBus
from quantflow.core.errors import LockAcquisitionError

pytestmark = pytest.mark.integration


class TestValues:
    async def test_set_and_get_round_trip(self, cache: Cache) -> None:
        await cache.set("greeting", {"hello": "world"})
        assert await cache.get("greeting") == {"hello": "world"}

    async def test_missing_key_returns_none(self, cache: Cache) -> None:
        assert await cache.get("nope") is None

    async def test_decimal_survives_as_a_string(self, cache: Cache) -> None:
        # Serialising a Decimal as a JSON float would silently lose precision, so the
        # codec writes it as a string and the caller reconstructs it.
        await cache.set("price", {"last": Decimal("50000.12345678")})
        stored = await cache.get("price")
        assert stored is not None
        assert stored == {"last": "50000.12345678"}
        assert Decimal(stored["last"]) == Decimal("50000.12345678")

    async def test_datetime_serialises_to_iso(self, cache: Cache) -> None:
        moment = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
        await cache.set("at", {"t": moment})
        stored = await cache.get("at")
        assert stored is not None
        assert datetime.fromisoformat(stored["t"]) == moment

    async def test_ttl_expires_the_key(self, cache: Cache) -> None:
        await cache.set("ephemeral", 1, ttl_seconds=0.2)
        assert await cache.get("ephemeral") == 1
        await asyncio.sleep(0.35)
        assert await cache.get("ephemeral") is None

    async def test_delete_and_exists(self, cache: Cache) -> None:
        await cache.set("a", 1)
        await cache.set("b", 2)
        assert await cache.exists("a")
        assert await cache.delete("a", "b") == 2
        assert not await cache.exists("a")

    async def test_delete_without_keys_is_a_noop(self, cache: Cache) -> None:
        assert await cache.delete() == 0

    async def test_keys_are_namespaced(self, cache: Cache) -> None:
        await cache.set("scoped", 1)
        assert await cache.client.exists(cache.key("scoped")) == 1
        assert await cache.client.exists("scoped") == 0

    async def test_incr_counts_and_expires(self, cache: Cache) -> None:
        assert await cache.incr("orders", ttl_seconds=60) == 1
        assert await cache.incr("orders", ttl_seconds=60) == 2
        assert await cache.incr("orders", amount=5) == 7
        assert 0 < await cache.client.ttl(cache.key("orders")) <= 60

    async def test_incr_ttl_is_not_extended_by_later_increments(self, cache: Cache) -> None:
        await cache.incr("window", ttl_seconds=100)
        await asyncio.sleep(0.05)
        await cache.incr("window", ttl_seconds=100)
        # NX on the expiry means the window does not slide.
        assert await cache.client.ttl(cache.key("window")) <= 100

    async def test_get_or_set_computes_once(self, cache: Cache) -> None:
        calls = 0

        def factory() -> int:
            nonlocal calls
            calls += 1
            return 42

        assert await cache.get_or_set("lazy", factory) == 42
        assert await cache.get_or_set("lazy", factory) == 42
        assert calls == 1

    async def test_get_or_set_accepts_an_async_factory(self, cache: Cache) -> None:
        async def factory() -> str:
            await asyncio.sleep(0)
            return "computed"

        assert await cache.get_or_set("async-lazy", factory) == "computed"

    async def test_hash_operations(self, cache: Cache) -> None:
        await cache.hset("book", "BTC/USDT", {"bid": "50000"})
        await cache.hset("book", "ETH/USDT", {"bid": "3000"})
        assert await cache.hget("book", "BTC/USDT") == {"bid": "50000"}
        assert await cache.hget("book", "MISSING") is None
        assert set(await cache.hgetall("book")) == {"BTC/USDT", "ETH/USDT"}

    async def test_ping(self, cache: Cache) -> None:
        assert await cache.ping() is True


class TestDistributedLock:
    async def test_acquire_and_release(self, cache: Cache) -> None:
        lock = cache.lock("engine")
        assert await lock.acquire()
        assert lock.held
        assert await lock.release()
        assert not lock.held

    async def test_second_holder_is_blocked(self, cache: Cache) -> None:
        first = cache.lock("engine", ttl_seconds=5)
        second = cache.lock("engine", ttl_seconds=5, acquire_timeout_seconds=0.2)
        assert await first.acquire()
        try:
            assert await second.acquire() is False
        finally:
            await first.release()

    async def test_lock_is_available_after_release(self, cache: Cache) -> None:
        first = cache.lock("engine")
        await first.acquire()
        await first.release()
        second = cache.lock("engine", acquire_timeout_seconds=0.5)
        assert await second.acquire()
        await second.release()

    async def test_ttl_frees_a_crashed_holder(self, cache: Cache) -> None:
        crashed = cache.lock("engine", ttl_seconds=0.2)
        assert await crashed.acquire()
        # Deliberately never released, as if the process died.
        survivor = cache.lock("engine", acquire_timeout_seconds=1.0, retry_interval_seconds=0.05)
        assert await survivor.acquire()
        await survivor.release()

    async def test_release_only_succeeds_for_the_owner(self, cache: Cache) -> None:
        owner = cache.lock("engine", ttl_seconds=5)
        await owner.acquire()
        impostor = DistributedLock(cache.client, cache.key("lock", "engine"))
        impostor._token = "not-the-real-token"
        assert await impostor.release() is False
        assert await cache.client.exists(cache.key("lock", "engine")) == 1
        assert await owner.release() is True

    async def test_release_without_holding_returns_false(self, cache: Cache) -> None:
        assert await cache.lock("engine").release() is False

    async def test_extend_only_works_while_held(self, cache: Cache) -> None:
        lock = cache.lock("engine", ttl_seconds=1)
        await lock.acquire()
        assert await lock.extend(ttl_seconds=30) is True
        assert await cache.client.pttl(cache.key("lock", "engine")) > 1_000
        await lock.release()
        assert await lock.extend(ttl_seconds=30) is False

    async def test_context_manager_releases(self, cache: Cache) -> None:
        async with cache.lock("engine") as lock:
            assert lock.held
        assert await cache.client.exists(cache.key("lock", "engine")) == 0

    async def test_context_manager_releases_on_exception(self, cache: Cache) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with cache.lock("engine"):
                raise RuntimeError("boom")
        assert await cache.client.exists(cache.key("lock", "engine")) == 0

    async def test_context_manager_raises_on_timeout(self, cache: Cache) -> None:
        holder = cache.lock("engine", ttl_seconds=5)
        await holder.acquire()
        try:
            with pytest.raises(LockAcquisitionError, match="could not acquire"):
                async with cache.lock("engine", acquire_timeout_seconds=0.2):
                    pass
        finally:
            await holder.release()

    async def test_only_one_of_many_contenders_wins(self, cache: Cache) -> None:
        winners = 0

        async def contend() -> None:
            nonlocal winners
            lock = cache.lock("engine", ttl_seconds=5, acquire_timeout_seconds=0.1)
            if await lock.acquire():
                winners += 1
                await asyncio.sleep(0.3)
                await lock.release()

        await asyncio.gather(*(contend() for _ in range(10)))
        assert winners == 1


class TestPubSub:
    async def test_publish_and_receive(self, cache: Cache) -> None:
        received: list[tuple[str, object]] = []

        async def listen() -> None:
            async with cache.subscribe("fills") as stream:
                async for channel, message in stream:
                    received.append((channel, message))
                    break

        task = asyncio.create_task(listen())
        await asyncio.sleep(0.2)  # let the subscription establish
        assert await cache.publish("fills", {"order_id": "abc"}) == 1
        await asyncio.wait_for(task, timeout=3)

        assert received == [("fills", {"order_id": "abc"})]

    async def test_publish_with_no_subscribers(self, cache: Cache) -> None:
        assert await cache.publish("nobody-listening", {"x": 1}) == 0

    async def test_event_bus_envelope(self, cache: Cache) -> None:
        bus = EventBus(cache)
        received: list[object] = []

        async def listen() -> None:
            async with bus.subscribe(EventBus.CHANNEL_RISK) as stream:
                async for _, message in stream:
                    received.append(message)
                    break

        task = asyncio.create_task(listen())
        await asyncio.sleep(0.2)
        await bus.publish_risk_event({"rule": "max_drawdown"})
        await asyncio.wait_for(task, timeout=3)

        assert received == [{"type": "risk_event", "payload": {"rule": "max_drawdown"}}]


class TestStreams:
    async def test_enqueue_and_read(self, cache: Cache) -> None:
        first = await cache.enqueue("jobs", {"kind": "backtest", "id": "1"})
        await cache.enqueue("jobs", {"kind": "backtest", "id": "2"})

        entries = await cache.read_stream("jobs")
        assert [payload["id"] for _, payload in entries] == ["1", "2"]

        after_first = await cache.read_stream("jobs", last_id=first)
        assert [payload["id"] for _, payload in after_first] == ["2"]

    async def test_read_empty_stream(self, cache: Cache) -> None:
        assert await cache.read_stream("empty") == []


class TestNamespaceIsolation:
    async def test_flush_namespace_only_touches_our_prefix(self, cache: Cache) -> None:
        await cache.set("mine", 1)
        await cache.client.set("someone-elses-key", b"keep me")
        try:
            await cache.flush_namespace()
            assert await cache.get("mine") is None
            assert await cache.client.get("someone-elses-key") == b"keep me"
        finally:
            await cache.client.delete("someone-elses-key")
