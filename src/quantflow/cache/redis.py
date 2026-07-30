"""Redis client, distributed lock and event bus."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from types import TracebackType
from typing import Any, Final, Self, cast

import orjson
import redis.asyncio as aioredis
from redis.asyncio.client import Redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import RedisError

from quantflow.core.config import RedisSettings, Settings
from quantflow.core.errors import CacheError, LockAcquisitionError
from quantflow.core.logging import get_logger

logger = get_logger(__name__)

#: Compare-and-delete: only release a lock we still own. Without this, a lock that expired
#: mid-operation would be released by its *previous* holder, letting two workers run the
#: critical section at once.
_RELEASE_LOCK_SCRIPT: Final = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""

#: Compare-and-extend, same reasoning as above.
_EXTEND_LOCK_SCRIPT: Final = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
else
    return 0
end
"""


def _default(value: Any) -> Any:
    """Orjson fallback for types it cannot serialise natively."""
    if isinstance(value, Decimal):
        # Strings, not floats: a Decimal must survive the round trip exactly.
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return list(value)
    if hasattr(value, "slashed"):  # Symbol
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__} to JSON")


def dumps(value: Any) -> bytes:
    """Serialise to JSON bytes."""
    return orjson.dumps(value, default=_default)


def loads(payload: bytes | str) -> Any:
    """Deserialise JSON bytes."""
    return orjson.loads(payload)


def build_redis(settings: RedisSettings) -> Redis:
    """Create an async Redis client with a bounded connection pool.

    Retries use an explicit exponential backoff rather than the deprecated
    ``retry_on_timeout`` flag: redis-py 6+ retries ``TimeoutError`` by default, and a
    stated policy is clearer than relying on that default.
    """
    return aioredis.Redis(
        host=settings.host,
        port=settings.port,
        db=settings.db,
        password=settings.password.get_secret_value() if settings.password else None,
        max_connections=settings.max_connections,
        socket_timeout=settings.socket_timeout_seconds,
        socket_connect_timeout=settings.socket_timeout_seconds,
        retry=Retry(ExponentialBackoff(base=0.05, cap=1.0), retries=3),
        health_check_interval=30,
        decode_responses=False,
    )


class Cache:
    """Namespaced Redis facade.

    Every key is prefixed (``qf:...``) so a shared Redis instance cannot collide with
    another application's keyspace.
    """

    __slots__ = ("_client", "_prefix")

    def __init__(self, client: Redis, *, prefix: str = "qf") -> None:
        self._client = client
        self._prefix = prefix

    @classmethod
    def from_settings(cls, settings: Settings) -> Cache:
        """Build from the root settings object."""
        return cls(build_redis(settings.redis), prefix=settings.redis.key_prefix)

    @property
    def client(self) -> Redis:
        """The underlying Redis client, for operations this facade does not wrap."""
        return self._client

    def key(self, *parts: str) -> str:
        """Build a namespaced key from its parts."""
        return ":".join((self._prefix, *parts))

    # ------------------------------------------------------------------ #
    # Values
    # ------------------------------------------------------------------ #
    async def get(self, key: str) -> Any | None:
        """Read and decode a JSON value."""
        try:
            payload = await self._client.get(self.key(key))
        except RedisError as exc:
            raise CacheError(f"redis GET failed for {key!r}: {exc}") from exc
        return None if payload is None else loads(payload)

    async def set(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> None:
        """Write a JSON value with an optional TTL."""
        try:
            await self._client.set(
                self.key(key),
                dumps(value),
                px=int(ttl_seconds * 1000) if ttl_seconds else None,
            )
        except RedisError as exc:
            raise CacheError(f"redis SET failed for {key!r}: {exc}") from exc

    async def delete(self, *keys: str) -> int:
        """Delete keys. Returns the number removed."""
        if not keys:
            return 0
        try:
            return int(await self._client.delete(*(self.key(key) for key in keys)))
        except RedisError as exc:
            raise CacheError(f"redis DEL failed: {exc}") from exc

    async def exists(self, key: str) -> bool:
        """Whether a key is present."""
        try:
            return bool(await self._client.exists(self.key(key)))
        except RedisError as exc:
            raise CacheError(f"redis EXISTS failed for {key!r}: {exc}") from exc

    async def incr(self, key: str, *, amount: int = 1, ttl_seconds: float | None = None) -> int:
        """Atomically increment a counter, optionally setting a TTL on creation."""
        namespaced = self.key(key)
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incrby(namespaced, amount)
                if ttl_seconds is not None:
                    # NX so a repeated increment does not keep pushing the window out.
                    pipe.expire(namespaced, int(ttl_seconds), nx=True)
                results = await pipe.execute()
            return int(results[0])
        except RedisError as exc:
            raise CacheError(f"redis INCR failed for {key!r}: {exc}") from exc

    async def get_or_set(
        self, key: str, factory: Callable[[], Any], *, ttl_seconds: float | None = None
    ) -> Any:
        """Read a value, computing and caching it on a miss."""
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = factory()
        if asyncio.iscoroutine(value):
            value = await value
        await self.set(key, value, ttl_seconds=ttl_seconds)
        return value

    # ------------------------------------------------------------------ #
    # Hashes
    # ------------------------------------------------------------------ #
    async def hset(self, key: str, field: str, value: Any) -> None:
        """Set a hash field."""
        try:
            await self._client.hset(self.key(key), field, dumps(value))
        except RedisError as exc:
            raise CacheError(f"redis HSET failed for {key!r}: {exc}") from exc

    async def hget(self, key: str, field: str) -> Any | None:
        """Read a hash field."""
        try:
            payload = await self._client.hget(self.key(key), field)
        except RedisError as exc:
            raise CacheError(f"redis HGET failed for {key!r}: {exc}") from exc
        return None if payload is None else loads(payload)

    async def hgetall(self, key: str) -> dict[str, Any]:
        """Read every field in a hash."""
        try:
            raw = await self._client.hgetall(self.key(key))
        except RedisError as exc:
            raise CacheError(f"redis HGETALL failed for {key!r}: {exc}") from exc
        return {
            (field.decode() if isinstance(field, bytes) else field): loads(value)
            for field, value in raw.items()
        }

    # ------------------------------------------------------------------ #
    # Locks
    # ------------------------------------------------------------------ #
    def lock(
        self,
        name: str,
        *,
        ttl_seconds: float = 30.0,
        acquire_timeout_seconds: float = 10.0,
        retry_interval_seconds: float = 0.05,
    ) -> DistributedLock:
        """Build a distributed lock. See :class:`DistributedLock`."""
        return DistributedLock(
            self._client,
            self.key("lock", name),
            ttl_seconds=ttl_seconds,
            acquire_timeout_seconds=acquire_timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )

    # ------------------------------------------------------------------ #
    # Pub/sub
    # ------------------------------------------------------------------ #
    async def publish(self, channel: str, message: Any) -> int:
        """Publish a JSON message. Returns the number of subscribers reached."""
        try:
            return int(await self._client.publish(self.key("events", channel), dumps(message)))
        except RedisError as exc:
            raise CacheError(f"redis PUBLISH failed for {channel!r}: {exc}") from exc

    @asynccontextmanager
    async def subscribe(self, *channels: str) -> AsyncIterator[AsyncIterator[tuple[str, Any]]]:
        """Subscribe to channels, yielding ``(channel, message)`` pairs.

        Usage::

            async with cache.subscribe("fills", "risk") as stream:
                async for channel, message in stream:
                    ...
        """
        pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        namespaced = [self.key("events", channel) for channel in channels]
        prefix_length = len(self.key("events", ""))
        try:
            await pubsub.subscribe(*namespaced)

            async def stream() -> AsyncIterator[tuple[str, Any]]:
                async for raw in pubsub.listen():
                    if raw.get("type") != "message":
                        continue
                    channel = raw["channel"]
                    name = (channel.decode() if isinstance(channel, bytes) else channel)[
                        prefix_length:
                    ]
                    yield name, loads(raw["data"])

            yield stream()
        finally:
            with contextlib.suppress(RedisError):
                await pubsub.unsubscribe(*namespaced)
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------ #
    # Streams (durable work queue)
    # ------------------------------------------------------------------ #
    async def enqueue(self, stream: str, payload: dict[str, Any], *, maxlen: int = 10_000) -> str:
        """Append to a Redis stream, capping its length. Returns the entry id."""
        try:
            entry_id = await self._client.xadd(
                self.key("stream", stream),
                {"payload": dumps(payload)},
                maxlen=maxlen,
                approximate=True,
            )
        except RedisError as exc:
            raise CacheError(f"redis XADD failed for {stream!r}: {exc}") from exc
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

    async def read_stream(
        self, stream: str, *, last_id: str = "0-0", count: int = 100, block_ms: int | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        """Read entries from a stream after ``last_id``."""
        try:
            response: Any = await self._client.xread(
                {self.key("stream", stream): last_id}, count=count, block=block_ms
            )
        except RedisError as exc:
            raise CacheError(f"redis XREAD failed for {stream!r}: {exc}") from exc

        entries: list[tuple[str, dict[str, Any]]] = []
        for _, records in response or []:
            for entry_id, fields in records:
                payload = fields.get(b"payload") or fields.get("payload")
                entries.append(
                    (
                        entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id),
                        loads(payload) if payload else {},
                    )
                )
        return entries

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def ping(self) -> bool:
        """Check connectivity. Returns ``False`` rather than raising, for health checks."""
        try:
            return bool(await self._client.ping())
        except RedisError as exc:
            logger.warning("cache.ping_failed", error=str(exc))
            return False

    async def flush_namespace(self) -> int:
        """Delete every key under this prefix. Intended for tests and local resets."""
        deleted = 0
        async for key in self._client.scan_iter(match=f"{self._prefix}:*", count=500):
            deleted += int(await self._client.delete(key))
        return deleted

    async def aclose(self) -> None:
        """Close the client and its connection pool."""
        await self._client.aclose()


class DistributedLock:
    """Redis-backed mutex with a TTL and ownership-checked release.

    The TTL guarantees the lock is eventually released even if the holder crashes; the
    owner token guarantees a slow holder cannot release a lock that has since been
    reacquired by someone else.
    """

    __slots__ = (
        "_acquire_timeout",
        "_client",
        "_key",
        "_retry_interval",
        "_token",
        "_ttl_ms",
    )

    def __init__(
        self,
        client: Redis,
        key: str,
        *,
        ttl_seconds: float = 30.0,
        acquire_timeout_seconds: float = 10.0,
        retry_interval_seconds: float = 0.05,
    ) -> None:
        self._client = client
        self._key = key
        self._ttl_ms = int(ttl_seconds * 1000)
        self._acquire_timeout = acquire_timeout_seconds
        self._retry_interval = retry_interval_seconds
        self._token: str | None = None

    @property
    def held(self) -> bool:
        """Whether this instance currently believes it holds the lock."""
        return self._token is not None

    async def acquire(self) -> bool:
        """Try to acquire the lock, polling until the timeout.

        Returns:
            ``True`` on success, ``False`` if the timeout elapsed.

        """
        token = uuid.uuid4().hex
        deadline = asyncio.get_running_loop().time() + self._acquire_timeout
        while True:
            try:
                acquired = await self._client.set(self._key, token, nx=True, px=self._ttl_ms)
            except RedisError as exc:
                raise CacheError(f"redis SET NX failed for lock {self._key!r}: {exc}") from exc
            if acquired:
                self._token = token
                return True
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(self._retry_interval)

    async def release(self) -> bool:
        """Release the lock if we still own it."""
        if self._token is None:
            return False
        try:
            released = await self._client.eval(_RELEASE_LOCK_SCRIPT, 1, self._key, self._token)
        except RedisError as exc:
            raise CacheError(f"redis lock release failed for {self._key!r}: {exc}") from exc
        finally:
            self._token = None
        return bool(released)

    async def extend(self, *, ttl_seconds: float) -> bool:
        """Extend the TTL if we still own the lock."""
        if self._token is None:
            return False
        try:
            extended = await self._client.eval(
                _EXTEND_LOCK_SCRIPT, 1, self._key, self._token, int(ttl_seconds * 1000)
            )
        except RedisError as exc:
            raise CacheError(f"redis lock extend failed for {self._key!r}: {exc}") from exc
        return bool(extended)

    async def __aenter__(self) -> Self:
        """Acquire, raising if the lock cannot be obtained in time."""
        if not await self.acquire():
            raise LockAcquisitionError(
                f"could not acquire lock {self._key!r} within {self._acquire_timeout}s",
                lock=self._key,
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the lock."""
        await self.release()


class EventBus:
    """Typed pub/sub facade over :class:`Cache`.

    Channels are named constants so a typo in a publisher cannot silently produce messages
    nobody is listening for.
    """

    CHANNEL_SIGNALS: Final = "signals"
    CHANNEL_ORDERS: Final = "orders"
    CHANNEL_FILLS: Final = "fills"
    CHANNEL_POSITIONS: Final = "positions"
    CHANNEL_EQUITY: Final = "equity"
    CHANNEL_RISK: Final = "risk"
    CHANNEL_CANDLES: Final = "candles"
    CHANNEL_SYSTEM: Final = "system"

    __slots__ = ("_cache",)

    def __init__(self, cache: Cache) -> None:
        self._cache = cache

    async def publish(self, channel: str, event_type: str, payload: dict[str, Any]) -> int:
        """Publish an envelope of ``{type, payload}``."""
        return await self._cache.publish(channel, {"type": event_type, "payload": payload})

    @asynccontextmanager
    async def subscribe(self, *channels: str) -> AsyncIterator[AsyncIterator[tuple[str, Any]]]:
        """Subscribe to one or more channels."""
        async with self._cache.subscribe(*channels) as stream:
            yield stream

    async def publish_signal(self, payload: dict[str, Any]) -> int:
        """Publish a strategy signal."""
        return await self.publish(self.CHANNEL_SIGNALS, "signal", payload)

    async def publish_fill(self, payload: dict[str, Any]) -> int:
        """Publish an execution."""
        return await self.publish(self.CHANNEL_FILLS, "fill", payload)

    async def publish_risk_event(self, payload: dict[str, Any]) -> int:
        """Publish a risk decision."""
        return await self.publish(self.CHANNEL_RISK, "risk_event", payload)

    async def publish_equity(self, payload: dict[str, Any]) -> int:
        """Publish an equity update."""
        return await self.publish(self.CHANNEL_EQUITY, "equity", payload)


def cast_redis(client: object) -> Redis:
    """Narrow a duck-typed client (e.g. ``fakeredis``) to ``Redis`` for typing purposes."""
    return cast(Redis, client)
