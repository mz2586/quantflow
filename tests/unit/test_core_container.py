"""DI container semantics: caching, async factories, ordered teardown."""

from __future__ import annotations

import asyncio
from typing import Protocol

import pytest

from quantflow.core.container import Container
from quantflow.core.errors import DependencyNotRegisteredError


class Greeter(Protocol):
    def greet(self) -> str: ...


class EnglishGreeter:
    def greet(self) -> str:
        return "hello"


class FrenchGreeter:
    def greet(self) -> str:
        return "bonjour"


class Counter:
    def __init__(self) -> None:
        self.value = 0


class TestRegistration:
    async def test_register_instance(self) -> None:
        container = Container()
        instance = EnglishGreeter()
        container.register_instance(Greeter, instance)  # type: ignore[type-abstract]
        assert await container.resolve(Greeter) is instance  # type: ignore[type-abstract]

    async def test_register_sync_factory(self) -> None:
        container = Container()
        container.register_factory(Greeter, lambda _: EnglishGreeter())  # type: ignore[type-abstract]
        assert (await container.resolve(Greeter)).greet() == "hello"  # type: ignore[type-abstract]

    async def test_register_async_factory(self) -> None:
        container = Container()

        async def build(_: Container) -> EnglishGreeter:
            await asyncio.sleep(0)
            return EnglishGreeter()

        container.register_factory(Greeter, build)  # type: ignore[type-abstract]
        assert (await container.resolve(Greeter)).greet() == "hello"  # type: ignore[type-abstract]

    async def test_factory_receives_container_for_nested_resolution(self) -> None:
        container = Container()
        container.register_instance(Counter, Counter())

        async def build(inner: Container) -> EnglishGreeter:
            counter = await inner.resolve(Counter)
            counter.value += 1
            return EnglishGreeter()

        container.register_factory(Greeter, build)  # type: ignore[type-abstract]
        await container.resolve(Greeter)  # type: ignore[type-abstract]
        assert (await container.resolve(Counter)).value == 1

    def test_has(self) -> None:
        container = Container()
        assert not container.has(Greeter)
        container.register_instance(Greeter, EnglishGreeter())  # type: ignore[type-abstract]
        assert container.has(Greeter)


class TestResolution:
    async def test_unregistered_dependency_raises(self) -> None:
        with pytest.raises(DependencyNotRegisteredError, match="Greeter"):
            await Container().resolve(Greeter)  # type: ignore[type-abstract]

    async def test_singleton_is_built_once(self) -> None:
        container = Container()
        calls = 0

        def build(_: Container) -> Counter:
            nonlocal calls
            calls += 1
            return Counter()

        container.register_factory(Counter, build)
        first = await container.resolve(Counter)
        second = await container.resolve(Counter)
        assert first is second
        assert calls == 1

    async def test_non_singleton_builds_each_time(self) -> None:
        container = Container()
        container.register_factory(Counter, lambda _: Counter(), singleton=False)
        assert await container.resolve(Counter) is not await container.resolve(Counter)

    async def test_concurrent_resolution_builds_once(self) -> None:
        container = Container()
        calls = 0

        async def slow_build(_: Container) -> Counter:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return Counter()

        container.register_factory(Counter, slow_build)
        results = await asyncio.gather(*(container.resolve(Counter) for _ in range(10)))
        assert calls == 1
        assert all(result is results[0] for result in results)

    async def test_resolve_sync_requires_prior_resolution(self) -> None:
        container = Container()
        container.register_factory(Counter, lambda _: Counter())
        with pytest.raises(DependencyNotRegisteredError, match="not resolved"):
            container.resolve_sync(Counter)
        await container.resolve(Counter)
        assert isinstance(container.resolve_sync(Counter), Counter)

    async def test_override_replaces_registration(self) -> None:
        container = Container()
        container.register_instance(Greeter, EnglishGreeter())  # type: ignore[type-abstract]
        container.override(Greeter, FrenchGreeter())  # type: ignore[type-abstract]
        assert (await container.resolve(Greeter)).greet() == "bonjour"  # type: ignore[type-abstract]


class TestTeardown:
    async def test_teardown_runs_in_reverse_registration_order(self) -> None:
        container = Container()
        order: list[str] = []

        class First:
            pass

        class Second:
            pass

        container.register_instance(First, First(), teardown=lambda _: order.append("first"))
        container.register_instance(Second, Second(), teardown=lambda _: order.append("second"))
        await container.aclose()
        assert order == ["second", "first"]

    async def test_async_teardown_is_awaited(self) -> None:
        container = Container()
        closed = False

        async def close(_: Counter) -> None:
            nonlocal closed
            await asyncio.sleep(0)
            closed = True

        container.register_instance(Counter, Counter(), teardown=close)
        await container.aclose()
        assert closed

    async def test_unresolved_singletons_are_not_torn_down(self) -> None:
        container = Container()
        torn_down = False

        def close(_: Counter) -> None:
            nonlocal torn_down
            torn_down = True

        container.register_factory(Counter, lambda _: Counter(), teardown=close)
        await container.aclose()
        assert not torn_down

    async def test_failing_teardown_does_not_block_others(self) -> None:
        container = Container()
        second_closed = False

        class First:
            pass

        class Second:
            pass

        def boom(_: object) -> None:
            raise RuntimeError("teardown exploded")

        def close(_: object) -> None:
            nonlocal second_closed
            second_closed = True

        container.register_instance(Second, Second(), teardown=close)
        container.register_instance(First, First(), teardown=boom)
        await container.aclose()
        assert second_closed

    async def test_resolution_after_close_raises(self) -> None:
        container = Container()
        container.register_instance(Counter, Counter())
        await container.aclose()
        with pytest.raises(RuntimeError, match="closed container"):
            await container.resolve(Counter)

    async def test_close_is_idempotent(self) -> None:
        container = Container()
        calls = 0

        def close(_: object) -> None:
            nonlocal calls
            calls += 1

        container.register_instance(Counter, Counter(), teardown=close)
        await container.aclose()
        await container.aclose()
        assert calls == 1

    async def test_async_context_manager_closes(self) -> None:
        closed = False

        def close(_: object) -> None:
            nonlocal closed
            closed = True

        async with Container() as container:
            container.register_instance(Counter, Counter(), teardown=close)
        assert closed
