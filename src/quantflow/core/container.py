"""Minimal async-aware dependency-injection container.

Rationale for hand-rolling rather than pulling in a framework: the object graph here is
small and known at import time, we need async factories and async teardown in reverse
registration order, and a third-party container would add a runtime dependency plus its own
magic for no measurable benefit. Services are keyed by their type (usually a
:class:`typing.Protocol`), which keeps call sites statically typed.

Usage::

    container = Container()
    container.register_instance(Clock, SystemClock())
    container.register_factory(ExchangeGateway, build_gateway, teardown=close_gateway)

    gateway = await container.resolve(ExchangeGateway)   # built once, then cached
    await container.aclose()                             # tears down in reverse order
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self, TypeVar, cast

from quantflow.core.errors import DependencyNotRegisteredError
from quantflow.core.logging import get_logger

T = TypeVar("T")

Factory = Callable[["Container"], Any] | Callable[["Container"], Awaitable[Any]]
Teardown = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]

logger = get_logger(__name__)


@dataclass(slots=True)
class _Registration:
    """How a single service is built and disposed of."""

    factory: Factory
    singleton: bool
    teardown: Teardown | None
    instance: Any = None
    resolved: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Container:
    """Type-keyed service registry with singleton caching and ordered teardown."""

    __slots__ = ("_closed", "_order", "_registrations")

    def __init__(self) -> None:
        self._registrations: dict[type[Any], _Registration] = {}
        self._order: list[type[Any]] = []
        self._closed = False

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register_instance(
        self, key: type[T], instance: T, *, teardown: Teardown | None = None
    ) -> None:
        """Register an already-constructed singleton."""
        self._registrations[key] = _Registration(
            factory=lambda _: instance,
            singleton=True,
            teardown=teardown,
            instance=instance,
            resolved=True,
        )
        self._track(key)

    def register_factory(
        self,
        key: type[T],
        factory: Callable[[Container], T] | Callable[[Container], Awaitable[T]],
        *,
        singleton: bool = True,
        teardown: Teardown | None = None,
    ) -> None:
        """Register a (possibly async) factory.

        Args:
            key: Type or Protocol used to resolve the service.
            factory: Callable receiving the container, returning the service.
            singleton: Cache the first result and reuse it.
            teardown: Called with the instance during :meth:`aclose`.

        """
        self._registrations[key] = _Registration(
            factory=cast(Factory, factory), singleton=singleton, teardown=teardown
        )
        self._track(key)

    def _track(self, key: type[Any]) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def has(self, key: type[Any]) -> bool:
        """Whether ``key`` is registered."""
        return key in self._registrations

    def override(self, key: type[T], instance: T) -> None:
        """Replace a registration with a fixed instance. Intended for tests."""
        self._registrations.pop(key, None)
        self.register_instance(key, instance)

    # ------------------------------------------------------------------ #
    # Resolution
    # ------------------------------------------------------------------ #
    async def resolve(self, key: type[T]) -> T:
        """Resolve a service, constructing it on first use.

        Raises:
            DependencyNotRegisteredError: if ``key`` was never registered.
            RuntimeError: if the container has already been closed.

        """
        if self._closed:
            raise RuntimeError("cannot resolve from a closed container")
        registration = self._registrations.get(key)
        if registration is None:
            raise DependencyNotRegisteredError(
                f"no registration for {_name(key)}", requested=_name(key)
            )

        if registration.singleton and registration.resolved:
            return cast(T, registration.instance)

        if not registration.singleton:
            return cast(T, await self._build(registration))

        async with registration.lock:
            if registration.resolved:  # another task won the race
                return cast(T, registration.instance)
            instance = await self._build(registration)
            registration.instance = instance
            registration.resolved = True
            return cast(T, instance)

    def resolve_sync(self, key: type[T]) -> T:
        """Resolve an already-constructed singleton without awaiting.

        Raises:
            DependencyNotRegisteredError: if the service is absent or not yet built.

        """
        registration = self._registrations.get(key)
        if registration is None or not registration.resolved:
            raise DependencyNotRegisteredError(
                f"{_name(key)} is not resolved; await resolve() first",
                requested=_name(key),
            )
        return cast(T, registration.instance)

    async def _build(self, registration: _Registration) -> Any:
        result = registration.factory(self)
        if inspect.isawaitable(result):
            return await result
        return result

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def aclose(self) -> None:
        """Tear down every resolved singleton in reverse registration order.

        Teardown failures are logged and swallowed so one bad closer cannot leak the rest.
        """
        if self._closed:
            return
        self._closed = True
        for key in reversed(self._order):
            registration = self._registrations.get(key)
            if registration is None or not registration.resolved:
                continue
            if registration.teardown is None:
                continue
            try:
                outcome = registration.teardown(registration.instance)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                logger.warning("container.teardown_failed", service=_name(key), exc_info=True)
        self._registrations.clear()
        self._order.clear()

    async def __aenter__(self) -> Self:
        """Enter the container scope."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the container on scope exit."""
        await self.aclose()


def _name(key: type[Any]) -> str:
    return getattr(key, "__name__", repr(key))
