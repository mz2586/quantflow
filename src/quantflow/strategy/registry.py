"""Strategy registry.

Strategies are looked up by their stable ``strategy_id`` so the API, CLI, optimiser and
persisted session records can all refer to one by name without importing its class.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from quantflow.core.errors import NotFoundError, StrategyError
from quantflow.core.logging import get_logger
from quantflow.strategy.base import Strategy, StrategyParams

logger = get_logger(__name__)


class StrategyRegistry:
    """Name-to-class registry with parameter-schema introspection."""

    __slots__ = ("_strategies",)

    def __init__(self) -> None:
        self._strategies: dict[str, type[Strategy]] = {}

    def register(self, strategy_class: type[Strategy]) -> type[Strategy]:
        """Register a strategy class. Usable as a decorator.

        Raises:
            StrategyError: if the class has no id, or the id is already taken by a
                *different* class. Re-registering the same class is a no-op, so module
                reloads during development do not explode.

        """
        identifier = strategy_class.strategy_id
        if not identifier:
            raise StrategyError(f"{strategy_class.__name__} must declare a strategy_id")
        existing = self._strategies.get(identifier)
        if existing is not None and existing is not strategy_class:
            raise StrategyError(
                f"strategy id {identifier!r} is already registered to {existing.__name__}",
                strategy_id=identifier,
            )
        self._strategies[identifier] = strategy_class
        return strategy_class

    def get(self, strategy_id: str) -> type[Strategy]:
        """Look up a strategy class.

        Raises:
            NotFoundError: with the list of available ids, so a typo is self-correcting.

        """
        strategy_class = self._strategies.get(strategy_id)
        if strategy_class is None:
            available = ", ".join(sorted(self._strategies)) or "none registered"
            raise NotFoundError(
                f"unknown strategy {strategy_id!r}; available: {available}",
                strategy_id=strategy_id,
            )
        return strategy_class

    def create(
        self, strategy_id: str, params: dict[str, Any] | StrategyParams | None = None
    ) -> Strategy:
        """Instantiate a strategy by id with the given parameters."""
        return self.get(strategy_id)(params)

    def params_model(self, strategy_id: str) -> type[StrategyParams]:
        """The parameter schema for a strategy."""
        return self.get(strategy_id).params_model

    def json_schema(self, strategy_id: str) -> dict[str, Any]:
        """JSON Schema for a strategy's parameters, for the API and dashboard forms."""
        return self.params_model(strategy_id).model_json_schema()

    def describe(self, strategy_id: str) -> dict[str, Any]:
        """Metadata for one strategy, including its default parameters."""
        strategy_class = self.get(strategy_id)
        instance = strategy_class()
        return {
            "strategy_id": strategy_class.strategy_id,
            "description": strategy_class.description,
            "warmup_bars": instance.warmup_bars,
            "defaults": instance.params.to_dict(),
            "schema": strategy_class.params_model.model_json_schema(),
        }

    def describe_all(self) -> list[dict[str, Any]]:
        """Metadata for every registered strategy."""
        return [self.describe(identifier) for identifier in sorted(self._strategies)]

    def names(self) -> list[str]:
        """Every registered strategy id."""
        return sorted(self._strategies)

    def __contains__(self, strategy_id: object) -> bool:
        return strategy_id in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._strategies))

    def clear(self) -> None:
        """Remove every registration. Intended for tests."""
        self._strategies.clear()


#: Process-wide registry. The built-in library populates it on import.
registry = StrategyRegistry()


def register_strategy(strategy_class: type[Strategy]) -> type[Strategy]:
    """Decorator registering a strategy with the global registry."""
    return registry.register(strategy_class)


def load_builtin_strategies() -> StrategyRegistry:
    """Import the built-in library so its strategies self-register.

    Called by the API, CLI and engines at startup rather than at module import, keeping
    import side effects explicit and testable.
    """
    # The orchestrator registers like any other strategy so `--strategy orchestrator`
    # works, but it lives outside the library package because it composes it. Imported
    # here, inside the function, because it calls back into this module at construction
    # time — at module scope the two would import each other.
    from quantflow import orchestrator  # noqa: F401 — import triggers registration
    from quantflow.strategy import library  # noqa: F401 — import triggers registration

    logger.debug("strategy.registry_loaded", count=len(registry), names=registry.names())
    return registry
