"""Application state and FastAPI dependencies.

Every long-lived resource — the database pool, the Redis client, the exchange gateway, the
risk engine — is built once during startup and handed to routes by dependency injection.
Routes never construct their own; that is what keeps a request from opening its own
connection pool and what makes the whole graph substitutable in tests.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, Header, Request

from quantflow.cache.redis import Cache, EventBus
from quantflow.core.config import Environment, Settings
from quantflow.core.errors import AuthenticationError, ConfigurationError
from quantflow.core.logging import get_logger
from quantflow.exchange.base import ExchangeGateway
from quantflow.persistence.database import Database
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine
from quantflow.strategy.registry import StrategyRegistry

logger = get_logger(__name__)


@dataclass(slots=True)
class AppState:
    """Everything the application owns for its lifetime."""

    settings: Settings
    database: Database | None = None
    cache: Cache | None = None
    event_bus: EventBus | None = None
    gateway: ExchangeGateway | None = None
    risk: RiskEngine | None = None
    portfolio: PortfolioManager | None = None
    registry: StrategyRegistry | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def require_database(self) -> Database:
        """The database, or a clear error if it was not wired.

        Raises:
            ConfigurationError: if the database is unavailable.

        """
        if self.database is None:
            raise ConfigurationError("database is not configured")
        return self.database

    def require_cache(self) -> Cache:
        """The Redis cache, or a clear error.

        Raises:
            ConfigurationError: if the cache is unavailable.

        """
        if self.cache is None:
            raise ConfigurationError("redis is not configured")
        return self.cache

    def require_gateway(self) -> ExchangeGateway:
        """The exchange gateway, or a clear error.

        Raises:
            ConfigurationError: if the gateway is unavailable.

        """
        if self.gateway is None:
            raise ConfigurationError("exchange gateway is not configured")
        return self.gateway

    def require_risk(self) -> RiskEngine:
        """The risk engine, or a clear error.

        Raises:
            ConfigurationError: if the risk engine is unavailable.

        """
        if self.risk is None:
            raise ConfigurationError("risk engine is not configured")
        return self.risk

    def require_registry(self) -> StrategyRegistry:
        """The strategy registry, loading the built-ins on first use."""
        if self.registry is None:
            from quantflow.strategy.registry import load_builtin_strategies

            self.registry = load_builtin_strategies()
        return self.registry


def get_state(request: Request) -> AppState:
    """Read the application state from the request."""
    state: AppState = request.app.state.quantflow
    return state


StateDep = Annotated[AppState, Depends(get_state)]


def get_app_settings(state: StateDep) -> Settings:
    """The active settings."""
    return state.settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def get_database(state: StateDep) -> Database:
    """The database."""
    return state.require_database()


DatabaseDep = Annotated[Database, Depends(get_database)]


def get_optional_database(state: StateDep) -> Database | None:
    """The database, or ``None`` when none is configured.

    For endpoints that can answer from an in-process engine and treat the database as a
    fallback. Depending on `DatabaseDep` there would make persistence mandatory for a
    request that never needed it, and the caller would get "database is not configured"
    in place of the answer the process could have given.
    """
    return state.database


OptionalDatabaseDep = Annotated[Database | None, Depends(get_optional_database)]


def get_cache(state: StateDep) -> Cache:
    """The Redis cache."""
    return state.require_cache()


CacheDep = Annotated[Cache, Depends(get_cache)]


def get_gateway(state: StateDep) -> ExchangeGateway:
    """The exchange gateway."""
    return state.require_gateway()


GatewayDep = Annotated[ExchangeGateway, Depends(get_gateway)]


def get_risk(state: StateDep) -> RiskEngine:
    """The risk engine."""
    return state.require_risk()


RiskDep = Annotated[RiskEngine, Depends(get_risk)]


def get_registry(state: StateDep) -> StrategyRegistry:
    """The strategy registry."""
    return state.require_registry()


RegistryDep = Annotated[StrategyRegistry, Depends(get_registry)]


async def require_api_key(
    state: StateDep,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Authenticate a request in production-like environments.

    Development and test skip this so a local dashboard is not blocked by ceremony;
    staging and production enforce it, and settings validation already guarantees a key
    exists there. Comparison uses :func:`secrets.compare_digest` so a wrong key cannot be
    recovered one character at a time by timing the response.

    Raises:
        AuthenticationError: if the key is missing or wrong.

    """
    if not state.settings.env.is_production_like:
        return

    expected = state.settings.api_key
    if expected is None:  # pragma: no cover — settings validation forbids this
        raise ConfigurationError("api_key is required in this environment")

    if x_api_key is None:
        raise AuthenticationError("missing X-API-Key header")
    if not secrets.compare_digest(x_api_key, expected.get_secret_value()):
        logger.warning("auth.invalid_api_key")
        raise AuthenticationError("invalid API key")


AuthDep = Annotated[None, Depends(require_api_key)]


def is_mutating_allowed(settings: Settings) -> bool:
    """Whether state-changing endpoints may run.

    Read-only endpoints are always available; anything that can move money is gated so a
    misconfigured deployment cannot be poked into trading from a browser.
    """
    return settings.env is not Environment.PRODUCTION or settings.api_key is not None
