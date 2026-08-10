"""FastAPI application factory.

Resources are created in :func:`lifespan` and torn down in reverse order. Startup is
**tolerant**: if Redis or the exchange is unreachable the app still starts and reports
itself un-ready, rather than crash-looping. An API that refuses to boot cannot tell an
operator *why* it refused to boot, and during an incident that is exactly when you need it
to answer.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from quantflow import __version__
from quantflow.api import middleware
from quantflow.api.deps import AppState
from quantflow.api.routers import (
    account,
    analytics,
    backtest,
    marketdata,
    portfolio,
    risk,
    system,
)
from quantflow.cache.redis import Cache, EventBus
from quantflow.core.config import Settings, get_settings
from quantflow.core.logging import configure_logging, get_logger
from quantflow.exchange.base import ExchangeGateway
from quantflow.notifications.dispatcher import build_dispatcher
from quantflow.persistence.database import Database
from quantflow.risk.engine import RiskEngine
from quantflow.strategy.registry import load_builtin_strategies

logger = get_logger(__name__)


async def _connect_exchange(settings: Settings) -> ExchangeGateway | None:
    """Open an authenticated exchange gateway, or return None.

    Only attempted when credentials exist. A failure here degrades the account endpoints
    rather than refusing to boot - but it never falls back to stored paper state, because
    a dashboard silently showing yesterday's backtest as a live balance is worse than one
    showing an error.
    """
    if not settings.exchange.has_credentials:
        logger.info("startup.exchange_skipped", reason="no API credentials configured")
        return None
    try:
        from quantflow.exchange.bybit import BybitGateway

        gateway = BybitGateway(settings.exchange)
        await gateway.connect()
        logger.info("startup.exchange_connected", venue=gateway.name, testnet=gateway.is_testnet)
    except Exception as exc:
        logger.warning("startup.exchange_failed", error=str(exc))
        return None
    return gateway


# FastAPI serialises response models straight to JSON bytes via Pydantic, so a custom
# ORJSON response class is both unnecessary and (as of this version) deprecated.

DESCRIPTION = """
AI-powered algorithmic trading platform for Binance.

**Every order passes through the risk engine.** There is no endpoint that bypasses it,
and live trading requires an explicit arming token in the environment.

Monetary values are transmitted as **strings**, not JSON numbers, so no precision is lost
in transit. Parse them with a decimal type, never with `parseFloat`.
"""


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build and tear down application resources."""
    settings: Settings = app.state.settings
    state = AppState(settings=settings)
    app.state.quantflow = state

    settings.storage.ensure_directories()
    state.registry = load_builtin_strategies()

    # Database: required. Without it there is nothing to serve.
    try:
        state.database = Database.from_settings(settings)
        if not await state.database.ping():
            logger.error("startup.database_unreachable", dsn=settings.database.safe_dsn)
    except Exception as exc:
        logger.exception("startup.database_failed", error=str(exc))
        state.database = None

    # Redis: optional at startup; the app degrades rather than refusing to boot.
    try:
        state.cache = Cache.from_settings(settings)
        if await state.cache.ping():
            state.event_bus = EventBus(state.cache)
        else:
            logger.warning("startup.redis_unreachable", url=settings.redis.safe_url)
    except Exception as exc:
        logger.warning("startup.redis_failed", error=str(exc))
        state.cache = None

    state.gateway = await _connect_exchange(settings)

    dispatcher = build_dispatcher(settings.notifications)
    state.extras["dispatcher"] = dispatcher
    state.risk = RiskEngine(settings.risk, database=state.database, notifier=dispatcher)
    await state.risk.start()

    logger.info(
        "api.started",
        version=__version__,
        environment=settings.env.value,
        trading_mode=settings.trading.mode.value,
        live_armed=settings.is_live,
        strategies=len(state.registry),
        kill_switch_engaged=state.risk.kill_switch.engaged,
    )

    try:
        yield
    finally:
        if state.gateway is not None:
            with contextlib.suppress(Exception):
                await state.gateway.aclose()
        if state.cache is not None:
            with contextlib.suppress(Exception):
                await state.cache.aclose()
        if state.database is not None:
            with contextlib.suppress(Exception):
                await state.database.aclose()
        with contextlib.suppress(Exception):
            await dispatcher.aclose()
        logger.info("api.stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Override configuration. Defaults to the process settings singleton.

    """
    active = settings or get_settings()
    configure_logging(active, service="quantflow-api")

    # Hide the interactive docs on an unauthenticated production deployment: they are a
    # live description of every endpoint, including the kill switch.
    expose_docs = not (active.env.is_production_like and active.api_key is None)
    app = FastAPI(
        title=active.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if expose_docs else None,
        redoc_url="/redoc" if expose_docs else None,
        openapi_url="/openapi.json" if expose_docs else None,
    )
    app.state.settings = active

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
    )
    middleware.install(app)

    # System routes sit at the root: an orchestrator's probe should not have to know the
    # API version prefix.
    app.include_router(system.router)

    prefix = active.api_prefix
    app.include_router(marketdata.router, prefix=prefix)
    app.include_router(portfolio.router, prefix=prefix)
    app.include_router(risk.router, prefix=prefix)
    app.include_router(backtest.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)
    app.include_router(account.router, prefix=prefix)

    _install_websocket(app, prefix)
    return app


def _install_websocket(app: FastAPI, prefix: str) -> None:
    """Attach the live event stream used by the dashboard."""

    @app.websocket(f"{prefix}/ws")
    async def stream(websocket: WebSocket) -> None:
        """Relay engine events (fills, signals, risk, equity) to the dashboard.

        Backed by the Redis pub/sub bus, so any process in the deployment can publish and
        every connected dashboard sees it — polling would either lag behind a fill or
        hammer the API to avoid doing so.
        """
        await websocket.accept()
        state: AppState = websocket.app.state.quantflow

        if state.cache is None:
            await websocket.send_json({"type": "error", "message": "event bus is unavailable"})
            await websocket.close(code=1011)
            return

        channels = (
            EventBus.CHANNEL_FILLS,
            EventBus.CHANNEL_SIGNALS,
            EventBus.CHANNEL_RISK,
            EventBus.CHANNEL_EQUITY,
            EventBus.CHANNEL_SYSTEM,
        )
        try:
            async with state.cache.subscribe(*channels) as events:
                await websocket.send_json({"type": "connected", "channels": list(channels)})
                async for channel, message in events:
                    await websocket.send_json({"channel": channel, "data": message})
        except WebSocketDisconnect:
            logger.debug("ws.disconnected")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("ws.failed", error=str(exc))
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)


app = None
"""Module-level app is deliberately absent.

Uvicorn is pointed at the factory (`--factory quantflow.api.app:create_app`) so settings
are read at start time rather than at import time. A module-level app would evaluate
configuration during import, which breaks tests that need to override it.
"""
