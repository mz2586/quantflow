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
    dashboard,
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


#: Hard limit on the exchange handshake during startup.
#:
#: Uvicorn runs the lifespan to completion *before* it binds the listening socket, so a
#: slow handshake here is not a degraded account panel — it is an API that never answers
#: anything at all, including its own health probe. That happened: a contended venue call
#: left the process unreachable for twenty minutes while the container reported "starting".
#: Bounded, so a slow venue costs the account panel and nothing else.
EXCHANGE_CONNECT_TIMEOUT_SECONDS = 20.0

#: Minimum gap between attempts to re-establish a gateway that failed at startup.
EXCHANGE_RECONNECT_COOLDOWN_SECONDS = 60.0


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
        await asyncio.wait_for(gateway.connect(), timeout=EXCHANGE_CONNECT_TIMEOUT_SECONDS)
        logger.info("startup.exchange_connected", venue=gateway.name, testnet=gateway.is_testnet)
    except TimeoutError:
        logger.warning("startup.exchange_timeout", timeout_seconds=EXCHANGE_CONNECT_TIMEOUT_SECONDS)
        return None
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
    app.include_router(dashboard.router, prefix=prefix)

    _install_websocket(app, prefix)
    return app


#: How often the live snapshot is pushed. Two seconds is the operator-visible latency
#: target for account, position and order changes; below that the venue read itself becomes
#: the bottleneck and adds request load without adding information.
SNAPSHOT_INTERVAL_SECONDS = 2.0


async def _push_venue_snapshots(websocket: WebSocket, state: AppState) -> None:
    """Broadcast the venue's account and position state on a fixed interval.

    Reads through the same cache the REST endpoints use, so this adds no venue load beyond
    what the dashboard already generates — it changes who initiates the update, not how
    often the venue is asked.

    A failed read is sent as an explicit error frame rather than skipped. A client that
    stops receiving snapshots must be able to tell "nothing changed" from "the venue went
    away", and silence cannot distinguish them.
    """
    from quantflow.api.routers.dashboard import venue_snapshot

    while True:
        try:
            payload = await venue_snapshot(state)
        except Exception as exc:  # pragma: no cover - a read fault must not kill the socket
            payload = {"available": False, "error": str(exc)[:200]}
        await websocket.send_json({"channel": "venue", "data": payload})
        await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)


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

                # The engine publishes only on signals and fills — at most once per 15m
                # bar. A socket carrying just those is genuinely connected and genuinely
                # useless: prices move, unrealised PnL moves, positions open and close, and
                # none of it produces an event. Measured before this existed: connect,
                # handshake, then twenty-five seconds of silence.
                #
                # So the API pushes what it already reads. A ticker task broadcasts the
                # venue snapshot on a short interval, which is what makes account, position
                # and order changes appear without the client polling for them. Engine
                # events still arrive the instant they happen; this fills the gaps between.
                async def relay_engine_events() -> None:
                    async for channel, message in events:
                        await websocket.send_json({"channel": channel, "data": message})

                ticker = asyncio.create_task(_push_venue_snapshots(websocket, state))
                relay = asyncio.create_task(relay_engine_events())
                try:
                    done, pending = await asyncio.wait(
                        {ticker, relay}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        task.result()
                finally:
                    for task in (ticker, relay):
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await task
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
