"""Health, readiness and metrics endpoints.

Liveness and readiness are deliberately different things. ``/healthz`` says the process is
running; ``/readyz`` says it can actually serve traffic. Conflating them means an
orchestrator restarts a healthy process because the database blipped, or keeps routing
traffic to one that cannot reach any of its dependencies.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from quantflow import __version__
from quantflow.api.deps import StateDep
from quantflow.api.middleware import metrics_response
from quantflow.api.schemas import ComponentHealth, HealthResponse, ReadinessResponse

router = APIRouter(tags=["system"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz(state: StateDep) -> HealthResponse:
    """Report that the process is alive.

    Never touches a dependency: a liveness probe that fails when Postgres is briefly
    unreachable causes the orchestrator to kill a process that would have recovered.
    """
    return HealthResponse(version=__version__, environment=state.settings.env.value)


@router.get("/readyz", response_model=ReadinessResponse, summary="Readiness probe")
async def readyz(state: StateDep, response: Response) -> ReadinessResponse:
    """Report whether every dependency is reachable.

    Returns 503 when any component is down, so a load balancer stops sending traffic to an
    instance that cannot serve it.
    """
    components: list[ComponentHealth] = []

    if state.database is not None:
        started = time.perf_counter()
        healthy = await state.database.ping()
        components.append(
            ComponentHealth(
                name="postgres",
                healthy=healthy,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=None if healthy else "ping failed",
            )
        )

    if state.cache is not None:
        started = time.perf_counter()
        healthy = await state.cache.ping()
        components.append(
            ComponentHealth(
                name="redis",
                healthy=healthy,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=None if healthy else "ping failed",
            )
        )

    if state.gateway is not None:
        components.append(
            ComponentHealth(
                name="exchange",
                healthy=True,
                detail=f"{state.gateway.name}"
                + (" (testnet)" if state.gateway.is_testnet else " (production)"),
            )
        )

    kill_switch_engaged = state.risk.kill_switch.engaged if state.risk is not None else False
    ready = all(component.healthy for component in components)
    if not ready:
        response.status_code = HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=ready,
        components=tuple(components),
        trading_mode=state.settings.trading.mode.value,
        kill_switch_engaged=kill_switch_engaged,
    )


@router.get("/metrics", include_in_schema=False, summary="Prometheus metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics."""
    return metrics_response()
