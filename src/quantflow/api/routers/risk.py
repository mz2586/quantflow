"""Risk endpoints, including the kill switch.

The kill switch is the only endpoint in the API that can stop trading instantly, and it is
also the one most likely to be reached for under pressure. It is therefore: idempotent,
audited, and never silently a no-op.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from quantflow.api.deps import AuthDep, DatabaseDep, RiskDep, StateDep
from quantflow.api.schemas import (
    KillSwitchRequest,
    KillSwitchResponse,
    RiskEventResponse,
    RiskLimits,
    RiskStatusResponse,
)
from quantflow.core.errors import ValidationError
from quantflow.core.logging import get_logger
from quantflow.persistence.repositories import RiskEventRepository
from quantflow.risk.engine import summarise_headroom

logger = get_logger(__name__)

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/status", response_model=RiskStatusResponse, summary="Current risk state")
async def get_status(state: StateDep, risk: RiskDep) -> RiskStatusResponse:
    """Report limits, kill-switch state and remaining headroom.

    The switch is re-read from storage on every request rather than served from whatever
    this process loaded at startup. The switch is engaged and cleared from *other*
    processes — the CLI, the trading engine — so a cached copy goes stale the moment
    anyone else touches it, and a dashboard that reports HALTED while the engine trades
    normally is worse than one that reports nothing: it is confidently wrong about the one
    control an operator reaches for in an emergency.
    """
    settings = state.settings.risk
    await risk.refresh_kill_switch()
    switch = risk.kill_switch.state

    headroom: dict[str, str] = {}
    if state.portfolio is not None:
        try:
            headroom = summarise_headroom(state.portfolio.snapshot(), settings)
        except Exception as exc:
            logger.debug("risk.headroom_unavailable", reason=str(exc))

    described = risk.describe()
    return RiskStatusResponse(
        trading_halted=risk.is_halted,
        kill_switch=KillSwitchResponse(
            engaged=switch.engaged,
            reason=switch.reason,
            engaged_at=switch.engaged_at,
            engaged_by=switch.engaged_by,
        ),
        limits=RiskLimits(
            max_position_pct=settings.max_position_pct,
            max_total_exposure_pct=settings.max_total_exposure_pct,
            max_concurrent_positions=settings.max_concurrent_positions,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
            max_leverage=settings.max_leverage,
            require_stop_loss=settings.require_stop_loss,
            max_order_notional=settings.max_order_notional,
            max_orders_per_minute=settings.max_orders_per_minute,
        ),
        headroom=headroom,
        sizer=str(described["sizer"]),
        rules=tuple(described["rules"]),
    )


@router.post(
    "/kill-switch",
    response_model=KillSwitchResponse,
    summary="Engage or clear the kill switch",
)
async def set_kill_switch(
    request: KillSwitchRequest, risk: RiskDep, _auth: AuthDep
) -> KillSwitchResponse:
    """Latch or clear the emergency halt.

    Engaging requires a reason: a halt with no recorded cause is close to useless during
    the post-mortem that follows it. Clearing is deliberately a separate, explicit call —
    the switch never times out on its own.
    """
    if request.engaged:
        if not request.reason or not request.reason.strip():
            raise ValidationError("a reason is required when engaging the kill switch")
        state = await risk.kill_switch.engage(request.reason.strip(), actor=request.actor)
        logger.critical("api.kill_switch_engaged", actor=request.actor, reason=request.reason)
    else:
        state = await risk.kill_switch.clear(actor=request.actor)
        risk.resume()
        logger.warning("api.kill_switch_cleared", actor=request.actor)

    return KillSwitchResponse(
        engaged=state.engaged,
        reason=state.reason,
        engaged_at=state.engaged_at,
        engaged_by=state.engaged_by,
    )


@router.get("/events", response_model=list[RiskEventResponse], summary="Risk audit trail")
async def list_events(
    database: DatabaseDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    session_id: str | None = None,
) -> list[RiskEventResponse]:
    """Every risk decision that blocked an order or halted trading, newest first."""
    async with database.read_session() as session:
        events = await RiskEventRepository(session).list_recent(limit=limit, session_id=session_id)
    return [
        RiskEventResponse(
            rule=event.rule,
            severity=event.severity,
            message=event.message,
            symbol=event.symbol,
            observed_value=event.observed_value,
            limit_value=event.limit_value,
            blocked_order=event.blocked_order,
            halted_trading=event.halted_trading,
            created_at=event.created_at,
        )
        for event in events
    ]


@router.post("/resume", response_model=KillSwitchResponse, summary="Lift a daily halt")
async def resume_trading(risk: RiskDep, _auth: AuthDep) -> KillSwitchResponse:
    """Lift a daily-loss halt.

    Deliberately does **not** clear the kill switch: a daily halt and a latched drawdown
    breach are different severities, and one button that clears both invites clearing the
    serious one by reflex.
    """
    risk.resume()
    state = risk.kill_switch.state
    logger.warning("api.trading_resumed", kill_switch_still_engaged=state.engaged)
    return KillSwitchResponse(
        engaged=state.engaged,
        reason=state.reason,
        engaged_at=state.engaged_at,
        engaged_by=state.engaged_by,
    )
