"""Analytics endpoints.

Every response carries its own caveats. A win rate over eight trades and a win rate over
eight hundred look identical in JSON, and a dashboard that renders them the same way
invites the reader to act on noise.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Query

from quantflow.analytics.performance import (
    by_hour_of_day,
    by_strategy,
    by_symbol,
    review,
    rolling_win_rate,
)
from quantflow.api.deps import DatabaseDep, StateDep
from quantflow.core.clock import utc_now
from quantflow.core.errors import NotFoundError
from quantflow.notifications.dispatcher import describe as describe_dispatcher
from quantflow.persistence.repositories import ClosedTradeRepository, EquityRepository

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/review", summary="Full performance review")
async def performance_review(
    database: DatabaseDep,
    session_id: str | None = None,
    days: Annotated[int, Query(ge=1, le=3650)] = 90,
) -> dict[str, Any]:
    """Attribution, streaks, concentration and drawdown episodes.

    Scoped to one session when ``session_id`` is given, otherwise to the trailing
    ``days`` window.
    """
    async with database.read_session() as session:
        trade_repo = ClosedTradeRepository(session)
        if session_id:
            trades = await trade_repo.list_for_session(session_id, limit=10_000)
            curve = await EquityRepository(session).curve(session_id)
        else:
            now = utc_now()
            trades = await trade_repo.list_between(now - timedelta(days=days), now, limit=10_000)
            curve = []

    return review(trades, curve).to_dict()


@router.get("/attribution/strategy", summary="Performance by strategy")
async def strategy_attribution(
    database: DatabaseDep,
    days: Annotated[int, Query(ge=1, le=3650)] = 90,
) -> list[dict[str, Any]]:
    """Net PnL, win rate and fee drag per strategy."""
    now = utc_now()
    async with database.read_session() as session:
        trades = await ClosedTradeRepository(session).list_between(
            now - timedelta(days=days), now, limit=10_000
        )
    return [item.to_dict() for item in by_strategy(trades)]


@router.get("/attribution/symbol", summary="Performance by symbol")
async def symbol_attribution(
    database: DatabaseDep,
    days: Annotated[int, Query(ge=1, le=3650)] = 90,
) -> list[dict[str, Any]]:
    """Net PnL, win rate and fee drag per symbol."""
    now = utc_now()
    async with database.read_session() as session:
        trades = await ClosedTradeRepository(session).list_between(
            now - timedelta(days=days), now, limit=10_000
        )
    return [item.to_dict() for item in by_symbol(trades)]


@router.get("/attribution/hour", summary="Performance by UTC hour")
async def hourly_attribution(
    database: DatabaseDep,
    days: Annotated[int, Query(ge=1, le=3650)] = 90,
) -> list[dict[str, Any]]:
    """Performance by hour of entry.

    Crypto trades continuously but liquidity does not; a strategy can be quietly
    unprofitable in one session and hide it in the aggregate.
    """
    now = utc_now()
    async with database.read_session() as session:
        trades = await ClosedTradeRepository(session).list_between(
            now - timedelta(days=days), now, limit=10_000
        )
    return [item.to_dict() for item in by_hour_of_day(trades)]


@router.get("/win-rate", summary="Rolling win rate")
async def rolling_win_rate_series(
    database: DatabaseDep,
    session_id: str | None = None,
    window: Annotated[int, Query(ge=5, le=500)] = 20,
    days: Annotated[int, Query(ge=1, le=3650)] = 365,
) -> list[dict[str, str]]:
    """Win rate over a trailing window of trades.

    Reveals decay: a strategy whose win rate is drifting down is losing its edge, which a
    single aggregate figure hides completely.
    """
    async with database.read_session() as session:
        repository = ClosedTradeRepository(session)
        if session_id:
            trades = await repository.list_for_session(session_id, limit=10_000)
        else:
            now = utc_now()
            trades = await repository.list_between(now - timedelta(days=days), now, limit=10_000)
    return [
        {"timestamp": moment.isoformat(), "win_rate": str(rate)}
        for moment, rate in rolling_win_rate(trades, window=window)
    ]


@router.get("/session/{session_id}", summary="Analytics for one session")
async def session_analytics(session_id: str, database: DatabaseDep) -> dict[str, Any]:
    """Everything the dashboard needs for a single session."""
    async with database.read_session() as session:
        trades = await ClosedTradeRepository(session).list_for_session(session_id, limit=10_000)
        curve = await EquityRepository(session).curve(session_id)

    if not trades and not curve:
        raise NotFoundError(f"no analytics for session {session_id}", session_id=session_id)

    return {
        "session_id": session_id,
        "review": review(trades, curve).to_dict(),
        "equity_points": len(curve),
    }


@router.get("/notifications", summary="Notification transport status")
async def notification_status(state: StateDep) -> dict[str, Any]:
    """Which alert transports are live, and what has been suppressed."""
    dispatcher = state.extras.get("dispatcher")
    if dispatcher is None:
        return {"transports": [], "configured": False}
    described = await describe_dispatcher(dispatcher)
    return {**described, "configured": True}
