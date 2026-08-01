"""Portfolio, orders, trades and equity-curve endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from quantflow.api.deps import DatabaseDep, StateDep
from quantflow.api.schemas import (
    EquityPointResponse,
    OrderResponse,
    PortfolioResponse,
    PositionResponse,
    SessionResponse,
    TradeResponse,
)
from quantflow.core.errors import ConfigurationError, NotFoundError
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.instruments import Symbol
from quantflow.persistence.repositories import (
    ClosedTradeRepository,
    EquityRepository,
    OrderRepository,
    TradingSessionRepository,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("", response_model=PortfolioResponse, summary="Current portfolio")
async def get_portfolio(state: StateDep) -> PortfolioResponse:
    """Current cash, positions, exposure and PnL.

    Raises:
        ConfigurationError: if no trading session is active, so the caller sees a clear
            "nothing is running" rather than a portfolio of zeros that looks like a
            flat account.

    """
    manager = state.portfolio
    if manager is None:
        raise ConfigurationError("no active trading session; start a paper or live session first")

    snapshot = manager.snapshot()
    positions: list[PositionResponse] = []
    for position in snapshot.open_positions:
        mark = snapshot.mark_prices.get(position.symbol)
        positions.append(
            PositionResponse(
                symbol=position.symbol.slashed,
                side=position.side,
                quantity=position.quantity,
                average_entry_price=position.average_entry_price,
                mark_price=mark,
                unrealized_pnl=position.unrealized_pnl(mark) if mark else ZERO,
                unrealized_pnl_pct=position.unrealized_pnl_pct(mark) if mark else ZERO,
                realized_pnl=position.realized_pnl,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                opened_at=position.opened_at,
                strategy_id=position.strategy_id,
            )
        )

    equity = snapshot.equity
    return PortfolioResponse(
        base_currency=snapshot.base_currency,
        equity=equity,
        cash=snapshot.cash,
        starting_equity=manager.starting_equity,
        total_return_pct=safe_divide(equity - manager.starting_equity, manager.starting_equity),
        realized_pnl=snapshot.realized_pnl,
        unrealized_pnl=snapshot.unrealized_pnl,
        fees_paid=snapshot.fees_paid,
        gross_exposure=snapshot.gross_exposure,
        leverage=snapshot.leverage,
        drawdown_pct=snapshot.drawdown_pct,
        daily_pnl=snapshot.daily_pnl,
        position_count=snapshot.position_count,
        positions=tuple(positions),
    )


@router.get("/orders", response_model=list[OrderResponse], summary="Recent orders")
async def list_orders(
    database: DatabaseDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    symbol: str | None = None,
    session_id: str | None = None,
) -> list[OrderResponse]:
    """Recent orders, newest first."""
    parsed = Symbol.parse(symbol) if symbol else None
    async with database.read_session() as session:
        orders = await OrderRepository(session).list_recent(
            limit=limit,
            symbol=parsed if isinstance(parsed, Symbol) else None,
            session_id=session_id,
        )
    return [
        OrderResponse(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            venue_order_id=order.venue_order_id,
            symbol=order.symbol.slashed,
            side=order.side,
            order_type=order.order_type,
            status=order.status,
            quantity=order.quantity,
            price=order.price,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            fees_paid=order.fees_paid,
            stop_loss_price=order.stop_loss_price,
            strategy_id=order.strategy_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
            reject_reason=order.reject_reason,
        )
        for order in orders
    ]


@router.get("/orders/open", response_model=list[OrderResponse], summary="Working orders")
async def list_open_orders(
    database: DatabaseDep, session_id: str | None = None
) -> list[OrderResponse]:
    """Orders that can still receive fills."""
    async with database.read_session() as session:
        orders = await OrderRepository(session).list_open(session_id=session_id)
    return [
        OrderResponse(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            venue_order_id=order.venue_order_id,
            symbol=order.symbol.slashed,
            side=order.side,
            order_type=order.order_type,
            status=order.status,
            quantity=order.quantity,
            price=order.price,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            fees_paid=order.fees_paid,
            stop_loss_price=order.stop_loss_price,
            strategy_id=order.strategy_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
            reject_reason=order.reject_reason,
        )
        for order in orders
    ]


@router.get("/trades", response_model=list[TradeResponse], summary="Closed trades")
async def list_trades(
    database: DatabaseDep,
    session_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> list[TradeResponse]:
    """Completed round-trips."""
    async with database.read_session() as session:
        repository = ClosedTradeRepository(session)
        if session_id:
            trades = await repository.list_for_session(session_id, limit=limit)
        else:
            from datetime import timedelta

            from quantflow.core.clock import utc_now

            now = utc_now()
            trades = await repository.list_between(now - timedelta(days=365), now, limit=limit)
    return [
        TradeResponse(
            symbol=trade.symbol.slashed,
            side=trade.side,
            quantity=trade.quantity,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            gross_pnl=trade.gross_pnl,
            fees=trade.fees,
            net_pnl=trade.net_pnl,
            return_pct=trade.return_pct,
            holding_hours=trade.holding_period / 3600,
            strategy_id=trade.strategy_id,
        )
        for trade in trades
    ]


@router.get(
    "/equity/{session_id}",
    response_model=list[EquityPointResponse],
    summary="Equity curve",
)
async def get_equity_curve(
    session_id: str,
    database: DatabaseDep,
    limit: Annotated[int, Query(ge=1, le=100_000)] = 10_000,
) -> list[EquityPointResponse]:
    """The equity curve for one session."""
    async with database.read_session() as session:
        curve = await EquityRepository(session).curve(session_id, limit=limit)
    if not curve:
        raise NotFoundError(f"no equity curve for session {session_id}", session_id=session_id)
    return [
        EquityPointResponse(
            timestamp=point.timestamp,
            equity=point.equity,
            cash=point.cash,
            drawdown_pct=point.drawdown_pct,
            position_count=point.position_count,
        )
        for point in curve
    ]


@router.get("/sessions", response_model=list[SessionResponse], summary="Recent sessions")
async def list_sessions(
    database: DatabaseDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[SessionResponse]:
    """Recent trading sessions, newest first."""
    async with database.read_session() as session:
        records = await TradingSessionRepository(session).list_recent(limit=limit)
    return [
        SessionResponse(
            session_id=record.id,
            mode=record.mode,
            status=record.status,
            strategy_id=record.strategy_id,
            symbols=tuple(record.symbols),
            timeframe=record.timeframe,
            starting_equity=record.starting_equity,
            final_equity=record.final_equity,
            started_at=record.started_at,
            finished_at=record.finished_at,
            error=record.error,
        )
        for record in records
    ]
