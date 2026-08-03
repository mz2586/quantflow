"""Portfolio, orders, trades and equity-curve endpoints.

The portfolio is served from the **database** whenever the API process is not itself
running the trading engine. That is the normal deployment: the API, the worker and the
trading runner are separate processes, so an endpoint that only reads an in-process
manager would return "no session" forever in production while trading proceeded happily
next door.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Query

from quantflow.api.deps import DatabaseDep, OptionalDatabaseDep, StateDep
from quantflow.api.schemas import (
    EquityPointResponse,
    OrderResponse,
    PortfolioResponse,
    PositionResponse,
    SessionResponse,
    TradeResponse,
)
from quantflow.core.clock import utc_now
from quantflow.core.errors import NotFoundError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO, safe_divide
from quantflow.domain.enums import PositionSide
from quantflow.domain.instruments import Symbol
from quantflow.persistence.database import Database
from quantflow.persistence.repositories import (
    ClosedTradeRepository,
    EquityRepository,
    OrderRepository,
    PositionRepository,
    TradingSessionRepository,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _order_response(order: object) -> OrderResponse:
    """Map a domain order onto the wire schema."""
    from quantflow.domain.orders import Order

    assert isinstance(order, Order)
    return OrderResponse(
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


async def _portfolio_from_database(database: Database) -> PortfolioResponse:
    """Reconstruct the portfolio from the most recent session's persisted state.

    Raises:
        NotFoundError: if no session has ever run, which is genuinely different from a
            session that has run and is flat.

    """
    async with database.read_session() as session:
        sessions = await TradingSessionRepository(session).list_recent(limit=1)
        if not sessions:
            raise NotFoundError(
                "no trading session has run yet; start a paper session to populate this"
            )
        record = sessions[0]
        latest = await EquityRepository(session).latest(record.session_id)
        positions = await PositionRepository(session).list_open(session_id=record.session_id)

    starting = record.starting_equity
    equity = latest.equity if latest is not None else starting
    cash = latest.cash if latest is not None else starting

    responses: list[PositionResponse] = []
    gross_exposure = ZERO
    unrealized = ZERO
    for position in positions:
        # The last mark is not persisted per position, so the entry price is used as the
        # valuation reference. Labelled honestly rather than presented as a live mark.
        mark = position.average_entry_price
        gross_exposure += position.notional(mark)
        responses.append(
            PositionResponse(
                symbol=position.symbol.slashed,
                side=position.side,
                quantity=position.quantity,
                average_entry_price=position.average_entry_price,
                mark_price=None,
                unrealized_pnl=ZERO,
                unrealized_pnl_pct=ZERO,
                realized_pnl=position.realized_pnl,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                opened_at=position.opened_at,
                strategy_id=position.strategy_id,
            )
        )

    return PortfolioResponse(
        base_currency=record.base_currency,
        equity=equity,
        cash=cash,
        starting_equity=starting,
        total_return_pct=safe_divide(equity - starting, starting),
        realized_pnl=latest.realized_pnl if latest else ZERO,
        unrealized_pnl=unrealized,
        fees_paid=ZERO,
        gross_exposure=gross_exposure,
        leverage=safe_divide(gross_exposure, equity),
        drawdown_pct=latest.drawdown_pct if latest else ZERO,
        daily_pnl=ZERO,
        position_count=len(responses),
        positions=tuple(responses),
    )


@router.get("", response_model=PortfolioResponse, summary="Current portfolio")
async def get_portfolio(state: StateDep, database: OptionalDatabaseDep) -> PortfolioResponse:
    """Current cash, positions, exposure and PnL.

    Prefers the live in-process manager when the API is itself running an engine; falls
    back to persisted session state otherwise, which is the normal multi-process case.

    Raises:
        NotFoundError: when there is neither a running engine nor a persisted session.

    """
    manager = state.portfolio
    if manager is None:
        if database is None:
            # Neither a live engine nor persistence: that is "nothing has traded",
            # not a misconfiguration, and it should read that way to an operator.
            raise NotFoundError("no active trading session")
        return await _portfolio_from_database(database)

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
    return [_order_response(order) for order in orders]


@router.get("/orders/open", response_model=list[OrderResponse], summary="Working orders")
async def list_open_orders(
    database: DatabaseDep, session_id: str | None = None
) -> list[OrderResponse]:
    """Orders that can still receive fills."""
    async with database.read_session() as session:
        orders = await OrderRepository(session).list_open(session_id=session_id)
    return [_order_response(order) for order in orders]


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


@router.get("/equity", response_model=list[EquityPointResponse], summary="Latest equity curve")
async def get_latest_equity_curve(
    database: DatabaseDep,
    limit: Annotated[int, Query(ge=1, le=100_000)] = 5_000,
) -> list[EquityPointResponse]:
    """The equity curve of the most recent session.

    Saves the dashboard a round trip to discover the session id before it can draw a
    chart, which would otherwise mean an empty panel on first paint.
    """
    async with database.read_session() as session:
        sessions = await TradingSessionRepository(session).list_recent(limit=1)
        if not sessions:
            return []
        curve = await EquityRepository(session).curve(sessions[0].session_id, limit=limit)
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
        # Built inside the session on purpose: these are ORM records, and the read
        # session's rollback on exit expires them.
        return [
            SessionResponse(
                session_id=record.session_id,
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


def position_side_label(side: PositionSide) -> str:
    """Human label for a position side."""
    return side.value.upper()
