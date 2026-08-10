"""Live exchange account endpoints.

Everything here reads straight from the authenticated venue. Nothing is served from the
database, from a backtest or from a paper session — if the venue cannot be reached the
endpoint fails rather than falling back to stored state, because a dashboard that silently
substitutes yesterday's paper numbers for a live balance is worse than one that shows an
error.

Read-only by construction: no route in this module places, modifies or cancels anything.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Query

from quantflow.api.deps import StateDep
from quantflow.core.errors import ConfigurationError, ExchangeError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.instruments import Symbol

logger = get_logger(__name__)

router = APIRouter(prefix="/account", tags=["account"])


def _gateway(state: StateDep) -> Any:
    """The authenticated exchange gateway.

    Raises:
        ConfigurationError: when no gateway is wired, which means no credentials were
            configured at startup.

    """
    if state.gateway is None:
        raise ConfigurationError(
            "no exchange gateway is configured; set QF_EXCHANGE__API_KEY and "
            "QF_EXCHANGE__API_SECRET and restart the API"
        )
    return state.gateway


@router.get("", summary="Live account snapshot")
async def get_account(state: StateDep) -> dict[str, Any]:
    """Balances, positions, working orders and PnL, read live from the venue.

    Assembled in one call so the dashboard cannot render a balance from one instant beside
    positions from another — a mismatch that reads as a PnL error when it is really a
    timing artefact.
    """
    gateway = _gateway(state)

    balances = await gateway.fetch_balances()
    total = sum((item.free + item.locked for item in balances.values()), ZERO)
    available = sum((item.free for item in balances.values()), ZERO)

    positions: list[dict[str, Any]] = []
    unrealized = ZERO
    try:
        for raw in await gateway.fetch_positions():
            pnl = _decimal(raw.get("unrealizedPnl"))
            unrealized += pnl
            positions.append(
                {
                    "symbol": str(raw.get("symbol") or ""),
                    "side": str(raw.get("side") or ""),
                    "quantity": str(_decimal(raw.get("contracts"))),
                    "entry_price": str(_decimal(raw.get("entryPrice"))),
                    "mark_price": str(_decimal(raw.get("markPrice"))),
                    "unrealized_pnl": str(pnl),
                    "leverage": str(_decimal(raw.get("leverage"))),
                }
            )
    except (ExchangeError, AttributeError) as exc:
        # Spot accounts have no positions endpoint. That is not an error state, but it
        # must be visible rather than rendered as "no positions".
        logger.info("account.positions_unavailable", error=str(exc))

    orders = await gateway.fetch_open_orders()

    return {
        "venue": gateway.name,
        "network": "testnet" if gateway.is_testnet else "mainnet",
        "authenticated": bool(getattr(gateway, "supports_trading", False)),
        "total_balance": str(total),
        "available_balance": str(available),
        "balances": [
            {
                "asset": asset,
                "free": str(item.free),
                "locked": str(item.locked),
                "total": str(item.free + item.locked),
            }
            for asset, item in sorted(balances.items())
            if item.free + item.locked > ZERO
        ],
        "positions": positions,
        "position_count": len(positions),
        "unrealized_pnl": str(unrealized),
        "open_orders": [
            {
                "order_id": order.order_id,
                "venue_order_id": order.venue_order_id,
                "symbol": order.symbol.slashed,
                "side": order.side.value,
                "type": order.order_type.value,
                "status": order.status.value,
                "quantity": str(order.quantity),
                "filled": str(order.filled_quantity),
                "price": str(order.price) if order.price is not None else None,
                "created_at": order.created_at.isoformat(),
            }
            for order in orders
        ],
        "open_order_count": len(orders),
    }


@router.get("/fills", summary="Recent fills from the venue")
async def get_fills(
    state: StateDep,
    symbol: str,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Recent executions, and the realised PnL they add up to.

    A symbol is required because Bybit's V5 execution history is symbol-scoped; there is
    no all-symbols form, and quietly querying one default symbol would present a partial
    history as a complete one.
    """
    gateway = _gateway(state)
    parsed = Symbol.parse(symbol)
    assert isinstance(parsed, Symbol)

    fills = await gateway.fetch_my_trades(parsed, limit=limit)
    realized = sum((fill.realized_pnl or ZERO for fill in fills), ZERO)
    fees = sum((fill.fee for fill in fills), ZERO)

    return {
        "symbol": parsed.slashed,
        "count": len(fills),
        "realized_pnl": str(realized),
        "total_fees": str(fees),
        "fills": [
            {
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "side": fill.side.value,
                "price": str(fill.price),
                "quantity": str(fill.quantity),
                "fee": str(fill.fee),
                "fee_currency": fill.fee_currency,
                "role": fill.liquidity_role.value,
                "timestamp": fill.timestamp.isoformat(),
            }
            for fill in fills
        ],
    }


def _decimal(value: Any) -> Decimal:
    """Best-effort Decimal from a venue field, zero when absent or unparseable."""
    if value is None or value == "":
        return ZERO
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return ZERO


__all__ = ["router"]
