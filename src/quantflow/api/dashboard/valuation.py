"""Venue account state, with assets that are not the same unit kept apart.

The defect this module exists to remove
--------------------------------------
The previous account endpoint reported a "total balance" computed as::

    total = sum(item.free + item.locked for item in balances.values())

On the live demo account that is ``49,902.01 USDT + 50,000 USDC + 1 BTC + 1 ETH``, summed
as though a bitcoin and a dollar were the same thing, and rendered as ``99,904.01`` under
the label *Total balance*. The number is not merely imprecise, it is meaningless: it has no
unit. It also looks like a plausible account size, which is what makes it dangerous — an
operator reading it would believe they had roughly twice the capital the engine can
actually deploy, because the engine sizes in USDT and the two extra coins contribute
``2`` to that figure while being worth far more.

What replaces it
----------------
Three separate concepts, never combined implicitly:

1. **Trading equity (USDT)** and **available USDT** — the authoritative figures the engine
   sizes against, taken from the USDT balance alone.
2. **Other assets**, listed individually with their quantities, valued *only* where an
   authoritative current price is available from the venue.
3. **Total portfolio value (USDT)** — offered only when every non-USDT holding could be
   priced, and always accompanied by the valuation method and the timestamp of the prices
   used. When even one asset cannot be priced there is no total at all, because a total
   that silently omits a holding is the same class of error as the one being fixed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from quantflow.core.clock import utc_now
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.instruments import Symbol
from quantflow.domain.portfolio import Balance

logger = get_logger(__name__)

#: The unit the engine sizes, prices and reports in. Everything else is "another asset".
QUOTE_ASSET = "USDT"

#: Assets treated as the quote unit itself. Only USDT qualifies: a stablecoin that trades
#: near a dollar is *not* the quote unit, and assuming USDC == USDT is exactly the kind of
#: unstated conversion this module exists to prevent. USDC is valued at its traded rate
#: like any other asset.
QUOTE_EQUIVALENTS = frozenset({QUOTE_ASSET})


@dataclass(frozen=True, slots=True)
class AssetValuation:
    """One non-quote holding, and the price used to value it if one was obtainable.

    Attributes:
        asset: Asset code, e.g. ``BTC``.
        free: Quantity available to trade.
        locked: Quantity reserved against working orders.
        price: Authoritative current price in the quote asset, or ``None``.
        value: ``(free + locked) * price``, or ``None`` when unpriced.
        source: How the price was obtained, for display beside the number.
        reason: Why no price was obtainable, when there is none.

    """

    asset: str
    free: Decimal
    locked: Decimal
    price: Decimal | None
    value: Decimal | None
    source: str | None
    reason: str | None

    @property
    def total(self) -> Decimal:
        """Total quantity held."""
        return self.free + self.locked

    def to_dict(self) -> dict[str, Any]:
        """Wire form, with every quantity as an exact decimal string."""
        return {
            "asset": self.asset,
            "free": str(self.free),
            "locked": str(self.locked),
            "quantity": str(self.total),
            "price_usdt": str(self.price) if self.price is not None else None,
            "value_usdt": str(self.value) if self.value is not None else None,
            "valuation_source": self.source,
            "unpriced_reason": self.reason,
        }


async def _price_of(gateway: Any, asset: str) -> tuple[Decimal | None, str | None, str | None]:
    """Fetch an authoritative current price for one asset, in the quote asset.

    The mid of the venue's own book is used rather than the last trade: the mid is the
    price the account could realistically transact near, and it is what the engine's own
    cost model works from.

    Args:
        gateway: Authenticated exchange gateway.
        asset: Asset code to price.

    Returns:
        ``(price, source, reason)``. Exactly one of ``price`` and ``reason`` is set.

    """
    symbol = Symbol(base=asset, quote=QUOTE_ASSET)
    try:
        ticker = await gateway.fetch_ticker(symbol)
    except Exception as exc:
        logger.info("dashboard.price_unavailable", asset=asset, error=str(exc))
        return None, None, f"no current price for {symbol.slashed}: {exc}"

    mid = ticker.mid
    if mid <= ZERO:
        return None, None, f"venue returned a non-positive mid for {symbol.slashed}"
    return mid, f"{gateway.name} {symbol.slashed} book mid @ {ticker.timestamp.isoformat()}", None


async def value_balances(gateway: Any, balances: dict[str, Balance]) -> dict[str, Any]:
    """Split an account into quote-asset equity and separately-valued other holdings.

    Args:
        gateway: Authenticated exchange gateway, used to price non-quote assets.
        balances: Balances keyed by asset code, as returned by the gateway.

    Returns:
        A JSON-safe mapping. ``total_portfolio_value_usdt`` is ``None`` whenever any held
        asset could not be priced; ``unpriced_assets`` then names them.

    """
    quote = balances.get(QUOTE_ASSET, Balance(asset=QUOTE_ASSET, free=ZERO, locked=ZERO))

    others = [
        balance
        for asset, balance in sorted(balances.items())
        if asset not in QUOTE_EQUIVALENTS and balance.total > ZERO
    ]

    # Priced concurrently: serially this is one venue round trip per asset, and the
    # endpoint's deadline is shared with the balance and position reads.
    prices = await asyncio.gather(
        *(_price_of(gateway, balance.asset) for balance in others),
        return_exceptions=False,
    )

    valuations: list[AssetValuation] = []
    for balance, (price, source, reason) in zip(others, prices, strict=True):
        valuations.append(
            AssetValuation(
                asset=balance.asset,
                free=balance.free,
                locked=balance.locked,
                price=price,
                value=(balance.total * price) if price is not None else None,
                source=source,
                reason=reason,
            )
        )

    unpriced = [item.asset for item in valuations if item.value is None]
    other_value = sum((item.value or ZERO for item in valuations), ZERO)

    # The total is offered only when it is complete. A total missing one holding is the
    # same defect as a total that added unlike units: confidently wrong, and trusted.
    total_value: Decimal | None = None
    valuation_method: str | None = None
    valued_at: str | None = None
    if not unpriced:
        total_value = quote.total + other_value
        valuation_method = (
            f"USDT balance held at par, plus each other asset valued at its current "
            f"{gateway.name} order-book mid against {QUOTE_ASSET}"
        )
        valued_at = utc_now().isoformat()

    return {
        # The two figures the engine actually sizes against. Never a cross-asset sum.
        "trading_equity_usdt": str(quote.total),
        "available_usdt": str(quote.free),
        "locked_usdt": str(quote.locked),
        "quote_asset": QUOTE_ASSET,
        "other_assets": [item.to_dict() for item in valuations],
        "other_assets_value_usdt": str(other_value) if not unpriced else None,
        "unpriced_assets": unpriced,
        # Present only when complete, and never labelled merely "total balance".
        "total_portfolio_value_usdt": str(total_value) if total_value is not None else None,
        "valuation_method": valuation_method,
        "valued_at": valued_at,
    }


def _protection_by_symbol(orders: list[Any] | None) -> dict[str, list[Any]]:
    """Index resting reduce-only trigger orders by bare ``BASE/QUOTE`` symbol."""
    grouped: dict[str, list[Any]] = {}
    for order in orders or []:
        if not getattr(order, "reduce_only", False):
            continue
        if getattr(order, "trigger_price", None) is None:
            continue
        key = str(getattr(order, "symbol", "")).split(":")[0]
        grouped.setdefault(key, []).append(order)
    return grouped


def _protection_for(
    row_symbol: str, entry: Decimal, grouped: dict[str, list[Any]]
) -> tuple[Any | None, Any | None]:
    """The stop and take-profit protecting one position, read from its resting orders.

    Returns ``(stop, take_profit)``, either of which may be ``None``.

    Which is which is decided by the order's TYPE: a take-profit is a limit order and
    carries a price, a stop is a market order and does not.

    Deliberately NOT by where the trigger sits relative to entry. That test fails exactly
    when it matters — once the trail ratchets a stop into profit it sits above entry on a
    long and reads as a take-profit. In the intrabar manager the same mistake closed a
    live winner at its trail level instead of its target; here it would merely mislabel the
    panel, but it would mislabel it at the very moment the operator most wants the truth.
    """
    candidates = grouped.get(str(row_symbol).split(":")[0], [])
    if not candidates or entry <= ZERO:
        return None, None

    stop = take_profit = None
    for order in candidates:
        if order.price:
            take_profit = order
        else:
            stop = order
    return stop, take_profit


def position_rows(
    raw_positions: list[dict[str, Any]], protective_orders: list[Any] | None = None
) -> tuple[list[dict[str, Any]], Decimal]:
    """Normalise the venue's raw position payloads and total their unrealised PnL.

    The venue returns ccxt's unified dicts, whose numbers are floats. They are converted
    through ``str`` so a float's binary representation is never carried into a decimal.

    Args:
        raw_positions: Raw position dicts from the gateway.
        protective_orders: Resting orders on the account, used to find each position's
            stop and target when the venue does not carry them on the position row.

    Returns:
        ``(rows, total_unrealised)``.

    """
    grouped = _protection_by_symbol(protective_orders)
    rows: list[dict[str, Any]] = []
    total = ZERO
    for raw in raw_positions:
        quantity = _decimal(raw.get("contracts"))
        if quantity == ZERO:
            # Venues commonly return closed positions as zero-size rows. Rendering those
            # as open positions would contradict the position count everywhere else.
            continue
        pnl = _decimal(raw.get("unrealizedPnl"))
        total += pnl
        info = raw.get("info")
        row_entry = _decimal(raw.get("entryPrice"))
        stop_order, target_order = _protection_for(str(raw.get("symbol") or ""), row_entry, grouped)
        _row_stop = _optional_decimal_str(info.get("stopLoss")) if isinstance(info, dict) else None
        _row_target = (
            _optional_decimal_str(info.get("takeProfit")) if isinstance(info, dict) else None
        )
        _stop_price = str(stop_order.trigger_price) if stop_order is not None else _row_stop
        _target_price = str(target_order.trigger_price) if target_order is not None else _row_target
        _stop_id = stop_order.venue_order_id if stop_order is not None else None
        _target_id = target_order.venue_order_id if target_order is not None else None
        rows.append(
            {
                "symbol": str(raw.get("symbol") or ""),
                "side": str(raw.get("side") or ""),
                "quantity": str(quantity),
                "entry_price": str(_decimal(raw.get("entryPrice"))),
                "mark_price": str(_decimal(raw.get("markPrice"))),
                "notional_usdt": str(_decimal(raw.get("notional"))),
                "unrealized_pnl": str(pnl),
                "leverage": str(_decimal(raw.get("leverage"))),
                "liquidation_price": _optional_decimal_str(raw.get("liquidationPrice")),
                "margin_mode": raw.get("marginMode"),
                # Present only when protection actually exists; an unprotected position is
                # a real and important condition, so it is never defaulted to a number.
                #
                # Two places have to be consulted, because the venue keeps protection in
                # one of two forms. With Bybit's `tpslMode: Full` the stop and target live
                # on the position row. With `Partial` — which is what a maker take-profit
                # requires, and therefore what this engine uses — they are separate
                # reduce-only trigger orders and the position row's fields stay empty.
                # Reading only the row reported a live WLD position with a venue stop at
                # 0.3573 and a target at 0.3834 as having neither, which is the most
                # dangerous thing this panel could get wrong.
                "venue_stop_loss": _stop_price,
                "venue_take_profit": _target_price,
                "venue_stop_order_id": _stop_id,
                "venue_take_profit_order_id": _target_id,
                "protected": _stop_price is not None,
                "opened_at": raw.get("datetime"),
            }
        )
    return rows, total


def order_rows(orders: list[Any]) -> list[dict[str, Any]]:
    """Normalise working orders read from the venue.

    Status is whatever the venue says it is. The dashboard's contract is that an order
    panel never shows ``NEW`` for something the venue has already filled or cancelled,
    which is only achievable by reading status from the venue on every refresh rather than
    from the local order store.

    Args:
        orders: Domain ``Order`` objects from the gateway.

    Returns:
        JSON-safe rows.

    """
    return [
        {
            "order_id": order.order_id,
            "venue_order_id": order.venue_order_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol.slashed,
            "side": order.side.value,
            "type": order.order_type.value,
            "status": order.status.value,
            "time_in_force": order.time_in_force.value,
            "quantity": str(order.quantity),
            "filled_quantity": str(order.filled_quantity),
            "remaining_quantity": str(order.remaining_quantity),
            "price": str(order.price) if order.price is not None else None,
            "trigger_price": (
                str(order.trigger_price) if order.trigger_price is not None else None
            ),
            "average_fill_price": str(order.average_fill_price),
            "reduce_only": order.reduce_only,
            # A stop and a target are both a reduce-only sell once normalised; without
            # this a protective bracket reads as a duplicated exit.
            "purpose": order.metadata.get("purpose"),
            "created_at": order.created_at.isoformat(),
        }
        for order in orders
    ]


def _decimal(value: Any) -> Decimal:
    """Best-effort ``Decimal`` from a venue field; zero when absent or unparseable."""
    if value is None or value == "":
        return ZERO
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return ZERO


def _optional_decimal_str(value: Any) -> str | None:
    """Exact decimal string, or ``None`` when the venue did not supply the field."""
    if value is None or value in {"", "0"}:
        return None
    try:
        return str(Decimal(str(value)))
    except (ArithmeticError, ValueError):
        return None


__all__ = [
    "QUOTE_ASSET",
    "QUOTE_EQUIVALENTS",
    "AssetValuation",
    "order_rows",
    "position_rows",
    "value_balances",
]
