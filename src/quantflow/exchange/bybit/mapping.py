"""Translation between CCXT/Bybit V5 wire formats and the QuantFlow domain.

Kept in one module so venue quirks are contained: everywhere else in the codebase, an
order status is an :class:`OrderStatus`, never the string ``"New"``.

Bybit V5 differs from other venues in ways that matter here:

* Its market segments are **spot**, **linear** and **inverse**, not "future". CCXT wants
  ``linear`` for USDT-margined perpetuals, and passing ``future`` silently selects the
  wrong category.
* Its status vocabulary is CamelCase (``PartiallyFilled``) and includes conditional-order
  states (``Untriggered``, ``Triggered``, ``Deactivated``) that no other venue reports.
  Both the raw forms and CCXT's lowercase normalisations are accepted, because which one
  arrives depends on whether the value came through CCXT's unified parser or straight off
  the wire in a websocket frame.
* ``PostOnly`` is a time-in-force value rather than an order flag.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.core.clock import from_epoch_ms, utc_now
from quantflow.core.config import MarketType
from quantflow.core.errors import (
    ExchangeAuthenticationError,
    ExchangeConnectionError,
    ExchangeError,
    ExchangeTimeoutError,
    InsufficientFundsError,
    InvalidSymbolError,
    OrderRejectedError,
    RateLimitError,
)
from quantflow.core.precision import ZERO, step_from_precision, to_decimal
from quantflow.domain.enums import (
    LiquidityRole,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.domain.orders import Fill, Order

# --------------------------------------------------------------------------- #
# Enum translation
# --------------------------------------------------------------------------- #
ORDER_TYPE_TO_CCXT: dict[OrderType, str] = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
    OrderType.STOP_MARKET: "stop_market",
    OrderType.STOP_LIMIT: "stop_limit",
    OrderType.TAKE_PROFIT_MARKET: "take_profit_market",
    OrderType.TAKE_PROFIT_LIMIT: "take_profit_limit",
}

CCXT_TO_ORDER_TYPE: dict[str, OrderType] = {
    "market": OrderType.MARKET,
    "limit": OrderType.LIMIT,
    # Bybit V5 returns CamelCase order types on the raw wire.
    "Market": OrderType.MARKET,
    "Limit": OrderType.LIMIT,
    "stop": OrderType.STOP_LIMIT,
    "stop_market": OrderType.STOP_MARKET,
    "stop_loss": OrderType.STOP_MARKET,
    "stop_loss_limit": OrderType.STOP_LIMIT,
    "stop_limit": OrderType.STOP_LIMIT,
    "take_profit": OrderType.TAKE_PROFIT_MARKET,
    "take_profit_market": OrderType.TAKE_PROFIT_MARKET,
    "take_profit_limit": OrderType.TAKE_PROFIT_LIMIT,
    "limit_maker": OrderType.LIMIT,
}

#: Bybit V5's status vocabulary plus the lowercase forms CCXT normalises to. Both are
#: needed: CCXT's unified parser lowercases, while a raw websocket frame does not.
#:
#: Keyed on the *lowercased* spelling, because :func:`parse_order_status` lowercases before
#: it looks up. Written CamelCase here — as the venue writes it — the V5 entries below were
#: simply unreachable, and every one of them fell through to the ``NEW`` default: a
#: ``PartiallyFilled`` frame read as untouched, a ``Deactivated`` or
#: ``PartiallyFilledCanceled`` order read as still working. Three of the four statuses the
#: OMS has to tell apart were therefore indistinguishable on the raw path, and an order the
#: venue had cancelled stayed open in the book forever.
CCXT_TO_ORDER_STATUS: dict[str, OrderStatus] = {
    # CCXT unified
    "new": OrderStatus.NEW,
    "open": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "partial": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.PENDING_CANCEL,
    "rejected": OrderStatus.REJECTED,
    "expired": OrderStatus.EXPIRED,
    # Bybit V5 raw, casefolded: "New", "PartiallyFilled", "Filled", "Cancelled", "Rejected"
    "partiallyfilled": OrderStatus.PARTIALLY_FILLED,
    # A conditional order that has not triggered is working, not filled.
    "untriggered": OrderStatus.NEW,
    "triggered": OrderStatus.NEW,
    # Deactivated is Bybit's word for a conditional order cancelled before triggering.
    "deactivated": OrderStatus.CANCELLED,
    # Partially filled then cancelled: the remainder is gone, so the order is terminal.
    # Bybit spells it with one "l" in Canceled here and two in "Cancelled" above; both
    # spellings appear in V5 payloads, so both are carried.
    "partiallyfilledcanceled": OrderStatus.CANCELLED,
    "partiallyfilledcancelled": OrderStatus.CANCELLED,
}

#: Bybit V5 accepts GTC, IOC, FOK and PostOnly. It has no GTD, so a good-till-date order
#: is sent as GTC rather than silently rejected by the venue - the expiry is enforced on
#: our side, which is where the clock we trust lives anyway.
TIME_IN_FORCE_TO_CCXT: dict[TimeInForce, str] = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.GTD: "GTC",
}

CCXT_TO_TIME_IN_FORCE: dict[str, TimeInForce] = {
    "GTC": TimeInForce.GTC,
    "IOC": TimeInForce.IOC,
    "FOK": TimeInForce.FOK,
    "GTD": TimeInForce.GTD,
    "PostOnly": TimeInForce.GTC,
    "PO": TimeInForce.GTC,
}


def to_ccxt_symbol(symbol: Symbol, market_type: MarketType = MarketType.SPOT) -> str:
    """Render a symbol the way CCXT expects it for Bybit.

    ``BTC/USDT`` for spot, ``BTC/USDT:USDT`` for a USDT-margined linear perpetual. The
    settlement suffix is what tells CCXT to route to Bybit's ``linear`` category; without
    it a futures order would be placed against the spot book.
    """
    if market_type is MarketType.FUTURE:
        return f"{symbol.slashed}:{symbol.quote}"
    return symbol.slashed


def bybit_category(market_type: MarketType) -> str:
    """Bybit V5's category name for a market segment.

    V5 organises everything by category and rejects a request whose category does not
    match the symbol. CCXT's ``defaultType`` must therefore be ``linear``, never
    ``future`` - Bybit has no such category and the request fails at the venue.
    """
    return "linear" if market_type is MarketType.FUTURE else "spot"


def from_ccxt_symbol(raw: str) -> Symbol:
    """Parse a CCXT symbol, discarding any settlement suffix."""
    base_part = raw.split(":", 1)[0]
    parsed = Symbol.parse(base_part)
    assert isinstance(parsed, Symbol)
    return parsed


def parse_order_status(raw: str | None) -> OrderStatus:
    """Map a venue status string onto :class:`OrderStatus`.

    Unknown values map to ``NEW`` rather than raising: a status we do not recognise still
    represents a live order, and dropping it would orphan real exposure. That default is
    also why every spelling the venue actually emits has to be in the table — an entry the
    lookup cannot reach is not a missing case, it is a *wrong* answer that says "working".
    """
    if not raw:
        return OrderStatus.NEW
    return CCXT_TO_ORDER_STATUS.get(raw.strip().lower(), OrderStatus.NEW)


def parse_order_type(raw: str | None) -> OrderType:
    """Map a venue order-type string onto :class:`OrderType`."""
    if not raw:
        return OrderType.MARKET
    return CCXT_TO_ORDER_TYPE.get(raw.strip().lower(), OrderType.MARKET)


def parse_side(raw: str | None) -> OrderSide:
    """Map a venue side string onto :class:`OrderSide`."""
    return OrderSide.SELL if (raw or "").strip().lower() == "sell" else OrderSide.BUY


# --------------------------------------------------------------------------- #
# Structure translation
# --------------------------------------------------------------------------- #
def _decimal_or(value: Any, default: Decimal = ZERO) -> Decimal:
    """Coerce a possibly-``None`` CCXT numeric to Decimal."""
    if value is None or value == "":
        return default
    return to_decimal(value)


def _optional_decimal(value: Any) -> Decimal | None:
    """Coerce a possibly-``None`` CCXT numeric, preserving ``None``."""
    if value is None or value == "":
        return None
    return to_decimal(value)


def _timestamp(value: Any, *, fallback: datetime | None = None) -> datetime:
    """Convert a CCXT millisecond timestamp, falling back when absent."""
    if value is None:
        return fallback or utc_now()
    return from_epoch_ms(int(value))


def parse_instrument(market: dict[str, Any]) -> Instrument | None:
    """Build an :class:`Instrument` from a CCXT market dict.

    Returns ``None`` for markets we do not trade (options, dated futures, inverse
    contracts, or entries missing the precision data an order would need).

    Options and dated futures are rejected *by identity*, not merely by preference. Bybit
    reports its USDT options as ``linear`` markets, and their CCXT symbols
    (``BTC/USDT:USDT-260821-52000-C``) collapse onto the perpetual's symbol once the
    settlement suffix is stripped. Accepting them lets an option's tradability rules
    overwrite the perpetual's in the instrument registry — which is exactly how BTC/USDT
    came to carry a 0.01 lot minimum (the option's) instead of the perpetual's 0.001, and
    a 0.00001 price tick instead of 0.1. Every downstream size, rounding and minimum check
    was then measured against a contract the engine never trades.
    """
    if not market.get("spot") and not market.get("linear"):
        return None
    # An expiring contract is a different instrument from the perpetual it shares a base
    # symbol with. ``expiry`` is None for spot and for perpetual swaps, and set for every
    # option and dated future, so it separates them on the venue's own field rather than
    # on a guess about the symbol's shape.
    if market.get("option") or market.get("future") or market.get("expiry") is not None:
        return None
    raw_symbol = market.get("symbol")
    if not isinstance(raw_symbol, str):
        return None

    try:
        symbol = from_ccxt_symbol(raw_symbol)
    except Exception:  # an unparseable venue symbol is simply skipped
        return None

    limits = market.get("limits") or {}
    amount_limits = limits.get("amount") or {}
    cost_limits = limits.get("cost") or {}
    precision = market.get("precision") or {}
    venue = _venue_filters(market)

    price_tick = venue.get("price_tick") or _precision_to_step(precision.get("price"))
    quantity_step = venue.get("quantity_step") or _precision_to_step(precision.get("amount"))
    if price_tick is None or quantity_step is None:
        return None

    min_quantity = venue.get("min_quantity") or _decimal_or(amount_limits.get("min"), quantity_step)
    return Instrument(
        symbol=symbol,
        market_type=MarketType.FUTURE if market.get("linear") else MarketType.SPOT,
        price_tick=price_tick,
        quantity_step=quantity_step,
        min_quantity=max(min_quantity, quantity_step),
        max_quantity=_optional_decimal(amount_limits.get("max")),
        min_notional=venue.get("min_notional")
        or _decimal_or(cost_limits.get("min"), Decimal("0.00000001")),
        max_notional=_optional_decimal(cost_limits.get("max")),
        maker_fee=_decimal_or(market.get("maker"), Decimal("0.001")),
        taker_fee=_decimal_or(market.get("taker"), Decimal("0.001")),
        max_leverage=_decimal_or((limits.get("leverage") or {}).get("max"), Decimal("1")),
        contract_size=_decimal_or(market.get("contractSize"), Decimal("1")),
        active=bool(market.get("active", True)),
    )


def _venue_filters(market: dict[str, Any]) -> dict[str, Decimal]:
    """Read Bybit's own lot and price filters out of the raw payload.

    CCXT's ``precision`` is a *derived* field, and the derivation is lossy: a step of
    ``1`` (FARTCOIN) or ``100`` (1000PEPE) is indistinguishable from a decimal-place count
    once it lands there, and :func:`_precision_to_step` reads it as one — turning a lot
    step of 100 into 1e-100. Bybit states the same numbers unambiguously in
    ``info.lotSizeFilter`` and ``info.priceFilter``, so those are used when present and the
    CCXT-derived values are kept only as a fallback for venues or fixtures without them.

    ``minNotionalValue`` matters just as much: CCXT leaves ``limits.cost.min`` empty for
    Bybit perpetuals, which defaulted the instrument to a 1e-8 floor and let an order the
    venue would reject for being under 5 USDT pass every local check.
    """
    info = market.get("info")
    if not isinstance(info, dict):
        return {}
    lot = info.get("lotSizeFilter")
    price = info.get("priceFilter")
    filters: dict[str, Decimal] = {}

    if isinstance(price, dict):
        tick = _positive_decimal(price.get("tickSize"))
        if tick is not None:
            filters["price_tick"] = tick

    if isinstance(lot, dict):
        # Spot reports its lot step as ``basePrecision``; linear reports ``qtyStep``.
        step = _positive_decimal(lot.get("qtyStep")) or _positive_decimal(lot.get("basePrecision"))
        if step is not None:
            filters["quantity_step"] = step
        minimum = _positive_decimal(lot.get("minOrderQty"))
        if minimum is not None:
            filters["min_quantity"] = minimum
        # ``minNotionalValue`` on linear, ``minOrderAmt`` on spot.
        notional = _positive_decimal(lot.get("minNotionalValue")) or _positive_decimal(
            lot.get("minOrderAmt")
        )
        if notional is not None:
            filters["min_notional"] = notional

    return filters


def _positive_decimal(value: Any) -> Decimal | None:
    """Coerce a venue numeric string to a positive Decimal, or ``None``."""
    if value is None or value == "":
        return None
    try:
        number = to_decimal(value)
    except Exception:  # a malformed venue field falls back rather than crashing the load
        return None
    return number if number > ZERO else None


def _precision_to_step(value: Any) -> Decimal | None:
    """Normalise CCXT precision to a step size.

    CCXT reports precision either as a decimal-place count (``2``) or as an actual tick
    size (``0.01``), depending on the venue and the mode; both forms appear for Binance.
    """
    if value is None:
        return None
    number = to_decimal(value)
    if number <= ZERO:
        return None
    if number >= 1 and number == number.to_integral_value():
        return step_from_precision(int(number))
    return number


def parse_fill(raw: dict[str, Any], *, order_id: str, symbol: Symbol) -> Fill:
    """Build a :class:`Fill` from a CCXT trade dict."""
    fee_info = raw.get("fee") or {}
    # Bybit reports per-execution realised PnL as `closedPnl` in the raw payload; CCXT does
    # not surface it, so it is read from `info` rather than invented downstream.
    closed_pnl = (raw.get("info") or {}).get("closedPnl")
    realized = None
    if closed_pnl not in (None, ""):
        try:
            realized = Decimal(str(closed_pnl))
        except (ArithmeticError, ValueError):
            realized = None
    return Fill(
        fill_id=str(raw.get("id") or raw.get("tradeId") or ""),
        order_id=order_id,
        symbol=symbol,
        side=parse_side(raw.get("side")),
        quantity=_decimal_or(raw.get("amount")),
        price=_decimal_or(raw.get("price")),
        fee=_decimal_or(fee_info.get("cost")),
        fee_currency=str(fee_info.get("currency") or symbol.quote),
        timestamp=_timestamp(raw.get("timestamp")),
        role=LiquidityRole.MAKER if raw.get("takerOrMaker") == "maker" else LiquidityRole.TAKER,
        realized_pnl=realized,
    )


def parse_order(
    raw: dict[str, Any],
    *,
    local_order_id: str | None = None,
    strategy_id: str | None = None,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    fallback_quantity: Decimal | None = None,
) -> Order:
    """Build an :class:`Order` from a CCXT order dict.

    ``local_order_id`` preserves our own identifier so an order fetched back from the venue
    still reconciles against the local OMS record.

    ``fallback_quantity`` is the quantity we asked for. Bybit V5's create_order acknowledges
    with only ``orderId``/``orderLinkId`` - it does not echo the amount - so parsing the
    acknowledgement alone yields zero and the Order invariant rejects it. Raising there
    would report a *failure for an order the venue has already accepted*, which is the
    orphan case: real position, no local record. The requested size is the correct fallback
    because it is what was sent.
    """
    symbol = from_ccxt_symbol(str(raw.get("symbol", "")))
    venue_order_id = str(raw.get("id") or "")
    created = _timestamp(raw.get("timestamp"))
    updated = _timestamp(raw.get("lastTradeTimestamp") or raw.get("timestamp"), fallback=created)

    quantity = _decimal_or(raw.get("amount"))
    if quantity <= ZERO and fallback_quantity is not None:
        quantity = fallback_quantity
    filled = _decimal_or(raw.get("filled"))
    # A venue that reports more filled than ordered would violate the Order invariant;
    # clamping keeps a reconciliation pass from crashing on a bad payload.
    filled = min(filled, quantity) if quantity > ZERO else filled

    fills = tuple(
        parse_fill(trade, order_id=local_order_id or venue_order_id, symbol=symbol)
        for trade in (raw.get("trades") or [])
    )

    average = _decimal_or(raw.get("average"))
    if average == ZERO and fills:
        gross = sum((fill.quantity * fill.price for fill in fills), ZERO)
        total = sum((fill.quantity for fill in fills), ZERO)
        average = gross / total if total > ZERO else ZERO

    fee_info = raw.get("fee") or {}
    return Order(
        order_id=local_order_id or venue_order_id,
        client_order_id=str(raw.get("clientOrderId") or venue_order_id),
        symbol=symbol,
        side=parse_side(raw.get("side")),
        order_type=parse_order_type(raw.get("type")),
        quantity=quantity,
        status=parse_order_status(raw.get("status")),
        created_at=created,
        updated_at=updated,
        price=_optional_decimal(raw.get("price")),
        trigger_price=_optional_decimal(raw.get("stopPrice") or raw.get("triggerPrice")),
        time_in_force=CCXT_TO_TIME_IN_FORCE.get(
            str(raw.get("timeInForce") or "GTC").upper(), TimeInForce.GTC
        ),
        filled_quantity=filled,
        average_fill_price=average,
        fees_paid=_decimal_or(fee_info.get("cost")),
        fills=fills,
        venue_order_id=venue_order_id or None,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        reduce_only=bool(raw.get("reduceOnly", False)),
        strategy_id=strategy_id,
        metadata=_conditional_metadata(raw),
    )


#: Bybit's name for a conditional order's job, mapped to ours.
_STOP_ORDER_PURPOSE = {"stoploss": "stop_loss", "takeprofit": "take_profit"}


def _conditional_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Record what a conditional order is *for*.

    A stop and a target on the same position are both "sell market" once CCXT has
    normalised them; without this they are indistinguishable in any list, which is how a
    correct protective bracket reads as a duplicated exit.
    """
    stop_order_type = str((raw.get("info") or {}).get("stopOrderType") or "").strip()
    purpose = _STOP_ORDER_PURPOSE.get(stop_order_type.lower().replace(" ", ""))
    return {"purpose": purpose} if purpose else {}


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #
def translate_exception(exc: BaseException) -> ExchangeError:  # noqa: PLR0911
    """Convert a CCXT exception into a QuantFlow error.

    Done by class name rather than by importing every CCXT exception type, so a CCXT
    upgrade that adds or renames an error class degrades to a generic ``ExchangeError``
    instead of crashing the translator.
    """
    name = type(exc).__name__
    message = str(exc)

    if name in {"AuthenticationError", "PermissionDenied", "AccountSuspended"}:
        return ExchangeAuthenticationError(f"exchange rejected our credentials: {message}")
    if name in {"RateLimitExceeded", "DDoSProtection"}:
        return RateLimitError(f"exchange rate limit hit: {message}")
    if name in {"RequestTimeout"}:
        return ExchangeTimeoutError(f"exchange request timed out: {message}")
    if name in {"NetworkError", "ExchangeNotAvailable", "OnMaintenance"}:
        return ExchangeConnectionError(f"exchange unreachable: {message}")
    if name in {"InsufficientFunds"}:
        return InsufficientFundsError(f"insufficient balance: {message}")
    if name in {"BadSymbol"}:
        return InvalidSymbolError(f"unknown symbol: {message}")
    if name in {"InvalidOrder", "OrderNotFound", "OrderImmediatelyFillable", "OrderNotFillable"}:
        return OrderRejectedError(f"order rejected: {message}", venue_error=name)
    return ExchangeError(f"exchange error ({name}): {message}")
