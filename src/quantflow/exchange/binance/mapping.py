"""Translation between CCXT/Binance wire formats and the QuantFlow domain.

Kept in one module so venue quirks are contained: everywhere else in the codebase, an
order status is an :class:`OrderStatus`, never the string ``"NEW"``.
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

#: Binance's own status vocabulary, plus the lowercase forms CCXT normalises to.
CCXT_TO_ORDER_STATUS: dict[str, OrderStatus] = {
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
    "expired_in_match": OrderStatus.EXPIRED,
}

TIME_IN_FORCE_TO_CCXT: dict[TimeInForce, str] = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.GTD: "GTD",
}

CCXT_TO_TIME_IN_FORCE: dict[str, TimeInForce] = {
    "GTC": TimeInForce.GTC,
    "IOC": TimeInForce.IOC,
    "FOK": TimeInForce.FOK,
    "GTD": TimeInForce.GTD,
    "PO": TimeInForce.GTC,
}


def to_ccxt_symbol(symbol: Symbol, market_type: MarketType = MarketType.SPOT) -> str:
    """Render a symbol the way CCXT expects it.

    CCXT uses ``BTC/USDT`` for spot and ``BTC/USDT:USDT`` for linear perpetuals.
    """
    if market_type is MarketType.FUTURE:
        return f"{symbol.slashed}:{symbol.quote}"
    return symbol.slashed


def from_ccxt_symbol(raw: str) -> Symbol:
    """Parse a CCXT symbol, discarding any settlement suffix."""
    base_part = raw.split(":", 1)[0]
    parsed = Symbol.parse(base_part)
    assert isinstance(parsed, Symbol)
    return parsed


def parse_order_status(raw: str | None) -> OrderStatus:
    """Map a venue status string onto :class:`OrderStatus`.

    Unknown values map to ``NEW`` rather than raising: a status we do not recognise still
    represents a live order, and dropping it would orphan real exposure.
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

    Returns ``None`` for markets we do not trade (options, inverse contracts, or entries
    missing the precision data an order would need).
    """
    if not market.get("spot") and not market.get("linear"):
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

    price_tick = _precision_to_step(precision.get("price"))
    quantity_step = _precision_to_step(precision.get("amount"))
    if price_tick is None or quantity_step is None:
        return None

    min_quantity = _decimal_or(amount_limits.get("min"), quantity_step)
    return Instrument(
        symbol=symbol,
        market_type=MarketType.FUTURE if market.get("linear") else MarketType.SPOT,
        price_tick=price_tick,
        quantity_step=quantity_step,
        min_quantity=max(min_quantity, quantity_step),
        max_quantity=_optional_decimal(amount_limits.get("max")),
        min_notional=_decimal_or(cost_limits.get("min"), Decimal("0.00000001")),
        max_notional=_optional_decimal(cost_limits.get("max")),
        maker_fee=_decimal_or(market.get("maker"), Decimal("0.001")),
        taker_fee=_decimal_or(market.get("taker"), Decimal("0.001")),
        max_leverage=_decimal_or((limits.get("leverage") or {}).get("max"), Decimal("1")),
        contract_size=_decimal_or(market.get("contractSize"), Decimal("1")),
        active=bool(market.get("active", True)),
    )


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
    )


def parse_order(
    raw: dict[str, Any],
    *,
    local_order_id: str | None = None,
    strategy_id: str | None = None,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
) -> Order:
    """Build an :class:`Order` from a CCXT order dict.

    ``local_order_id`` preserves our own identifier so an order fetched back from the venue
    still reconciles against the local OMS record.
    """
    symbol = from_ccxt_symbol(str(raw.get("symbol", "")))
    venue_order_id = str(raw.get("id") or "")
    created = _timestamp(raw.get("timestamp"))
    updated = _timestamp(raw.get("lastTradeTimestamp") or raw.get("timestamp"), fallback=created)

    quantity = _decimal_or(raw.get("amount"))
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
    )


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
