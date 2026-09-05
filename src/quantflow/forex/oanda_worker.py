"""OANDA v20 transport — the route that runs on a headless Linux box.

**EXPERIMENTAL — this module has never placed an order.** No FX credentials exist in
this project; it is written to the venue's published API and tested against fakes only.

Unlike :mod:`quantflow.forex.mt5_worker`, nothing here needs Windows, a GUI terminal, a JVM
or a local gateway process. It is plain HTTPS: a REST host for requests and a separate
streaming host that holds a chunked connection open for prices. That is the whole reason
this transport exists — it is the one FX venue we verified that a cloud container can drive
end to end with a free, self-service practice account.

Endpoint shapes, field names and conventions below were taken from OANDA's published v20
documentation. **They have not been exercised against a live account** — no credentials
exist in this project — so every response parser is written to the documented shape and
tested against fixtures built from that documentation, not against recorded traffic. The
functions carrying the most risk if the docs are stale are marked in their docstrings.

Two conventions are worth knowing before reading further:

* **OANDA trades in units of the base currency, not lots.** One unit of ``EUR_USD`` is one
  euro; a standard lot is 100,000 units. The rest of QuantFlow speaks lots, so every
  boundary crossing goes through :func:`units_from_lots` / :func:`lots_from_units` with
  ``contract_size`` as the single conversion constant.
* **Direction is the sign of ``units``.** Positive is long, negative is short — in order
  requests, in trades and in fills. There is no side field to read.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final, Protocol

import httpx
import structlog

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO, to_decimal
from quantflow.domain.enums import OrderSide
from quantflow.forex.errors import (
    ForexAuthenticationError,
    ForexCapabilityError,
    ForexConnectionError,
    ForexOrderRejectedError,
    ForexSymbolError,
)
from quantflow.forex.instruments import ForexInstrument, TradeMode, prioritise_symbols
from quantflow.forex.protocol import (
    AccountInfo,
    BrokerDescription,
    ForexBar,
    ForexFill,
    ForexOrder,
    ForexOrderRequest,
    ForexOrderStatus,
    ForexOrderType,
    ForexPosition,
    ForexTick,
    ForexTimeframe,
    OrderAck,
)

logger = structlog.get_logger(__name__)

#: Environment variables the operator must set.
REQUIRED_ENV_VARS: Final[tuple[str, ...]] = ("QF_OANDA_TOKEN", "QF_OANDA_ACCOUNT_ID")

OANDA_PRACTICE_REST: Final = "https://api-fxpractice.oanda.com"
OANDA_PRACTICE_STREAM: Final = "https://stream-fxpractice.oanda.com"
OANDA_LIVE_REST: Final = "https://api-fxtrade.oanda.com"
OANDA_LIVE_STREAM: Final = "https://stream-fxtrade.oanda.com"

#: Units of base currency in one standard lot. OANDA has no lot concept; this constant is
#: the entire bridge between its ``units`` and QuantFlow's ``lots``.
OANDA_UNITS_PER_LOT: Final = Decimal("100000")

#: Documented v20 limits: 120 REST requests/second per IP, at most 2 new connections per
#: second, and no more than 20 concurrent streams.
REST_REQUESTS_PER_SECOND: Final = 120
NEW_CONNECTIONS_PER_SECOND: Final = 2
MAX_CONCURRENT_STREAMS: Final = 20

#: Maximum candles per request.
MAX_CANDLE_COUNT: Final = 5000

_HTTP_UNAUTHORISED: Final = 401
_HTTP_FORBIDDEN: Final = 403
_HTTP_NOT_FOUND: Final = 404
_HTTP_TOO_MANY: Final = 429
_HTTP_SERVER_ERROR: Final = 500

#: OANDA order types that are attachments to a trade rather than standalone orders.
_PROTECTIVE_ORDER_TYPES: Final = frozenset(
    {"TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP_LOSS", "GUARANTEED_STOP_LOSS"}
)

_OANDA_ORDER_TYPES: Final[dict[str, ForexOrderType]] = {
    "MARKET": ForexOrderType.MARKET,
    "LIMIT": ForexOrderType.LIMIT,
    "STOP": ForexOrderType.STOP,
    "MARKET_IF_TOUCHED": ForexOrderType.STOP,
}

_OANDA_ORDER_STATES: Final[dict[str, ForexOrderStatus]] = {
    "PENDING": ForexOrderStatus.PLACED,
    "FILLED": ForexOrderStatus.FILLED,
    "TRIGGERED": ForexOrderStatus.PLACED,
    "CANCELLED": ForexOrderStatus.CANCELLED,
}


class OandaEnvironment(StrEnum):
    """Which OANDA environment to talk to."""

    PRACTICE = "practice"
    LIVE = "live"

    @property
    def rest_host(self) -> str:
        """REST base URL for this environment."""
        return OANDA_PRACTICE_REST if self is OandaEnvironment.PRACTICE else OANDA_LIVE_REST

    @property
    def stream_host(self) -> str:
        """Streaming base URL for this environment."""
        return OANDA_PRACTICE_STREAM if self is OandaEnvironment.PRACTICE else OANDA_LIVE_STREAM


@dataclass(frozen=True, slots=True)
class OandaCapabilities:
    """Whether this host can run the OANDA transport, and what is missing if not.

    Deliberately has no platform blocker: that is the point of this transport.
    """

    platform: str
    missing_env: tuple[str, ...]
    blockers: tuple[str, ...]
    environment: OandaEnvironment
    notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """Whether the transport can be started here."""
        return not self.blockers

    @property
    def linux_compatible(self) -> bool:
        """Always true — this transport is plain HTTPS on any OS."""
        return True

    def describe(self) -> str:
        """A multi-line, paste-into-a-runbook explanation."""
        if self.ready:
            head = f"OANDA transport ready ({self.environment.value}, host {self.platform})."
        else:
            head = f"OANDA transport cannot run yet ({self.platform}):"
        lines = [head]
        lines.extend(f"  - {blocker}" for blocker in self.blockers)
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)

    def raise_if_not_ready(self) -> None:
        """Raise a capability error carrying every blocker.

        Raises:
            ForexCapabilityError: if anything is missing.

        """
        if not self.ready:
            raise ForexCapabilityError(self.describe(), blockers=list(self.blockers))


def capabilities(env: Mapping[str, str] | None = None) -> OandaCapabilities:
    """Report whether the OANDA transport can run here.

    The only blockers are credentials — there is no platform or package gate, which is what
    makes this the Linux route.
    """
    import platform as platform_module

    resolved = os.environ if env is None else env
    missing = tuple(name for name in REQUIRED_ENV_VARS if not resolved.get(name))
    environment = _environment_from_env(resolved)

    blockers: list[str] = []
    if missing:
        blockers.append(
            "credentials: "
            + ", ".join(missing)
            + " are unset. Open a free OANDA fxTrade Practice account, then in the account "
            "HUB go to Manage API Access and generate a personal access token; the account "
            "id has the form 001-001-1234567-001."
        )
    if environment is OandaEnvironment.LIVE and not _allow_live(resolved):
        blockers.append(
            "environment: QF_OANDA_ENVIRONMENT=live points at real money. Set it to "
            "'practice', or set QF_OANDA_ALLOW_LIVE=1 to deliberately override."
        )

    return OandaCapabilities(
        platform=platform_module.system(),
        missing_env=missing,
        blockers=tuple(blockers),
        environment=environment,
        notes=(
            "OANDA's docs state the v20 API is available to all divisions except OANDA "
            "Global Markets and OANDA TMS BROKERS S.A. Confirm a token can actually be "
            "generated for your division before relying on this transport.",
        ),
    )


def _environment_from_env(env: Mapping[str, str]) -> OandaEnvironment:
    """Read the target environment, defaulting to practice."""
    raw = env.get("QF_OANDA_ENVIRONMENT", "practice").strip().lower()
    return OandaEnvironment.LIVE if raw == "live" else OandaEnvironment.PRACTICE


def _allow_live(env: Mapping[str, str]) -> bool:
    """Whether the operator explicitly opted in to real money."""
    return env.get("QF_OANDA_ALLOW_LIVE", "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True, slots=True)
class OandaCredentials:
    """OANDA API credentials. The token never appears in a repr or a log line."""

    token: str = field(repr=False)
    account_id: str
    environment: OandaEnvironment = OandaEnvironment.PRACTICE
    allow_live: bool = False

    @property
    def is_practice(self) -> bool:
        """Whether this points at the practice environment."""
        return self.environment is OandaEnvironment.PRACTICE

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OandaCredentials:
        """Build credentials from the environment.

        Raises:
            ForexCapabilityError: if a required variable is unset.

        """
        resolved = os.environ if env is None else env
        missing = [name for name in REQUIRED_ENV_VARS if not resolved.get(name)]
        if missing:
            raise ForexCapabilityError(
                "missing OANDA credentials: " + ", ".join(missing), missing_env=missing
            )
        return cls(
            token=resolved["QF_OANDA_TOKEN"],
            account_id=resolved["QF_OANDA_ACCOUNT_ID"],
            environment=_environment_from_env(resolved),
            allow_live=_allow_live(resolved),
        )


# --------------------------------------------------------------------------- #
# Units <-> lots
# --------------------------------------------------------------------------- #
def units_from_lots(lots: Decimal, side: OrderSide, instrument: ForexInstrument) -> Decimal:
    """Convert lots and a side into OANDA's signed base-currency units."""
    if lots <= ZERO:
        raise ValidationError(f"lots must be positive, got {lots}", symbol=instrument.symbol)
    units = lots * instrument.contract_size
    return units if side is OrderSide.BUY else -units


def lots_from_units(units: Decimal, instrument: ForexInstrument) -> Decimal:
    """Convert OANDA's signed units back into an unsigned lot size."""
    return abs(units) / instrument.contract_size


def side_from_units(units: Decimal) -> OrderSide:
    """Read direction off the sign of ``units``.

    Raises:
        ValidationError: on zero units, which carries no direction.

    """
    if units == ZERO:
        raise ValidationError("zero units carries no direction")
    return OrderSide.BUY if units > ZERO else OrderSide.SELL


def _format_units(units: Decimal) -> str:
    """Render units the way v20 expects them — a decimal string, sign included."""
    return format(units.normalize(), "f")


# --------------------------------------------------------------------------- #
# Payload mapping — pure functions over documented v20 JSON
# --------------------------------------------------------------------------- #
def _get(payload: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a key, raising a field-naming error when it is required and absent."""
    if key in payload:
        return payload[key]
    if default is None and key not in payload:
        raise ValidationError(f"OANDA payload is missing required field {key!r}", field=key)
    return default


def _dec(payload: Mapping[str, Any], key: str, default: str | None = None) -> Decimal:
    """Read a v20 numeric string as a Decimal. v20 sends numbers as strings by design."""
    raw = payload.get(key)
    if raw is None:
        if default is None:
            raise ValidationError(f"OANDA payload is missing numeric field {key!r}", field=key)
        return Decimal(default)
    return to_decimal(str(raw))


def _time(payload: Mapping[str, Any], key: str) -> datetime:
    """Parse a v20 RFC3339 timestamp.

    v20 emits nanosecond precision (``...T00:00:00.000000000Z``) which
    :meth:`datetime.fromisoformat` will not accept, so the fraction is truncated to
    microseconds first.
    """
    raw = str(_get(payload, key))
    return parse_oanda_time(raw)


def parse_oanda_time(raw: str) -> datetime:
    """Parse an RFC3339 timestamp with up to nanosecond precision into a UTC datetime."""
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, _, tail = text.partition(".")
        fraction, sign, offset = _split_offset(tail)
        text = f"{head}.{fraction[:6]:0<6}{sign}{offset}"
    try:
        return datetime.fromisoformat(text).astimezone(UTC)
    except ValueError as exc:
        raise ValidationError(f"unparseable OANDA timestamp: {raw!r}") from exc


def _split_offset(tail: str) -> tuple[str, str, str]:
    """Split a fractional-seconds tail into (fraction, sign, offset)."""
    for sign in ("+", "-"):
        if sign in tail:
            fraction, _, offset = tail.partition(sign)
            return fraction, sign, offset
    return tail, "", ""


def instrument_from_oanda(
    payload: Mapping[str, Any],
    *,
    home_conversion_factor: Decimal | None = None,
    venue: str = "oanda",
) -> ForexInstrument:
    """Map a v20 ``Instrument`` onto a :class:`ForexInstrument`.

    ``pipLocation`` and ``displayPrecision`` are cross-checked rather than trusted
    individually: the pip size QuantFlow derives from the price precision must equal
    ``10 ** pipLocation``, and a disagreement means the symbol is not a conventional FX
    quote and should not be sized with FX pip maths.

    ``home_conversion_factor`` converts one unit of quote-currency profit into the account
    currency. Without it the tick value is only correct when the quote currency *is* the
    account currency (every ``*_USD`` pair on a USD account); supply it from
    ``/pricing?includeHomeConversions=true`` for everything else.
    """
    name = str(_get(payload, "name"))
    base, _, quote = name.partition("_")
    if not quote:
        raise ForexSymbolError(f"unexpected OANDA instrument name: {name!r}", symbol=name)

    digits = int(_get(payload, "displayPrecision"))
    point = Decimal(1).scaleb(-digits)
    pip_location = int(_get(payload, "pipLocation"))
    units_precision = int(payload.get("tradeUnitsPrecision", 0))
    minimum_units = _dec(payload, "minimumTradeSize", "1")
    maximum_units = _dec(payload, "maximumOrderUnits", str(OANDA_UNITS_PER_LOT * 100))

    conversion = Decimal("1") if home_conversion_factor is None else home_conversion_factor
    instrument = ForexInstrument(
        symbol=name,
        base=base,
        quote=quote,
        contract_size=OANDA_UNITS_PER_LOT,
        min_lot=minimum_units / OANDA_UNITS_PER_LOT,
        max_lot=maximum_units / OANDA_UNITS_PER_LOT,
        lot_step=Decimal(1).scaleb(-units_precision) / OANDA_UNITS_PER_LOT,
        digits=digits,
        point=point,
        tick_size=point,
        tick_value=point * OANDA_UNITS_PER_LOT * conversion,
        margin_rate=_dec(payload, "marginRate", "0"),
        trade_mode=TradeMode.FULL,
        commission_per_lot=_commission_per_lot(payload),
        venue=venue,
    )

    expected_pip = Decimal(1).scaleb(pip_location)
    if instrument.pip_size != expected_pip:
        raise ForexSymbolError(
            f"{name} pipLocation {pip_location} implies a pip of {expected_pip}, but its "
            f"{digits}-digit price grid implies {instrument.pip_size}; refusing to apply FX "
            "pip maths to it",
            symbol=name,
        )
    return instrument


def _commission_per_lot(payload: Mapping[str, Any]) -> Decimal:
    """Read the documented per-unit commission and scale it to one lot.

    v20's ``commission`` block is optional and absent on most retail practice accounts, so
    an unreported commission stays zero rather than becoming an invented figure.
    """
    commission = payload.get("commission")
    if not isinstance(commission, Mapping):
        return ZERO
    per_units = to_decimal(str(commission.get("unitsTraded", "0")))
    if per_units <= ZERO:
        return ZERO
    rate = to_decimal(str(commission.get("commission", "0")))
    return rate * OANDA_UNITS_PER_LOT / per_units


def tick_from_oanda(payload: Mapping[str, Any]) -> ForexTick:
    """Map a v20 ``ClientPrice`` onto a :class:`ForexTick`.

    Only the top of each book is taken. ``closeoutBid``/``closeoutAsk`` are used as a
    fallback for the rare price with empty ladders.
    """
    bids = payload.get("bids") or []
    asks = payload.get("asks") or []
    bid = to_decimal(str(bids[0]["price"])) if bids else _dec(payload, "closeoutBid")
    ask = to_decimal(str(asks[0]["price"])) if asks else _dec(payload, "closeoutAsk")
    return ForexTick(
        symbol=str(_get(payload, "instrument")),
        bid=bid,
        ask=ask,
        timestamp=_time(payload, "time"),
    )


def bar_from_oanda(
    instrument: str, timeframe: ForexTimeframe, candle: Mapping[str, Any]
) -> ForexBar:
    """Map a v20 ``Candlestick`` onto a :class:`ForexBar`.

    Prefers the mid ladder, falling back to bid then ask, so a request made with
    ``price=BA`` still parses.
    """
    ohlc = candle.get("mid") or candle.get("bid") or candle.get("ask")
    if not isinstance(ohlc, Mapping):
        raise ValidationError("OANDA candle carries no mid/bid/ask ladder", symbol=instrument)
    return ForexBar(
        symbol=instrument,
        timeframe=timeframe,
        open_time=_time(candle, "time"),
        open=to_decimal(str(_get(ohlc, "o"))),
        high=to_decimal(str(_get(ohlc, "h"))),
        low=to_decimal(str(_get(ohlc, "l"))),
        close=to_decimal(str(_get(ohlc, "c"))),
        tick_volume=int(candle.get("volume", 0)),
    )


def account_from_oanda(payload: Mapping[str, Any], *, is_practice: bool) -> AccountInfo:
    """Map a v20 ``AccountSummary`` onto an :class:`AccountInfo`.

    v20 reports NAV rather than MT5's equity; they are the same quantity (balance plus
    unrealised PnL) and are mapped onto ``equity``.
    """
    account_id = str(_get(payload, "id"))
    margin_rate = _dec(payload, "marginRate", "0")
    leverage = int(Decimal("1") / margin_rate) if margin_rate > ZERO else 0
    return AccountInfo(
        login=_numeric_account_login(account_id),
        server=account_id,
        currency=str(_get(payload, "currency")),
        balance=_dec(payload, "balance"),
        equity=_dec(payload, "NAV", "0") or _dec(payload, "balance"),
        margin_used=_dec(payload, "marginUsed", "0"),
        margin_free=_dec(payload, "marginAvailable", "0"),
        margin_level=_dec(payload, "marginCloseoutPercent", "0"),
        leverage=leverage,
        trade_allowed=True,
        is_demo=is_practice,
        name=str(payload.get("alias", "")),
    )


def _numeric_account_login(account_id: str) -> int:
    """Derive a numeric login from an OANDA ``xxx-xxx-xxxxxxx-xxx`` account id.

    QuantFlow's :class:`AccountInfo` carries an integer login because MT5 does; OANDA's
    identifier is a string, so the digits are concatenated to give a stable number. The
    original string is preserved verbatim in ``server``.
    """
    digits = "".join(character for character in account_id if character.isdigit())
    return int(digits) if digits else 0


def position_from_oanda_trade(
    trade: Mapping[str, Any], instrument_contract_size: Decimal
) -> ForexPosition:
    """Map a v20 ``Trade`` onto a :class:`ForexPosition`.

    OANDA's own *positions* are netted per instrument and carry no ticket, so open **trades**
    are what map onto QuantFlow's per-ticket position model. A hedging-disabled account has
    at most one trade per instrument, which makes the two views identical in practice.
    """
    units = _dec(trade, "currentUnits")
    stop_loss = trade.get("stopLossOrder")
    take_profit = trade.get("takeProfitOrder")
    open_price = _dec(trade, "price")
    return ForexPosition(
        ticket=int(_get(trade, "id")),
        symbol=str(_get(trade, "instrument")),
        side=side_from_units(units),
        lots=abs(units) / instrument_contract_size,
        entry_price=open_price,
        current_price=open_price,
        opened_at=_time(trade, "openTime"),
        stop_loss=_dec(stop_loss, "price") if isinstance(stop_loss, Mapping) else None,
        take_profit=_dec(take_profit, "price") if isinstance(take_profit, Mapping) else None,
        swap=_dec(trade, "financing", "0"),
        profit=_dec(trade, "unrealizedPL", "0"),
    )


def order_from_oanda(order: Mapping[str, Any], contract_size: Decimal) -> ForexOrder:
    """Map a v20 ``Order`` onto a :class:`ForexOrder`.

    Raises:
        ValidationError: for an order type this transport does not model.

    """
    raw_type = str(_get(order, "type"))
    order_type = _OANDA_ORDER_TYPES.get(raw_type)
    if order_type is None:
        raise ValidationError(f"unsupported OANDA order type: {raw_type}", order_type=raw_type)
    units = _dec(order, "units")
    return ForexOrder(
        ticket=int(_get(order, "id")),
        symbol=str(_get(order, "instrument")),
        side=side_from_units(units),
        order_type=order_type,
        status=_OANDA_ORDER_STATES.get(str(order.get("state", "")), ForexOrderStatus.PENDING),
        lots=abs(units) / contract_size,
        price=_dec(order, "price", "0") or None,
        created_at=_time(order, "createTime") if "createTime" in order else None,
    )


def fill_from_oanda(transaction: Mapping[str, Any], contract_size: Decimal) -> ForexFill:
    """Map a v20 ``ORDER_FILL`` transaction onto a :class:`ForexFill`."""
    units = _dec(transaction, "units")
    return ForexFill(
        ticket=int(_get(transaction, "id")),
        order_ticket=int(transaction.get("orderID", 0)),
        symbol=str(_get(transaction, "instrument")),
        side=side_from_units(units),
        lots=abs(units) / contract_size,
        price=_dec(transaction, "price"),
        timestamp=_time(transaction, "time"),
        commission=_dec(transaction, "commission", "0"),
        swap=_dec(transaction, "financing", "0"),
        profit=_dec(transaction, "pl", "0"),
        is_entry=_dec(transaction, "pl", "0") == ZERO,
    )


# --------------------------------------------------------------------------- #
# HTTP boundary
# --------------------------------------------------------------------------- #
class HttpTransport(Protocol):
    """The narrow HTTP surface this worker needs, so tests can substitute a fake."""

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> tuple[int, Mapping[str, Any]]:
        """Perform a request and return ``(status_code, decoded_json)``."""
        ...

    def stream_lines(self, url: str, *, params: Mapping[str, str] | None = None) -> Iterator[str]:
        """Hold a chunked connection open, yielding one JSON document per line."""
        ...

    def close(self) -> None:
        """Release any underlying connection pool."""
        ...


class HttpxTransport:
    """The real HTTP transport, over :mod:`httpx`."""

    def __init__(self, token: str, *, timeout: float = 20.0) -> None:
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }
        self._client = httpx.Client(timeout=timeout, headers=self._headers)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> tuple[int, Mapping[str, Any]]:
        """Perform a request and return ``(status_code, decoded_json)``."""
        try:
            response = self._client.request(method, url, params=params, json=json_body)
        except httpx.HTTPError as exc:
            raise ForexConnectionError(f"OANDA request failed: {exc}", url=url) from exc
        try:
            body = response.json()
        except ValueError:
            body = {"errorMessage": response.text}
        return response.status_code, body if isinstance(body, Mapping) else {"data": body}

    def stream_lines(self, url: str, *, params: Mapping[str, str] | None = None) -> Iterator[str]:
        """Hold a chunked connection open, yielding one JSON document per line."""
        try:
            with self._client.stream("GET", url, params=params, timeout=None) as response:
                yield from response.iter_lines()
        except httpx.HTTPError as exc:
            raise ForexConnectionError(f"OANDA stream failed: {exc}", url=url) from exc

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()


def raise_for_oanda_error(status: int, body: Mapping[str, Any]) -> None:
    """Translate a v20 error response into a typed QuantFlow error.

    v20 has no single error envelope: ``errorMessage`` is always present but ``errorCode``
    is optional, and a rejected order additionally carries a ``*RejectTransaction`` whose
    ``rejectReason`` is the part worth surfacing.

    Raises:
        ForexAuthenticationError: on 401/403.
        ForexOrderRejectedError: on 400/404, or any body carrying a reject transaction.
        ForexConnectionError: on 429 and 5xx.

    """
    if status < 400:  # noqa: PLR2004 — HTTP success boundary
        return
    message = str(body.get("errorMessage", "OANDA returned an error"))
    code = str(body.get("errorCode", ""))
    reject_reason = _reject_reason(body)
    if reject_reason:
        message = f"{message} ({reject_reason})"

    if status in {_HTTP_UNAUTHORISED, _HTTP_FORBIDDEN}:
        raise ForexAuthenticationError(
            f"OANDA rejected the credentials: {message}. Check the token, the account id "
            "format (001-001-1234567-001) and that the host matches the account type.",
            status=status,
            venue_code=code,
        )
    if status == _HTTP_TOO_MANY:
        raise ForexConnectionError(
            f"OANDA rate limit hit: {message} (limit is {REST_REQUESTS_PER_SECOND}/s per IP)",
            status=status,
        )
    if status >= _HTTP_SERVER_ERROR:
        raise ForexConnectionError(f"OANDA server error: {message}", status=status)
    raise ForexOrderRejectedError(message, status=status, venue_code=code or reject_reason)


def _reject_reason(body: Mapping[str, Any]) -> str:
    """Pull ``rejectReason`` out of whichever reject transaction the body carries."""
    for key, value in body.items():
        if key.endswith("RejectTransaction") and isinstance(value, Mapping):
            return str(value.get("rejectReason", ""))
    return ""


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #
class OandaWorker:
    """A :class:`~quantflow.forex.protocol.ForexBroker` over OANDA's v20 REST API.

    Stateless between calls apart from an instrument cache, which exists because every
    units/lots conversion needs the instrument's contract size and precision, and refetching
    the instrument list per order would burn the request budget for no benefit.
    """

    description = BrokerDescription(
        venue="oanda",
        transport="v20 REST + HTTP streaming",
        supports_streaming=True,
        supports_partial_close=True,
        notes=(
            "Runs on headless Linux; no terminal, gateway or GUI.",
            "Trades in units of base currency; lots are a QuantFlow-side convention.",
            "Endpoint shapes are from published docs and are UNVERIFIED against a live account.",
        ),
    )

    def __init__(
        self,
        credentials: OandaCredentials,
        *,
        transport: HttpTransport | None = None,
    ) -> None:
        if not credentials.is_practice and not credentials.allow_live:
            raise ForexCapabilityError(
                "refusing to run against the OANDA live environment. Set "
                "QF_OANDA_ENVIRONMENT=practice, or QF_OANDA_ALLOW_LIVE=1 to override "
                "(real money).",
                environment=credentials.environment.value,
            )
        self._credentials = credentials
        self._transport: HttpTransport = transport or HttpxTransport(credentials.token)
        self._instruments: dict[str, ForexInstrument] = {}

    # ----------------------------------------------------------------- plumbing
    @property
    def _account_path(self) -> str:
        """REST path prefix for the configured account."""
        return f"/v3/accounts/{self._credentials.account_id}"

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Perform a REST call and raise on any error status."""
        url = f"{self._credentials.environment.rest_host}{path}"
        status, body = self._transport.request(method, url, params=params, json_body=json_body)
        raise_for_oanda_error(status, body)
        return body

    def disconnect(self) -> None:
        """Release the HTTP connection pool. There is no session to tear down."""
        self._transport.close()

    def _contract_size(self, symbol: str) -> Decimal:
        """Contract size for a symbol, falling back to the standard-lot constant.

        The fallback matters: it keeps a fill for an instrument we never listed from
        blowing up reconciliation, and for FX it is the right number anyway.
        """
        instrument = self._instruments.get(symbol)
        return instrument.contract_size if instrument else OANDA_UNITS_PER_LOT

    def _instrument_or_raise(self, symbol: str) -> ForexInstrument:
        """Fetch an instrument from the cache, loading the list once if needed.

        Raises:
            ForexSymbolError: if the venue does not offer the symbol.

        """
        if symbol not in self._instruments:
            self.get_symbols([symbol])
        instrument = self._instruments.get(symbol)
        if instrument is None:
            raise ForexSymbolError(f"OANDA does not offer {symbol}", symbol=symbol)
        return instrument

    # -------------------------------------------------------------- ForexBroker
    def get_account(self) -> AccountInfo:
        """Current account balance, NAV and margin."""
        body = self._call("GET", f"{self._account_path}/summary")
        account = body.get("account")
        if not isinstance(account, Mapping):
            raise ForexConnectionError("OANDA account summary carried no 'account' object")
        return account_from_oanda(account, is_practice=self._credentials.is_practice)

    def get_symbols(self, symbols: Sequence[str] | None = None) -> tuple[ForexInstrument, ...]:
        """Discover tradable instruments, majors first.

        Instruments whose pip and price grids disagree are skipped rather than returned with
        maths that would not apply to them.
        """
        params = {"instruments": ",".join(symbols)} if symbols else None
        body = self._call("GET", f"{self._account_path}/instruments", params=params)
        raw = body.get("instruments")
        if not isinstance(raw, list):
            raise ForexConnectionError("OANDA instrument list carried no 'instruments' array")

        parsed: dict[str, ForexInstrument] = {}
        for payload in raw:
            if not isinstance(payload, Mapping):
                continue
            try:
                instrument = instrument_from_oanda(payload)
            except (ForexSymbolError, ValidationError) as exc:
                logger.warning("oanda_instrument_skipped", error=str(exc))
                continue
            parsed[instrument.symbol] = instrument
        self._instruments.update(parsed)
        return tuple(parsed[name] for name in prioritise_symbols(parsed))

    def subscribe_ticks(self, symbols: Sequence[str]) -> Iterator[ForexTick]:
        """Stream prices from the streaming host.

        The stream interleaves ``PRICE`` documents with ``HEARTBEAT`` documents roughly
        every five seconds. Heartbeats are dropped here, but their absence is the intended
        liveness signal for a supervising loop.
        """
        if not symbols:
            return
        url = f"{self._credentials.environment.stream_host}{self._account_path}/pricing/stream"
        params = {"instruments": ",".join(symbols), "snapshot": "true"}
        for line in self._transport.stream_lines(url, params=params):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                logger.warning("oanda_stream_undecodable_line")
                continue
            if not isinstance(payload, Mapping) or payload.get("type") != "PRICE":
                continue
            if payload.get("tradeable") is False:
                continue
            yield tick_from_oanda(payload)

    def get_bars(
        self,
        symbol: str,
        timeframe: ForexTimeframe,
        count: int,
        end: datetime | None = None,
    ) -> tuple[ForexBar, ...]:
        """Fetch the most recent ``count`` candles at or before ``end``, oldest first."""
        if not 0 < count <= MAX_CANDLE_COUNT:
            raise ValidationError(
                f"count must be between 1 and {MAX_CANDLE_COUNT}, got {count}", symbol=symbol
            )
        params: dict[str, str] = {
            "granularity": timeframe.oanda_granularity,
            "count": str(count),
            "price": "M",
        }
        if end is not None:
            params["to"] = end.astimezone(UTC).isoformat().replace("+00:00", "Z")
        body = self._call(
            "GET", f"{self._account_path}/instruments/{symbol}/candles", params=params
        )
        candles = body.get("candles")
        if not isinstance(candles, list):
            raise ForexConnectionError(f"no candles returned for {symbol}", symbol=symbol)
        return tuple(
            bar_from_oanda(symbol, timeframe, candle)
            for candle in candles
            if isinstance(candle, Mapping) and candle.get("complete", True)
        )

    def submit_order(self, request: ForexOrderRequest) -> OrderAck:
        """Send an order to OANDA.

        Market orders are sent ``FOK`` — v20 restricts market time-in-force to ``FOK`` or
        ``IOC``, and a partially-filled FX entry would leave the position size disagreeing
        with what was sized for.
        """
        instrument = self._instrument_or_raise(request.symbol)
        units = units_from_lots(request.lots, request.side, instrument)
        order: dict[str, Any] = {
            "type": _forex_type_to_oanda(request.order_type),
            "instrument": request.symbol,
            "units": _format_units(units),
            "positionFill": "DEFAULT",
        }
        if request.order_type is ForexOrderType.MARKET:
            order["timeInForce"] = "FOK"
        else:
            order["timeInForce"] = "GTC"
            order["price"] = str(instrument.round_price(_require_price(request)))
        if request.stop_loss is not None:
            order["stopLossOnFill"] = {
                "price": str(instrument.round_price(request.stop_loss)),
                "timeInForce": "GTC",
            }
        if request.take_profit is not None:
            order["takeProfitOnFill"] = {
                "price": str(instrument.round_price(request.take_profit)),
                "timeInForce": "GTC",
            }
        if request.client_tag:
            order["clientExtensions"] = {"tag": request.client_tag}

        body = self._call("POST", f"{self._account_path}/orders", json_body={"order": order})
        return _ack_from_order_response(body, instrument.contract_size)

    def modify_stop(
        self,
        ticket: int,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> OrderAck:
        """Attach or move the stop and/or take-profit on an open trade."""
        if stop_loss is None and take_profit is None:
            raise ValidationError("modify_stop needs a stop_loss or a take_profit")
        payload: dict[str, Any] = {}
        if stop_loss is not None:
            payload["stopLoss"] = {"price": str(stop_loss), "timeInForce": "GTC"}
        if take_profit is not None:
            payload["takeProfit"] = {"price": str(take_profit), "timeInForce": "GTC"}
        body = self._call("PUT", f"{self._account_path}/trades/{ticket}/orders", json_body=payload)
        return OrderAck(
            accepted=True,
            status=ForexOrderStatus.PLACED,
            ticket=ticket,
            message=str(body.get("lastTransactionID", "")),
        )

    def close_position(self, ticket: int, lots: Decimal | None = None) -> OrderAck:
        """Close a trade fully, or partially when ``lots`` is given."""
        if lots is None:
            units_field = "ALL"
        else:
            if lots <= ZERO:
                raise ValidationError(f"lots must be positive, got {lots}", ticket=ticket)
            contract_size = self._contract_size(self._symbol_for_ticket(ticket))
            units_field = _format_units(lots * contract_size)
        body = self._call(
            "PUT", f"{self._account_path}/trades/{ticket}/close", json_body={"units": units_field}
        )
        fill = body.get("orderFillTransaction")
        filled = (
            abs(_dec(fill, "units", "0")) / self._contract_size(str(fill.get("instrument", "")))
            if isinstance(fill, Mapping)
            else ZERO
        )
        return OrderAck(
            accepted=True,
            status=ForexOrderStatus.FILLED if filled > ZERO else ForexOrderStatus.PLACED,
            ticket=ticket,
            filled_lots=filled,
            average_price=_dec(fill, "price", "0") or None if isinstance(fill, Mapping) else None,
        )

    def get_orders(self, symbol: str | None = None) -> tuple[ForexOrder, ...]:
        """List working orders, excluding stop/take-profit attachments."""
        body = self._call("GET", f"{self._account_path}/pendingOrders")
        raw = body.get("orders")
        if not isinstance(raw, list):
            return ()
        orders: list[ForexOrder] = []
        for payload in raw:
            if not isinstance(payload, Mapping):
                continue
            if str(payload.get("type", "")) in _PROTECTIVE_ORDER_TYPES:
                continue
            instrument_name = str(payload.get("instrument", ""))
            if symbol is not None and instrument_name != symbol:
                continue
            orders.append(order_from_oanda(payload, self._contract_size(instrument_name)))
        return tuple(orders)

    def get_positions(self, symbol: str | None = None) -> tuple[ForexPosition, ...]:
        """List open trades as positions."""
        body = self._call("GET", f"{self._account_path}/openTrades")
        raw = body.get("trades")
        if not isinstance(raw, list):
            return ()
        positions: list[ForexPosition] = []
        for payload in raw:
            if not isinstance(payload, Mapping):
                continue
            instrument_name = str(payload.get("instrument", ""))
            if symbol is not None and instrument_name != symbol:
                continue
            positions.append(
                position_from_oanda_trade(payload, self._contract_size(instrument_name))
            )
        return tuple(positions)

    def get_fills(
        self,
        since: datetime,
        until: datetime | None = None,
        symbol: str | None = None,
    ) -> tuple[ForexFill, ...]:
        """List ``ORDER_FILL`` transactions in a time range, oldest first.

        v20's time-ranged transaction endpoint answers with a list of *page URLs* rather
        than the transactions themselves, so each page is fetched in turn. For continuous
        reconciliation prefer :meth:`get_fills_since_id`, which is a single call.
        """
        params = {
            "from": _rfc3339(since),
            "to": _rfc3339(until or datetime.now(tz=UTC)),
            "type": "ORDER_FILL",
        }
        body = self._call("GET", f"{self._account_path}/transactions", params=params)
        pages = body.get("pages")
        fills: list[ForexFill] = []
        if isinstance(pages, list):
            for page in pages:
                fills.extend(self._fills_from_page_url(str(page)))
        return tuple(fill for fill in fills if symbol is None or fill.symbol == symbol)

    def get_fills_since_id(self, transaction_id: str) -> tuple[tuple[ForexFill, ...], str]:
        """Fetch fills after a transaction id, with the new watermark.

        This is the reconciliation path: persist the returned id and pass it back next time,
        and no fill can be missed between polls.
        """
        body = self._call(
            "GET",
            f"{self._account_path}/transactions/sinceid",
            params={"id": transaction_id, "type": "ORDER_FILL"},
        )
        return (
            self._fills_from_transactions(body.get("transactions")),
            str(body.get("lastTransactionID", transaction_id)),
        )

    def _fills_from_page_url(self, url: str) -> tuple[ForexFill, ...]:
        """Fetch one transaction page by its absolute URL."""
        status, body = self._transport.request("GET", url)
        raise_for_oanda_error(status, body)
        return self._fills_from_transactions(body.get("transactions"))

    def _fills_from_transactions(self, transactions: Any) -> tuple[ForexFill, ...]:
        """Parse the ``ORDER_FILL`` entries out of a transaction array."""
        if not isinstance(transactions, list):
            return ()
        return tuple(
            fill_from_oanda(item, self._contract_size(str(item.get("instrument", ""))))
            for item in transactions
            if isinstance(item, Mapping) and item.get("type") == "ORDER_FILL"
        )

    def _symbol_for_ticket(self, ticket: int) -> str:
        """Look up which instrument a trade belongs to.

        Raises:
            ForexOrderRejectedError: if the trade is not open.

        """
        body = self._call("GET", f"{self._account_path}/trades/{ticket}")
        trade = body.get("trade")
        if not isinstance(trade, Mapping):
            raise ForexOrderRejectedError(f"no open trade with id {ticket}", ticket=ticket)
        return str(trade.get("instrument", ""))


def _rfc3339(moment: datetime) -> str:
    """Render an instant the way v20's ``from``/``to`` parameters expect."""
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_price(request: ForexOrderRequest) -> Decimal:
    """Return the price a non-market order must carry.

    Raises:
        ValidationError: if it is absent.

    """
    if request.price is None:
        raise ValidationError(
            f"{request.order_type.value} orders need a price", symbol=request.symbol
        )
    return request.price


def _forex_type_to_oanda(order_type: ForexOrderType) -> str:
    """Map a QuantFlow order type onto v20's spelling."""
    if order_type is ForexOrderType.STOP_LIMIT:
        raise ValidationError("OANDA v20 has no stop-limit order type")
    return {
        ForexOrderType.MARKET: "MARKET",
        ForexOrderType.LIMIT: "LIMIT",
        ForexOrderType.STOP: "STOP",
    }[order_type]


def _ack_from_order_response(body: Mapping[str, Any], contract_size: Decimal) -> OrderAck:
    """Normalise a v20 order-create response.

    A ``201`` does not imply a fill: ``orderFillTransaction`` is only present when the order
    filled immediately, so its absence is reported as ``PLACED`` rather than assumed filled.
    """
    reject = body.get("orderRejectTransaction")
    if isinstance(reject, Mapping):
        return OrderAck(
            accepted=False,
            status=ForexOrderStatus.REJECTED,
            message=str(reject.get("rejectReason", "rejected")),
            venue_code=str(reject.get("type", "")),
        )
    fill = body.get("orderFillTransaction")
    create = body.get("orderCreateTransaction")
    ticket_source = fill if isinstance(fill, Mapping) else create
    ticket = (
        int(ticket_source.get("orderID") or ticket_source.get("id") or 0)
        if isinstance(ticket_source, Mapping)
        else None
    )
    if isinstance(fill, Mapping):
        return OrderAck(
            accepted=True,
            status=ForexOrderStatus.FILLED,
            ticket=ticket,
            filled_lots=abs(_dec(fill, "units", "0")) / contract_size,
            average_price=_dec(fill, "price", "0") or None,
            venue_code=str(body.get("lastTransactionID", "")),
        )
    return OrderAck(
        accepted=True,
        status=ForexOrderStatus.PLACED,
        ticket=ticket,
        venue_code=str(body.get("lastTransactionID", "")),
    )
