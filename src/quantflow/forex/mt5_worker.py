"""MetaTrader 5 transport — one optional implementation of :class:`ForexBroker`.

**EXPERIMENTAL — this module has never placed an order.** No FX credentials exist in
this project; it is written to the venue's published API and tested against fakes only.

Bybit's FX product (``EURUSD+``, ``GBPUSD+`` …) is a CFD on MetaTrader 5. It is **not**
reachable from Bybit's V5 REST API — the ``fx``/``forex``/``tradfi``/``cfd`` categories all
return ``retCode=10001 "Illegal category"``, ``/v5/tradfi/*`` 404s, and CCXT carries no FX
endpoints — so it needs a separate MT5 account and the terminal's own Python bridge.

That bridge is the constraint this module is shaped around. The ``MetaTrader5`` package
publishes **win_amd64 wheels only** and talks to a running MT5 terminal over local IPC, so
this transport can only run on Windows. It is therefore:

* **not a dependency.** ``MetaTrader5`` is never declared in ``pyproject.toml`` — adding it
  would break ``pip install`` on every non-Windows machine. It is imported lazily, by name,
  through :func:`importlib.import_module`.
* **not the only route.** See :mod:`quantflow.forex.oanda_worker` for a transport that runs
  on a headless Linux host. This module stays because an operator who already has the MT5
  account should not have to abandon it.
* **explicit about why it cannot run.** :func:`capabilities` reports platform, package and
  credential blockers as actionable sentences before anything is attempted, so the failure
  mode is a readable message rather than an ``ImportError`` from three frames down.

Everything below :func:`capabilities` that parses a payload is a pure function taking a
duck-typed object, which is what lets the whole mapping layer be tested against fakes on a
machine that can never run the real terminal.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final

import structlog
from fastapi import FastAPI, HTTPException

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO, to_decimal
from quantflow.domain.enums import OrderSide
from quantflow.forex.errors import (
    ForexCapabilityError,
    ForexConnectionError,
    ForexOrderRejectedError,
)
from quantflow.forex.instruments import (
    ForexInstrument,
    TradeMode,
    mt5_weekday_to_python,
    prioritise_symbols,
)
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

#: Environment variables the operator must set on the Windows host.
REQUIRED_ENV_VARS: Final[tuple[str, ...]] = (
    "QF_MT5_LOGIN",
    "QF_MT5_PASSWORD",
    "QF_MT5_SERVER",
)

#: The only platform the ``MetaTrader5`` wheel supports.
SUPPORTED_PLATFORM: Final = "Windows"

MT5_PACKAGE: Final = "MetaTrader5"

#: ``ACCOUNT_TRADE_MODE_REAL``. Anything else (demo, contest) is not real money.
MT5_ACCOUNT_TRADE_MODE_REAL: Final = 2

#: ``TRADE_RETCODE_DONE`` — the only unambiguous success code.
MT5_TRADE_RETCODE_DONE: Final = 10009
MT5_TRADE_RETCODE_PLACED: Final = 10008

_DEFAULT_MAGIC: Final = 770_001


# --------------------------------------------------------------------------- #
# Capability reporting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Capabilities:
    """Whether this host can run the MT5 transport, and precisely what is missing."""

    platform: str
    python_version: str
    package_available: bool
    package_version: str | None
    missing_env: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        """Whether the transport can be started here."""
        return not self.blockers

    def describe(self) -> str:
        """A multi-line, paste-into-a-runbook explanation."""
        if self.ready:
            return (
                f"MT5 transport ready: {self.platform}, Python {self.python_version}, "
                f"MetaTrader5 {self.package_version}."
            )
        lines = [f"MT5 transport cannot run on this host ({self.platform}):"]
        lines.extend(f"  - {blocker}" for blocker in self.blockers)
        return "\n".join(lines)

    def raise_if_not_ready(self) -> None:
        """Raise a capability error carrying every blocker.

        Raises:
            ForexCapabilityError: if anything at all is missing.

        """
        if not self.ready:
            raise ForexCapabilityError(self.describe(), blockers=list(self.blockers))


def _probe_metatrader5() -> tuple[bool, str | None]:
    """Report whether the ``MetaTrader5`` package is importable, and at what version."""
    if importlib.util.find_spec(MT5_PACKAGE) is None:
        return False, None
    try:
        from importlib.metadata import version

        return True, version(MT5_PACKAGE)
    except Exception:
        return True, None


def capabilities(
    env: Mapping[str, str] | None = None,
    *,
    platform_system: str | None = None,
    package_probe: Callable[[], tuple[bool, str | None]] | None = None,
) -> Capabilities:
    """Report whether the MT5 transport can run here, and why not if it cannot.

    Every input is injectable so the blocked and unblocked paths are both testable from a
    machine that can never satisfy the real ones.
    """
    resolved_env = os.environ if env is None else env
    resolved_platform = platform.system() if platform_system is None else platform_system
    probe = _probe_metatrader5 if package_probe is None else package_probe

    blockers: list[str] = []

    if resolved_platform != SUPPORTED_PLATFORM:
        blockers.append(
            f"platform: the {MT5_PACKAGE} package publishes win_amd64 wheels only and drives "
            f"a local MT5 terminal over IPC, but this host reports {resolved_platform!r}. "
            "Run this worker on a Windows VM/VPS with the MT5 terminal installed, or use the "
            "OANDA transport (quantflow.forex.oanda_worker), which runs on headless Linux."
        )

    package_available, package_version = probe()
    if not package_available:
        blockers.append(
            f"package: {MT5_PACKAGE} is not importable. On the Windows host run "
            f"'py -3.12 -m pip install {MT5_PACKAGE}'. It is deliberately absent from "
            "pyproject.toml because it cannot install on Linux or macOS."
        )

    missing_env = tuple(name for name in REQUIRED_ENV_VARS if not resolved_env.get(name))
    if missing_env:
        blockers.append(
            "credentials: " + ", ".join(missing_env) + " are unset. Create the Bybit MT5 CFD "
            "account, choose a DEMO server, and export the login id, password and server name "
            "on the Windows host before starting the worker."
        )

    return Capabilities(
        platform=resolved_platform,
        python_version=platform.python_version(),
        package_available=package_available,
        package_version=package_version,
        missing_env=missing_env,
        blockers=tuple(blockers),
    )


@dataclass(frozen=True, slots=True)
class MT5Credentials:
    """MT5 login details. The password never appears in a repr or a log line."""

    login: int
    password: str = field(repr=False)
    server: str
    terminal_path: str | None = None
    timeout_ms: int = 60_000
    allow_live: bool = False

    @property
    def is_demo_server(self) -> bool:
        """Whether the server name identifies a demo/practice server.

        A name-based check is a guard rail, not proof. The authoritative check is the
        account's own ``trade_mode`` after connecting — see :meth:`MT5Worker.connect`.
        """
        return "demo" in self.server.lower()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MT5Credentials:
        """Build credentials from the environment.

        Raises:
            ForexCapabilityError: if a required variable is unset.
            ValidationError: if the login is not numeric.

        """
        resolved = os.environ if env is None else env
        missing = [name for name in REQUIRED_ENV_VARS if not resolved.get(name)]
        if missing:
            raise ForexCapabilityError(
                "missing MT5 credentials: " + ", ".join(missing),
                missing_env=missing,
            )
        raw_login = resolved["QF_MT5_LOGIN"]
        try:
            login = int(raw_login)
        except ValueError as exc:
            raise ValidationError(
                f"QF_MT5_LOGIN must be the numeric MT5 login id, got {raw_login!r}"
            ) from exc
        return cls(
            login=login,
            password=resolved["QF_MT5_PASSWORD"],
            server=resolved["QF_MT5_SERVER"],
            terminal_path=resolved.get("QF_MT5_PATH") or None,
            timeout_ms=int(resolved.get("QF_MT5_TIMEOUT_MS", "60000")),
            allow_live=resolved.get("QF_MT5_ALLOW_LIVE", "").strip().lower()
            in {"1", "true", "yes"},
        )


# --------------------------------------------------------------------------- #
# Payload mapping — pure functions over duck-typed MT5 structs
# --------------------------------------------------------------------------- #
_MISSING: Final = object()


def _raw(payload: Any, name: str, default: Any = _MISSING) -> Any:
    """Read ``name`` off an MT5 struct, a numpy row or a plain mapping.

    MT5 returns named tuples for symbols and accounts but numpy structured rows for bars,
    so both access styles have to work. A genuinely absent field raises naming itself,
    which is far easier to act on than an ``AttributeError`` on ``None``.
    """
    value = getattr(payload, name, _MISSING)
    if value is _MISSING:
        try:
            value = payload[name]
        except (TypeError, KeyError, IndexError, ValueError):
            value = _MISSING
    if value is _MISSING:
        if default is _MISSING:
            raise ValidationError(f"MT5 payload is missing required field {name!r}", field=name)
        return default
    return value


def _dec(payload: Any, name: str, default: Any = _MISSING) -> Decimal:
    """Read a field and coerce it to :class:`~decimal.Decimal` without float drift."""
    return to_decimal(_raw(payload, name, default))


def _optional_price(payload: Any, name: str) -> Decimal | None:
    """Read a price field where MT5 uses ``0.0`` to mean "not set"."""
    value = to_decimal(_raw(payload, name, 0))
    return value if value > ZERO else None


def _timestamp(payload: Any, name: str) -> datetime:
    """Read an MT5 epoch-seconds field as a UTC datetime."""
    return datetime.fromtimestamp(int(_raw(payload, name)), tz=UTC)


def instrument_from_symbol_info(info: Any, venue: str = "mt5") -> ForexInstrument:
    """Map an MT5 ``symbol_info`` struct onto a :class:`ForexInstrument`.

    ``margin_initial`` is only adopted as a margin *rate* when it looks like a fraction.
    MT5 otherwise reports it as an absolute per-lot amount in the margin currency, which
    cannot be turned into a rate without a price — so it is left unknown rather than
    guessed, and :attr:`ForexInstrument.leverage` stays ``None``.
    """
    trade_mode = TradeMode.from_mt5(int(_raw(info, "trade_mode")))
    margin_initial = to_decimal(_raw(info, "margin_initial", 0))
    margin_rate = margin_initial if ZERO < margin_initial <= Decimal("1") else ZERO
    visible = bool(_raw(info, "visible", True)) and bool(_raw(info, "select", True))

    return ForexInstrument(
        symbol=str(_raw(info, "name")),
        base=str(_raw(info, "currency_base")),
        quote=str(_raw(info, "currency_profit")),
        contract_size=_dec(info, "trade_contract_size"),
        min_lot=_dec(info, "volume_min"),
        max_lot=_dec(info, "volume_max"),
        lot_step=_dec(info, "volume_step"),
        digits=int(_raw(info, "digits")),
        point=_dec(info, "point"),
        tick_size=_dec(info, "trade_tick_size"),
        tick_value=_dec(info, "trade_tick_value"),
        margin_rate=margin_rate,
        trade_mode=trade_mode,
        spread_points=_dec(info, "spread", 0),
        swap_long=_dec(info, "swap_long", 0),
        swap_short=_dec(info, "swap_short", 0),
        triple_swap_weekday=mt5_weekday_to_python(int(_raw(info, "swap_rollover3days", 3))),
        tradable=visible and trade_mode is not TradeMode.DISABLED,
        venue=venue,
    )


def tick_from_mt5(symbol: str, raw: Any) -> ForexTick:
    """Map an MT5 tick struct onto a :class:`ForexTick`."""
    last = to_decimal(_raw(raw, "last", 0))
    return ForexTick(
        symbol=symbol,
        bid=_dec(raw, "bid"),
        ask=_dec(raw, "ask"),
        timestamp=_timestamp(raw, "time"),
        last=last if last > ZERO else None,
        volume=to_decimal(_raw(raw, "volume", 0)),
    )


def bar_from_mt5(symbol: str, timeframe: ForexTimeframe, row: Any) -> ForexBar:
    """Map one row of an MT5 ``copy_rates_*`` result onto a :class:`ForexBar`."""
    return ForexBar(
        symbol=symbol,
        timeframe=timeframe,
        open_time=_timestamp(row, "time"),
        open=_dec(row, "open"),
        high=_dec(row, "high"),
        low=_dec(row, "low"),
        close=_dec(row, "close"),
        tick_volume=int(_raw(row, "tick_volume", 0)),
        spread_points=_dec(row, "spread", 0),
        real_volume=_dec(row, "real_volume", 0),
    )


#: ``ORDER_TYPE_*`` codes, indexed by (side, order type).
_MT5_ORDER_TYPES: Final[dict[tuple[OrderSide, ForexOrderType], int]] = {
    (OrderSide.BUY, ForexOrderType.MARKET): 0,
    (OrderSide.SELL, ForexOrderType.MARKET): 1,
    (OrderSide.BUY, ForexOrderType.LIMIT): 2,
    (OrderSide.SELL, ForexOrderType.LIMIT): 3,
    (OrderSide.BUY, ForexOrderType.STOP): 4,
    (OrderSide.SELL, ForexOrderType.STOP): 5,
    (OrderSide.BUY, ForexOrderType.STOP_LIMIT): 6,
    (OrderSide.SELL, ForexOrderType.STOP_LIMIT): 7,
}

_MT5_ORDER_TYPES_INVERSE: Final[dict[int, tuple[OrderSide, ForexOrderType]]] = {
    code: pair for pair, code in _MT5_ORDER_TYPES.items()
}

#: ``ORDER_STATE_*`` codes.
_MT5_ORDER_STATES: Final[dict[int, ForexOrderStatus]] = {
    0: ForexOrderStatus.PENDING,
    1: ForexOrderStatus.PLACED,
    2: ForexOrderStatus.CANCELLED,
    3: ForexOrderStatus.PARTIALLY_FILLED,
    4: ForexOrderStatus.FILLED,
    5: ForexOrderStatus.REJECTED,
    6: ForexOrderStatus.EXPIRED,
    7: ForexOrderStatus.PENDING,
    8: ForexOrderStatus.PENDING,
    9: ForexOrderStatus.PENDING,
}


def mt5_order_type(side: OrderSide, order_type: ForexOrderType) -> int:
    """Map a side/type pair onto the MT5 ``ORDER_TYPE_*`` code."""
    code = _MT5_ORDER_TYPES.get((side, order_type))
    if code is None:  # pragma: no cover — the table is exhaustive over both enums
        raise ValidationError(f"unsupported order {side.value}/{order_type.value}")
    return code


def side_and_type_from_mt5(code: int) -> tuple[OrderSide, ForexOrderType]:
    """Map an MT5 ``ORDER_TYPE_*`` code back to a side/type pair."""
    pair = _MT5_ORDER_TYPES_INVERSE.get(code)
    if pair is None:
        raise ValidationError(f"unrecognised MT5 order type code: {code}", code=code)
    return pair


def order_status_from_mt5(state: int) -> ForexOrderStatus:
    """Map an MT5 ``ORDER_STATE_*`` code onto a status, defaulting unknown to pending."""
    return _MT5_ORDER_STATES.get(state, ForexOrderStatus.PENDING)


def position_from_mt5(raw: Any) -> ForexPosition:
    """Map an MT5 position struct onto a :class:`ForexPosition`."""
    return ForexPosition(
        ticket=int(_raw(raw, "ticket")),
        symbol=str(_raw(raw, "symbol")),
        side=OrderSide.SELL if int(_raw(raw, "type")) == 1 else OrderSide.BUY,
        lots=_dec(raw, "volume"),
        entry_price=_dec(raw, "price_open"),
        current_price=_dec(raw, "price_current"),
        opened_at=_timestamp(raw, "time"),
        stop_loss=_optional_price(raw, "sl"),
        take_profit=_optional_price(raw, "tp"),
        swap=_dec(raw, "swap", 0),
        profit=_dec(raw, "profit", 0),
        magic=int(_raw(raw, "magic", 0)),
        comment=str(_raw(raw, "comment", "")),
    )


def order_from_mt5(raw: Any) -> ForexOrder:
    """Map an MT5 order struct onto a :class:`ForexOrder`."""
    side, order_type = side_and_type_from_mt5(int(_raw(raw, "type")))
    initial = _dec(raw, "volume_initial")
    remaining = _dec(raw, "volume_current", initial)
    return ForexOrder(
        ticket=int(_raw(raw, "ticket")),
        symbol=str(_raw(raw, "symbol")),
        side=side,
        order_type=order_type,
        status=order_status_from_mt5(int(_raw(raw, "state"))),
        lots=initial,
        filled_lots=initial - remaining,
        price=_optional_price(raw, "price_open"),
        stop_loss=_optional_price(raw, "sl"),
        take_profit=_optional_price(raw, "tp"),
        created_at=_timestamp(raw, "time_setup"),
        magic=int(_raw(raw, "magic", 0)),
        comment=str(_raw(raw, "comment", "")),
    )


def fill_from_mt5(raw: Any) -> ForexFill:
    """Map an MT5 deal struct onto a :class:`ForexFill`."""
    return ForexFill(
        ticket=int(_raw(raw, "ticket")),
        order_ticket=int(_raw(raw, "order", 0)),
        symbol=str(_raw(raw, "symbol")),
        side=OrderSide.SELL if int(_raw(raw, "type")) == 1 else OrderSide.BUY,
        lots=_dec(raw, "volume"),
        price=_dec(raw, "price"),
        timestamp=_timestamp(raw, "time"),
        commission=_dec(raw, "commission", 0),
        swap=_dec(raw, "swap", 0),
        profit=_dec(raw, "profit", 0),
        is_entry=int(_raw(raw, "entry", 0)) == 0,
        magic=int(_raw(raw, "magic", 0)),
        comment=str(_raw(raw, "comment", "")),
    )


def account_from_mt5(raw: Any) -> AccountInfo:
    """Map an MT5 ``account_info`` struct onto an :class:`AccountInfo`."""
    return AccountInfo(
        login=int(_raw(raw, "login")),
        server=str(_raw(raw, "server")),
        currency=str(_raw(raw, "currency")),
        balance=_dec(raw, "balance"),
        equity=_dec(raw, "equity"),
        margin_used=_dec(raw, "margin", 0),
        margin_free=_dec(raw, "margin_free", 0),
        margin_level=_dec(raw, "margin_level", 0),
        leverage=int(_raw(raw, "leverage", 0)),
        trade_allowed=bool(_raw(raw, "trade_allowed", False)),
        is_demo=int(_raw(raw, "trade_mode", 0)) != MT5_ACCOUNT_TRADE_MODE_REAL,
        name=str(_raw(raw, "name", "")),
    )


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #
class MT5Worker:
    """A :class:`~quantflow.forex.protocol.ForexBroker` backed by a local MT5 terminal.

    Not thread-safe by design: the ``MetaTrader5`` package keeps terminal state per process
    and expects calls from one thread. The HTTP service below funnels every call through a
    single worker thread for exactly this reason.
    """

    description = BrokerDescription(
        venue="bybit-mt5",
        transport="MetaTrader5 IPC",
        supports_streaming=False,
        notes=("Windows-only.", "Requires a running MT5 terminal with algo trading enabled."),
    )

    def __init__(
        self,
        credentials: MT5Credentials | None = None,
        *,
        magic: int = _DEFAULT_MAGIC,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._credentials = credentials
        self._magic = magic
        self._env = env
        self._mt5: Any = None
        self._connected = False
        if credentials is not None:
            self._assert_demo_intent(credentials)

    @staticmethod
    def _assert_demo_intent(credentials: MT5Credentials) -> None:
        """Refuse a server that does not look like demo unless explicitly overridden."""
        if not credentials.is_demo_server and not credentials.allow_live:
            raise ForexCapabilityError(
                f"server {credentials.server!r} does not look like a demo server. "
                "This worker is demo-only: point QF_MT5_SERVER at the MT5 demo server, or "
                "set QF_MT5_ALLOW_LIVE=1 to deliberately override (real money).",
                server=credentials.server,
            )

    # ---------------------------------------------------------------- lifecycle
    def _require_mt5(self) -> Any:
        """Return the live terminal handle, or explain why there is none.

        Raises:
            ForexCapabilityError: if not connected.

        """
        if self._mt5 is None or not self._connected:
            capabilities(self._env).raise_if_not_ready()
            raise ForexCapabilityError(
                "MT5 worker is not connected; call connect() first",
            )
        return self._mt5

    def connect(self) -> AccountInfo:
        """Start the terminal session and verify the account is not a real-money one.

        Raises:
            ForexCapabilityError: if the host cannot run MT5, or the account is real money
                without an explicit override.
            ForexConnectionError: if the terminal refuses to initialise or log in.

        """
        capabilities(self._env).raise_if_not_ready()
        credentials = self._credentials or MT5Credentials.from_env(self._env)
        self._assert_demo_intent(credentials)

        import importlib

        self._mt5 = importlib.import_module(MT5_PACKAGE)

        kwargs: dict[str, Any] = {
            "login": credentials.login,
            "password": credentials.password,
            "server": credentials.server,
            "timeout": credentials.timeout_ms,
        }
        if credentials.terminal_path:
            kwargs["path"] = credentials.terminal_path
        if not self._mt5.initialize(**kwargs):
            code, message = self._last_error()
            raise ForexConnectionError(
                f"MT5 terminal did not initialise: {message}",
                venue_code=code,
                server=credentials.server,
            )
        self._connected = True

        account = account_from_mt5(self._mt5.account_info())
        if not account.is_demo and not credentials.allow_live:
            self.disconnect()
            raise ForexCapabilityError(
                f"account {account.login} on {account.server} is a REAL-money account. "
                "Refusing to trade it. Set QF_MT5_ALLOW_LIVE=1 only if that is intended.",
                login=account.login,
            )
        logger.info(
            "mt5_connected", login=account.login, server=account.server, demo=account.is_demo
        )
        return account

    def disconnect(self) -> None:
        """Shut the terminal session down. Safe to call when not connected."""
        if self._mt5 is not None:
            with suppress(Exception):
                self._mt5.shutdown()
        self._connected = False

    def _last_error(self) -> tuple[str, str]:
        """Read the terminal's last error as a (code, message) pair."""
        if self._mt5 is None:
            return "", "MT5 module not loaded"
        try:
            code, message = self._mt5.last_error()
        except Exception:
            return "", "unknown MT5 error"
        return str(code), str(message)

    # ------------------------------------------------------------ ForexBroker
    def get_account(self) -> AccountInfo:
        """Current account balance, equity and margin."""
        return account_from_mt5(self._require_mt5().account_info())

    def get_symbols(self, symbols: Sequence[str] | None = None) -> tuple[ForexInstrument, ...]:
        """Discover tradable instruments, majors first.

        The MAJORS ranking only reorders what the terminal actually reports — a symbol the
        venue does not offer is never conjured into the result.
        """
        terminal = self._require_mt5()
        raw_symbols = terminal.symbols_get() or ()
        by_name = {str(_raw(item, "name")): item for item in raw_symbols}
        wanted = list(by_name) if symbols is None else [s for s in symbols if s in by_name]
        instruments: list[ForexInstrument] = []
        for name in prioritise_symbols(wanted):
            terminal.symbol_select(name, True)
            info = terminal.symbol_info(name) or by_name[name]
            instruments.append(instrument_from_symbol_info(info))
        return tuple(instruments)

    def subscribe_ticks(self, symbols: Sequence[str]) -> Iterator[ForexTick]:
        """Poll the terminal for quotes.

        The MT5 bridge exposes no push channel, so this is a polling loop presented as a
        stream. It yields one round of quotes per pass and stops; callers that want a
        continuous feed drive it from their own loop.
        """
        terminal = self._require_mt5()
        for symbol in symbols:
            raw = terminal.symbol_info_tick(symbol)
            if raw is not None:
                yield tick_from_mt5(symbol, raw)

    def get_bars(
        self,
        symbol: str,
        timeframe: ForexTimeframe,
        count: int,
        end: datetime | None = None,
    ) -> tuple[ForexBar, ...]:
        """Fetch the most recent ``count`` bars at or before ``end``, oldest first."""
        terminal = self._require_mt5()
        granularity = getattr(terminal, timeframe.mt5_constant)
        rows = (
            terminal.copy_rates_from_pos(symbol, granularity, 0, count)
            if end is None
            else terminal.copy_rates_from(symbol, granularity, end, count)
        )
        if rows is None:
            code, message = self._last_error()
            raise ForexConnectionError(
                f"no bars returned for {symbol}: {message}", symbol=symbol, venue_code=code
            )
        return tuple(bar_from_mt5(symbol, timeframe, row) for row in rows)

    def submit_order(self, request: ForexOrderRequest) -> OrderAck:
        """Send an order to the terminal."""
        terminal = self._require_mt5()
        is_market = request.order_type is ForexOrderType.MARKET
        payload: dict[str, Any] = {
            "action": terminal.TRADE_ACTION_DEAL if is_market else terminal.TRADE_ACTION_PENDING,
            "symbol": request.symbol,
            "volume": float(request.lots),
            "type": mt5_order_type(request.side, request.order_type),
            "deviation": int(request.deviation_points),
            "magic": request.magic or self._magic,
            "comment": request.comment or request.client_tag,
            "type_time": terminal.ORDER_TIME_GTC,
            "type_filling": terminal.ORDER_FILLING_IOC,
        }
        if request.price is not None:
            payload["price"] = float(request.price)
        if request.stop_loss is not None:
            payload["sl"] = float(request.stop_loss)
        if request.take_profit is not None:
            payload["tp"] = float(request.take_profit)
        return self._send(payload)

    def modify_stop(
        self,
        ticket: int,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> OrderAck:
        """Attach or move the stop and/or take-profit on an open position."""
        if stop_loss is None and take_profit is None:
            raise ValidationError("modify_stop needs a stop_loss or a take_profit")
        terminal = self._require_mt5()
        position = self._position_or_raise(ticket)
        payload: dict[str, Any] = {
            "action": terminal.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": float(stop_loss if stop_loss is not None else (position.stop_loss or ZERO)),
            "tp": float(take_profit if take_profit is not None else (position.take_profit or ZERO)),
        }
        return self._send(payload)

    def close_position(self, ticket: int, lots: Decimal | None = None) -> OrderAck:
        """Close a position fully, or partially when ``lots`` is given."""
        terminal = self._require_mt5()
        position = self._position_or_raise(ticket)
        closing_lots = position.lots if lots is None else lots
        if closing_lots <= ZERO or closing_lots > position.lots:
            raise ValidationError(
                f"cannot close {closing_lots} of a {position.lots} lot position",
                ticket=ticket,
            )
        closing_side = position.side.opposite
        payload: dict[str, Any] = {
            "action": terminal.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": float(closing_lots),
            "type": mt5_order_type(closing_side, ForexOrderType.MARKET),
            "deviation": 20,
            "magic": self._magic,
            "type_time": terminal.ORDER_TIME_GTC,
            "type_filling": terminal.ORDER_FILLING_IOC,
        }
        return self._send(payload)

    def get_orders(self, symbol: str | None = None) -> tuple[ForexOrder, ...]:
        """List working (pending) orders."""
        terminal = self._require_mt5()
        raw = (terminal.orders_get(symbol=symbol) if symbol else terminal.orders_get()) or ()
        return tuple(order_from_mt5(item) for item in raw)

    def get_positions(self, symbol: str | None = None) -> tuple[ForexPosition, ...]:
        """List open positions."""
        terminal = self._require_mt5()
        raw = (terminal.positions_get(symbol=symbol) if symbol else terminal.positions_get()) or ()
        return tuple(position_from_mt5(item) for item in raw)

    def get_fills(
        self,
        since: datetime,
        until: datetime | None = None,
        symbol: str | None = None,
    ) -> tuple[ForexFill, ...]:
        """List executions in a time range, oldest first."""
        terminal = self._require_mt5()
        raw = terminal.history_deals_get(since, until or datetime.now(tz=UTC)) or ()
        fills = tuple(fill_from_mt5(item) for item in raw)
        if symbol is not None:
            fills = tuple(fill for fill in fills if fill.symbol == symbol)
        return fills

    # ------------------------------------------------------------------ helpers
    def _position_or_raise(self, ticket: int) -> ForexPosition:
        """Fetch one position by ticket.

        Raises:
            ForexOrderRejectedError: if the venue has no such open position.

        """
        raw = self._require_mt5().positions_get(ticket=ticket)
        if not raw:
            raise ForexOrderRejectedError(f"no open position with ticket {ticket}", ticket=ticket)
        return position_from_mt5(raw[0])

    def _send(self, payload: dict[str, Any]) -> OrderAck:
        """Post a trade request and normalise the terminal's answer."""
        result = self._require_mt5().order_send(payload)
        if result is None:
            code, message = self._last_error()
            raise ForexOrderRejectedError(
                f"MT5 rejected the request before routing: {message}", venue_code=code
            )
        retcode = int(_raw(result, "retcode"))
        accepted = retcode in {MT5_TRADE_RETCODE_DONE, MT5_TRADE_RETCODE_PLACED}
        filled = to_decimal(_raw(result, "volume", 0))
        price = to_decimal(_raw(result, "price", 0))
        return OrderAck(
            accepted=accepted,
            status=(
                ForexOrderStatus.FILLED
                if accepted and filled > ZERO
                else (ForexOrderStatus.PLACED if accepted else ForexOrderStatus.REJECTED)
            ),
            ticket=int(_raw(result, "order", 0)) or None,
            filled_lots=filled,
            average_price=price if price > ZERO else None,
            message=str(_raw(result, "comment", "")),
            venue_code=str(retcode),
        )


# --------------------------------------------------------------------------- #
# Standalone service
# --------------------------------------------------------------------------- #
class _WorkerState:
    """Holds the worker and the single thread every terminal call is funnelled through."""

    def __init__(self, worker: MT5Worker | None) -> None:
        self.worker = worker
        self.executor: ThreadPoolExecutor | None = None
        self.error: str = ""


def _require_worker(state: _WorkerState) -> MT5Worker:
    """Return the live worker, or a 503 explaining exactly what is missing.

    Raises:
        HTTPException: 503 carrying the capability blockers.

    """
    if state.worker is None:
        caps = capabilities()
        raise HTTPException(
            status_code=503,
            detail={
                "message": state.error or "MT5 worker is not available on this host",
                "blockers": list(caps.blockers),
                "platform": caps.platform,
            },
        )
    return state.worker


def to_jsonable(value: Any) -> Any:
    """Render dataclasses, Decimals, datetimes and enums as JSON-safe primitives.

    Decimals become strings, never floats — round-tripping a price through a float is
    exactly the corruption the rest of this package exists to avoid.
    """
    if isinstance(value, str):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return {name: to_jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return _scalar_to_jsonable(value)


def _scalar_to_jsonable(value: Any) -> Any:
    """Render a single non-container value."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def build_app(worker: MT5Worker | None = None) -> FastAPI:
    """Build the standalone HTTP boundary for this transport.

    Runs on the Windows host next to the terminal; QuantFlow core talks to it over local
    HTTP so nothing else in the system needs to be Windows-aware. ``/health`` and
    ``/capabilities`` always answer, even when the terminal is unreachable — that is what
    makes the blocker diagnosable from the other side of the boundary.
    """
    state = _WorkerState(worker)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        """Bring up the single MT5 thread, and the worker if this host can run one."""
        state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5")
        if state.worker is None:
            caps = capabilities()
            if caps.ready:
                try:
                    candidate = MT5Worker(MT5Credentials.from_env())
                    await _run(state, candidate.connect)
                    state.worker = candidate
                except (ForexCapabilityError, ForexConnectionError) as exc:
                    state.error = str(exc)
            else:
                state.error = caps.describe()
        try:
            yield
        finally:
            if state.worker is not None:
                state.worker.disconnect()
            state.executor.shutdown(wait=False)

    app = FastAPI(title="QuantFlow Forex worker (MT5)", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness, independent of the terminal."""
        return {"status": "ok", "worker_ready": state.worker is not None}

    @app.get("/capabilities")
    async def read_capabilities() -> dict[str, Any]:
        """Report platform, package and credential blockers."""
        caps = capabilities()
        return {**to_jsonable(caps), "ready": caps.ready, "error": state.error}

    @app.get("/account")
    async def read_account() -> Any:
        """Account balance, equity and margin."""
        worker = _require_worker(state)
        return to_jsonable(await _run(state, worker.get_account))

    @app.get("/symbols")
    async def read_symbols() -> Any:
        """Discovered instruments, majors first."""
        worker = _require_worker(state)
        return to_jsonable(await _run(state, worker.get_symbols))

    @app.get("/positions")
    async def read_positions() -> Any:
        """Open positions."""
        worker = _require_worker(state)
        return to_jsonable(await _run(state, worker.get_positions))

    @app.get("/orders")
    async def read_orders() -> Any:
        """Working orders."""
        worker = _require_worker(state)
        return to_jsonable(await _run(state, worker.get_orders))

    return app


async def _run(state: _WorkerState, function: Callable[..., Any], *args: Any) -> Any:
    """Run a terminal call on the single MT5 thread."""
    import asyncio

    if state.executor is None:  # pragma: no cover — lifespan always sets it
        raise HTTPException(status_code=503, detail={"message": "worker not started"})
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(state.executor, function, *args)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: report capabilities, or serve the worker.

    Returns a non-zero exit code from ``--check`` when the host cannot run the transport,
    so a deployment script can gate on it.
    """
    parser = argparse.ArgumentParser(description="QuantFlow Forex worker (MetaTrader 5)")
    parser.add_argument(
        "--check", action="store_true", help="report why the worker can or cannot run, then exit"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)

    caps = capabilities()
    sys.stdout.write(caps.describe() + "\n")
    if args.check:
        return 0 if caps.ready else 1
    if not caps.ready:
        return 1

    import uvicorn

    uvicorn.run(build_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
