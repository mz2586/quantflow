"""Notification contract and event types.

A notifier is a best-effort side channel. Delivery failures are **logged and swallowed**,
never raised: an unreachable Telegram API must not take down the trading loop, and an
exception propagating out of a "tell the operator about this fill" call would do exactly
that at the worst possible moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from quantflow.core.config import Severity
from quantflow.core.errors import ValidationError


@dataclass(frozen=True, slots=True)
class Notification:
    """One message bound for an operator."""

    title: str
    body: str
    severity: Severity = Severity.INFO
    event_type: str = "generic"
    timestamp: datetime | None = None
    fields: dict[str, str] = field(default_factory=dict)
    url: str | None = None

    def __post_init__(self) -> None:
        """Validate the message."""
        if not self.title.strip():
            raise ValidationError("notification requires a title")

    @property
    def is_urgent(self) -> bool:
        """Whether this warrants interrupting someone."""
        return self.severity is Severity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        """Serialise for logging and the event bus."""
        return {
            "title": self.title,
            "body": self.body,
            "severity": self.severity.value,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "fields": self.fields,
        }


@runtime_checkable
class Notifier(Protocol):
    """A transport that can deliver a notification."""

    @property
    def name(self) -> str:
        """Transport identifier, used in logs."""
        ...

    @property
    def enabled(self) -> bool:
        """Whether the transport is configured and usable."""
        ...

    async def send(self, notification: Notification) -> bool:
        """Deliver a notification.

        Returns:
            ``True`` on success. Implementations must **not** raise on a delivery failure;
            an unreachable notification service cannot be allowed to break trading.

        """
        ...

    async def aclose(self) -> None:
        """Release any transport resources."""
        ...


class NullNotifier:
    """A notifier that discards everything.

    The default. It means every call site can notify unconditionally without checking
    whether notifications are configured, which removes a whole category of
    `if notifier is not None` noise from the trading path.
    """

    __slots__ = ("sent",)

    def __init__(self) -> None:
        self.sent: list[Notification] = []

    @property
    def name(self) -> str:
        """Transport identifier."""
        return "null"

    @property
    def enabled(self) -> bool:
        """Always false: nothing is delivered."""
        return False

    async def send(self, notification: Notification) -> bool:
        """Record and discard."""
        self.sent.append(notification)
        return True

    async def aclose(self) -> None:
        """Nothing to release."""


# --------------------------------------------------------------------------- #
# Event builders
#
# Centralised so the wording, severity and field set of each event type are consistent
# wherever they are raised — an operator should never have to work out whether two
# differently-phrased alerts mean the same thing.
# --------------------------------------------------------------------------- #
def fill_notification(
    *,
    symbol: str,
    side: str,
    quantity: Decimal,
    price: Decimal,
    strategy_id: str | None,
    timestamp: datetime,
    fee: Decimal | None = None,
) -> Notification:
    """An execution."""
    fields = {
        "Symbol": symbol,
        "Side": side.upper(),
        "Quantity": f"{quantity:,.8f}".rstrip("0").rstrip("."),
        "Price": f"{price:,.2f}",
        "Notional": f"{quantity * price:,.2f}",
    }
    if fee is not None:
        fields["Fee"] = f"{fee:,.4f}"
    if strategy_id:
        fields["Strategy"] = strategy_id

    return Notification(
        title=f"Filled {side.upper()} {symbol}",
        body=f"{quantity} {symbol} at {price}",
        severity=Severity.INFO,
        event_type="fill",
        timestamp=timestamp,
        fields=fields,
    )


def risk_notification(
    *,
    rule: str,
    message: str,
    symbol: str | None = None,
    observed: Decimal | None = None,
    limit: Decimal | None = None,
    halted: bool = False,
) -> Notification:
    """A risk rule that blocked an order or halted trading."""
    fields = {"Rule": rule}
    if symbol:
        fields["Symbol"] = symbol
    if observed is not None:
        fields["Observed"] = f"{observed}"
    if limit is not None:
        fields["Limit"] = f"{limit}"
    if halted:
        fields["Action"] = "TRADING HALTED"

    return Notification(
        title="Trading halted" if halted else f"Risk limit: {rule}",
        body=message,
        severity=Severity.CRITICAL if halted else Severity.WARNING,
        event_type="risk",
        fields=fields,
    )


def kill_switch_notification(*, engaged: bool, reason: str | None, actor: str) -> Notification:
    """The kill switch changing state.

    Always CRITICAL in both directions. Someone turning trading back *on* is as
    important to know about as someone turning it off.
    """
    if engaged:
        return Notification(
            title="KILL SWITCH ENGAGED",
            body=reason or "trading halted",
            severity=Severity.CRITICAL,
            event_type="kill_switch",
            fields={"Actor": actor, "State": "ENGAGED"},
        )
    return Notification(
        title="Kill switch cleared",
        body="trading may resume",
        severity=Severity.CRITICAL,
        event_type="kill_switch",
        fields={"Actor": actor, "State": "CLEARED"},
    )


def daily_digest(
    *,
    equity: Decimal,
    daily_pnl: Decimal,
    daily_pnl_pct: Decimal,
    trades: int,
    wins: int,
    open_positions: int,
    drawdown_pct: Decimal,
) -> Notification:
    """The end-of-day summary."""
    direction = "up" if daily_pnl >= 0 else "down"
    win_rate = (wins / trades) if trades else 0.0
    return Notification(
        title=f"Daily summary — {direction} {abs(daily_pnl_pct):.2%}",
        body=f"Equity {equity:,.2f}, {trades} trades closed",
        severity=Severity.INFO,
        event_type="digest",
        fields={
            "Equity": f"{equity:,.2f}",
            "Daily PnL": f"{daily_pnl:+,.2f} ({daily_pnl_pct:+.2%})",
            "Trades": str(trades),
            "Win rate": f"{win_rate:.1%}" if trades else "n/a",
            "Open positions": str(open_positions),
            "Drawdown": f"{drawdown_pct:.2%}",
        },
    )


def error_notification(*, component: str, message: str) -> Notification:
    """An unrecoverable component failure."""
    return Notification(
        title=f"{component} failed",
        body=message,
        severity=Severity.CRITICAL,
        event_type="error",
        fields={"Component": component},
    )


def session_notification(
    *, session_id: str, mode: str, strategy_id: str, started: bool
) -> Notification:
    """A trading session starting or stopping."""
    action = "started" if started else "stopped"
    return Notification(
        title=f"{mode.title()} session {action}",
        body=f"{strategy_id} ({session_id[:8]})",
        severity=Severity.INFO,
        event_type="session",
        fields={"Mode": mode, "Strategy": strategy_id, "Session": session_id[:8]},
    )
