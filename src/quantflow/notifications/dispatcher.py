"""Notification dispatcher.

Fans one event out to every configured transport, applies severity filtering, and
suppresses duplicates.

De-duplication is the point of this class. A risk rule that blocks an order will block the
*next* one too, and the one after that — without suppression an operator receives the same
alert hundreds of times, learns to ignore the channel, and misses the one that mattered.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from quantflow.core.clock import Clock, SystemClock
from quantflow.core.config import NotificationSettings, Severity
from quantflow.core.logging import get_logger
from quantflow.notifications.base import (
    Notification,
    Notifier,
    daily_digest,
    error_notification,
    fill_notification,
    kill_switch_notification,
    risk_notification,
    session_notification,
)

logger = get_logger(__name__)

#: How long an identical alert is suppressed after being sent.
DEFAULT_DEDUPE_WINDOW = timedelta(minutes=15)

#: Ceiling on outbound messages per minute across all transports, independent of any one
#: transport's own limit. Protects the operator's attention, not just the API.
DEFAULT_RATE_LIMIT = 20


@dataclass(slots=True)
class DispatchStats:
    """Counters for observability."""

    sent: int = 0
    suppressed_duplicate: int = 0
    suppressed_severity: int = 0
    suppressed_rate_limit: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialise for the API."""
        return {
            "sent": self.sent,
            "suppressed_duplicate": self.suppressed_duplicate,
            "suppressed_severity": self.suppressed_severity,
            "suppressed_rate_limit": self.suppressed_rate_limit,
            "failed": self.failed,
        }


@dataclass(slots=True)
class NotificationDispatcher:
    """Routes notifications to transports with filtering and de-duplication."""

    notifiers: Sequence[Notifier]
    settings: NotificationSettings = field(default_factory=NotificationSettings)
    clock: Clock = field(default_factory=SystemClock)
    dedupe_window: timedelta = DEFAULT_DEDUPE_WINDOW

    _recent: dict[str, datetime] = field(default_factory=dict, init=False)
    _sent_times: list[datetime] = field(default_factory=list, init=False)
    stats: DispatchStats = field(default_factory=DispatchStats, init=False)

    @property
    def enabled_notifiers(self) -> tuple[Notifier, ...]:
        """Every transport that is actually configured."""
        return tuple(notifier for notifier in self.notifiers if notifier.enabled)

    async def dispatch(self, notification: Notification) -> int:
        """Send a notification to every enabled transport.

        Returns:
            The number of transports that accepted it.

        """
        if notification.severity.rank < self.settings.min_severity.rank:
            self.stats.suppressed_severity += 1
            return 0

        now = self.clock.now()
        key = self._dedupe_key(notification)

        # A CRITICAL alert is never suppressed as a duplicate: repetition is noise for a
        # routine event and signal for an emergency.
        if not notification.is_urgent and self._is_duplicate(key, now):
            self.stats.suppressed_duplicate += 1
            logger.debug("notify.suppressed_duplicate", event_type=notification.event_type)
            return 0

        if not self._within_rate_limit(now):
            self.stats.suppressed_rate_limit += 1
            logger.warning(
                "notify.rate_limited",
                event_type=notification.event_type,
                limit_per_minute=self.settings.rate_limit_per_minute,
            )
            return 0

        targets = self.enabled_notifiers
        if not targets:
            return 0

        results = await asyncio.gather(
            *(self._send_safely(notifier, notification) for notifier in targets)
        )
        delivered = sum(1 for ok in results if ok)

        self._recent[key] = now
        self._sent_times.append(now)
        if delivered:
            self.stats.sent += 1
        else:
            self.stats.failed += 1
        return delivered

    async def _send_safely(self, notifier: Notifier, notification: Notification) -> bool:
        """Send through one transport, containing any failure.

        A transport that raises must not prevent the others from delivering, and must
        never propagate into the trading loop.
        """
        try:
            return await notifier.send(notification)
        except Exception as exc:
            logger.warning("notify.transport_failed", transport=notifier.name, error=str(exc))
            return False

    def _dedupe_key(self, notification: Notification) -> str:
        """Identity used for duplicate suppression.

        Title plus event type, deliberately excluding the body: two "max position limit"
        alerts differing only in the observed number are the same alert to a human.
        """
        return f"{notification.event_type}:{notification.title}"

    def _is_duplicate(self, key: str, now: datetime) -> bool:
        previous = self._recent.get(key)
        if previous is None:
            return False
        if now - previous >= self.dedupe_window:
            del self._recent[key]
            return False
        return True

    def _within_rate_limit(self, now: datetime) -> bool:
        cutoff = now - timedelta(minutes=1)
        self._sent_times = [stamp for stamp in self._sent_times if stamp > cutoff]
        return len(self._sent_times) < self.settings.rate_limit_per_minute

    def reset(self) -> None:
        """Clear suppression state. Intended for tests and for a session restart."""
        self._recent.clear()
        self._sent_times.clear()

    async def aclose(self) -> None:
        """Close every transport."""
        for notifier in self.notifiers:
            with contextlib.suppress(Exception):
                await notifier.aclose()

    # ------------------------------------------------------------------ #
    # Typed helpers
    # ------------------------------------------------------------------ #
    async def notify_fill(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        timestamp: datetime,
        strategy_id: str | None = None,
        fee: Decimal | None = None,
    ) -> int:
        """Announce an execution."""
        return await self.dispatch(
            fill_notification(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                strategy_id=strategy_id,
                timestamp=timestamp,
                fee=fee,
            )
        )

    async def notify_risk(
        self,
        *,
        rule: str,
        message: str,
        symbol: str | None = None,
        observed: Decimal | None = None,
        limit: Decimal | None = None,
        halted: bool = False,
    ) -> int:
        """Announce a risk decision."""
        return await self.dispatch(
            risk_notification(
                rule=rule,
                message=message,
                symbol=symbol,
                observed=observed,
                limit=limit,
                halted=halted,
            )
        )

    async def notify_kill_switch(self, *, engaged: bool, reason: str | None, actor: str) -> int:
        """Announce a kill-switch change."""
        return await self.dispatch(
            kill_switch_notification(engaged=engaged, reason=reason, actor=actor)
        )

    async def notify_digest(
        self,
        *,
        equity: Decimal,
        daily_pnl: Decimal,
        daily_pnl_pct: Decimal,
        trades: int,
        wins: int,
        open_positions: int,
        drawdown_pct: Decimal,
    ) -> int:
        """Send the end-of-day summary."""
        return await self.dispatch(
            daily_digest(
                equity=equity,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                trades=trades,
                wins=wins,
                open_positions=open_positions,
                drawdown_pct=drawdown_pct,
            )
        )

    async def notify_error(self, *, component: str, message: str) -> int:
        """Announce a component failure."""
        return await self.dispatch(error_notification(component=component, message=message))

    async def notify_session(
        self, *, session_id: str, mode: str, strategy_id: str, started: bool
    ) -> int:
        """Announce a session starting or stopping."""
        return await self.dispatch(
            session_notification(
                session_id=session_id,
                mode=mode,
                strategy_id=strategy_id,
                started=started,
            )
        )


def build_dispatcher(
    settings: NotificationSettings, *, clock: Clock | None = None
) -> NotificationDispatcher:
    """Construct a dispatcher with every configured transport attached."""
    from quantflow.notifications.telegram import build_notifier

    notifier = build_notifier(settings)
    return NotificationDispatcher(
        notifiers=[notifier], settings=settings, clock=clock or SystemClock()
    )


async def describe(dispatcher: NotificationDispatcher) -> dict[str, Any]:
    """Report which transports are live, for the API's status endpoint."""
    return {
        "transports": [
            {"name": notifier.name, "enabled": notifier.enabled}
            for notifier in dispatcher.notifiers
        ],
        "min_severity": dispatcher.settings.min_severity.value,
        "rate_limit_per_minute": dispatcher.settings.rate_limit_per_minute,
        "stats": dispatcher.stats.to_dict(),
    }


def severity_at_least(threshold: Severity) -> tuple[Severity, ...]:
    """Every severity at or above ``threshold``, for filtering UI."""
    return tuple(level for level in Severity if level.rank >= threshold.rank)
