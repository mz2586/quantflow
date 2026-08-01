"""Notifications: the notifier protocol, Telegram transport and dispatcher."""

from __future__ import annotations

from quantflow.notifications.base import (
    Notification,
    Notifier,
    NullNotifier,
    daily_digest,
    fill_notification,
    kill_switch_notification,
    risk_notification,
)
from quantflow.notifications.dispatcher import (
    DispatchStats,
    NotificationDispatcher,
    build_dispatcher,
)
from quantflow.notifications.telegram import TelegramNotifier, build_notifier

__all__ = [
    "DispatchStats",
    "Notification",
    "NotificationDispatcher",
    "Notifier",
    "NullNotifier",
    "TelegramNotifier",
    "build_dispatcher",
    "build_notifier",
    "daily_digest",
    "fill_notification",
    "kill_switch_notification",
    "risk_notification",
]
