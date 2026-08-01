"""Telegram transport.

Written directly against the Bot API over ``httpx`` rather than through a bot framework.
We need exactly one endpoint (``sendMessage``), and a framework would pull in a polling
loop, an update dispatcher and a scheduler that this system has no use for.
"""

from __future__ import annotations

import asyncio
import html
from typing import Any, Final

import httpx

from quantflow.core.config import NotificationSettings, Severity
from quantflow.core.logging import get_logger

logger = get_logger(__name__)

API_BASE: Final = "https://api.telegram.org"

#: Telegram truncates anything longer. Better to send a clipped message than to have the
#: API reject the whole thing and deliver nothing.
MAX_MESSAGE_LENGTH: Final = 4096

#: Telegram allows ~30 messages/second globally and ~20/minute to one group. A burst of
#: fills could trip that and get the bot temporarily blocked, so sends are throttled.
MIN_SEND_INTERVAL_SECONDS: Final = 1.0

# Emoji rather than words: they survive Telegram's HTML parser and are legible at a
# glance on a phone lock screen, which is where these are actually read.
SEVERITY_EMOJI: Final[dict[Severity, str]] = {
    Severity.DEBUG: "🔍",
    Severity.INFO: "ℹ️",  # noqa: RUF001 - deliberate emoji, not a Latin "i"
    Severity.WARNING: "⚠️",
    Severity.CRITICAL: "🚨",
}


class TelegramNotifier:
    """Delivers notifications to a Telegram chat."""

    __slots__ = ("_client", "_last_sent", "_lock", "_settings")

    def __init__(
        self, settings: NotificationSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.telegram_timeout_seconds)
        )
        self._lock = asyncio.Lock()
        self._last_sent = 0.0

    @property
    def name(self) -> str:
        """Transport identifier."""
        return "telegram"

    @property
    def enabled(self) -> bool:
        """Whether a bot token and chat id are configured."""
        return (
            self._settings.telegram_enabled
            and self._settings.telegram_bot_token is not None
            and self._settings.telegram_chat_id is not None
        )

    async def send(self, notification: Any) -> bool:
        """Deliver a notification.

        Returns ``False`` on any failure rather than raising: an unreachable Telegram API
        must never interrupt trading.
        """
        if not self.enabled:
            return False
        if notification.severity.rank < self._settings.min_severity.rank:
            return False

        token = self._settings.telegram_bot_token
        chat_id = self._settings.telegram_chat_id
        assert token is not None
        assert chat_id is not None

        text = render(notification)
        await self._throttle()

        try:
            response = await self._client.post(
                f"{API_BASE}/bot{token.get_secret_value()}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                    # Routine events arrive silently; only a CRITICAL alert buzzes a phone,
                    # so notification fatigue does not train the operator to ignore them.
                    "disable_notification": not notification.is_urgent,
                },
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "telegram.send_failed", error=str(exc), event_type=notification.event_type
            )
            return False

        if response.status_code != httpx.codes.OK:
            logger.warning(
                "telegram.rejected",
                status=response.status_code,
                body=response.text[:200],
                event_type=notification.event_type,
            )
            return False

        logger.debug("telegram.sent", event_type=notification.event_type)
        return True

    async def _throttle(self) -> None:
        """Space out sends to stay inside Telegram's rate limits."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            elapsed = loop.time() - self._last_sent
            if elapsed < MIN_SEND_INTERVAL_SECONDS:
                await asyncio.sleep(MIN_SEND_INTERVAL_SECONDS - elapsed)
            self._last_sent = loop.time()

    async def verify(self) -> bool:
        """Check the bot token by calling ``getMe``.

        Called at startup so a misconfigured token surfaces immediately, rather than at
        the moment an alert actually needed to go out.
        """
        if not self.enabled:
            return False
        token = self._settings.telegram_bot_token
        assert token is not None
        try:
            response = await self._client.get(f"{API_BASE}/bot{token.get_secret_value()}/getMe")
        except httpx.HTTPError as exc:
            logger.warning("telegram.verify_failed", error=str(exc))
            return False
        if response.status_code != httpx.codes.OK:
            logger.warning("telegram.verify_rejected", status=response.status_code)
            return False
        logger.info("telegram.verified")
        return True

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def render(notification: Any) -> str:
    """Render a notification as Telegram-flavoured HTML.

    Every interpolated value is HTML-escaped: a strategy name or rejection reason can
    contain ``<`` or ``&``, and an unescaped one makes Telegram reject the entire message
    — meaning the alert is silently never delivered.
    """
    emoji = SEVERITY_EMOJI.get(notification.severity, "")
    lines = [f"{emoji} <b>{html.escape(notification.title)}</b>"]

    if notification.body:
        lines.append(html.escape(notification.body))

    if notification.fields:
        lines.append("")
        for key, value in notification.fields.items():
            lines.append(f"<b>{html.escape(key)}:</b> <code>{html.escape(str(value))}</code>")

    if notification.timestamp:
        lines.append("")
        lines.append(f"<i>{notification.timestamp:%Y-%m-%d %H:%M:%S UTC}</i>")

    text = "\n".join(lines)
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[: MAX_MESSAGE_LENGTH - 20] + "\n<i>… truncated</i>"
    return text


def build_notifier(settings: NotificationSettings) -> Any:
    """Construct the configured notifier, or a null one.

    Returning a working object rather than ``None`` means call sites never need to guard
    their notification calls.
    """
    from quantflow.notifications.base import NullNotifier

    if settings.telegram_enabled:
        return TelegramNotifier(settings)
    return NullNotifier()
