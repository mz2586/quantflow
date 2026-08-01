"""Notification transports, rendering and dispatch policy.

The behaviour that matters most: a notification failure must never propagate into the
trading loop, and a repeating alert must not train the operator to ignore the channel.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import httpx
import pytest
import respx

from quantflow.core.clock import FrozenClock
from quantflow.core.config import NotificationSettings, Severity
from quantflow.core.errors import ValidationError
from quantflow.notifications.base import (
    Notification,
    NullNotifier,
    daily_digest,
    error_notification,
    fill_notification,
    kill_switch_notification,
    risk_notification,
    session_notification,
)
from quantflow.notifications.dispatcher import NotificationDispatcher, build_dispatcher
from quantflow.notifications.telegram import (
    MAX_MESSAGE_LENGTH,
    TelegramNotifier,
    build_notifier,
    render,
)
from tests.conftest import REFERENCE_TIME

TOKEN = "123456:test-token"
CHAT_ID = "-100999"


def telegram_settings(**overrides: object) -> NotificationSettings:
    kwargs: dict[str, object] = {
        "telegram_enabled": True,
        "telegram_bot_token": TOKEN,
        "telegram_chat_id": CHAT_ID,
    }
    kwargs.update(overrides)
    return NotificationSettings(**kwargs)  # type: ignore[arg-type]


class RecordingNotifier:
    """A transport that records what it was asked to send."""

    def __init__(self, *, enabled: bool = True, fail: bool = False, raises: bool = False) -> None:
        self._enabled = enabled
        self._fail = fail
        self._raises = raises
        self.received: list[Notification] = []
        self.closed = False

    @property
    def name(self) -> str:
        return "recording"

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def send(self, notification: Notification) -> bool:
        if self._raises:
            raise RuntimeError("transport exploded")
        self.received.append(notification)
        return not self._fail

    async def aclose(self) -> None:
        self.closed = True


class TestNotificationBuilders:
    def test_fill(self) -> None:
        notification = fill_notification(
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            strategy_id="ema_cross",
            timestamp=REFERENCE_TIME,
            fee=Decimal("25"),
        )
        assert notification.event_type == "fill"
        assert notification.severity is Severity.INFO
        assert notification.fields["Notional"] == "25,000.00"
        assert notification.fields["Strategy"] == "ema_cross"
        assert not notification.is_urgent

    def test_risk_halt_is_critical(self) -> None:
        notification = risk_notification(
            rule="max_drawdown",
            message="drawdown 18% exceeds 15%",
            observed=Decimal("0.18"),
            limit=Decimal("0.15"),
            halted=True,
        )
        assert notification.severity is Severity.CRITICAL
        assert notification.is_urgent
        assert notification.fields["Action"] == "TRADING HALTED"

    def test_risk_block_is_only_a_warning(self) -> None:
        notification = risk_notification(rule="max_position_pct", message="too big")
        assert notification.severity is Severity.WARNING

    def test_kill_switch_is_critical_in_both_directions(self) -> None:
        # Someone turning trading back ON matters as much as turning it off.
        engaged = kill_switch_notification(engaged=True, reason="drawdown", actor="risk")
        cleared = kill_switch_notification(engaged=False, reason=None, actor="operator")
        assert engaged.severity is Severity.CRITICAL
        assert cleared.severity is Severity.CRITICAL
        assert "ENGAGED" in engaged.title

    def test_digest(self) -> None:
        notification = daily_digest(
            equity=Decimal("10500"),
            daily_pnl=Decimal("500"),
            daily_pnl_pct=Decimal("0.05"),
            trades=10,
            wins=6,
            open_positions=2,
            drawdown_pct=Decimal("0.02"),
        )
        assert "up" in notification.title
        assert notification.fields["Win rate"] == "60.0%"

    def test_digest_without_trades(self) -> None:
        notification = daily_digest(
            equity=Decimal("10000"),
            daily_pnl=Decimal("0"),
            daily_pnl_pct=Decimal("0"),
            trades=0,
            wins=0,
            open_positions=0,
            drawdown_pct=Decimal("0"),
        )
        assert notification.fields["Win rate"] == "n/a"

    def test_error_and_session(self) -> None:
        assert error_notification(component="worker", message="died").is_urgent
        started = session_notification(
            session_id="abcdef123456", mode="paper", strategy_id="ema_cross", started=True
        )
        assert "started" in started.title

    def test_empty_title_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires a title"):
            Notification(title="  ", body="x")


class TestRendering:
    def test_includes_title_body_and_fields(self) -> None:
        text = render(
            Notification(
                title="Filled BUY BTC/USDT",
                body="0.5 at 50000",
                fields={"Symbol": "BTC/USDT"},
                timestamp=REFERENCE_TIME,
            )
        )
        assert "<b>Filled BUY BTC/USDT</b>" in text
        assert "0.5 at 50000" in text
        assert "<code>BTC/USDT</code>" in text
        assert "2026-01-01" in text

    def test_html_is_escaped(self) -> None:
        # An unescaped `<` makes Telegram reject the whole message, so the alert is
        # silently never delivered — the worst possible failure mode for an alert.
        text = render(
            Notification(
                title="<script>alert(1)</script>",
                body="a < b & c",
                fields={"Reason": "<injected>"},
            )
        )
        assert "<script>" not in text
        assert "&lt;script&gt;" in text
        assert "&lt;injected&gt;" in text

    def test_long_messages_are_truncated(self) -> None:
        text = render(Notification(title="x", body="y" * 10_000))
        assert len(text) <= MAX_MESSAGE_LENGTH
        assert "truncated" in text

    def test_severity_emoji(self) -> None:
        critical = render(Notification(title="t", body="", severity=Severity.CRITICAL))
        assert critical.startswith("🚨")


class TestTelegramTransport:
    def test_disabled_without_configuration(self) -> None:
        assert not TelegramNotifier(NotificationSettings()).enabled

    def test_enabled_when_configured(self) -> None:
        assert TelegramNotifier(telegram_settings()).enabled

    @respx.mock
    async def test_sends_a_message(self) -> None:
        route = respx.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )

        notifier = TelegramNotifier(telegram_settings())
        try:
            sent = await notifier.send(
                Notification(title="Test", body="hello", severity=Severity.WARNING)
            )
        finally:
            await notifier.aclose()

        assert sent
        assert route.called
        payload = route.calls[0].request.content.decode()
        assert "Test" in payload
        assert CHAT_ID in payload

    @respx.mock
    async def test_only_critical_alerts_buzz_the_phone(self) -> None:
        # Notification fatigue is what makes an operator stop reading the channel.
        route = respx.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        notifier = TelegramNotifier(telegram_settings())
        try:
            await notifier.send(Notification(title="routine", body="", severity=Severity.INFO))
            await notifier.send(Notification(title="urgent", body="", severity=Severity.CRITICAL))
        finally:
            await notifier.aclose()

        import json

        first = json.loads(route.calls[0].request.content)
        second = json.loads(route.calls[1].request.content)
        assert first["disable_notification"] is True
        assert second["disable_notification"] is False

    @respx.mock
    async def test_network_failure_returns_false_without_raising(self) -> None:
        # The property the trading loop depends on.
        respx.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage").mock(
            side_effect=httpx.ConnectError("network down")
        )
        notifier = TelegramNotifier(telegram_settings())
        try:
            assert await notifier.send(Notification(title="t", body="")) is False
        finally:
            await notifier.aclose()

    @respx.mock
    async def test_api_rejection_returns_false(self) -> None:
        respx.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage").mock(
            return_value=httpx.Response(400, json={"ok": False, "description": "bad chat"})
        )
        notifier = TelegramNotifier(telegram_settings())
        try:
            assert await notifier.send(Notification(title="t", body="")) is False
        finally:
            await notifier.aclose()

    @respx.mock
    async def test_severity_below_the_threshold_is_dropped(self) -> None:
        route = respx.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        notifier = TelegramNotifier(telegram_settings(min_severity=Severity.CRITICAL))
        try:
            assert (
                await notifier.send(Notification(title="t", body="", severity=Severity.INFO))
                is False
            )
        finally:
            await notifier.aclose()
        assert not route.called

    @respx.mock
    async def test_verify_checks_the_token(self) -> None:
        respx.get(f"https://api.telegram.org/bot{TOKEN}/getMe").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        notifier = TelegramNotifier(telegram_settings())
        try:
            assert await notifier.verify() is True
        finally:
            await notifier.aclose()

    @respx.mock
    async def test_verify_reports_a_bad_token(self) -> None:
        respx.get(f"https://api.telegram.org/bot{TOKEN}/getMe").mock(
            return_value=httpx.Response(401, json={"ok": False})
        )
        notifier = TelegramNotifier(telegram_settings())
        try:
            assert await notifier.verify() is False
        finally:
            await notifier.aclose()

    def test_build_notifier_falls_back_to_null(self) -> None:
        # Call sites can notify unconditionally, with no None checks.
        assert isinstance(build_notifier(NotificationSettings()), NullNotifier)
        assert isinstance(build_notifier(telegram_settings()), TelegramNotifier)


class TestDispatcher:
    def _dispatcher(
        self, notifier: RecordingNotifier, clock: FrozenClock, **overrides: object
    ) -> NotificationDispatcher:
        return NotificationDispatcher(
            notifiers=[notifier],
            settings=NotificationSettings(**overrides),  # type: ignore[arg-type]
            clock=clock,
        )

    async def test_dispatches_to_enabled_transports(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock)
        assert await dispatcher.dispatch(Notification(title="t", body="b")) == 1
        assert len(notifier.received) == 1

    async def test_disabled_transports_are_skipped(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier(enabled=False)
        dispatcher = self._dispatcher(notifier, clock)
        assert await dispatcher.dispatch(Notification(title="t", body="b")) == 0
        assert notifier.received == []

    async def test_a_raising_transport_is_contained(self, clock: FrozenClock) -> None:
        # An exception here would propagate straight into the trading loop.
        notifier = RecordingNotifier(raises=True)
        dispatcher = self._dispatcher(notifier, clock)
        assert await dispatcher.dispatch(Notification(title="t", body="b")) == 0
        assert dispatcher.stats.failed == 1

    async def test_duplicates_are_suppressed(self, clock: FrozenClock) -> None:
        # A blocking risk rule fires on every subsequent order; without suppression the
        # operator gets hundreds of identical alerts and stops reading the channel.
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock)
        alert = risk_notification(rule="max_position_pct", message="too big")

        assert await dispatcher.dispatch(alert) == 1
        assert await dispatcher.dispatch(alert) == 0
        assert await dispatcher.dispatch(alert) == 0
        assert len(notifier.received) == 1
        assert dispatcher.stats.suppressed_duplicate == 2

    async def test_suppression_expires(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock)
        dispatcher.dedupe_window = timedelta(minutes=5)
        alert = risk_notification(rule="max_position_pct", message="too big")

        await dispatcher.dispatch(alert)
        clock.advance(delta=timedelta(minutes=6))
        assert await dispatcher.dispatch(alert) == 1
        assert len(notifier.received) == 2

    async def test_critical_alerts_are_never_deduplicated(self, clock: FrozenClock) -> None:
        # Repetition is noise for a routine event and signal for an emergency.
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock)
        alert = kill_switch_notification(engaged=True, reason="drawdown", actor="risk")

        await dispatcher.dispatch(alert)
        await dispatcher.dispatch(alert)
        assert len(notifier.received) == 2

    async def test_severity_threshold(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock, min_severity=Severity.WARNING)

        assert (
            await dispatcher.dispatch(
                Notification(title="chatter", body="", severity=Severity.INFO)
            )
            == 0
        )
        assert (
            await dispatcher.dispatch(
                Notification(title="problem", body="", severity=Severity.WARNING)
            )
            == 1
        )
        assert dispatcher.stats.suppressed_severity == 1

    async def test_rate_limit(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock, rate_limit_per_minute=3)

        for index in range(6):
            await dispatcher.dispatch(Notification(title=f"alert {index}", body=""))

        assert len(notifier.received) == 3
        assert dispatcher.stats.suppressed_rate_limit == 3

    async def test_rate_limit_window_slides(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock, rate_limit_per_minute=2)

        await dispatcher.dispatch(Notification(title="a", body=""))
        await dispatcher.dispatch(Notification(title="b", body=""))
        await dispatcher.dispatch(Notification(title="c", body=""))
        assert len(notifier.received) == 2

        clock.advance(seconds=61)
        await dispatcher.dispatch(Notification(title="d", body=""))
        assert len(notifier.received) == 3

    async def test_typed_helpers(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock)

        await dispatcher.notify_fill(
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("50000"),
            timestamp=REFERENCE_TIME,
        )
        await dispatcher.notify_kill_switch(engaged=True, reason="test", actor="ops")
        await dispatcher.notify_error(component="worker", message="died")

        kinds = {notification.event_type for notification in notifier.received}
        assert kinds == {"fill", "kill_switch", "error"}

    async def test_close_closes_every_transport(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock)
        await dispatcher.aclose()
        assert notifier.closed

    async def test_reset_clears_suppression(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier()
        dispatcher = self._dispatcher(notifier, clock)
        alert = risk_notification(rule="r", message="m")
        await dispatcher.dispatch(alert)
        dispatcher.reset()
        assert await dispatcher.dispatch(alert) == 1

    def test_build_dispatcher_defaults_to_null(self) -> None:
        dispatcher = build_dispatcher(NotificationSettings())
        assert dispatcher.enabled_notifiers == ()


class TestStructlogCompatibility:
    """`event` is structlog's reserved key for the message itself.

    Passing it as a field raises TypeError at call time, which for a notification path
    means the alert is lost *and* the caller sees an unexpected exception.
    """

    async def test_logging_a_failed_send_does_not_raise(self, clock: FrozenClock) -> None:
        notifier = RecordingNotifier(raises=True)
        dispatcher = NotificationDispatcher(
            notifiers=[notifier], settings=NotificationSettings(), clock=clock
        )
        # Would raise TypeError if the logger were passed a reserved `event` kwarg.
        await dispatcher.dispatch(Notification(title="t", body="b"))

    @respx.mock
    async def test_telegram_failure_logging_does_not_raise(self) -> None:
        respx.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage").mock(
            side_effect=httpx.ConnectError("down")
        )
        notifier = TelegramNotifier(telegram_settings())
        try:
            assert await notifier.send(Notification(title="t", body="")) is False
        finally:
            await notifier.aclose()
