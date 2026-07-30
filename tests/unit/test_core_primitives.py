"""Clock, precision and logging primitives."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import structlog

from quantflow.core.clock import (
    EPOCH,
    FrozenClock,
    SystemClock,
    floor_to_interval,
    from_epoch_ms,
    start_of_utc_day,
    to_epoch_ms,
)
from quantflow.core.config import Environment, Settings
from quantflow.core.errors import ValidationError
from quantflow.core.logging import (
    REDACTED,
    configure_logging,
    get_logger,
    log_context,
)
from quantflow.core.precision import (
    ZERO,
    clamp,
    decimal_places,
    normalize,
    pct_change,
    quantize_down,
    quantize_nearest,
    quantize_up,
    round_price,
    round_quantity,
    safe_divide,
    step_from_precision,
    to_decimal,
)


class TestClock:
    def test_system_clock_is_utc_aware(self) -> None:
        now = SystemClock().now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    def test_frozen_clock_does_not_drift(self) -> None:
        clock = FrozenClock(datetime(2026, 6, 1, tzinfo=UTC))
        assert clock.now() == clock.now()

    def test_frozen_clock_advance(self) -> None:
        clock = FrozenClock(datetime(2026, 6, 1, tzinfo=UTC))
        clock.advance(seconds=90)
        assert clock.now() == datetime(2026, 6, 1, 0, 1, 30, tzinfo=UTC)
        assert clock.monotonic() == pytest.approx(90.0)

    def test_frozen_clock_rejects_backwards_motion(self) -> None:
        clock = FrozenClock(datetime(2026, 6, 1, tzinfo=UTC))
        with pytest.raises(ValueError, match="backwards"):
            clock.advance(delta=timedelta(seconds=-1))
        with pytest.raises(ValueError, match="backwards"):
            clock.set(datetime(2026, 5, 1, tzinfo=UTC))

    def test_frozen_clock_rejects_naive_start(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            FrozenClock(datetime(2026, 6, 1))  # noqa: DTZ001

    async def test_frozen_clock_sleep_advances_virtual_time(self) -> None:
        clock = FrozenClock(datetime(2026, 6, 1, tzinfo=UTC))
        await clock.sleep(3600)
        assert clock.now() == datetime(2026, 6, 1, 1, 0, tzinfo=UTC)

    def test_epoch_round_trip(self) -> None:
        moment = datetime(2026, 3, 14, 15, 9, 26, tzinfo=UTC)
        assert from_epoch_ms(to_epoch_ms(moment)) == moment

    def test_epoch_ms_of_epoch_is_zero(self) -> None:
        assert to_epoch_ms(EPOCH) == 0

    def test_to_epoch_ms_rejects_naive(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            to_epoch_ms(datetime(2026, 1, 1))  # noqa: DTZ001

    @pytest.mark.parametrize(
        ("moment", "interval", "expected"),
        [
            (
                datetime(2026, 1, 1, 12, 37, tzinfo=UTC),
                timedelta(hours=1),
                datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            ),
            (
                datetime(2026, 1, 1, 12, 37, tzinfo=UTC),
                timedelta(minutes=15),
                datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
            ),
            (
                datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                timedelta(hours=4),
                datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            ),
        ],
    )
    def test_floor_to_interval(
        self, moment: datetime, interval: timedelta, expected: datetime
    ) -> None:
        assert floor_to_interval(moment, interval) == expected

    def test_floor_rejects_non_positive_interval(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            floor_to_interval(datetime(2026, 1, 1, tzinfo=UTC), timedelta(0))

    def test_start_of_utc_day(self) -> None:
        assert start_of_utc_day(datetime(2026, 5, 5, 23, 59, 59, tzinfo=UTC)) == datetime(
            2026, 5, 5, tzinfo=UTC
        )


class TestToDecimal:
    def test_float_avoids_binary_expansion(self) -> None:
        assert to_decimal(0.1) == Decimal("0.1")
        assert str(to_decimal(1.1)) == "1.1"

    def test_accepts_str_int_decimal(self) -> None:
        assert to_decimal("2.5") == Decimal("2.5")
        assert to_decimal(3) == Decimal("3")
        assert to_decimal(Decimal("4.25")) == Decimal("4.25")

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_rejects_non_finite(self, value: str) -> None:
        with pytest.raises(ValidationError, match="non-finite"):
            to_decimal(value)

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValidationError, match="cannot convert"):
            to_decimal("not-a-number")


class TestQuantisation:
    @pytest.mark.parametrize(
        ("value", "step", "expected"),
        [
            ("1.23456789", "0.001", "1.234"),
            ("0.999", "1", "0"),
            ("100", "10", "100"),
            ("104", "10", "100"),
            ("1.5", "0.5", "1.5"),
        ],
    )
    def test_quantize_down(self, value: str, step: str, expected: str) -> None:
        assert quantize_down(Decimal(value), Decimal(step)) == Decimal(expected)

    @pytest.mark.parametrize(
        ("value", "step", "expected"),
        [("1.2341", "0.001", "1.235"), ("100", "10", "100"), ("101", "10", "110")],
    )
    def test_quantize_up(self, value: str, step: str, expected: str) -> None:
        assert quantize_up(Decimal(value), Decimal(step)) == Decimal(expected)

    def test_quantize_nearest_ties_to_even(self) -> None:
        assert quantize_nearest(Decimal("0.5"), Decimal("1")) == Decimal("0")
        assert quantize_nearest(Decimal("1.5"), Decimal("1")) == Decimal("2")

    def test_quantize_rejects_non_positive_step(self) -> None:
        with pytest.raises(ValidationError, match="step must be positive"):
            quantize_down(Decimal("1"), ZERO)

    def test_round_price_is_conservative_per_side(self) -> None:
        tick = Decimal("0.01")
        assert round_price(Decimal("100.567"), tick, side_is_buy=True) == Decimal("100.56")
        assert round_price(Decimal("100.561"), tick, side_is_buy=False) == Decimal("100.57")

    def test_round_quantity_never_rounds_up(self) -> None:
        result = round_quantity(Decimal("0.123456789"), Decimal("0.00001"))
        assert result == Decimal("0.12345")
        assert result <= Decimal("0.123456789")

    @pytest.mark.parametrize(
        ("step", "places"),
        [("0.001", 3), ("0.01", 2), ("1", 0), ("10", 0), ("0.00000001", 8)],
    )
    def test_decimal_places(self, step: str, places: int) -> None:
        assert decimal_places(Decimal(step)) == places

    def test_step_from_precision_round_trip(self) -> None:
        for precision in range(0, 9):
            assert decimal_places(step_from_precision(precision)) == precision

    def test_step_from_precision_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            step_from_precision(-1)


class TestDecimalHelpers:
    def test_safe_divide_by_zero_returns_default(self) -> None:
        assert safe_divide(Decimal("1"), ZERO) == ZERO
        assert safe_divide(Decimal("1"), ZERO, default=Decimal("9")) == Decimal("9")

    def test_pct_change(self) -> None:
        assert pct_change(Decimal("100"), Decimal("110")) == Decimal("0.1")
        assert pct_change(Decimal("100"), Decimal("90")) == Decimal("-0.1")
        assert pct_change(ZERO, Decimal("5")) == ZERO

    def test_pct_change_uses_absolute_base(self) -> None:
        assert pct_change(Decimal("-100"), Decimal("-90")) == Decimal("0.1")

    def test_clamp(self) -> None:
        assert clamp(Decimal("5"), Decimal("1"), Decimal("3")) == Decimal("3")
        assert clamp(Decimal("0"), Decimal("1"), Decimal("3")) == Decimal("1")
        assert clamp(Decimal("2"), Decimal("1"), Decimal("3")) == Decimal("2")

    def test_clamp_rejects_empty_range(self) -> None:
        with pytest.raises(ValidationError, match="empty clamp range"):
            clamp(Decimal("1"), Decimal("3"), Decimal("1"))

    @pytest.mark.parametrize(
        ("value", "expected"), [("1.500", "1.5"), ("1E+2", "100"), ("3.000", "3")]
    )
    def test_normalize(self, value: str, expected: str) -> None:
        assert str(normalize(Decimal(value))) == expected


class TestLogging:
    """`configure_logging` owns the root handler, so these assert on the rendered
    stderr stream rather than on `caplog` (whose handler is intentionally replaced)."""

    @pytest.mark.parametrize(
        "key", ["api_key", "password", "secret_key", "authorization", "Bot-Token", "signature"]
    )
    def test_sensitive_keys_are_redacted(
        self, settings: Settings, capsys: pytest.CaptureFixture[str], key: str
    ) -> None:
        configure_logging(settings)
        get_logger("test.redaction").info("connecting", **{key: "SUPER-SECRET-VALUE"})
        captured = capsys.readouterr().err
        assert "SUPER-SECRET-VALUE" not in captured
        assert REDACTED in captured

    def test_non_sensitive_keys_survive(
        self, settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(settings)
        get_logger("test.plain").info("connecting", host="binance.com")
        assert "binance.com" in capsys.readouterr().err

    def test_nested_secrets_are_redacted(
        self, settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(settings)
        get_logger("test.nested").info("wired", config={"password": "leak-me", "host": "db"})
        assert "leak-me" not in capsys.readouterr().err

    def test_json_format_emits_parseable_lines(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        configure_logging(Settings(_env_file=None, env=Environment.TEST, log_format="json"))
        get_logger("test.json").warning("halted", reason="max_drawdown")
        payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert payload["event"] == "halted"
        assert payload["reason"] == "max_drawdown"
        assert payload["level"] == "warning"
        assert payload["service"] == "quantflow"

    def test_log_context_binds_and_unwinds(self, settings: Settings) -> None:
        configure_logging(settings)
        structlog.contextvars.clear_contextvars()
        with log_context(run_id="abc"):
            assert structlog.contextvars.get_contextvars()["run_id"] == "abc"
        assert "run_id" not in structlog.contextvars.get_contextvars()

    def test_configure_is_idempotent(self, settings: Settings) -> None:
        configure_logging(settings)
        before = len(logging.getLogger().handlers)
        configure_logging(settings)
        assert len(logging.getLogger().handlers) == before
