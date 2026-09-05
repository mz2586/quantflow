"""Live-trading arming gates.

Live trading must be **off by default** and must require every gate to pass. These tests
exist to make a regression here impossible to merge quietly: the failure mode is sending
real orders from a system nobody intended to arm.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.core.config import (
    LIVE_CONFIRMATION_TOKEN,
    DatabaseSettings,
    Environment,
    ExchangeSettings,
    Settings,
    TradingMode,
    TradingSettings,
)
from quantflow.core.errors import LiveTradingNotArmedError, ValidationError
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.live.runner import (
    LIVE_TRADING_ENV_VAR,
    LiveArmingCheck,
    RunnerConfig,
    TradingRunner,
    check_live_arming,
    describe_arming,
    live_trading_env_enabled,
)


def settings_for(
    *,
    mode: TradingMode = TradingMode.PAPER,
    armed_token: bool = False,
    testnet: bool = True,
    credentials: bool = False,
) -> Settings:
    trading = TradingSettings(
        mode=mode,
        live_confirmation=LIVE_CONFIRMATION_TOKEN if armed_token else None,  # type: ignore[arg-type]
    )
    exchange = ExchangeSettings(
        testnet=testnet,
        api_key="key" if credentials else None,  # type: ignore[arg-type]
        api_secret="secret" if credentials else None,  # type: ignore[arg-type]
    )
    return Settings(
        _env_file=None,
        env=Environment.TEST,
        trading=trading,
        exchange=exchange,
        database=DatabaseSettings(),
    )


def fully_armed_settings() -> Settings:
    return settings_for(mode=TradingMode.LIVE, armed_token=True, testnet=False, credentials=True)


class TestEnvironmentFlag:
    def test_absent_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        assert live_trading_env_enabled() is False

    def test_exactly_true_enables_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, "true")
        assert live_trading_env_enabled() is True

    @pytest.mark.parametrize("value", ["TRUE", "True", "  true  "])
    def test_case_and_whitespace_are_tolerated(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, value)
        assert live_trading_env_enabled() is True

    @pytest.mark.parametrize("value", ["1", "yes", "y", "on", "enabled", "TRUE-ish", ""])
    def test_nothing_else_counts(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        # A gate this consequential should require the operator to have typed the word.
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, value)
        assert live_trading_env_enabled() is False

    def test_the_variable_is_not_part_of_settings(self) -> None:
        # Deliberately outside the Settings model so it cannot be set by editing a config
        # file that something else copies.
        assert not any(
            LIVE_TRADING_ENV_VAR.lower() in field.lower() for field in Settings.model_fields
        )


class TestArmingCheck:
    def test_default_configuration_is_not_armed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        check = check_live_arming(settings_for())
        assert not check.armed
        assert len(check.blockers()) >= 4

    def test_every_gate_must_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, "true")
        check = check_live_arming(fully_armed_settings())
        assert check.armed
        assert check.blockers() == []

    def test_missing_env_flag_alone_blocks_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        check = check_live_arming(fully_armed_settings())
        assert not check.armed
        assert any(LIVE_TRADING_ENV_VAR in reason for reason in check.blockers())

    def test_paper_mode_alone_blocks_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, "true")
        check = check_live_arming(
            settings_for(mode=TradingMode.PAPER, testnet=False, credentials=True)
        )
        assert not check.armed
        assert any("MODE" in reason for reason in check.blockers())

    def test_missing_credentials_alone_blocks_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An "armed" session that cannot reach the venue looks like it is trading and is
        # not — a worse state than a refusal.
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, "true")
        check = check_live_arming(
            settings_for(mode=TradingMode.LIVE, armed_token=True, testnet=False, credentials=False)
        )
        assert not check.armed
        assert any("credentials" in reason for reason in check.blockers())

    def test_testnet_alone_blocks_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, "true")
        # Settings validation itself forbids live-on-testnet in production, so this is
        # constructed in a test environment to exercise the runner's own gate.
        check = LiveArmingCheck(
            env_flag=True,
            mode_is_live=True,
            confirmation_token=True,
            has_credentials=True,
            not_testnet=False,
        )
        assert not check.armed
        assert any("testnet" in reason for reason in check.blockers())

    def test_serialisation_lists_every_blocker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        described = check_live_arming(settings_for()).to_dict()
        assert described["armed"] is False
        assert isinstance(described["blockers"], list)
        assert described["env_flag"] is False


class TestRunnerRefusal:
    def _config(self, btc: Symbol, mode: TradingMode) -> RunnerConfig:
        return RunnerConfig(
            strategy_id="ema_cross",
            symbols=(btc,),
            timeframe=Timeframe.H1,
            mode=mode,
            starting_equity=Decimal("10000"),
            persist=False,
        )

    async def test_live_runner_refuses_without_the_env_flag(
        self, btc: Symbol, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        runner = TradingRunner(fully_armed_settings(), self._config(btc, TradingMode.LIVE))
        with pytest.raises(LiveTradingNotArmedError, match=LIVE_TRADING_ENV_VAR):
            await runner.start()

    async def test_the_refusal_lists_every_unmet_requirement(
        self, btc: Symbol, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # So the operator can fix them in one pass rather than one restart at a time.
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        runner = TradingRunner(
            settings_for(mode=TradingMode.LIVE, armed_token=True),
            self._config(btc, TradingMode.LIVE),
        )
        with pytest.raises(LiveTradingNotArmedError) as excinfo:
            await runner.start()
        blockers = excinfo.value.details["blockers"]
        assert len(blockers) >= 3

    def test_paper_runner_needs_no_gates(self, btc: Symbol) -> None:
        runner = TradingRunner(settings_for(), self._config(btc, TradingMode.PAPER))
        # No exception: the gate applies only to live mode.
        runner._assert_mode_permitted()

    def test_backtest_mode_is_rejected_by_the_config(self, btc: Symbol) -> None:
        with pytest.raises(ValidationError, match="backtest engine"):
            RunnerConfig(
                strategy_id="ema_cross",
                symbols=(btc,),
                timeframe=Timeframe.H1,
                mode=TradingMode.BACKTEST,
            )

    def test_a_session_needs_at_least_one_symbol(self) -> None:
        with pytest.raises(ValidationError, match="at least one symbol"):
            RunnerConfig(strategy_id="ema_cross", symbols=(), timeframe=Timeframe.H1)

    def test_snapshot_before_start_reports_arming(self, btc: Symbol) -> None:
        runner = TradingRunner(settings_for(), self._config(btc, TradingMode.PAPER))
        snapshot = runner.snapshot()
        assert snapshot["mode"] == "paper"
        assert snapshot["arming"]["armed"] is False


class TestDescribeArming:
    def test_reports_disabled_with_reasons(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        text = describe_arming(settings_for())
        assert "DISABLED" in text
        assert LIVE_TRADING_ENV_VAR in text

    def test_reports_armed_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, "true")
        text = describe_arming(fully_armed_settings())
        assert "ARMED" in text
        assert "real orders" in text


class TestDefaultsAreSafe:
    def test_shipped_defaults_are_paper_and_testnet(self) -> None:
        settings = Settings(_env_file=None, env=Environment.TEST)
        assert settings.trading.mode is TradingMode.PAPER
        assert settings.exchange.testnet is True
        assert settings.is_live is False

    def test_live_mode_cannot_be_set_without_the_token(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError, match="LIVE_CONFIRMATION"):
            TradingSettings(mode=TradingMode.LIVE)

    def test_env_example_does_not_arm_anything(self) -> None:
        """The shipped example must not arm live trading, in any of its three ways.

        Only *active* assignments count. The file documents the live block in full so an
        operator can see exactly what arming costs, and every line of it is commented out
        - a commented `QF_TRADING__LIVE_CONFIRMATION=<token>` is documentation, not a
        setting, and reading it as one would fail this test on the safest possible file.
        """
        from pathlib import Path

        example = Path(__file__).resolve().parents[2] / ".env.example"
        active: dict[str, str] = {}
        for raw in example.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            # Trailing `# ...` comments are used throughout the file to annotate values.
            active[key.strip()] = value.split("#")[0].strip()

        assert active.get("QF_TRADING__MODE") == "paper"
        assert active.get("QF_EXCHANGE__ENV") == "demo"
        assert active.get("QF_TRADING__LIVE_CONFIRMATION", "") != LIVE_CONFIRMATION_TOKEN
        # No credential ships filled in.
        for key, value in active.items():
            if key.endswith(("API_KEY", "API_SECRET", "BOT_TOKEN", "SECRET_KEY")):
                assert value == "", f"{key} ships with a value"
