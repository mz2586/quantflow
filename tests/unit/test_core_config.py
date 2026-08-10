"""Settings loading, precedence and production guardrails."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from quantflow.core.config import (
    LIVE_CONFIRMATION_TOKEN,
    AISettings,
    DatabaseSettings,
    Environment,
    NotificationSettings,
    RedisSettings,
    RiskSettings,
    Settings,
    Severity,
    TradingMode,
    TradingSettings,
    get_settings,
    reset_settings_cache,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestNestedEnvironmentLoading:
    def test_nested_delimiter_populates_sections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QF_DATABASE__HOST", "db.internal")
        monkeypatch.setenv("QF_DATABASE__PORT", "6543")
        monkeypatch.setenv("QF_REDIS__DB", "7")
        settings = _settings()
        assert settings.database.host == "db.internal"
        assert settings.database.port == 6543
        assert settings.redis.db == 7

    def test_constructor_kwargs_beat_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QF_API_PORT", "9999")
        assert _settings(api_port=8123).api_port == 8123

    def test_cors_origins_accepts_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QF_CORS_ORIGINS", "http://a.test, http://b.test ,")
        assert _settings().cors_origins == ("http://a.test", "http://b.test")

    def test_unknown_variables_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QF_TOTALLY_UNKNOWN", "x")
        assert _settings().app_name == "QuantFlow"

    def test_settings_are_immutable(self) -> None:
        settings = _settings()
        with pytest.raises(PydanticValidationError):
            settings.api_port = 1234  # type: ignore[misc]


class TestSecretHandling:
    def test_password_is_not_in_repr(self) -> None:
        database = DatabaseSettings(password="hunter2")  # type: ignore[arg-type]
        assert "hunter2" not in repr(database)

    def test_safe_dsn_redacts_password(self) -> None:
        database = DatabaseSettings(password="hunter2")  # type: ignore[arg-type]
        assert "hunter2" not in database.safe_dsn
        assert "***" in database.safe_dsn
        assert "hunter2" in database.async_dsn

    def test_dsn_drivers(self) -> None:
        database = DatabaseSettings()
        assert database.async_dsn.startswith("postgresql+asyncpg://")
        assert database.sync_dsn.startswith("postgresql+psycopg://")

    def test_redis_safe_url_redacts_credentials(self) -> None:
        redis = RedisSettings(password="s3cret")  # type: ignore[arg-type]
        assert "s3cret" not in redis.safe_url
        assert "s3cret" in redis.url

    def test_redis_url_without_password(self) -> None:
        assert RedisSettings(host="h", port=1, db=2).url == "redis://h:1/2"


class TestBlankSecrets:
    """A blank `.env` placeholder must read as *absent*, not as an empty credential.

    Otherwise the system believes it has API keys, signs requests with an empty string,
    and fails at the venue with an error that points nowhere near the real cause.
    """

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_exchange_credentials_are_absent(self, blank: str) -> None:
        from quantflow.core.config import ExchangeSettings

        settings = ExchangeSettings(api_key=blank, api_secret=blank)  # type: ignore[arg-type]
        assert settings.api_key is None
        assert settings.api_secret is None
        assert not settings.has_credentials

    def test_real_credentials_are_present(self) -> None:
        from quantflow.core.config import ExchangeSettings

        settings = ExchangeSettings(api_key="abc", api_secret="def")  # type: ignore[arg-type]
        assert settings.has_credentials

    def test_blank_ai_key_disables_the_provider(self) -> None:
        # A blank key with provider='anthropic' would otherwise pass validation and then
        # fail on the first API call.
        with pytest.raises(PydanticValidationError, match="anthropic_api_key"):
            AISettings(provider="anthropic", anthropic_api_key="")  # type: ignore[arg-type]

    def test_blank_live_confirmation_does_not_arm(self) -> None:
        with pytest.raises(PydanticValidationError, match="LIVE_CONFIRMATION"):
            TradingSettings(mode=TradingMode.LIVE, live_confirmation="  ")  # type: ignore[arg-type]

    def test_blank_env_var_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QF_EXCHANGE__API_KEY", "")
        monkeypatch.setenv("QF_EXCHANGE__API_SECRET", "")
        assert not _settings().exchange.has_credentials


class TestMarketDataRouting:
    """Bybit's testnet carries thin history and synthetic prices.

    Backtesting or warming up against it produces results that mean nothing, so public
    data is read from production by default even when trading on testnet.
    """

    def test_testnet_reads_production_market_data_by_default(self) -> None:
        from quantflow.core.config import ExchangeSettings

        assert ExchangeSettings(testnet=True).use_production_market_data

    def test_production_needs_no_redirect(self) -> None:
        from quantflow.core.config import ExchangeSettings

        assert not ExchangeSettings(testnet=False).use_production_market_data

    def test_redirect_can_be_disabled(self) -> None:
        from quantflow.core.config import ExchangeSettings

        settings = ExchangeSettings(testnet=True, market_data_from_production=False)
        assert not settings.use_production_market_data


class TestLiveTradingArming:
    def test_live_mode_without_token_is_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="LIVE_CONFIRMATION"):
            TradingSettings(mode=TradingMode.LIVE)

    def test_live_mode_with_wrong_token_is_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            TradingSettings(mode=TradingMode.LIVE, live_confirmation="nope")  # type: ignore[arg-type]

    def test_live_mode_with_correct_token_arms(self) -> None:
        trading = TradingSettings(
            mode=TradingMode.LIVE,
            live_confirmation=LIVE_CONFIRMATION_TOKEN,  # type: ignore[arg-type]
        )
        assert trading.is_live_armed

    def test_paper_mode_is_never_armed(self) -> None:
        trading = TradingSettings(
            mode=TradingMode.PAPER,
            live_confirmation=LIVE_CONFIRMATION_TOKEN,  # type: ignore[arg-type]
        )
        assert not trading.is_live_armed

    def test_base_currency_is_upper_cased(self) -> None:
        assert TradingSettings(base_currency="usdt").base_currency == "USDT"


class TestRiskCoherence:
    def test_defaults_are_coherent(self) -> None:
        assert RiskSettings().require_stop_loss is True

    def test_position_cap_cannot_exceed_total_exposure(self) -> None:
        with pytest.raises(PydanticValidationError, match="max_total_exposure_pct"):
            RiskSettings(max_position_pct=Decimal("0.5"), max_total_exposure_pct=Decimal("0.2"))

    def test_default_stop_cannot_exceed_max_stop(self) -> None:
        with pytest.raises(PydanticValidationError, match="max_stop_loss_pct"):
            RiskSettings(default_stop_loss_pct=Decimal("0.3"), max_stop_loss_pct=Decimal("0.1"))

    def test_daily_loss_cannot_exceed_max_drawdown(self) -> None:
        # The limits form a chain: daily <= weekly <= drawdown. Raising the weekly cap
        # alongside the daily one isolates the drawdown check being asserted here.
        with pytest.raises(PydanticValidationError, match="max_drawdown_pct"):
            RiskSettings(
                max_daily_loss_pct=Decimal("0.2"),
                max_weekly_loss_pct=Decimal("0.2"),
                max_drawdown_pct=Decimal("0.1"),
            )

    def test_daily_loss_cannot_exceed_weekly_loss(self) -> None:
        with pytest.raises(PydanticValidationError, match="max_weekly_loss_pct"):
            RiskSettings(max_daily_loss_pct=Decimal("0.2"), max_weekly_loss_pct=Decimal("0.1"))

    def test_notional_bounds_must_be_ordered(self) -> None:
        with pytest.raises(PydanticValidationError, match="min_order_notional"):
            RiskSettings(min_order_notional=Decimal("100"), max_order_notional=Decimal("50"))

    @pytest.mark.parametrize("value", [Decimal("0"), Decimal("-0.1"), Decimal("1.5")])
    def test_fractions_are_bounded(self, value: Decimal) -> None:
        with pytest.raises(PydanticValidationError):
            RiskSettings(max_position_pct=value)


class TestProductionGuardrails:
    def _prod_kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "env": Environment.PRODUCTION,
            "debug": False,
            "api_key": "k" * 32,
            "secret_key": "s" * 32,
            "database": DatabaseSettings(password="a-real-password"),  # type: ignore[arg-type]
        }
        base.update(overrides)
        return base

    def test_valid_production_config_passes(self) -> None:
        settings = _settings(**self._prod_kwargs())
        assert settings.env is Environment.PRODUCTION

    def test_production_forces_json_logging(self) -> None:
        settings = _settings(**self._prod_kwargs(log_format="console"))
        assert settings.log_format == "json"

    def test_production_rejects_debug(self) -> None:
        with pytest.raises(PydanticValidationError, match="debug must be false"):
            _settings(**self._prod_kwargs(debug=True))

    def test_production_requires_api_key(self) -> None:
        with pytest.raises(PydanticValidationError, match="api_key is required"):
            _settings(**self._prod_kwargs(api_key=None))

    def test_production_rejects_default_database_password(self) -> None:
        with pytest.raises(PydanticValidationError, match=r"database.password"):
            _settings(**self._prod_kwargs(database=DatabaseSettings()))

    def test_live_cannot_run_against_testnet(self) -> None:
        from quantflow.core.config import ExchangeSettings

        with pytest.raises(PydanticValidationError, match="testnet"):
            _settings(
                **self._prod_kwargs(
                    trading=TradingSettings(
                        mode=TradingMode.LIVE,
                        live_confirmation=LIVE_CONFIRMATION_TOKEN,  # type: ignore[arg-type]
                    ),
                    exchange=ExchangeSettings(testnet=True),
                )
            )

    def test_development_skips_production_checks(self) -> None:
        assert _settings(env=Environment.DEVELOPMENT, debug=True).debug is True


class TestAISettings:
    def test_anthropic_requires_key(self) -> None:
        with pytest.raises(PydanticValidationError, match="anthropic_api_key"):
            AISettings(provider="anthropic")

    def test_llm_enabled_only_with_key(self) -> None:
        assert not AISettings().llm_enabled
        assert AISettings(provider="anthropic", anthropic_api_key="k").llm_enabled  # type: ignore[arg-type]

    def test_news_provider_requires_key(self) -> None:
        with pytest.raises(PydanticValidationError, match="news_api_key"):
            AISettings(news_provider="cryptopanic")


class TestNotificationSettings:
    def test_telegram_requires_token_and_chat(self) -> None:
        with pytest.raises(PydanticValidationError, match="telegram_bot_token"):
            NotificationSettings(telegram_enabled=True)

    def test_telegram_enabled_with_both(self) -> None:
        settings = NotificationSettings(
            telegram_enabled=True,
            telegram_bot_token="t",  # type: ignore[arg-type]
            telegram_chat_id="1",
        )
        assert settings.telegram_enabled

    def test_severity_ordering(self) -> None:
        assert Severity.DEBUG.rank < Severity.INFO.rank < Severity.WARNING.rank
        assert Severity.WARNING.rank < Severity.CRITICAL.rank


class TestSettingsSingleton:
    def test_get_settings_is_cached(self) -> None:
        reset_settings_cache()
        assert get_settings() is get_settings()

    def test_reset_clears_cache(self) -> None:
        first = get_settings()
        reset_settings_cache()
        assert get_settings() is not first


def test_environment_production_like() -> None:
    assert Environment.PRODUCTION.is_production_like
    assert Environment.STAGING.is_production_like
    assert not Environment.DEVELOPMENT.is_production_like
    assert not Environment.TEST.is_production_like
