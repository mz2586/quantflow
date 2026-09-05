"""Stage B: environment resolution must never send a non-mainnet session to production."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from quantflow.core.config import EXCHANGE_HOSTS, ExchangeEnv, ExchangeSettings


def settings(**overrides: object) -> ExchangeSettings:
    base: dict[str, object] = {
        "api_key": "main-key",
        "api_secret": "main-secret",
        "demo_api_key": "demo-key",
        "demo_api_secret": "demo-secret",
    }
    base.update(overrides)
    return ExchangeSettings(**base)  # type: ignore[arg-type]


class TestEndpointResolution:
    @pytest.mark.parametrize(
        ("env", "host"),
        [
            (ExchangeEnv.MAINNET, "https://api.bybit.com"),
            (ExchangeEnv.TESTNET, "https://api-testnet.bybit.com"),
            (ExchangeEnv.DEMO, "https://api-demo.bybit.com"),
        ],
    )
    def test_each_env_maps_to_its_own_host(self, env: ExchangeEnv, host: str) -> None:
        assert settings(env=env).endpoint == host

    def test_env_takes_precedence_over_the_legacy_testnet_flag(self) -> None:
        """`testnet=False` must not drag a demo session onto production."""
        config = settings(env=ExchangeEnv.DEMO, testnet=False)
        assert config.resolved_env is ExchangeEnv.DEMO
        assert config.endpoint == EXCHANGE_HOSTS[ExchangeEnv.DEMO]

    def test_the_legacy_flag_still_decides_when_env_is_unset(self) -> None:
        assert settings(env=None, testnet=True).resolved_env is ExchangeEnv.TESTNET
        assert settings(env=None, testnet=False).resolved_env is ExchangeEnv.MAINNET


class TestCredentialSelection:
    def test_demo_uses_the_demo_pair(self) -> None:
        config = settings(env=ExchangeEnv.DEMO)
        assert config.active_api_key is not None
        assert config.active_api_key.get_secret_value() == "demo-key"
        assert config.active_api_secret is not None
        assert config.active_api_secret.get_secret_value() == "demo-secret"

    def test_mainnet_uses_the_mainnet_pair(self) -> None:
        config = settings(env=ExchangeEnv.MAINNET)
        assert config.active_api_key is not None
        assert config.active_api_key.get_secret_value() == "main-key"

    def test_a_demo_session_without_demo_credentials_has_none(self) -> None:
        """It must not silently fall back to the mainnet key."""
        config = ExchangeSettings(
            env=ExchangeEnv.DEMO, api_key=SecretStr("main-key"), api_secret=SecretStr("main-secret")
        )
        assert config.active_api_key is None
        assert config.has_credentials is False


class TestMainnetRefusal:
    def test_demo_passes_the_assertion(self) -> None:
        settings(env=ExchangeEnv.DEMO).assert_not_mainnet()

    def test_testnet_passes_the_assertion(self) -> None:
        settings(env=ExchangeEnv.TESTNET).assert_not_mainnet()

    def test_mainnet_is_refused(self) -> None:
        """The harness must abort rather than trade real money."""
        with pytest.raises(ValueError, match="mainnet"):
            settings(env=ExchangeEnv.MAINNET).assert_not_mainnet()

    def test_the_legacy_flag_alone_can_still_trip_the_refusal(self) -> None:
        with pytest.raises(ValueError, match="mainnet"):
            settings(env=None, testnet=False).assert_not_mainnet()


class TestGatewayHonoursTheEnv:
    def test_a_demo_gateway_is_not_pointed_at_production(self) -> None:
        from quantflow.exchange.bybit.rest import BybitGateway

        gateway = BybitGateway(settings(env=ExchangeEnv.DEMO))
        resolved = str(gateway._client.urls.get("api", ""))
        assert "api.bybit.com" not in resolved or "api-demo" in resolved

    def test_a_demo_gateway_carries_the_demo_key(self) -> None:
        from quantflow.exchange.bybit.rest import BybitGateway

        gateway = BybitGateway(settings(env=ExchangeEnv.DEMO))
        assert gateway._client.apiKey == "demo-key"

    def test_public_market_data_is_read_from_production_without_credentials(self) -> None:
        """Demo books are thin; public reads use production and must carry no key."""
        from quantflow.exchange.bybit.rest import BybitGateway

        gateway = BybitGateway(settings(env=ExchangeEnv.DEMO))
        assert gateway._data_client is not gateway._client
        assert not gateway._data_client.apiKey
