"""The account panel must name the environment it is actually connected to.

``network`` was a two-way flag — ``testnet`` or ``mainnet`` — written before the DEMO
environment existed. Demo is not testnet, so the flag was false and a demo session was
labelled **mainnet** on the dashboard, next to a real balance and four real open positions.

On a trading screen that is the single worst field to get wrong: it is the one an operator
reads to decide whether the money is real. Nothing was actually at risk here, and that is
the point — the label offered no way to tell.

The live rotation is the full registry: every registered strategy competes, minus the two
that cannot be members of themselves — ``orchestrator`` and the ``buy_and_hold`` benchmark.
That is asserted here so a future change cannot silently shrink the live pool.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from quantflow.core.config import ExchangeEnv, ExchangeSettings, MarketType
from quantflow.exchange.bybit.rest import BybitGateway
from quantflow.orchestrator.strategy import DEFAULT_EXCLUDED, StrategyOrchestrator
from quantflow.strategy.registry import load_builtin_strategies


def gateway_for(env: ExchangeEnv) -> BybitGateway:
    return BybitGateway(
        ExchangeSettings(
            name="bybit",
            api_key=SecretStr("k" * 18),
            api_secret=SecretStr("s" * 36),
            demo_api_key=SecretStr("d" * 18),
            demo_api_secret=SecretStr("e" * 36),
            env=env,
            market_type=MarketType.FUTURE,
        )
    )


class TestNetworkLabel:
    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            (ExchangeEnv.DEMO, "demo"),
            (ExchangeEnv.TESTNET, "testnet"),
            (ExchangeEnv.MAINNET, "mainnet"),
        ],
    )
    def test_gateway_reports_its_own_environment(self, env: ExchangeEnv, expected: str) -> None:
        assert gateway_for(env).network == expected

    def test_demo_is_never_reported_as_mainnet(self) -> None:
        """The specific defect: a demo session read 'mainnet' on the dashboard."""
        assert gateway_for(ExchangeEnv.DEMO).network != "mainnet"


class TestLiveRotationIsTheFullRegistry:
    def test_every_registered_strategy_competes(self) -> None:
        """No silent shrinking: the pool is the registry minus the two non-members."""
        registered = set(load_builtin_strategies().names())
        expected = registered - DEFAULT_EXCLUDED

        members = {member.strategy_id for member in StrategyOrchestrator().members}

        assert members == expected

    def test_only_the_benchmark_and_itself_are_excluded(self) -> None:
        """buy_and_hold is a yardstick, and an orchestrator of orchestrators recurses."""
        assert set(DEFAULT_EXCLUDED) == {"buy_and_hold", "orchestrator"}

    def test_the_newest_strategies_are_included(self) -> None:
        """The nine added in commit 1c8641f trade alongside the rest."""
        members = {member.strategy_id for member in StrategyOrchestrator().members}

        assert {"adx_trend", "regime_adaptive", "vwap_momentum"} <= members
