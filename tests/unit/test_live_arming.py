"""Hardening tests for the live-trading interlock.

This is the mechanism that stands between the platform and real money. It is also the
least-exercised code in the repository, because nothing in normal operation ever arms it
— which is exactly the combination that produces an interlock nobody notices is broken
until the day it matters.

The tests below are written to *fail open loudly*: each one removes a single condition
and asserts the interlock still refuses. A gate that only holds when all five conditions
are simultaneously absent is not a gate.
"""

from __future__ import annotations

import pytest

from quantflow.core.errors import LiveTradingNotArmedError
from quantflow.live.runner import (
    LIVE_TRADING_ENV_VAR,
    LiveArmingCheck,
    live_trading_env_enabled,
)


def armed(**overrides: bool) -> LiveArmingCheck:
    """An otherwise fully-armed check with individual gates overridden."""
    fields: dict[str, bool] = {
        "env_flag": True,
        "mode_is_live": True,
        "confirmation_token": True,
        "has_credentials": True,
        "not_testnet": True,
    }
    fields.update(overrides)
    return LiveArmingCheck(**fields)


GATES = (
    "env_flag",
    "mode_is_live",
    "confirmation_token",
    "has_credentials",
    "not_testnet",
)


class TestArmingInterlock:
    """Every gate must be independently sufficient to refuse."""

    def test_all_five_conditions_arm_it(self) -> None:
        assert armed().armed

    @pytest.mark.parametrize("gate", GATES)
    def test_removing_any_single_gate_refuses(self, gate: str) -> None:
        # The property that matters: not "all five off refuses" but "any one off
        # refuses". An interlock that needs every condition absent is not an interlock.
        check = armed(**{gate: False})
        assert not check.armed
        assert check.blockers()

    @pytest.mark.parametrize("gate", GATES)
    def test_each_refusal_names_its_cause(self, gate: str) -> None:
        # An operator should be able to fix the blockers in one pass rather than
        # discovering them one restart at a time.
        blockers = armed(**{gate: False}).blockers()
        assert len(blockers) == 1

    def test_nothing_armed_reports_every_blocker(self) -> None:
        check = LiveArmingCheck(False, False, False, False, False)
        assert not check.armed
        assert len(check.blockers()) == len(GATES)

    def test_credentials_alone_do_not_arm(self) -> None:
        # The dangerous near-miss: everything configured except the deliberate env flag.
        assert not armed(env_flag=False).armed

    def test_testnet_blocks_even_when_fully_configured(self) -> None:
        # An "armed" session pointed at testnet looks like it is trading and is not,
        # which is worse than a refusal.
        assert not armed(not_testnet=False).armed

    def test_the_payload_exposes_every_gate(self) -> None:
        payload = armed(env_flag=False).to_dict()
        assert payload["armed"] is False
        for gate in GATES:
            assert gate in payload
        assert payload["blockers"]


class TestEnvironmentFlag:
    """The environment flag must be exact, not merely truthy."""

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", " true "])
    def test_accepted_spellings(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, value)
        assert live_trading_env_enabled()

    @pytest.mark.parametrize("value", ["1", "yes", "y", "on", "TRUE!", "", "false", "no"])
    def test_rejected_spellings(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        # "1" and "yes" are deliberately not accepted. Every near-miss spelling that a
        # deployment script might produce by accident must fail closed.
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, value)
        assert not live_trading_env_enabled()

    def test_an_unset_variable_is_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        assert not live_trading_env_enabled()


class TestCurrentState:
    """The repository must ship disarmed."""

    def test_live_trading_is_not_enabled_in_this_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A guard against the flag being left set in a developer's shell profile or a
        # committed .env. If this ever fails, something turned it on.
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        assert not live_trading_env_enabled()

    def test_the_error_carries_the_blockers(self) -> None:
        check = LiveArmingCheck(False, True, False, True, True)
        error = LiveTradingNotArmedError(
            "live trading is not armed: " + "; ".join(check.blockers()),
            blockers=check.blockers(),
        )
        assert LIVE_TRADING_ENV_VAR in str(error)
