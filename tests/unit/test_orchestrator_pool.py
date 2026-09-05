"""The orchestrator's candidate pool must be restrictable through its params.

The live runner builds its strategy from the registry (``registry.create("orchestrator")``,
which calls ``cls(params)``), so the existing keyword-only ``members`` argument is
unreachable from a running session. Restricting the live rotation to a chosen few therefore
has to travel as a parameter.

Rejecting an unknown name matters more than it looks: a typo that silently fell back to
"every strategy" would put twenty retired strategies back into a live rotation while the
log claimed two.
"""

from __future__ import annotations

import pytest

from quantflow.core.errors import ValidationError
from quantflow.orchestrator.strategy import StrategyOrchestrator
from quantflow.strategy.registry import load_builtin_strategies

WINNERS = ["dual_thrust", "ema_cross"]


class TestPoolRestriction:
    def test_pool_limits_members_to_the_named_strategies(self) -> None:
        orchestrator = StrategyOrchestrator({"pool": WINNERS})

        assert sorted(member.strategy_id for member in orchestrator.members) == sorted(WINNERS)

    def test_pool_reaches_the_orchestrator_through_the_registry(self) -> None:
        """The path the live runner actually uses."""
        registry = load_builtin_strategies()

        orchestrator = registry.create("orchestrator", {"pool": WINNERS})
        assert isinstance(orchestrator, StrategyOrchestrator)

        assert sorted(member.strategy_id for member in orchestrator.members) == sorted(WINNERS)

    def test_no_pool_keeps_the_full_roster(self) -> None:
        """Absent a pool, nothing changes: every registered strategy still competes."""
        full = StrategyOrchestrator()
        restricted = StrategyOrchestrator({"pool": WINNERS})

        assert len(full.members) > len(restricted.members)

    def test_unknown_strategy_name_is_rejected(self) -> None:
        """A typo must fail loudly, not silently restore the whole roster."""
        with pytest.raises(ValidationError):
            StrategyOrchestrator({"pool": ["dual_thrust", "no_such_strategy"]})

    def test_empty_pool_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StrategyOrchestrator({"pool": []})

    def test_orchestrator_cannot_include_itself(self) -> None:
        """Recursion here would be an infinite regress, not a clever meta-strategy."""
        with pytest.raises(ValidationError):
            StrategyOrchestrator({"pool": ["orchestrator"]})
